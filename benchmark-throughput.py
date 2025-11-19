#!/usr/bin/env python3
"""
benchmark_multi_engine_option_c.py

Benchmark token-per-second (TPS) throughput for **vLLM** *or* **Ollama**.

This version implements "Option C":

* For *streaming* requests, it counts generated tokens incrementally
as chunks arrive, so even if we stop reading before the backend
sends its final usage / counters, we still get a non-zero estimate
of completion tokens.

* We **do not** use aiohttp's total timeout for requests; instead we
use `max_time` only for:
	- passing down to the backend (vLLM → max_time,
	Ollama → options.max_predict_time)
	- limiting how long we read the streaming response loop.

That way, we never lose partial generations due to client-side
timeouts raising exceptions.
"""

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import aiohttp
from tqdm.asyncio import tqdm_asyncio


# --------------------------------------------------------------------------- #
#  Token counting helper
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str, model: Optional[str] = None) -> int:
	"""
	Estimate the number of tokens in `text`.

	If `tiktoken` is installed and a model name is supplied, we use a
	model-appropriate tokenizer. Otherwise we fall back to a very
	rough approximation based on whitespace.

	This is only used when the backend does NOT give us explicit
	token counts (e.g. when we stop reading a stream before the
	final usage / eval counters).
	"""
	if not text:
		return 0

	try:
		import tiktoken  # type: ignore

		encoding = None
		if model:
			try:
				encoding = tiktoken.encoding_for_model(model)
			except Exception:
				# Fall back to a generic encoding if model is unknown
				pass
		if encoding is None:
			encoding = tiktoken.get_encoding("cl100k_base")
		return len(encoding.encode(text))
	except Exception:
		# Fallback: simple heuristic (tokens ≈ words)
		return len(text.split())


# --------------------------------------------------------------------------- #
#  URL / payload builders for the two back-ends
# --------------------------------------------------------------------------- #
def vllm_url(host: str, port: int) -> str:
	return f"http://{host}:{port}/v1/completions"


def ollama_url(host: str, port: int) -> str:
	# Ollama uses `/api/generate` for plain completions.
	return f"http://{host}:{port}/api/generate"


def vllm_payload(
	prompt: str,
	max_tokens: int,
	model: str,
	stream: bool = False,
	max_time: float | None = None,
) -> Dict[str, Any]:
	"""OpenAI-compatible payload for vLLM."""
	payload: Dict[str, Any] = {
		"model": model,
		"prompt": prompt,
		"max_tokens": max_tokens,
		"temperature": 0.0,
		"top_p": 1.0,
		"stream": stream,
		"n": 1,
		"logprobs": None,
	}
	if max_time is not None:
		payload["max_time"] = max_time
	return payload


def ollama_payload(
	prompt: str,
	max_tokens: int,
	model: str,
	stream: bool = False,
	max_time: float | None = None,
) -> Dict[str, Any]:
	"""
	Ollama payload.

	Ollama expects:
		{
			"model": "...",
			"prompt": "...",
			"stream": true|false,
			"options": {"num_predict": max_tokens}
		}
	"""
	options: Dict[str, Any] = {
		"num_predict": max_tokens,
		"temperature": 0.0,
		"top_p": 1.0,
	}
	if max_time is not None:
		options["max_predict_time"] = max_time + 10
	return {
		"model": model,
		"prompt": prompt,
		"stream": stream,
		"options": options,
	}


# --------------------------------------------------------------------------- #
#  Response parsers – both return a dict with a `completion_tokens` int
# --------------------------------------------------------------------------- #
def parse_vllm_usage(resp_json: Dict[str, Any]) -> int:
	"""
	vLLM always includes a `usage` dict:
		{"completion_tokens": X, "prompt_tokens": Y, "total_tokens": Z}
	"""
	usage = resp_json.get("usage", {})
	return int(usage.get("completion_tokens", 0))


def parse_ollama_usage(resp_json: Dict[str, Any]) -> int:
	"""
	Ollama returns:
		{
			"model": "...",
			"created_at": "...",
			"response": "...",
			"done": true,
			"eval_count": total_generated_tokens,
			"prompt_eval_count": tokens_in_prompt,
			...
		}

	Completion tokens = eval_count - prompt_eval_count.
	"""
	total = int(resp_json.get("eval_count", 0))
	prompt = int(resp_json.get("prompt_eval_count", 0))
	return max(total - prompt, 0)


# --------------------------------------------------------------------------- #
#  Async request helpers – one pair for each engine
# --------------------------------------------------------------------------- #
async def _vllm_completion(
	session: aiohttp.ClientSession, url: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
	async with session.post(url, json=payload) as resp:
		if resp.status != 200:
			txt = await resp.text()
			raise RuntimeError(f"vLLM error {resp.status}: {txt}")
		return await resp.json()


async def _vllm_completion_stream(
	session: aiohttp.ClientSession,
	url: str,
	payload: Dict[str, Any],
	*,
	model: str,
	max_time: float | None,
) -> Dict[str, Any]:
	"""
	Streaming for vLLM with robust SSE parsing.

	We parse Server-Sent Events of the form:

		data: { ...json... }

	Events are separated by a blank line (`\\n\\n`).

	We accumulate all generated text across events. If vLLM sends a
	final event with a `usage` block, we use that. If not, we estimate
	completion tokens from the accumulated text.
	"""
	async with session.post(url, json=payload) as resp:
		if resp.status != 200:
			txt = await resp.text()
			raise RuntimeError(f"vLLM error {resp.status}: {txt}")

		usage: Dict[str, Any] = {}
		generated_chunks: List[str] = []

		start = time.perf_counter()
		buffer = ""

		async for raw in resp.content:
			now = time.perf_counter()
			if max_time is not None and (now - start) > max_time:
				# Stop reading further – we'll estimate tokens from what we have.
				break

			buffer += raw.decode("utf-8")

			# Process complete SSE events separated by a blank line
			while "\n\n" in buffer:
				event, buffer = buffer.split("\n\n", 1)

				# Each event can have multiple lines; we care about data: lines
				for line in event.splitlines():
					line = line.strip()
					if not line.startswith("data:"):
						continue

					data_str = line[5:].strip()
					if not data_str or data_str == "[DONE]":
						# [DONE] just signals end of stream; ignore as content
						continue

					try:
						data = json.loads(data_str)
					except json.JSONDecodeError:
						continue

					# Accumulate generated text for token estimation
					choices = data.get("choices")
					if choices:
						choice0 = choices[0] or {}
						text_piece = (
							choice0.get("text")
							or choice0.get("delta", {}).get("content", "")
							or ""
						)
						if text_piece:
							generated_chunks.append(text_piece)

					# If vLLM includes usage, prefer that and we can stop
					if "usage" in data:
						usage = data["usage"]
						# We could keep reading, but for benchmarking there's
						# no need once we have final usage.
						break

				if usage:
					# Break outer `while "\n\n" in buffer` if usage found
					break

			if usage:
				# Break outer async for loop if usage found
				break

		# After the loop: if no usage from backend, estimate tokens
		if not usage:
			generated_text = "".join(generated_chunks)
			comp_tokens = estimate_tokens(generated_text, model=model)
			usage = {
				"completion_tokens": comp_tokens,
				"prompt_tokens": 0,          # unknown; doesn't affect TPS
				"total_tokens": comp_tokens,
			}

		return {"usage": usage}


async def _ollama_completion(
	session: aiohttp.ClientSession, url: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
	async with session.post(url, json=payload) as resp:
		if resp.status != 200:
			txt = await resp.text()
			raise RuntimeError(f"Ollama error {resp.status}: {txt}")
		return await resp.json()


async def _ollama_completion_stream(
	session: aiohttp.ClientSession,
	url: str,
	payload: Dict[str, Any],
	*,
	model: str,
	max_time: float | None,
) -> Dict[str, Any]:
	"""
	Ollama streams via newline-delimited JSON.

	Each line is a JSON object with a `response` field.
	The final line contains `"done": true` and also the token counters:
		"eval_count" and "prompt_eval_count".

	We now accumulate `response` text as it arrives, and if we stop
	reading before seeing `"done": true` (e.g. due to `max_time` on
	the client side), we estimate the completion token count from
	that partial text.
	"""
	async with session.post(url, json=payload) as resp:
		if resp.status != 200:
			txt = await resp.text()
			raise RuntimeError(f"Ollama error {resp.status}: {txt}")

		final_obj: Dict[str, Any] = {}
		generated_chunks: List[str] = []

		start = time.perf_counter()

		async for raw in resp.content:
			now = time.perf_counter()
			if max_time is not None and (now - start) > max_time:
				# Stop reading further – we'll estimate tokens from what we have.
				break

			line = raw.decode("utf-8").strip()
			if not line:
				continue

			try:
				data = json.loads(line)
			except json.JSONDecodeError:
				continue

			# Accumulate incremental text
			text_piece = data.get("response") or data.get("thinking", "")
			if text_piece:
				generated_chunks.append(text_piece)

			if data.get("done", False):
				final_obj = data
				break

		if final_obj:
			# Backend produced counters; use them.
			return final_obj

		# Otherwise, estimate from partial text.
		generated_text = "".join(generated_chunks)
		comp_tokens = estimate_tokens(generated_text, model=model)
		return {
			"model": payload.get("model"),
			"response": generated_text,
			"done": False,
			"eval_count": comp_tokens,
			"prompt_eval_count": 0,
		}


# --------------------------------------------------------------------------- #
#  Dispatcher that hides the engine differences from the benchmark core
# --------------------------------------------------------------------------- #
async def single_completion(
	session: aiohttp.ClientSession,
	engine: str,
	url: str,
	payload: Dict[str, Any],
	stream: bool,
	model: str,
	max_time: float | None,
) -> Dict[str, Any]:
	"""
	Returns a dict with at least:
		{"completion_tokens": int}

	For streaming requests, this uses incremental token estimation
	if the backend does not provide explicit usage counters (e.g. if
	we stop reading due to a local `max_time` limit).
	"""
	if engine == "vllm":
		if stream:
			resp = await _vllm_completion_stream(
				session,
				url,
				payload,
				model=model,
				max_time=max_time,
			)
		else:
			resp = await _vllm_completion(session, url, payload)
		comp_tokens = parse_vllm_usage(resp)
	elif engine == "ollama":
		if stream:
			resp = await _ollama_completion_stream(
				session,
				url,
				payload,
				model=model,
				max_time=max_time,
			)
		else:
			resp = await _ollama_completion(session, url, payload)
		comp_tokens = parse_ollama_usage(resp)
	else:
		raise ValueError(f"Unsupported engine: {engine}")

	return {"completion_tokens": comp_tokens, "raw": resp}


# --------------------------------------------------------------------------- #
#  Benchmark orchestrator (updated to support Option C streaming)
# --------------------------------------------------------------------------- #
async def run_benchmark(
	host: str,
	port: int,
	engine: str,
	model: str,
	prompt: str,
	max_tokens: int,
	total_requests: int,
	parallel: int,
	stream: bool,
	max_time: float | None = None,
) -> Dict[str, Any]:
	# Resolve URL and payload according to engine
	if engine == "vllm":
		url = vllm_url(host, port)
		payload = vllm_payload(
			prompt,
			max_tokens,
			model,
			stream=stream,
			max_time=max_time,
		)
	elif engine == "ollama":
		url = ollama_url(host, port)
		payload = ollama_payload(
			prompt,
			max_tokens,
			model,
			stream=stream,
			max_time=max_time,
		)
	else:
		raise ValueError(f"Engine must be 'vllm' or 'ollama', got {engine}")

	# IMPORTANT CHANGE FOR OPTION C:
	# We do NOT set aiohttp's total timeout from `max_time` – doing so
	# would raise exceptions before we can account for partial tokens.
	connector = aiohttp.TCPConnector(limit=parallel * 2)
	client_timeout = aiohttp.ClientTimeout(total=None)  # no global timeout

	async with aiohttp.ClientSession(
		connector=connector,
		timeout=client_timeout,
	) as session:
		semaphore = asyncio.Semaphore(parallel)

		async def _run_one(idx: int) -> Dict[str, Any]:
			start = time.perf_counter()
			try:
				result = await single_completion(
					session,
					engine,
					url,
					payload,
					stream=stream,
					model=model,
					max_time=max_time,
				)
				elapsed = time.perf_counter() - start
				return {
					"idx": idx,
					"elapsed": elapsed,
					"completion_tokens": result["completion_tokens"],
					"error": None,
				}
			except Exception as exc:  # pragma: no cover
				return {
					"idx": idx,
					"elapsed": time.perf_counter() - start,
					"completion_tokens": 0,
					"error": str(exc),
				}

		async def _bounded(idx: int):
			async with semaphore:
				return await _run_one(idx)

		tasks = [_bounded(i) for i in range(total_requests)]
		results = await tqdm_asyncio.gather(
			*tasks,
			desc="Requests",
			unit="req",
			leave=False,
		)

	# --------------------  Summarise  -------------------- #
	total_tokens = sum(r["completion_tokens"] for r in results)
	wall_time = max(r["elapsed"] for r in results)  # approximation of real wall-clock
	overall_tps = total_tokens / wall_time if wall_time > 0 else 0.0
	per_req = [r["elapsed"] for r in results if r["error"] is None]

	print("\n=== Benchmark Summary ===")
	print(f"Engine                 : {engine}")
	print(f"Host / port            : {host}:{port}")
	print(f"Model                  : {model}")
	print(f"Prompt (chars)         : {len(prompt)}")
	print(f"Max generated tokens   : {max_tokens}")
	print(f"Total requests sent    : {total_requests}")
	print(f"Parallelism            : {parallel}")
	print(f"Stream mode            : {stream}")
	print(f"Total completion tokens: {total_tokens}")
	print(f"Wall-clock time        : {wall_time:.2f} s")
	print(f"Max inference time (s) : {max_time if max_time is not None else 'unlimited'}")
	print(f"Overall TPS            : {overall_tps:.2f} tokens/s")
	if per_req:
		print(f"Mean latency per request: {sum(per_req)/len(per_req):.3f} s")
		print(
			"  (min/median/max)    : "
			f"{min(per_req):.3f}/"
			f"{sorted(per_req)[len(per_req)//2]:.3f}/"
			f"{max(per_req):.3f} s"
		)
	if any(r["error"] for r in results):
		print("\n⚠️  Some requests failed:")
		for r in results:
			if r["error"]:
				print(f"  - idx {r['idx']}: {r['error']}")

	return {
		"engine": engine,
		"host": host,
		"port": port,
		"model": model,
		"prompt": prompt,
		"max_tokens": max_tokens,
		"total_requests": total_requests,
		"parallel": parallel,
		"stream": stream,
		"max_time": max_time,
		"total_completion_tokens": total_tokens,
		"wall_clock_seconds": wall_time,
		"overall_tps": overall_tps,
		"per_request_seconds": per_req,
		"errors": [r for r in results if r["error"]],
	}


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Token-per-second benchmark for vLLM or Ollama (Option C streaming token counting)"
	)
	parser.add_argument(
		"--engine",
		choices=["vllm", "ollama"],
		default="vllm",
		help="Backend to test (default: vllm). Use 'ollama' for Ollama on port 11434.",
	)
	parser.add_argument("--host", default="localhost", help="Server hostname")
	parser.add_argument(
		"--port",
		type=int,
		help="Server port (default: 9000 for vLLM, 11434 for Ollama)",
	)
	parser.add_argument(
		"--model",
		default="vllm",  # for Ollama you can pass e.g. "llama2"
		help="Model name as registered in the backend",
	)
	parser.add_argument(
		"--prompt",
		default="Write a short poem about the sunrise.",
		help="Prompt that will be sent to the model",
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=128,
		help="Maximum number of tokens the model may generate per request",
	)
	parser.add_argument(
		"--requests",
		type=int,
		default=None,
		help="Total number of inference requests (default = parallel)",
	)
	parser.add_argument(
		"--parallel",
		type=int,
		default=8,
		help="Number of concurrent requests",
	)
	parser.add_argument(
		"--stream",
		action="store_true",
		help="Use streaming API (both vLLM and Ollama support it). "
		"Option C's partial token counting is only meaningful when streaming.",
	)
	parser.add_argument(
		"--max-time",
		type=float,
		default=None,
		help=(
			"Maximum wall-clock time (seconds) allowed for reading a single "
			"streamed response. If the generation does not finish within "
			"this limit, we stop reading further chunks but still estimate "
			"completion tokens from the partial text. The flag is also "
			"passed to the backend (vLLM → max_time, Ollama → "
			"options.max_predict_time)."
		),
	)
	parser.add_argument(
		"--output",
		default=None,
		help="Write full JSON result to this file",
	)
	return parser.parse_args()


async def main():
	args = parse_args()

	# Pick sensible defaults for the port if the user omitted it
	if args.port is None:
		args.port = 9000 if args.engine == "vllm" else 11434

	total_requests = args.requests if args.requests is not None else args.parallel

	result = await run_benchmark(
		host=args.host,
		port=args.port,
		engine=args.engine,
		model=args.model,
		prompt=args.prompt,
		max_tokens=args.max_tokens,
		total_requests=total_requests,
		parallel=args.parallel,
		stream=args.stream,
		max_time=args.max_time,
	)

	if args.output:
		with open(args.output, "w", encoding="utf-8") as fp:
			json.dump(result, fp, indent=2)
		print(f"\nFull result written to {args.output}")


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		print("\nBenchmark interrupted by user.")

