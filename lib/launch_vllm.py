#!/usr/bin/env python3

import os, sys, time, threading, signal
import subprocess, atexit, socket
import logging as LOG
from contextlib import closing

def wait_for_port(host: str, port: int, timeout: float = 600.0):
	start = time.time()
	while time.time() - start < timeout:
		try:
			with socket.create_connection((host, port), timeout=2):
				return
		except OSError:
			time.sleep(1)
	raise TimeoutError(f"vLLM server at {host}:{port} did not come up within {timeout}s")

def pump(stream):
	try:
		buf = b''
		while True:
			chunk = stream.read(1)
			if not chunk:
				break
			if os.is_vllm_server_ready:
				continue
			buf += chunk
			if chunk in b'\r\n':
				sys.stderr.buffer.write(buf)
				sys.stderr.buffer.flush()
				buf = b''
	finally:
		stream.close()

def launch_vllm_server(model_name: str, host="127.0.0.1", port=0,
					cuda_visible_devices="0", max_model_len=25000,
					tensor_parallel_size=1, gpu_memory_utilization=0.85):
	os.is_vllm_server_ready = False
	env = os.environ.copy()
	env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

	# Find a free port
	if port == 0:
		s = socket.socket()
		s.bind(('',0))
		port = s.getsockname()[1]
		s.close()

	cmd = [
		os.path.dirname(sys.executable)+"/trl", "vllm-serve",
		"--model", model_name,
		"--host", host,
		"--port", str(port),
		"--tensor-parallel-size", str(tensor_parallel_size),
		"--gpu-memory-utilization", str(gpu_memory_utilization),
		"--max-model-len", str(max_model_len),
		"--log-level", "error",
		"--trust-remote-code",
		# "--vllm-model-impl", "transformers",
	]

	LOG.info(f"Launching vLLM server: {' '.join(cmd)}")
	proc = subprocess.Popen(
		cmd,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=False,
	)

	threading.Thread(target=pump, args=(proc.stdout,), daemon=True).start()
	threading.Thread(target=pump, args=(proc.stderr,), daemon=True).start()

	def _cleanup():
		if proc.poll() is None:
			os.killpg(proc.pid, signal.SIGKILL)

	atexit.register(_cleanup)
	wait_for_port(host, port, timeout=600.0)
	os.is_vllm_server_ready = True
	proc.port_num = port
	return proc