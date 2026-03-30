#!/usr/bin/env python3

import os, sys, time
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

def launch_vllm_server(model_name: str, host="127.0.0.1", port=8000,
					cuda_visible_devices="0", max_model_len=20000,
					tensor_parallel_size=1, gpu_memory_utilization=0.85):
	log_level = LOG.getLogger(__name__).getEffectiveLevel()
	env = os.environ.copy()
	env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

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
		stdout=sys.stdout if log_level==LOG.DEBUG else subprocess.DEVNULL,
		stderr=sys.stderr if log_level==LOG.DEBUG else subprocess.DEVNULL,
	)

	def _cleanup():
		if proc.poll() is None:
			proc.terminate()
			try:
				proc.wait(timeout=10)
			except subprocess.TimeoutExpired:
				proc.kill()

	atexit.register(_cleanup)
	wait_for_port(host, port, timeout=600.0)
	return proc