#!/bin/bash

if [ $# == 0 ]; then
	echo "Usage: $0 <model_path> <port-number=9000> (other_options ...)
This will run vLLM model on the specified port.
e.g., $0 /home/LLM_models/gpt-oss-20b 9000
Useful options:
	limit GPU memory usage: --gpu-memory-utilization 0.8
	Model context length (prompt and output, default: from model config): --max-model-len 65536
"
	exit
fi

#docker run --rm -it   --gpus "device=0" --ipc=host   -p 10120:8000   -e TIKTOKEN_ENCODINGS_BASE=/encodings   -e TIKTOKEN_RS_CACHE_DIR=/tiktoken_cache   -v /home/LLM_models/harmony_encodings:/encodings:ro   -v /home/LLM_models/tiktoken_cache:/tiktoken_cache   -v /home/LLM_models/gpt-oss-120b:/model   vllm/vllm-openai:latest-aarch64   --model /model   --dtype bfloat16   --tensor-parallel-size 1   --max-model-len 65536

model_base="`basename $1`"

GPU_OPT=all
if [ "$CUDA_VISIBLE_DEVICES" ]; then
	GPU_OPT="device=$CUDA_VISIBLE_DEVICES"
fi

docker run --rm -it \
  --gpus "$GPU_OPT" --ipc=host -p ${2:-9000}:8000 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e TIKTOKEN_ENCODINGS_BASE=/encodings \
  -e TIKTOKEN_RS_CACHE_DIR=/tiktoken_cache \
  -v /home/LLM_models/harmony_encodings:/encodings:ro \
  -v /home/LLM_models/tiktoken_cache:/tiktoken_cache:rw \
  -v ${1%/}:/$model_base \
  vllm/vllm-openai:latest-aarch64 \
  --model /$model_base \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-sleep-mode "${@:3}" \
  --trust-remote-code
