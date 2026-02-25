## LLM training and fine-tuning sample codes (tested working on GH200 NVL2 server)

This repository provides code samples for LLM generation, training/fine-tuning and GRPO, both with and without unsloth.

The GH200 server has Grace Hopper architecture that allows very fast data transfer between CPU memory and GPU memory. Thus, it is possible to use the entire 1.2TB (=(480+144)x2) unified memory for training/fine-tuning large GPT models at the expense of slower training speed (about 3-5 times slower than running purely on GPU, as compared to 50-100 times slower on PCIe GPU servers). This can be done using the RMM (RAPIDS Memory Manager) trick.

generate.py : generate text using a pre-trained model.

hf_download.sh : fast download a model from huggingface (you need to define your own `HF_TOKEN` in `.secret.sh` and accept the licenses of the models you want to download on huggingface website).

benchmark-throughput.py : run benchmarks on vLLM/Ollama instances.

start-vllm.sh : start vLLM server using a pre-trained model.

cpt_lora_gpt_oss.py : run continued pre-training for gpt-oss-* model with LoRA using transformers without unsloth.

cpt_qlora_unsloth_gpt_oss.py : run continued pre-training for gpt-oss-* model with QLoRA using unsloth.

train_rag_grpo_unsloth.py : train a RAG model with GRPO using unsloth.

train_rag_grpo.py : train a RAG model with GRPO using transformers without unsloth.

train_rag_grpo_rmm.py : train a RAG model with GRPO without unsloth using the RMM trick.

train_rag_grpo_rmm_torchcompile.py : train a RAG model with GRPO without unsloth using the RMM trick with torch.compile=True.

## Environment setup

```
!pip install --upgrade -qqq uv
try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
except: get_numpy = "numpy"
!uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes "transformers>=4.55.3" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers
!uv pip install --no-deps trl==0.22.2
```

## Troubleshoot:
- Many codes will NOT run in IDE debugger unless env TORCHDYNAMO\_DISABLE is set. This is because both debugpy and torch dynamo uses frame hooking and they conflict with each other.
- unsloth 4-bit-quantized models cannot be loaded properly using transformers' `AutoModelForCausalLM.from_pretrained()`, must use unsloth's `FastLanguageModel.from_pretrained()` to load.
- currently, RMM trick with `torch.compile=True` will alter the NVidia driver behavior afterwards which slows down certain CUDA processes (e.g., vLLM will take much longer to load) until a reboot. The bug might be solved soon.
- For the CUDA process with RMM trick applied, `nvidia-smi` and `top` will not correctly show its GPU/CPU memory usage because it changes the default CUDA allocator to use unified memory allocator at kernel driver level. You can use `free` to get a rough estimate of its memory usage.

