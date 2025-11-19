
## Environment setup

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


## Troubleshoot:
- Many codes will NOT run in IDE debugger unless env TORCHDYNAMO\_DISABLE is set. This is because both debugpy and torch dynamo uses frame hooking and they conflict with each other.
- unsloth 4-bit-quantized models cannot be loaded properly using transformers' `AutoModelForCausalLM.from_pretrained()`, must use unsloth's `FastLanguageModel.from_pretrained()` to load.

