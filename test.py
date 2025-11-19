import os, sys
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, Mxfp4Config

model_id = os.getenv('MODEL_ID', '/home/LLM_models/unsloth/gpt-oss-120b-unsloth-bnb-4bit')
# model_id = os.getenv('MODEL_ID', '/home/LLM_models/gpt-oss-120b')

max_new_tokens = int(os.getenv('MAX_SEQ_LEN', '65536'))
reasoning_effort = os.getenv('REASONING', 'high')

# messages = [{"role": "user", "content": "Explain quantum mechanics clearly and concisely."}]
messages = [{"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}]

if 'unsloth' in model_id:
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name = model_id,
		dtype = None, # None for auto detection
		max_seq_length = max_new_tokens,
		device_map = "cuda",
		# load_in_4bit = True, # False for LoRA 16bit
		offload_embedding = False, # Reduces VRAM by 1GB
		full_finetuning = False
	)