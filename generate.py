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

	inputs = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt = True,
		return_tensors = "pt",
		# return_dict = True,
		reasoning_effort = reasoning_effort,
	).to(model.device)

	t_out = model.generate(inputs, max_new_tokens = max_new_tokens, streamer = TextStreamer(tokenizer))

	outputs = t_out.tolist()

else:
	tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map="cuda",
		torch_dtype="auto",
	)

	formatted_chat = tokenizer.apply_chat_template(
		messages,
		tokenize=False)
	print(formatted_chat)

	input_ids = tokenizer.apply_chat_template(
		messages,
		return_tensors="pt",
		reasoning_effort = reasoning_effort
	).to(model.device)
	outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, streamer = TextStreamer(tokenizer))

print('################## Completed ###################')
print('################## Without special tokens ###################')
txts_raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(txts_raw)
print('################## With special tokens ###################')
txts = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(txts)

