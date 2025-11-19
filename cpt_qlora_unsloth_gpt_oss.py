#!/usr/bin/env python

import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID

from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
from unsloth import FastLanguageModel, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import TextStreamer

in_model = os.getenv("IN_MODEL", "/home/LLM_models/unsloth/gpt-oss-120b-unsloth-bnb-4bit")
max_seq_length = 2048

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }



## Main Start

model, tokenizer = FastLanguageModel.from_pretrained(
	model_name = in_model,
	max_seq_length = max_seq_length,
	device_map = "cuda:0",
	load_in_4bit = True, # False for LoRA 16bit
	offload_embedding = False, # Reduces VRAM by 1GB
)

# for reasoning_effort in ['low', 'medium', 'high']:
# 	messages = [
# 		{"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
# 	]
# 	inputs = tokenizer.apply_chat_template(
# 		messages,
# 		add_generation_prompt = True,
# 		return_tensors = "pt",
# 		return_dict = True,
# 		reasoning_effort = reasoning_effort, # **NEW!** Set reasoning effort to low, medium or high
# 	).to("cuda")

# 	_ = model.generate(**inputs, max_new_tokens = 64, streamer = TextStreamer(tokenizer))


# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
	model,
	r = 32,
	target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
					"gate_proj", "up_proj", "down_proj",],
	lora_alpha = 32,
	lora_dropout = 0, # Supports any, but = 0 is optimized
	bias = "none",    # Supports any, but = "none" is optimized
	# [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
	use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
	random_state = 3407,
	use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



# Train the model
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)

trainer = SFTTrainer(
	model = model,
	tokenizer = tokenizer,
	train_dataset = dataset,
	args = SFTConfig(
		per_device_train_batch_size = 1,
		gradient_accumulation_steps = 4,
		warmup_steps = 5,
		# num_train_epochs = 1, # Set this for 1 full training run.
		max_steps = 30,
		learning_rate = 2e-4,
		logging_steps = 1,
		optim = "adamw_8bit",
		weight_decay = 0.01,
		lr_scheduler_type = "linear",
		seed = 3407,
		output_dir = "outputs",
		report_to = "none", # Use this for WandB etc
	),
)

gpt_oss_kwargs = dict(instruction_part = "<|start|>user<|message|>", response_part="<|start|>assistant<|channel|>final<|message|>")
trainer = train_on_responses_only(
    trainer,
    **gpt_oss_kwargs,
)

trainer_stats = trainer.train()
model.save_pretrained("finetuned_model")
# model.save_pretrained_merged("finetuned_model", tokenizer, save_method="mxfp4")
