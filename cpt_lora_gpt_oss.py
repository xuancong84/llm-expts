import os, math, random
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	DataCollatorForLanguageModeling,
	TrainingArguments,
	Trainer,
	set_seed,
	Mxfp4Config,
)
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint

# -----------------------------
# Config (edit or pass via env)
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "/home/LLM_models/gpt-oss-20b")
OPTIM = os.getenv("OPTIM", "adamw_torch") # use "adamw_8bit" to save GPU memory
DATASET_NAME = os.getenv("DATASET_NAME", "cerebras/SlimPajama-627B")  # or 'tiiuae/falcon-refinedweb'
DATASET_STREAMING = os.getenv("DATASET_STREAMING", "true").lower() == "true"
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")            # SlimPajama has 'train'
SEQ_LEN = int(os.getenv("SEQ_LEN", "4096"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))                  # per device; keep small for consumer GPUs
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))                  # effective batch = BATCH_SIZE * GRAD_ACCUM * devices
LR = float(os.getenv("LR", "2e-4"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.01"))
NUM_STEPS = int(os.getenv("NUM_STEPS", "5000"))                 # stop by steps; change as needed
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "1000"))
LOG_STEPS = int(os.getenv("LOG_STEPS", "20"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./gptoss_20b_cpt_qlora")
SEED = int(os.getenv("SEED", "42"))

# LoRA config
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.0"))
LORA_TARGET_MODULES = "all-linear"

set_seed(SEED)

# -----------------------------
# Dataset: stream & pack tokens
# -----------------------------
# We build an iterable that yields packed fixed-length sequences to avoid padding waste.
class PackedTextIterable(IterableDataset):
	def __init__(self, tokenizer, dataset_iter, seq_len=4096, buffer_len=1_000_000):
		self.tokenizer = tokenizer
		self.dataset_iter = dataset_iter
		self.seq_len = seq_len
		self.buffer_len = buffer_len

	def __iter__(self):
		token_buffer = []
		toks = 0

		for ex in self.dataset_iter:
			# The SlimPajama config uses 'text' column
			text = ex.get("text") or ""
			if not text.strip():
				continue

			ids = self.tokenizer(
				text,
				add_special_tokens=False,
			)["input_ids"]

			token_buffer.extend(ids)
			toks += len(ids)

			# When buffer is big enough, yield fixed-length chunks
			while len(token_buffer) >= self.seq_len:
				chunk = token_buffer[:self.seq_len]
				token_buffer = token_buffer[self.seq_len:]
				yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

		# tail: drop remainder (continued pretraining prefers clean packing)
		return

def build_iterable_dataset(tokenizer):
	ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=DATASET_STREAMING)
	# Shuffle the stream deterministically for better mixing
	ds = ds.shuffle(seed=SEED, buffer_size=100_000) if DATASET_STREAMING else ds.shuffle(SEED)
	return PackedTextIterable(tokenizer, iter(ds), seq_len=SEQ_LEN)

# -----------------------------
# Load tokenizer & 4-bit model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# ensure an EOS exists
eos = tokenizer.eos_token or tokenizer.pad_token or "</s>"
if tokenizer.pad_token is None:
	tokenizer.pad_token = eos

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=Mxfp4Config(dequantize=True), device_map="auto")

peft_cfg = LoraConfig(
	r=LORA_R,
	lora_alpha=LORA_ALPHA,
	lora_dropout=LORA_DROPOUT,
	target_modules=LORA_TARGET_MODULES,
	task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# -----------------------------
# Data collator (no MLM)
# -----------------------------
@dataclass
class PackedCollator:
	tokenizer: AutoTokenizer

	def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
		# features already fixed length; create labels equal to input_ids
		input_ids = torch.stack([f["input_ids"] for f in features])
		# attention is all ones (packed)
		attention_mask = torch.ones_like(input_ids)
		labels = input_ids.clone()
		return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_dataset = build_iterable_dataset(tokenizer)
collator = PackedCollator(tokenizer)

# -----------------------------
# Trainer
# -----------------------------
args = TrainingArguments(
	output_dir=OUTPUT_DIR,
	per_device_train_batch_size=BATCH_SIZE,
	gradient_accumulation_steps=GRAD_ACCUM,
	learning_rate=LR,
	warmup_ratio=WARMUP_RATIO,
	lr_scheduler_type="cosine",
	logging_steps=LOG_STEPS,
	save_steps=SAVE_STEPS,
	save_total_limit=2,
	bf16=True,
	dataloader_drop_last=True,
	dataloader_pin_memory=True,
	gradient_checkpointing=True,
	optim=OPTIM,
	max_steps=NUM_STEPS,
	report_to="none",
)

trainer = Trainer(
	model=model,
	args=args,
	train_dataset=train_dataset,
	eval_dataset=None,
	tokenizer=tokenizer,
	data_collator=collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done. Your LoRA-QLoRA CPT adapters are in:", OUTPUT_DIR)


## Merge and export
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

base_or_merged = "./gptoss_20b_cpt_qlora"
model = AutoPeftModelForCausalLM.from_pretrained(base_or_merged, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()  # merges LoRA into base weights (in RAM); for inference-only export
model.save_pretrained("./gptoss_20b_cpt_merged_bf16", safe_serialization=True)
tok = AutoTokenizer.from_pretrained(base_or_merged, use_fast=True)
tok.save_pretrained("./gptoss_20b_cpt_merged_bf16")
