import os, sys, re, glob, json

# Change CUDA allocator to RMM
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch
rmm.reinitialize(pool_allocator=True, managed_memory=True)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import pypdf


# ==========================================
# Data Loading
# ==========================================
SYSTEM_PROMPT = """You are a medical assistant. You will be provided with a patient case description and a clinical guideline document (PDF content).
Your task is to:
1. Identify relevant quotes from the guideline that apply to the case.
2. Formulate a recommended action based on the guideline and the case.
3. Output your reasoning, relevant quotes, and the final recommendation in the following XML format:

<analysis>
<reasoning>
Explain why the guideline applies to this patient.
</reasoning>
<relevant_quotes>
<quote page="1">Exact text from document</quote>
</relevant_quotes>
</analysis>
<recommended_action>
The specific action to take.
</recommended_action>
"""

def extract_text_from_pdf(pdf_path):
	try:
		reader = pypdf.PdfReader(pdf_path)
		text = ""
		for i, page in enumerate(reader.pages):
			page_text = page.extract_text()
			if page_text:
				text += f"\n--- Page {i+1} ---\n{page_text}"
		return text
	except Exception as e:
		print(f"Error reading {pdf_path}: {e}")
		return ""

def load_rag_dataset(data_dir):
	json_files = glob.glob(os.path.join(data_dir, "*_filtered.json"))
	data = []
	
	for json_file in json_files:
		base_name = os.path.basename(json_file).replace("_filtered.json", "")
		# Try finding the pdf
		pdf_files = glob.glob(os.path.join(data_dir, f"{base_name}*.pdf"))
		if not pdf_files:
			continue
		pdf_file = pdf_files[0]
		
		pdf_content = extract_text_from_pdf(pdf_file)
		if not pdf_content:
			continue
			
		with open(json_file, 'r') as f:
			cases = json.load(f)
			
		for case in cases:
			# Construct the prompt inputs
			# Only strictly need 'prompt' and 'ground_truth' for GPRO? 
			# GPRO Trainer expects 'prompt' column.
			
			user_content = f"Patient Description: {case['description']}\n\nClinical Guideline:\n{pdf_content}"
			
			conversation = [
				{"role": "system", "content": SYSTEM_PROMPT},
				{"role": "user", "content": user_content}
			]
			
			# We store ground truth for reward calculation
			gt_action = case.get('recommended_action', '')
			gt_quotes = [r['quote'] for r in case.get('reference', [])]
			
			data.append({
				"prompt": conversation, 
				"ground_truth_action": gt_action,
				"ground_truth_quotes": gt_quotes,
				"pdf_content": pdf_content # valid for citation checking
			})
			
	return Dataset.from_list(data)

# ==========================================
# Reward Functions
# ==========================================

def _get_text(completion):
	if isinstance(completion, str):
		return completion
	if isinstance(completion, list):
		# If it's a list, it might be a conversation list [{'role':..., 'content':...}]
		# or just a list of one string.
		if len(completion) > 0:
			if isinstance(completion[0], dict) and 'content' in completion[0]:
				return completion[0]['content']
			if isinstance(completion[0], str):
				return completion[0]
		return str(completion)
	return str(completion)

def xml_format_reward(prompts, completions, **kwargs):
	rewards = []
	xml_regex = r"<analysis>.*<reasoning>.*</reasoning>.*<relevant_quotes>.*</relevant_quotes>.*</analysis>.*<recommended_action>.*</recommended_action>"
	for completion in completions:
		text = _get_text(completion)
		# Simple regex check for structure (allowing newlines with DOTALL)
		if re.search(xml_regex, text, re.DOTALL):
			rewards.append(1.0)
		else:
			rewards.append(0.0)
	return rewards

def content_reward(prompts, completions, ground_truth_action, **kwargs):
	# Reward for matching the ground truth action similarity
	# Simple overlap for now, ideally use embedding or ROUGE
	rewards = []
	for completion, gt_action in zip(completions, ground_truth_action):
		try:
			text = _get_text(completion)
			# Extract generated action
			match = re.search(r"<recommended_action>(.*?)</recommended_action>", text, re.DOTALL)
			if match:
				gen_action = match.group(1).strip()
				# Dummy metric: word overlap Jaccard
				gen_words = set(gen_action.lower().split())
				gt_words = set(gt_action.lower().split())
				if not gt_words:
					score = 0.0
				else:
					score = len(gen_words.intersection(gt_words)) / len(gen_words.union(gt_words))
				rewards.append(score)
			else:
				rewards.append(0.0)
		except:
			rewards.append(0.0)
	return rewards

def citation_reward(prompts, completions, pdf_content, **kwargs):
	# Reward for hallucinatory quotes: Check if quotes exist in PDF
	rewards = []
	for completion, context in zip(completions, pdf_content):
		try:
			text = _get_text(completion)
			quotes = re.findall(r"<quote.*?>(.*?)</quote>", text, re.DOTALL)
			if not quotes:
				# If no quotes found but structure exists, neutral or small penalty? 
				# Let's say 0.0 if structure requires quotes.
				rewards.append(0.0) 
				continue
			
			valid_quotes = 0
			for q in quotes:
				clean_q = q.strip()
				# Fuzzy check: strict substring is hard with PDF extraction artifacts.
				# We'll normalize whitespace.
				clean_q_norm = " ".join(clean_q.split())
				context_norm = " ".join(context.split())
				
				if clean_q_norm in context_norm:
					valid_quotes += 1
			
			# Score = percentage of valid quotes
			rewards.append(valid_quotes / len(quotes))
		except:
			rewards.append(0.0)
	return rewards

# ==========================================
# Training
# ==========================================
def main():
	import argparse
	parser = argparse.ArgumentParser(description="GPRO RAG Training")
	parser.add_argument("--model_name", type=str, default="/home/LLM_models/gpt-oss-120b", help="Model name or path")
	parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
	parser.add_argument("--max_seq_len", type=int, default=20000, help="Max sequence length")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
	parser.add_argument("--epochs", type=int, default=1, help="Num epochs")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Grad accumulation steps")
	args = parser.parse_args()

	# Check GPU
	gpu_count = torch.cuda.device_count()
	print(f"Detected {gpu_count} GPUs.")
	print(f"Loading model: {args.model_name}")

	# Disable warmup for custom allocator to run
	import transformers.modeling_utils
	transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

	# Load Model
	tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
	model = AutoModelForCausalLM.from_pretrained(
		args.model_name,
		device_map="cuda",
		dtype="auto",
		quantization_config = Mxfp4Config(dequantize=True),
		trust_remote_code=True,
	)

	peft_config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		inference_mode=False,
		r=8,
		lora_alpha=32,
		lora_dropout=0.1,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
	)
	
	model = get_peft_model(model, peft_config)

	# Load Dataset
	dataset = load_rag_dataset("data")
	print(f"Loaded {len(dataset)} items.")
	if len(dataset) == 0:
		print("No data found! Exiting.")
		return

	# Trainer Config
	training_args = GRPOConfig(
		output_dir = args.output_dir,
		run_name = "gpro_rag_run",
		learning_rate = 5e-6,
		per_device_train_batch_size = args.batch_size,
		gradient_accumulation_steps = args.gradient_accumulation_steps,
		num_train_epochs = args.epochs,
		logging_steps = 1,
		max_prompt_length = args.max_seq_len // 2, # Heuristic
		max_completion_length = 2048, # Enough for XML output
		num_generations = 4, # Number of GPRO samples
		report_to = "none",
		save_strategy = "steps",
		save_steps = 100,
		bf16 = True, # Use BF16 for GH200
		torch_compile=False,
		# deepspeed="ds_config.json",
	)
	
	trainer = GRPOTrainer(
		model = model,
		processing_class = tokenizer,
		reward_funcs = [xml_format_reward, content_reward, citation_reward],
		args = training_args,
		train_dataset = dataset,
	)
	
	print("Starting Training...")
	trainer.train()
	print("Training Complete.")
	
	# Save
	model.save_pretrained(os.path.join(args.output_dir, "final_model"))
	tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
	main()
