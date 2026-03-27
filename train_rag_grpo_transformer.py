import os, sys, re, glob, json
import logging as LOG

# Change CUDA allocator to RMM
from lib.cuda_use_rmm import *

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Mxfp4Config, TextStreamer
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from tqdm import tqdm

from lib.common import *
from lib.sentence_similarity import *
from lib.convert_pdf import *

# ==========================================
# Data Loading
# ==========================================
SYSTEM_PROMPT = """You are a medical assistant. You will be provided with a patient case description and a clinical guideline document (converted from PDF).
Your task is to:
1. Extract relevant quotes from the guideline that apply to the case.
2. Formulate a recommended action based on the guideline and the case.
3. Output your relevant quotes (each with justification), your reasoning, and the final recommendation.

IMPORTANT: You must output everything in the following XML format:
<output>
<analysis>
<relevant_quotes>
<quote>
<text>Exact text from the document</text>
<explanation>Explain why the extracted text from the document applies to this patient.</explanation>
</quote>
<quote>
<text>Exact text from the document</text>
<explanation>Explain why the extracted text from the document applies to this patient.</explanation>
</quote>
</relevant_quotes>
<reasoning>
Explain how the extracted text from the document are combined to form the final recommendation.
</reasoning>
</analysis>
<recommended_action>
The specific action to take based on the extracted text from the document.
</recommended_action>
</output>
"""

extract_text_from_pdf = convert_doc
def process_text(text):
	text = html.unescape(text)
	text = text.replace('<!-- image -->\n', '')
	lines = text.splitlines()
	# Remove expert group section
	title_set = set('Dr Ms Mr Adj A/Prof Assoc Prof'.split())
	idx = lines.index('## Expert group')
	if idx >= 0:
		lines.pop(idx)
		while idx < len(lines):
			if not lines[idx].strip():
				pass
			elif lines[idx].strip().startswith('##'):
				pass
			elif lines[idx].split()[0] in title_set:
				pass
			else:
				break
			lines.pop(idx)
	return '\n'.join(lines)

def load_rag_dataset(data_dir):
	json_files = glob.glob(os.path.join(data_dir, "*_filtered.json"))
	data = []

	md_cache_dir = data_dir+'.cache'
	os.makedirs(md_cache_dir, exist_ok=True)
	
	for json_file in tqdm(json_files, desc="Loading RAG dataset"):
		base_name = os.path.basename(json_file).replace("_filtered.json", "")

		# Load PDF content
		# Try finding the pdf
		pdf_files = glob.glob(os.path.join(data_dir, f"{base_name}*.pdf"))
		if not pdf_files:
			LOG.warning(f"No PDF found for {base_name}, skipped!")
			continue
		pdf_file = pdf_files[0]
		# Load PDF content from cache (if exists) or run converter
		pdf_cache_fn = os.path.join(md_cache_dir, os.path.basename(pdf_file)[:-4]+'.md')
		if os.path.exists(pdf_cache_fn) and os.path.getsize(pdf_cache_fn) > 0:
			with open(pdf_cache_fn, 'r') as f:
				pdf_content = f.read().strip()
		else:
			pdf_content = extract_text_from_pdf(pdf_file)
			with open(pdf_cache_fn, 'w') as f:
				f.write(pdf_content)
		if not pdf_content:
			LOG.warning(f"Empty PDF content for {base_name}, skipped!")
			continue

		pdf_content = process_text(pdf_content)

		# Load JSON
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
			gt_quotes = case.get('reference', [])
			
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
		return completion.strip()
	if isinstance(completion, list):
		# If it's a list, it might be a conversation list [{'role':..., 'content':...}]
		# or just a list of one string.
		if len(completion) > 0:
			if isinstance(completion[0], dict) and 'content' in completion[0]:
				return completion[0]['content'].strip()
			if isinstance(completion[0], str):
				return completion[0].strip()
		return str(completion).strip()
	return str(completion).strip()

def xml_format_reward(prompts, completions):
	global tokenizer
	rewards = []
	xml_elems_lst = re.findall(r'<[^<>]*>', SYSTEM_PROMPT)
	xml_elems_lst = [re.sub(r' [^>]*>', ' ', e) for e in xml_elems_lst]
	xml_elems_unique_lst = []
	temp_set = set()
	for e in xml_elems_lst:
		if e not in temp_set:
			xml_elems_unique_lst.append(e)
			temp_set.add(e)
	xml_regex = '.*'.join(xml_elems_unique_lst)
	for completion in completions:
		text = _get_text(completion)
		score = (sum([1 for i in xml_elems_unique_lst if i in text])/len(xml_elems_unique_lst))*0.5
		# Simple regex check for structure (allowing newlines with DOTALL)
		score += 0.5 if re.match(xml_regex, text, re.DOTALL) else 0
		rewards.append(score)
	return rewards

def content_reward(prompts, completions, ground_truth_action):
	# Reward for matching the ground truth action similarity
	# Simple overlap for now, ideally use embedding or ROUGE
	global tokenizer
	rewards = []
	for completion, gt_action in zip(completions, ground_truth_action):
		try:
			text = _get_text(completion)
			# Extract generated action
			match = re.search(r"<recommended_action>(.*?)(</recommended_action>|$)", text, re.DOTALL)
			if match:
				gen_action = match.group(1).strip()
				# Dummy metric: word overlap Jaccard
				gen_words = set(gen_action.lower().split())
				gt_words = set(gt_action.lower().split())
				score = len(gen_words.intersection(gt_words)) / len(gen_words.union(gt_words)) if gt_words else 0
				# Sentence similarity measures
				score += 2*sentence_similarity_crossEncoder(gen_action, gt_action)
				score += sentence_similarity_cased(gen_action, gt_action)
				rewards.append(score*0.25)
			else:
				rewards.append(0.0)
		except:
			rewards.append(0.0)
	return rewards

def citation_reward(prompts, completions, pdf_content, ground_truth_quotes):
	global tokenizer
	rewards = []
	for completion, context, gt_quotes1 in zip(completions, pdf_content, ground_truth_quotes):
		try:
			text = _get_text(completion)
			context_norm = "".join(context.split())
			quote_blocks = re.findall(r"<quote>(.*?)</quote>", text, re.DOTALL)
			if not quote_blocks:
				rewards.append(0.0)
				continue

			gt_quote_explain = [(q['quote'], q['why_relevant']) for q in gt_quotes1]

			reward = 0
			for block in quote_blocks:
				quote = explain = ''
				m = re.search(r"<text>(.*?)</text>", block, re.DOTALL)
				if m:
					quote = " ".join(m.group(1).split())
				m = re.search(r"<explanation>(.*?)</explanation>", block, re.DOTALL)
				if m:
					explain = " ".join(m.group(1).split())
				if not quote or not gt_quote_explain:
					reward -= 1
					continue

				score, idx = match_quote(quote, [q1 for q1, e1 in gt_quote_explain], normalize=True)
				if idx >= 0:
					reward += (score + sentence_similarity_crossEncoder(explain, gt_quote_explain[idx][1]))/2
					gt_quote_explain.pop(idx)
				else:
					reward -= 1
					
			rewards.append(reward / len(gt_quotes1))
		except Exception:
			rewards.append(-1.0)
	return rewards

def reward_func(prompts, pdf_content, ground_truth_action, ground_truth_quotes, **kwargs):
	global tokenizer, args
	completions = get_text_from_ids(kwargs['completion_ids'], tokenizer)
	r1 = xml_format_reward(prompts, completions)
	r2 = content_reward(prompts, completions, ground_truth_action)
	r3 = citation_reward(prompts, completions, pdf_content, ground_truth_quotes)
	if args.reward == 'xml':
		return r1
	elif args.reward == 'content':
		return r2
	elif args.reward == 'citation':
		return r3
	else:
		return [(r1[i]+r2[i]+r3[i])/3 for i in range(len(r1))]

# ==========================================
# Training
# ==========================================
def main():
	global tokenizer, args
	import argparse
	parser = argparse.ArgumentParser(description="GPRO RAG Training")
	parser.add_argument("--model_name", type=str, default="/home/LLM_models/gpt-oss-20b", help="Model name or path")
	parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
	parser.add_argument("--max_seq_len", type=int, default=20000, help="Max sequence length")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
	parser.add_argument("--data_dir", type=str, default="data", help="dataset directory")
	parser.add_argument("--epochs", type=int, default=1, help="Num epochs")
	parser.add_argument("--reward", default='all', choices=['xml', 'content', 'citation', 'all'], help="type of reward function")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Grad accumulation steps")
	parser.add_argument("--verbose", "-v", choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help="Logging level")
	args = parser.parse_args()

	LOG.basicConfig(level=Try(lambda: int(args.verbose), eval('LOG.'+args.verbose.upper())),
					format='%(levelname)s %(asctime)s: %(message)s')

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
	dataset = load_rag_dataset(args.data_dir)
	print(f"Loaded {len(dataset)} items.")
	if len(dataset) == 0:
		LOG.error("No data found! Exiting.")
		return

	# Determine maximum prompt length from dataset
	max_dct = get_max_prompt_length(dataset, tokenizer)
	max_prompt_len = max_dct['max_prompt_length']
	if max_prompt_len >= args.max_seq_len:
		LOG.error(f"max_prompt_len (max_prompt_len) is more than max_seq_len (args.max_seq_len) !!!")
	else:
		LOG.info(f"Maximum prompt length is {max_prompt_len}")

	if False:
		outputs = model.generate(tokenizer.apply_chat_template(
				dataset[0]['prompt'],
				add_generation_prompt=True,
				return_tensors="pt",
				# reasoning_effort = "high"
			).to(model.device), 
			max_new_tokens = args.max_seq_len,
			streamer = TextStreamer(tokenizer)
		)
		print(tokenizer.decode(outputs[0], skip_special_tokens=False))

	# Trainer Config
	training_args = GRPOConfig(
		output_dir = args.output_dir,
		run_name = "gpro_rag_run",
		learning_rate = 5e-6,
		per_device_train_batch_size = args.batch_size,
		gradient_accumulation_steps = args.gradient_accumulation_steps,
		num_train_epochs = args.epochs,
		logging_steps = 1,
		# max_prompt_length = None,	# option removed in new version TRL
		max_completion_length = int(args.max_seq_len-max_prompt_len),
		num_generations = 2, # Number of GPRO samples
		report_to = "none",
		save_strategy = "steps",
		save_steps = 10,
		bf16 = True, # Use BF16 for GH200
		torch_compile=True,
		# use_vllm=False,
		# vllm_mode="colocate",
	)
	
	trainer = GRPOTrainer(
		model = model,
		processing_class = tokenizer,
		reward_funcs = reward_func,
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
	if False:
		xml_format_reward([SYSTEM_PROMPT], [SYSTEM_PROMPT[SYSTEM_PROMPT.find('<analysis>'):]])
	if True:
		pass
	main()
