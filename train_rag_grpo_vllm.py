import os, sys, re, glob, json
import logging as LOG

# Change CUDA allocator to RMM
from lib.cuda_use_rmm import *

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Mxfp4Config, TextStreamer, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from tqdm import tqdm

from lib.common import *
from lib.sentence_similarity import *
from lib.convert_pdf import *
from lib.launch_vllm import launch_vllm_server

# ==========================================
# Data Loading
# ==========================================
SYSTEM_PROMPT = """You are a medical assistant. You will be provided with a patient case description and a clinical guideline document (converted from PDF to markdown with some errors).
Your task is to:
1. Extract relevant quotes from the guideline that apply to the case.
2. Formulate a recommended action based on the guideline and the case.
3. Output your relevant quotes (each with justification), your reasoning, and the final recommendation.

IMPORTANT: You must output everything in the following XML format:
<output>
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
<recommended_action>
The specific action to take based on the extracted text from the document.
</recommended_action>
</output>
"""

extract_text_from_pdf = convert_doc
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
			pdf_content = load_txt(pdf_cache_fn)
		else:
			pdf_content, pdf_content_raw = extract_text_from_pdf(pdf_file)
			save_txt(pdf_cache_fn, pdf_content)
			save_txt(pdf_cache_fn[:-3]+'_raw.md', pdf_content_raw)
		if not pdf_content:
			LOG.warning(f"Empty PDF content for {base_name}, skipped!")
			continue

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

			# Sanity check for whether the ground truth quote occurs in the pdf_content
			valid_gt_quotes = []
			for gt_quote in gt_quotes:
				pdf_content_norm = norm_for_match(pdf_content)
				if norm_for_match(gt_quote['quote']) in pdf_content_norm:
					valid_gt_quotes.append(gt_quote)
					continue
				for L in gt_quote['quote'].splitlines():
					valid_gt_quotes.append({'quote': L, 'why_relevant': gt_quote['why_relevant']})
					if norm_for_match(L) not in pdf_content_norm:
						LOG.debug(f"Ground truth quote line '{L}' not found in pdf_content for {base_name}!")
						
			data.append({
				"prompt": conversation, 
				"ground_truth_action": gt_action,
				"ground_truth_quotes": valid_gt_quotes,
				"pdf_content": pdf_content, # valid for citation checking
				"file_id": base_name
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

def reward_xml_format(prompts, completions, **kwargs):
	global tokenizer
	if not kwargs.get('completion_extracted', False):
		completions = get_text_from_ids(kwargs['completion_ids'], tokenizer)
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
		score = 0.8 * (sum([1 for i in xml_elems_unique_lst if i in text])/len(xml_elems_unique_lst))
		# Simple regex check for structure (allowing newlines with DOTALL)
		score += 0.2 if re.match(xml_regex, text, re.DOTALL) else 0
		rewards.append(score)
	return rewards

def reward_content(prompts, completions, ground_truth_action, **kwargs):
	# Reward for matching the ground truth action similarity
	# Simple overlap for now, ideally use embedding or ROUGE
	global tokenizer
	if not kwargs.get('completion_extracted', False):
		completions = get_text_from_ids(kwargs['completion_ids'], tokenizer)
	rewards = []
	for completion, gt_action in zip(completions, ground_truth_action):
		try:
			text = _get_text(completion)
			# Extract generated action
			match = re.search(r"<recommended_action>(.*?)(</recommended_action>|$)", text, re.DOTALL)
			if match:
				gen_action = xml_unesc(match.group(1).strip())
				# Dummy metric: word overlap Jaccard
				gen_words = set(gen_action.lower().split())
				gt_words = set(gt_action.lower().split())
				score = len(gen_words.intersection(gt_words)) / len(gen_words.union(gt_words)) if gt_words else 0
				# Sentence similarity measures
				score += 2*sentence_similarity_crossEncoder(gen_action, gt_action)
				score += sentence_similarity_cased(gen_action, gt_action)
				rewards.append(score*0.25)
			else:
				rewards.append(-1)
		except Exception as e:
			LOG.error(f"Error in reward_content: {e}")
			rewards.append(-1)
	return rewards

def reward_citation(prompts, completions, pdf_content, ground_truth_quotes, **kwargs):
	global tokenizer
	if not kwargs.get('completion_extracted', False):
		completions = get_text_from_ids(kwargs['completion_ids'], tokenizer)
	rewards = []
	for completion, context, gt_quotes1 in zip(completions, pdf_content, ground_truth_quotes):
		try:
			text = _get_text(completion)
			quote_blocks = re.findall(r"<quote>(.*?)</quote>", text, re.DOTALL)
			if not quote_blocks:
				rewards.append(0.0)
				continue

			gt_quote_explain = [(q['quote'], q['why_relevant']) for q in gt_quotes1]

			matches = []
			for block in quote_blocks:
				quote = explain = ''
				m = re.search(r"<text>(.*?)</text>", block, re.DOTALL)
				if m:
					quote = xml_unesc(" ".join(m.group(1).split()))
				m = re.search(r"<explanation>(.*?)</explanation>", block, re.DOTALL)
				if m:
					explain = xml_unesc(" ".join(m.group(1).split()))
				if not quote or not gt_quote_explain:
					continue

				# score, idx = match_quote_alnum(quote, [q1 for q1, e1 in gt_quote_explain], normalize=True)
				score, idx = match_quote_bow(quote, [q1 for q1, e1 in gt_quote_explain])
				if idx >= 0:
					matches.append(score)
					# matches.append((score + sentence_similarity_crossEncoder(explain, gt_quote_explain[idx][1]))/2)
					gt_quote_explain.pop(idx)

			prec = len(matches) / max(1, len(quote_blocks))
			w_prec = sum(matches) / max(1, len(quote_blocks))
			recall = len(matches) / max(1, len(gt_quotes1))
			w_recall = sum(matches) / max(1, len(gt_quotes1))
			f1 = 2 * prec * recall / max(1e-8, prec + recall)
			w_f1 = 2 * w_prec * w_recall / max(1e-8, w_prec + w_recall)
			rewards.append((f1 + w_f1)/2)
		except Exception as e:
			LOG.error(f"Error in reward_citation: {e}")
			rewards.append(-1.0)
	return rewards

def reward_func(prompts, pdf_content, ground_truth_action, ground_truth_quotes, **kwargs):
	global tokenizer, args
	completions = get_text_from_ids(kwargs['completion_ids'], tokenizer)
	R = {'xml': reward_xml_format(prompts, completions, completion_extracted=True),
		'content': reward_content(prompts, completions, ground_truth_action, completion_extracted=True),
		'citation': reward_citation(prompts, completions, pdf_content, ground_truth_quotes, completion_extracted=True)}
	if args.reward == 'all':
		return [(R['xml'][i]+R['content'][i]+R['citation'][i])/3 for i in range(len(R['xml']))]
	rs = [R[arg] for arg in args.reward.split(':')]
	N_rs = len(rs)
	return [sum([rs[j][i] for j in range(N_rs)])/N_rs for i in range(len(completions))]

# ==========================================
# Training
# ==========================================
def main():
	global tokenizer, args
	import argparse
	parser = argparse.ArgumentParser(description="GPRO RAG Training")
	parser.add_argument("--model-name", '-m', type=str, default="/home/LLM_models/gpt-oss-20b", help="Model name or path")
	parser.add_argument("--output-dir", '-o', type=str, default="outputs", help="Output directory")
	parser.add_argument("--max-seq-len", '-l', type=int, default=25000, help="Max sequence length")
	parser.add_argument("--batch-size", '-b', type=int, default=1, help="Batch size per device")
	parser.add_argument("--data-dir", '-d', type=str, default="data", help="dataset directory")
	parser.add_argument("--epochs", '-e', type=int, default=10, help="Num epochs")
	parser.add_argument("--lora-r", '-Lr', type=int, default=16, help="LoRA r")
	parser.add_argument("--lora-alpha", '-La', type=int, default=24, help="LoRA alpha")
	parser.add_argument("--lora-dropout", '-Ld', type=float, default=0.1, help="LoRA dropout")
	parser.add_argument("--learning-rate", '-lr', type=float, default=1e-5, help="Learning rate")
	parser.add_argument("--log-steps", '-Gls', type=int, default=1, help="GRPO logging steps")
	parser.add_argument("--vllm-gpu", '-vg', default='1', help="GPU ID for the vLLM server, set to empty to not use vLLM server")
	parser.add_argument("--grpo-num-samples", '-Gns', type=int, default=8, help="GRPO Num samples")
	parser.add_argument("--reward", '-r', default='citation', help="type of reward function: xml/content/citation/all separated by a colon.")
	parser.add_argument("--gradient-accumulation-steps", '-grad-acc-steps', type=int, default=8, help="Grad accumulation steps")
	parser.add_argument("--verbose", "-v", choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help="Logging level")
	parser.add_argument("--test-ratio", '-tr', type=float, default=0.1, help="Ratio of the dataset to use for testing (0 to 1)")
	parser.add_argument("--save-best-model", '-best', action='store_true', help="Save best model")
	parser.add_argument("--eval-save-every", '-eses', default=10, type=int, help="save every N steps")
	parser.add_argument("--seed", '-s', type=int, default=1234, help="Random seed")
	args = parser.parse_args()

	LOG.basicConfig(level=Try(lambda: int(args.verbose), eval('LOG.'+args.verbose.upper())),
					format='%(levelname)s %(asctime)s: %(message)s', force=True)

	# Check GPU
	gpu_count = torch.cuda.device_count()
	print(f"Detected {gpu_count} GPUs.")
	print(f"Loading model: {args.model_name}")

	# Disable warmup for custom allocator to run
	import transformers.modeling_utils
	transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
	
	# Load Dataset
	dataset = load_rag_dataset(args.data_dir)
	print(f"Loaded {len(dataset)} items from the entire dataset.")
	if len(dataset) == 0:
		LOG.error("No data found! Exiting.")
		return

	# Determine maximum prompt length from dataset (use full dataset to be safe)
	max_dct = get_max_prompt_length(dataset, tokenizer)
	max_prompt_len = max_dct['max_prompt_length']
	if max_prompt_len >= args.max_seq_len:
		LOG.error(f"max_prompt_len (max_prompt_len) is more than max_seq_len (args.max_seq_len) !!!")
	else:
		LOG.info(f"Maximum prompt length is {max_prompt_len}")

	# Split into train/test sets
	if args.test_ratio > 0:
		import random
		file_ids = sorted(list(set(dataset['file_id'])))
		random.seed(args.seed)
		random.shuffle(file_ids)
		num_test = int(len(file_ids) * args.test_ratio)
		if num_test == 0 and args.test_ratio > 0 and len(file_ids) > 0:
			num_test = 1 # Ensuring at least one file is selected if test_ratio > 0
		test_file_ids = set(file_ids[:num_test])

		train_dataset = dataset.filter(lambda x: x['file_id'] not in test_file_ids)
		test_dataset = dataset.filter(lambda x: x['file_id'] in test_file_ids)
		print(f"Split dataset by files: {len(train_dataset)} train, {len(test_dataset)} test (test_ratio={args.test_ratio})")
		print(f"Train files: {len(file_ids) - num_test}, Test files: {num_test}")
	else:
		train_dataset = dataset
		test_dataset = None
		print(f"Using all {len(dataset)} items for training (no test split).")

	# Load Model
	model = AutoModelForCausalLM.from_pretrained(
		args.model_name,
		device_map="cuda",
		dtype="auto",
		quantization_config = Mxfp4Config(dequantize=True),
		trust_remote_code=True,
	)

	# Launch vLLM
	vllm_opt = {}
	if args.vllm_gpu:
		gpu_num, gpu_mem_use = (args.vllm_gpu.split(':')+['0.9'])[:2]
		# Save the de-quantized model for vLLM
		deq_model_path = args.model_name.rstrip('/') + ".deq"
		if not os.path.exists(deq_model_path):
			LOG.info(f'Saving de-quantized model to {deq_model_path} ...')
			model.save_pretrained(
				deq_model_path,
				safe_serialization=True,
				max_shard_size="8GB",
			)
			tokenizer.save_pretrained(deq_model_path)

		os.vllm_proc = launch_vllm_server(
			model_name=deq_model_path,
			host="127.0.0.1",
			cuda_visible_devices=gpu_num,
			max_model_len=args.max_seq_len,
			tensor_parallel_size=1,
			gpu_memory_utilization=float(gpu_mem_use),
		)
		torch.cuda.set_device(0)
		vllm_opt = {
			"use_vllm": True,
			"vllm_mode": "server",
			# "vllm_model_impl": "transformers",
			"vllm_server_host": "127.0.0.1",
			"vllm_server_port": os.vllm_proc.port_num,
			"vllm_server_timeout": 600.0,
		}

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

	# Wrapping Lora adapters will modify the model inplace, so must be done after vLLM server is launched
	peft_config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		inference_mode=False,
		r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
	)
	model = get_peft_model(model, peft_config)
	model.print_trainable_parameters()

	# Trainer Config
	training_args = GRPOConfig(
		output_dir = args.output_dir,
		run_name = "gpro_rag_run",
		learning_rate = args.learning_rate,
		per_device_train_batch_size = args.batch_size,
		gradient_accumulation_steps = args.gradient_accumulation_steps,
		num_train_epochs = args.epochs,
		logging_steps = args.log_steps,
		# max_prompt_length = None,	# option removed in new version TRL
		max_completion_length = int(args.max_seq_len-max_prompt_len),
		generation_batch_size = args.grpo_num_samples,
		num_generations = args.grpo_num_samples, # Number of GPRO samples
		report_to = "none",
		save_strategy = "steps",
		save_steps = args.eval_save_every,
		eval_strategy = "steps" if test_dataset is not None else "no",
		eval_steps = args.eval_save_every,
		eval_on_start = True,
		bf16 = True, # Use BF16 for GH200
		torch_compile = True,
		# Load best model at the end
		load_best_model_at_end = args.save_best_model,
		# metric_for_best_model = "eval_reward",
		# greater_is_better = True,
		# vllm options
		**vllm_opt,
	)
	
	trainer = RewardOnlyEvalGRPOTrainer(
		model = model,
		processing_class = tokenizer,
		reward_funcs = reward_func,
		# reward_funcs = [reward_xml_format, reward_content, reward_citation],
		args = training_args,
		train_dataset = train_dataset,
		eval_dataset = test_dataset,
	)
	
	print("Starting Training...")
	trainer.train()
	print("Training Complete.")
	
	# Save
	model.save_pretrained(os.path.join(args.output_dir, "final_model"))
	tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
	if False:
		reward_xml_format([SYSTEM_PROMPT], [SYSTEM_PROMPT[SYSTEM_PROMPT.find('<analysis>'):]])
	main()
