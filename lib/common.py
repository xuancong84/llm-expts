import os, sys, gzip, traceback, re, html
import numpy as np
import pandas as pd
from typing import Any
from xml.sax.saxutils import escape, unescape

def expand_path(fn):
	return os.path.expanduser(os.path.expandvars(fn))

def Try(*args):
	exc = ''
	for arg in args:
		try:
			return arg() if callable(arg) else arg
		except:
			exc = traceback.format_exc()
	return str(exc)

def Open(fn, mode='r', **kwargs):
	if fn == '-':
		return sys.stdin if mode.startswith('r') else sys.stdout
	fn = expand_path(fn)
	return gzip.open(fn, mode, **kwargs) if fn.lower().endswith('.gz') else open(fn, mode, **kwargs)

def save_txt(fn, text):
	with open(fn, 'wt') as f:
		f.write(text)

def load_txt(fn):
	with open(fn, 'rt') as f:
		return f.read().strip()


def prompt_token_length(example: dict[str, Any], tokenizer) -> int:
	prompt = example["prompt"]

	# Case 1: prompt is already final text
	if isinstance(prompt, str):
		return len(
			tokenizer(
				prompt,
				add_special_tokens=True,
				truncation=False,
			)["input_ids"]
		)

	# Case 2: prompt is still a chat/message list
	if isinstance(prompt, list):
		ids = tokenizer.apply_chat_template(
			prompt,
			tokenize=True,
			add_generation_prompt=True,  # match your training setup
		)
		return len(ids)

	raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def get_max_prompt_length(dataset, tokenizer):
	max_len = 0
	max_idx = -1

	for i, example in enumerate(dataset):
		n = prompt_token_length(example, tokenizer)
		if n > max_len:
			max_len = n
			max_idx = i

	return {
		"max_prompt_length": max_len,
		"max_index": max_idx,
		"longest_example": dataset[max_idx],
	}

# Prefer parsing the FINAL channel from raw text yourself for GPT-OSS
def extract_gpt_oss_final(raw_text: str) -> str:
	m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)", raw_text, re.S)
	ret = m.group(1).strip() if m else raw_text.strip()
	return ret[:-10].strip() if ret.endswith('<|return|>') else ret
	

# Use raw completion_ids in rewards instead of completion[0]["content"]
def get_text_from_ids(completion_ids_item, tokenizer):
	if completion_ids_item and isinstance(completion_ids_item[0], list):
		return [get_text_from_ids(item, tokenizer) for item in completion_ids_item]
	raw = tokenizer.decode(completion_ids_item, skip_special_tokens=False)
	return extract_gpt_oss_final(raw)

# Normalize for matching
norm_regex = re.compile(r'[^a-zA-Z0-9]')
def norm_for_match(text):
	return norm_regex.sub('', text)

def match_quote_alnum(quote, gt_quotes, normalize=False):
	# Match quote by alphanumeric characters: return matching score and index
	if normalize:
		quote = norm_for_match(quote)
		gt_quotes = [norm_for_match(gt_q) for gt_q in gt_quotes]
	if quote in gt_quotes:
		return 1, gt_quotes.index(quote)
	max_score = 0
	max_idx = -1
	for i, gt_quote in enumerate(gt_quotes):
		score = 0
		if quote in gt_quote:
			score = len(quote)/len(gt_quote)
		elif gt_quote in quote:
			score = len(gt_quote)/len(quote)
		if score > max_score:
			max_score = score
			max_idx = i
	return max_score, max_idx

def match_quote_bow(quote, gt_quotes, threshold=0.75):
	# Match quote by bag-of-words
	quote_bow = set(quote.split())
	gt_quotes_bow = [set(gt_q.split()) for gt_q in gt_quotes]
	if quote_bow in gt_quotes_bow:
		return 1, gt_quotes_bow.index(quote_bow)
	max_score = 0
	max_idx = -1
	for i, gt_quote_bow in enumerate(gt_quotes_bow):
		score = len(quote_bow & gt_quote_bow) / len(quote_bow | gt_quote_bow)
		if score > max_score and score > threshold:
			max_score = score
			max_idx = i
	return max_score, max_idx

xml_esc = lambda t: escape(t, {"'": "&apos;", '"': "&quot;"})
xml_unesc = lambda t: unescape(t, {"&apos;": "'", "&quot;": '"'})

import torch
from accelerate.utils import gather_object
from trl import GRPOTrainer
from trl.trainer.utils import nanstd

class RewardOnlyEvalGRPOTrainer(GRPOTrainer):
	@torch.no_grad()
	def _eval_reward_only(self, inputs):
		device = self.accelerator.device
		mode = "eval"

		prompts = [x["prompt"] for x in inputs]

		(
			prompt_ids_list,
			completion_ids_list,
			tool_mask_list,
			completions,
			_num_items_in_batch,
			sampling_per_token_logps_list,
			extra_fields,
		) = self._generate(prompts)

		prompts_text = self.processing_class.batch_decode(
			prompt_ids_list, skip_special_tokens=True
		)
		completions_text = self.processing_class.batch_decode(
			completion_ids_list, skip_special_tokens=True
		)

		if extra_fields:
			for i, inp in enumerate(inputs):
				for key, values in extra_fields.items():
					if isinstance(values, list) and i < len(values):
						inp[key] = values[i]
					elif not isinstance(values, list):
						inp[key] = values

		rewards_per_func = self._calculate_rewards(
			inputs, prompts, completions, completion_ids_list
		)

		rewards = (
			rewards_per_func
			* self.reward_weights.to(rewards_per_func.device).unsqueeze(0)
		).nansum(dim=1)

		mean_reward = rewards.mean()
		std_reward = rewards.std() if rewards.numel() > 1 else torch.zeros((), device=device)

		for i, reward_func_name in enumerate(self.reward_func_names):
			mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
			std_func_rewards = nanstd(rewards_per_func[:, i]).item()
			self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
			self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)

		self._metrics[mode]["reward"].append(mean_reward.item())
		self._metrics[mode]["reward_std"].append(std_reward.item())

		self._logs["prompt"].extend(gather_object(prompts_text))
		self._logs["completion"].extend(gather_object(completions_text))
		for i, name in enumerate(self.reward_func_names):
			self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

		self._logs["advantages"].extend([0.0] * rewards_per_func.shape[0])

		for column in sorted(self._pending_extra_logs):
			self._logs["extra"][column].extend(gather_object(self._pending_extra_logs[column]))
		self._pending_extra_logs.clear()

		for name in sorted(self._pending_metrics):
			values = self._pending_metrics[name]
			local_mean = sum(values) / len(values)
			global_mean = self.accelerator.gather(
				torch.tensor(local_mean, device=device)
			).mean().item()
			self._metrics[mode][name].append(global_mean)
		self._pending_metrics.clear()

		return mean_reward

	def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
		if model.training:
			return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

		with torch.no_grad():
			mean_reward = self._eval_reward_only(inputs)

		dummy_loss = torch.zeros((), device=mean_reward.device)
		return dummy_loss, None, None

	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		metrics = super().evaluate(
			eval_dataset=eval_dataset,
			ignore_keys=ignore_keys,
			metric_key_prefix=metric_key_prefix,
		)

		eval_store = self._metrics.get("eval", {})

		if "reward" in eval_store and eval_store["reward"]:
			metrics[f"{metric_key_prefix}_reward"] = float(sum(eval_store["reward"]) / len(eval_store["reward"]))

		if "reward_std" in eval_store and eval_store["reward_std"]:
			metrics[f"{metric_key_prefix}_reward_std"] = float(sum(eval_store["reward_std"]) / len(eval_store["reward_std"]))

		return metrics


if __name__ == '__main__':
	escaped_string = "&lt; &amp; &gt; &apos; &quot;"
	print(xml_unesc(escaped_string))
	print(xml_esc(xml_unesc(escaped_string)))