import os, sys, gzip, traceback, re, html
import numpy as np
import pandas as pd
from typing import Any

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

# Match quote: return matching score and index
def match_quote(quote, gt_quotes, normalize=False):
	if normalize:
		quote = ''.join(quote.split())
		gt_quotes = [''.join(gt_q.split()) for gt_q in gt_quotes]
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
		