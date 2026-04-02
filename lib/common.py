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

def match_quote_bow(quote, gt_quotes, threshold=0.8):
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

if __name__ == '__main__':
	escaped_string = "&lt; &amp; &gt; &apos; &quot;"
	print(xml_unesc(escaped_string))
	print(xml_esc(xml_unesc(escaped_string)))