#!/usr/bin/env python3

import sys
import os
import re
import ast
import math
import subprocess
from typing import List, Optional, Dict, Any

import pandas as pd

START_HINT = 'Starting Training...'

def set_display_from_tmux() -> None:
	"""Set DISPLAY from tmux before importing matplotlib."""
	try:
		result = subprocess.run(
			"tmux show-env | grep ^DISPLAY= | cut -d= -f2",
			shell=True,
			check=False,
			capture_output=True,
			text=True,
		)
		display = result.stdout.strip()
		if display:
			os.environ["DISPLAY"] = display
	except Exception:
		pass

set_display_from_tmux()

import matplotlib.pyplot as plt  # noqa: E402


def parse_dict_line(line: str) -> Optional[Dict[str, Any]]:
	try:
		obj = ast.literal_eval(line[line.find('{'):line.rfind('}')+1])
		return obj if type(obj)==dict else None
	except Exception:
		return None


def read_lines() -> List[str]:
	if len(sys.argv) > 1:
		with open(os.path.expanduser(sys.argv[1]), "r", encoding="utf-8", errors="replace") as f:
			return f.readlines()
	return sys.stdin.readlines()


def to_float(x: Any) -> float:
	try:
		v = float(x)
		return v
	except Exception:
		return math.nan


def backward_extract_dicts(lines: List[str], max_noneligible: int = 10) -> List[Dict[str, Any]]:
	"""
	Walk backward from the end and collect dictionary lines.
	Stop after more than `max_noneligible` consecutive non-eligible lines.
	Returned in original chronological order.
	"""
	collected: List[Dict[str, Any]] = []

	for line in reversed(lines):
		d = parse_dict_line(line)
		if d:
			collected.append(d)
		if START_HINT in line:
			break

	collected.reverse()
	return collected


def build_table(dicts: List[Dict[str, Any]]) -> pd.DataFrame:
	"""
	Split into reward track and eval track.

	Rule:
	- A dict with reward + reward_std creates a new row.
	- A dict with eval_reward + eval_reward_std is attached to the most recent prior reward row.
	"""
	rows: List[Dict[str, float]] = []

	for d in dicts:
		has_reward = "reward" in d and "reward_std" in d
		has_eval = "eval_reward" in d and "eval_reward_std" in d

		if has_reward:
			rows.append(
				{
					"reward": to_float(d["reward"]),
					"reward_std": to_float(d["reward_std"]),
					"eval_reward": math.nan,
					"eval_reward_std": math.nan,
				}
			)

		elif has_eval:
			if rows:
				rows[-1]["eval_reward"] = to_float(d["eval_reward"])
				rows[-1]["eval_reward_std"] = to_float(d["eval_reward_std"])
			else:
				rows.append(
					{
						"reward": math.nan,
						"reward_std": math.nan,
						"eval_reward": to_float(d["eval_reward"]),
						"eval_reward_std": to_float(d["eval_reward_std"]),
					}
				)

	return pd.DataFrame(
		rows,
		columns=["reward", "reward_std", "eval_reward", "eval_reward_std"],
	)


def plot_tracks(df: pd.DataFrame) -> None:
	if df.empty:
		print("No reward rows found; nothing to plot.", file=sys.stderr)
		return

	x = list(range(len(df)))

	plt.figure(figsize=(10, 6))

	reward = df["reward"].to_numpy()
	reward_std = df["reward_std"].to_numpy()
	plt.plot(x, reward, label="reward")
	plt.fill_between(x, reward - reward_std, reward + reward_std, alpha=0.25)

	eval_mask = df["eval_reward"].notna()
	if eval_mask.any():
		x_eval = df.index[eval_mask].to_numpy()
		eval_reward = df.loc[eval_mask, "eval_reward"].to_numpy()
		eval_reward_std = df.loc[eval_mask, "eval_reward_std"].to_numpy()
		plt.plot(x_eval, eval_reward, label="eval_reward")
		plt.fill_between(
			x_eval,
			eval_reward - eval_reward_std,
			eval_reward + eval_reward_std,
			alpha=0.25,
		)

	plt.xlabel("step")
	plt.ylabel("reward")
	plt.title("Reward and Eval Reward")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()


def main() -> None:
	lines = read_lines()
	dicts = backward_extract_dicts(lines, max_noneligible=10)
	df = build_table(dicts)

	print(df.to_string(index=False, na_rep="NAN"))
	plot_tracks(df)


if __name__ == "__main__":
	main()
