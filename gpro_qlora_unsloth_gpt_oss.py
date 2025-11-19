#!/usr/bin/env python

import os, sys, ast, sysconfig
import numpy as np
import torch
from pathlib import Path
from unsloth import FastLanguageModel


in_model = os.getenv("IN_MODEL", "/home/LLM_models/unsloth/gpt-oss-20b-unsloth-bnb-4bit")
max_seq_length = 65536 # Can increase for longer RL output
lora_rank = 4 # Larger rank = smarter, but slower
model, tokenizer = FastLanguageModel.from_pretrained(
	model_name = in_model,
    dtype = None, # None for auto detection
	max_seq_length = max_seq_length,
    device_map = "cuda",
	# load_in_4bit = True, # False for LoRA 16bit
	offload_embedding = False, # Reduces VRAM by 1GB
    full_finetuning = False
)


model = FastLanguageModel.get_peft_model(
	model,
	r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
	target_modules = [
		"q_proj", "k_proj", "v_proj", "o_proj",
		"gate_proj", "up_proj", "down_proj",
	],
	lora_alpha = lora_rank*2, # *2 speeds up training
	use_gradient_checkpointing = "unsloth", # Reduces memory usage
	random_state = 3407,
)


if True:
	def generate_random_matrices(seed = 3407, n = 256):
		random_state = np.random.RandomState(seed)
		n, k, m = random_state.randint(1, n+1, size = 3)
		A = np.random.uniform(-10, 10, size = (n, k))
		B = np.random.uniform(-10, 10, size = (k, m))
		return A, A.tolist(), B, B.tolist()

	A, A_list, B, B_list = generate_random_matrices(seed = 42, n = 5)
	print(A)
	print(B)
	print(np.matmul(A, B))

	def calculate_difference(pred, real):
		if pred is None: return 5, 5
		assert real is not None
		import numpy as np
		try:
			difference = pred - real
		except:
			return 5, 5
		amax_error = float(np.amax(difference))
		mse_error  = float(np.mean(np.square(difference)))
		return amax_error, mse_error

	def matmul(A, B):
		z, s = zip, sum
		Bt = list(z(*B))
		return [[s(a*b for a, b in z(row, col)) for col in Bt] for row in A]

	prediction = matmul(A_list, B_list)
	calculate_difference(prediction, np.matmul(A, B))


## Countering Reward Hacking 1: Stop laziness
#@title (Collapsible code)
def _stdlib_names():
    """
    Build a set of canonical stdlib top-level module/package names.
    Uses sys.stdlib_module_names when available (3.10+), with a
    filesystem fallback for older versions/edge cases.
    """
    names = {m.lower() for m in getattr(sys, "stdlib_module_names", set())}
    names |= {m.lower() for m in sys.builtin_module_names}
    names.add("__future__")  # special-case

    # Fallback/augmentation: scan the stdlib directory
    try:
        stdlib_dir = Path(sysconfig.get_path("stdlib"))
        if stdlib_dir.exists():
            for p in stdlib_dir.iterdir():
                if p.name == "site-packages":
                    continue
                if p.suffix == ".py":
                    names.add(p.stem.lower())
                elif p.is_dir() and (p / "__init__.py").exists():
                    names.add(p.name.lower())
    except Exception:
        # conservative fallback; the names set above will still work well
        pass

    return names

_STDLIB_SET = _stdlib_names()

def check_only_stdlib_imports(code: str):
    """
    Return (ok: bool, details: dict)

    ok == True  -> all absolute imports are from the stdlib.
    ok == False -> details['non_stdlib'] lists offending top-level modules.

    details includes:
      - stdlib: sorted list of stdlib imports found
      - non_stdlib: sorted list of non-stdlib imports found
      - relative_imports: count of relative imports (always allowed here)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "error": f"SyntaxError: {e}",
            "stdlib": [],
            "non_stdlib": [],
            "relative_imports": 0,
        }

    abs_imports = set()
    relative_count = 0

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                abs_imports.add(alias.name.split(".")[0])
        def visit_ImportFrom(self, node: ast.ImportFrom):
            nonlocal relative_count
            if (node.level or 0) > 0:
                # relative import
                relative_count += 1
            else:
                if node.module:
                    abs_imports.add(node.module.split(".")[0])

    Visitor().visit(tree)

    stdlib_found = sorted(m for m in abs_imports if m.lower() in _STDLIB_SET)
    non_stdlib = sorted(m for m in abs_imports if m.lower() not in _STDLIB_SET)

    return len(non_stdlib) == 0, {
        "stdlib": stdlib_found,
        "non_stdlib": non_stdlib,
        "relative_imports": relative_count,
    }

sample = """
def matmul(A, B):
    import numpy as np
    from torch import matmul
    z, s = zip, sum
    Bt = list(z(*B))
    return [[s(a*b for a, b in z(row, col)) for col in Bt] for row in A]
"""
ok, info = check_only_stdlib_imports(sample)
print("Only stdlib imports?", ok)
print(info)


## Countering Reward Hacking 2: Stop cheating
output_function = {}
exec(sample, {}, output_function)
output_function["matmul"]

import types
output_function["matmul"] = types.FunctionType(output_function["matmul"].__code__, {})

def import_numpy():
    np.matmul
    print("Success")

import_numpy()
import_numpy = types.FunctionType(import_numpy.__code__, {})
try:
    import_numpy()
except Exception as e:
    print(str(e))


