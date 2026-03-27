
# Change CUDA allocator to RMM
import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch
rmm.reinitialize(pool_allocator=True, managed_memory=True)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

PLUGGABLE_ALLOCATOR_ERR_PREFIX = "CUDAPluggableAllocator does not yet support"

def _is_pluggable_allocator_unsupported(e: BaseException) -> bool:
	return PLUGGABLE_ALLOCATOR_ERR_PREFIX in str(e)

def _safe_wrap(fn, fallback):
	def _inner(*a, **k):
		try:
			return fn(*a, **k)
		except RuntimeError as e:
			if _is_pluggable_allocator_unsupported(e):
				return fallback() if callable(fallback) else fallback
			raise
	return _inner

def patch_rmm_pluggable_allocator_compat():
	# 1) TorchInductor
	try:
		from torch._inductor.runtime import triton_heuristics as th
		_orig = th.CachingAutotuner.copy_args_to_cpu_if_needed

		def _patched(self, *args, **kwargs):
			try:
				return _orig(self, *args, **kwargs)
			except RuntimeError as e:
				if _is_pluggable_allocator_unsupported(e):
					return {}
				raise

		th.CachingAutotuner.copy_args_to_cpu_if_needed = _patched
		print("[patch] TorchInductor patched for pluggable allocator fallback.")
	except Exception as e:
		print(f"[patch] TorchInductor patch skipped: {e}")

	# 2) Patch top-level torch.cuda functions
	torch.cuda.memory_allocated = _safe_wrap(torch.cuda.memory_allocated, 0)
	torch.cuda.max_memory_allocated = _safe_wrap(torch.cuda.max_memory_allocated, 0)
	torch.cuda.memory_reserved = _safe_wrap(torch.cuda.memory_reserved, 0)
	torch.cuda.max_memory_reserved = _safe_wrap(torch.cuda.max_memory_reserved, 0)
	torch.cuda.memory_stats = _safe_wrap(torch.cuda.memory_stats, {})
	torch.cuda.memory_stats_as_nested_dict = _safe_wrap(torch.cuda.memory_stats_as_nested_dict, {})
	torch.cuda.reset_peak_memory_stats = _safe_wrap(torch.cuda.reset_peak_memory_stats, None)
	torch.cuda.reset_accumulated_memory_stats = _safe_wrap(
		torch.cuda.reset_accumulated_memory_stats, None
	)
	torch.cuda.reset_max_memory_allocated = _safe_wrap(
		torch.cuda.reset_max_memory_allocated, None
	)
	torch.cuda.reset_max_memory_cached = _safe_wrap(
		getattr(torch.cuda, "reset_max_memory_cached", lambda *a, **k: None), None
	)

	# 3) Patch torch.cuda.memory module too
	import torch.cuda.memory as tcm
	tcm.memory_allocated = _safe_wrap(tcm.memory_allocated, 0)
	tcm.max_memory_allocated = _safe_wrap(tcm.max_memory_allocated, 0)
	tcm.memory_reserved = _safe_wrap(tcm.memory_reserved, 0)
	tcm.max_memory_reserved = _safe_wrap(tcm.max_memory_reserved, 0)
	tcm.memory_stats = _safe_wrap(tcm.memory_stats, {})
	tcm.memory_stats_as_nested_dict = _safe_wrap(tcm.memory_stats_as_nested_dict, {})
	tcm.reset_peak_memory_stats = _safe_wrap(tcm.reset_peak_memory_stats, None)
	tcm.reset_accumulated_memory_stats = _safe_wrap(
		tcm.reset_accumulated_memory_stats, None
	)
	tcm.reset_max_memory_allocated = _safe_wrap(tcm.reset_max_memory_allocated, None)
	if hasattr(tcm, "reset_max_memory_cached"):
		tcm.reset_max_memory_cached = _safe_wrap(tcm.reset_max_memory_cached, None)

	# 4) DeepSpeed
	try:
		import deepspeed.runtime.utils as ds_utils
		_ds_orig = ds_utils.see_memory_usage

		def _ds_safe(*a, **k):
			try:
				return _ds_orig(*a, **k)
			except RuntimeError as e:
				if _is_pluggable_allocator_unsupported(e):
					return
				raise

		ds_utils.see_memory_usage = _ds_safe
		print("[patch] DeepSpeed memory logger patched.")
	except Exception as e:
		print(f"[patch] DeepSpeed patch skipped: {e}")

patch_rmm_pluggable_allocator_compat()
