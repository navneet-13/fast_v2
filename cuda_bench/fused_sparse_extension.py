"""
JIT CUDA extension for fused index-gather + RMSNorm (sparse forward path).

Enable from modeling with:
  export FAST_DLLM_FUSED_GATHER_INPUT_NORM=1

Requires: fused_gather_norm.cu next to this file, CUDA toolkit, nvcc visible to torch.utils.cpp_extension.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

_EXT: Any = None
_EXT_FAILED = False


def _cuda_bench_dir() -> Path:
    return Path(__file__).resolve().parent


def get_fused_ops():
    """Load extension once; return module or None if unavailable."""
    global _EXT, _EXT_FAILED
    if _EXT_FAILED:
        return None
    if _EXT is not None:
        return _EXT
    cu = _cuda_bench_dir() / "fused_gather_norm.cu"
    if not cu.is_file():
        _EXT_FAILED = True
        return None
    try:
        from torch.utils.cpp_extension import load

        _EXT = load(
            name="fused_gather_rmsnorm_model_v2",
            sources=[str(cu)],
            extra_cuda_cflags=["-O3"],
            verbose=os.environ.get("FAST_DLLM_FUSED_GATHER_VERBOSE", "0") == "1",
        )
    except Exception:
        _EXT_FAILED = True
        _EXT = None
    return _EXT


def fused_gather_input_rmsnorm_pair(
    hidden_states: torch.Tensor,
    token_indices: torch.Tensor,
    input_layernorm: torch.nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Match Fast_dLLM_QwenRMSNorm on gathered rows (same order as forward()):
      fp32 variance, normalized_fp32 = x * rsqrt(var + eps), then
      return weight * normalized_fp32.to(input_dtype) — not (x*rsqrt*w).to(dtype).

    Args:
        hidden_states: (B, S, H)
        token_indices: (B, T) int64
        input_layernorm: module with .weight and .variance_epsilon

    Returns:
        (selected_hidden, selected_norm) same device/dtype as hidden_states, or None to fall back to PyTorch.
    """
    if not hidden_states.is_cuda:
        return None
    ops = get_fused_ops()
    if ops is None:
        return None

    w = getattr(input_layernorm, "weight", None)
    if w is None:
        return None
    eps = float(getattr(input_layernorm, "variance_epsilon", 1e-6))

    # Kernel: gather + (x * rsqrt) in fp32; weight applied in dtype like Fast_dLLM_QwenRMSNorm.forward.
    h_f = hidden_states.float().contiguous()
    idx = token_indices.long().contiguous()
    pair = ops.fused_gather_input_rmsnorm_pair(h_f, idx, eps)
    gathered, normed_pre = pair[0], pair[1]
    dt = hidden_states.dtype
    normed = w * normed_pre.to(dt)
    return gathered.to(dt), normed
