#!/usr/bin/env python3
"""
Correctness checks for fused gather + input RMSNorm (cuda_bench).

Compares fused_sparse_extension.fused_gather_input_rmsnorm_pair against a
PyTorch reference: gather along seq dim, then RMSNorm in float32
(rsqrt(mean(x^2) + eps) * x * weight), matching fused_gather_norm.cu.

Run (CUDA + nvcc; PyTorch JIT needs the ``ninja`` **executable** on ``PATH`` —
``pip install ninja`` into your venv, then use that venv's ``python`` so
``.../venv/bin/ninja`` is found, or ``export PATH=.../venv/bin:$PATH``):

  cd fast_v2/cuda_bench && python test_correctness.py
  python test_correctness.py --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure cuda_bench is importable when run as `python test_correctness.py`
_CUDA_BENCH = Path(__file__).resolve().parent
if str(_CUDA_BENCH) not in sys.path:
    sys.path.insert(0, str(_CUDA_BENCH))

import fused_sparse_extension as fse  # noqa: E402


def ref_gather_input_rmsnorm(
    hidden: torch.Tensor,
    token_indices: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as Fast_dLLM_QwenRMSNorm.forward on gathered rows: fp32 rms, then weight * x.to(dtype)."""
    h = hidden.float().contiguous()
    b, _, dim = h.shape
    t = token_indices.shape[1]
    idx = token_indices.unsqueeze(-1).expand(b, t, dim)
    gathered_f = h.gather(1, idx)
    var = (gathered_f * gathered_f).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    normed_pre = gathered_f * inv
    dt = hidden.dtype
    normed = weight * normed_pre.to(dt)
    return gathered_f.to(dt), normed


class _MockInputNorm(nn.Module):
    """Minimal stand-in for Fast_dLLM_QwenRMSNorm (weight + variance_epsilon)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


def assert_close(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> None:
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        diff = (a.float() - b.float()).abs()
        raise AssertionError(
            f"{name}: max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e} "
            f"rtol={rtol} atol={atol}"
        )


def run_case(
    *,
    b: int,
    seq_len: int,
    hidden: int,
    num_tokens: int,
    dtype: torch.dtype,
    eps: float,
    rtol: float,
    atol: float,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    hidden_states = torch.randn(b, seq_len, hidden, device=device, dtype=dtype)
    # valid indices in [0, seq_len)
    token_indices = torch.randint(0, seq_len, (b, num_tokens), device=device, dtype=torch.long)
    mod = _MockInputNorm(hidden, eps=eps).to(device=device, dtype=dtype)

    ref_g, ref_n = ref_gather_input_rmsnorm(
        hidden_states, token_indices, mod.weight.data, float(mod.variance_epsilon)
    )

    pair = fse.fused_gather_input_rmsnorm_pair(hidden_states, token_indices, mod)
    if pair is None:
        raise RuntimeError(
            "fused_gather_input_rmsnorm_pair returned None (extension failed to load). "
            "Check CUDA, nvcc, and fused_gather_norm.cu next to fused_sparse_extension.py."
        )
    fused_g, fused_n = pair

    assert_close("gathered", ref_g, fused_g, rtol=rtol, atol=atol)
    assert_close("normed", ref_n, fused_n, rtol=rtol, atol=atol)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller shapes only (faster smoke test).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    # Extension JIT load once
    if fse.get_fused_ops() is None:
        print(
            "FAIL: could not load fused CUDA extension.\n"
            "  - pip install ninja\n"
            "  - nvcc on PATH, CUDA matches PyTorch build\n"
            "  - fused_gather_norm.cu next to fused_sparse_extension.py"
        )
        return 1

    # bf16 needs looser tol after round-trip through fp32 kernel
    cases_bf16 = [
        (1, 64, 128, 7, 1e-6, 2e-3, 2e-3),
        (2, 128, 256, 16, 1e-5, 2e-3, 2e-3),
    ]
    cases_fp32 = [
        (1, 32, 64, 5, 1e-6, 1e-5, 1e-5),
        (3, 48, 96, 11, 1e-6, 1e-5, 1e-5),
    ]
    if not args.quick:
        cases_bf16.extend(
            [
                (1, 512, 3584, 32, 1e-6, 5e-2, 5e-2),  # model-like H; wider tol for bf16
                (2, 256, 512, 24, 1e-6, 2e-3, 2e-3),
            ]
        )
        cases_fp32.append((1, 128, 512, 40, 1e-6, 1e-4, 1e-4))

    n_ok = 0
    for b, s, h, t, eps, rtol, atol in cases_bf16:
        run_case(b=b, seq_len=s, hidden=h, num_tokens=t, dtype=torch.bfloat16, eps=eps, rtol=rtol, atol=atol)
        n_ok += 1
        print(f"OK bf16  B={b} S={s} H={h} T={t} eps={eps}")

    for b, s, h, t, eps, rtol, atol in cases_fp32:
        run_case(b=b, seq_len=s, hidden=h, num_tokens=t, dtype=torch.float32, eps=eps, rtol=rtol, atol=atol)
        n_ok += 1
        print(f"OK fp32  B={b} S={s} H={h} T={t} eps={eps}")

    print(f"All {n_ok} cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
