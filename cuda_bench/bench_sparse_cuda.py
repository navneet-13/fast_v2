import sys
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_CUDA_BENCH = Path(__file__).resolve().parent
if str(_CUDA_BENCH) not in sys.path:
    sys.path.insert(0, str(_CUDA_BENCH))

# 1. Load the kernel (Ensure fused_gather_norm.cu is in the same directory)
fused_ops = load(
    name="fused_ops",
    sources=["fused_gather_norm.cu"],
    extra_cuda_cflags=['-O3'],
    verbose=True
)

def benchmark_gather_norm(batch=8, seq_len=4096, num_tokens=512, hidden_size=4096, iters=100):
    # Setup Data
    device = "cuda"
    dtype = torch.float32
    
    hidden_states = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    indices = torch.randint(0, seq_len, (batch, num_tokens), device=device).long()
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-5

    # --- APPROACH A: Standard PyTorch ---
    def pytorch_native():
        # Step 1: Gather
        idx_h = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        selected = hidden_states.gather(1, idx_h)
        # Step 2: RMSNorm (manual implementation for fair comparison)
        variance = selected.pow(2).mean(-1, keepdim=True)
        return selected * torch.rsqrt(variance + eps) * weight

    # --- APPROACH B: Fused CUDA ---
    def cuda_fused():
        return fused_ops.gather_rmsnorm_cuda(hidden_states, indices, weight, eps)

    # Warmup
    for _ in range(10):
        pytorch_native()
        cuda_fused()
    torch.cuda.synchronize()

    # Timing Function
    def time_func(fn):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iters):
            fn()
        end_event.record()
        
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / iters

    t_py = time_func(pytorch_native)
    t_cu = time_func(cuda_fused)

    # Verification
    out_py = pytorch_native()
    out_cu = cuda_fused()
    max_diff = (out_py - out_cu).abs().max().item()

    print(f"--- Benchmark Results (B={batch}, S={seq_len}, T={num_tokens}, H={hidden_size}) ---")
    print(f"PyTorch Native: {t_py:.4f} ms")
    print(f"CUDA Fused:     {t_cu:.4f} ms")
    print(f"Speedup:        {t_py/t_cu:.2f}x")
    print(f"Max Difference: {max_diff:.8e}")

def benchmark_scatter_roundtrip(batch=8, seq_len=4096, num_tokens=512, hidden_size=4096, iters=100):
    device = "cuda"
    dtype = torch.float32

    # Inputs: Computed sparse results
    sparse_values = torch.randn(batch, num_tokens, hidden_size, device=device, dtype=dtype)
    indices = torch.randint(0, seq_len, (batch, num_tokens), device=device).long()
    
    # The Persistent Cache
    global_cache = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)

    # --- APPROACH A: PyTorch "Round Trip" (Write then Read) ---
    def pytorch_roundtrip():
        # 1. Write to cache
        idx = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        global_cache.scatter_(1, idx, sparse_values)
        
        # 2. Read back the layer (This is what you'd pass to the next layer)
        # In PyTorch, 'global_cache' is already the full tensor, 
        # but accessing it involves memory synchronization overhead.
        return global_cache 

    # --- APPROACH B: CUDA "Direct Path" (Write and Return) ---
    def cuda_direct():
        # We pass a 'working' tensor that already has the previous layer's data.
        # The kernel updates the cache AND this working tensor simultaneously.
        working_tensor = global_cache.clone() 
        return fused_ops.scatter_and_return_cuda(
            sparse_values, indices, global_cache, working_tensor
        )

    # --- Timing Setup ---
    def time_func(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # Warmup
        for _ in range(10): fn()
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    t_py = time_func(pytorch_roundtrip)
    t_cu = time_func(cuda_direct)

    print(f"--- Scatter Round-Trip Benchmark (B={batch}, S={seq_len}, T={num_tokens}) ---")
    print(f"PyTorch (Scatter then Access): {t_py:.4f} ms")
    print(f"CUDA (Scatter & Return):       {t_cu:.4f} ms")
    print(f"Speedup:                       {t_py/t_cu:.2f}x")


def run_bench(B=8, S=4096, ratio=0.1, H=4096):
    T = int(S * ratio)
    device = "cuda"
    
    # Inputs
    hidden = torch.randn(B, S, H, device=device)
    indices = torch.randint(0, S, (B, T), device=device).long()
    weight = torch.randn(H, device=device)
    eps = 1e-5

    # --- PyTorch Native ---
    def pt_native():
        # Step 1: Gather
        idx_h = indices.unsqueeze(-1).expand(-1, -1, H)
        selected = hidden.gather(1, idx_h)
        # Step 2: RMSNorm
        variance = selected.pow(2).mean(-1, keepdim=True)
        return selected * torch.rsqrt(variance + eps) * weight

    # --- Fused CUDA ---
    def cuda_fused():
        return fused_ops.fused_select_norm(hidden, indices, weight, eps)

    # Warmup
    for _ in range(20):
        pt_native()
        cuda_fused()
    torch.cuda.synchronize()

    # Timing
    iters = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # PT Bench
    start.record()
    for _ in range(iters): pt_native()
    end.record()
    torch.cuda.synchronize()
    t_pt = start.elapsed_time(end) / iters

    # CUDA Bench
    start.record()
    for _ in range(iters): cuda_fused()
    end.record()
    torch.cuda.synchronize()
    t_cu = start.elapsed_time(end) / iters

    print(f"B={B}, S={S}, T={T}, H={H}")
    print(f"PyTorch Native: {t_pt:.4f} ms")
    print(f"CUDA Fused:     {t_cu:.4f} ms")
    print(f"Speedup:        {t_pt/t_cu:.2f}")


def benchmark_fused_gather_input_rmsnorm_pair(
    batch=8,
    seq_len=4096,
    num_tokens=1024,
    hidden_size=3584,
    dtype=torch.bfloat16,
    iters=100,
    warmup=20,
):
    """
    Time `fused_sparse_extension.fused_gather_input_rmsnorm_pair` (model path)
    vs PyTorch gather + fp32 RMSNorm + weight (same as test_correctness.ref_gather_input_rmsnorm).
    """
    import fused_sparse_extension as fse
    from test_correctness import _MockInputNorm, ref_gather_input_rmsnorm

    if not torch.cuda.is_available():
        print("benchmark_fused_gather_input_rmsnorm_pair: SKIP (no CUDA)")
        return

    if fse.get_fused_ops() is None:
        print(
            "benchmark_fused_gather_input_rmsnorm_pair: SKIP (extension failed to load; "
            "see test_correctness.py docstring for ninja/nvcc)."
        )
        return

    device = torch.device("cuda")
    torch.manual_seed(0)
    hidden_states = torch.randn(
        batch, seq_len, hidden_size, device=device, dtype=dtype
    )
    token_indices = torch.randint(
        0, seq_len, (batch, num_tokens), device=device, dtype=torch.long
    )
    input_layernorm = _MockInputNorm(hidden_size).to(device=device, dtype=dtype)

    w = input_layernorm.weight.data
    eps = float(input_layernorm.variance_epsilon)

    def pytorch_native():
        return ref_gather_input_rmsnorm(
            hidden_states, token_indices, w, eps
        )

    def cuda_fused():
        pair = fse.fused_gather_input_rmsnorm_pair(
            hidden_states, token_indices, input_layernorm
        )
        if pair is None:
            raise RuntimeError("fused_gather_input_rmsnorm_pair returned None")
        return pair

    for _ in range(warmup):
        pytorch_native()
        cuda_fused()
    torch.cuda.synchronize()

    def time_func(fn):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / iters

    t_py = time_func(pytorch_native)
    t_cu = time_func(cuda_fused)

    g_py, n_py = pytorch_native()
    g_cu, n_cu = cuda_fused()
    max_g = (g_py - g_cu).abs().max().item()
    max_n = (n_py - n_cu).abs().max().item()

    print(
        f"--- fused_gather_input_rmsnorm_pair vs PyTorch "
        f"(B={batch}, S={seq_len}, T={num_tokens}, H={hidden_size}, dtype={dtype}) ---"
    )
    print(f"PyTorch (gather + fp32 RMSNorm + weight): {t_py:.4f} ms")
    print(f"CUDA fused pair:                           {t_cu:.4f} ms")
    print(f"Speedup:                                   {t_py / t_cu:.2f}x")
    print(f"Max |diff| gathered: {max_g:.8e}  normed: {max_n:.8e}")


if __name__ == "__main__":
    # benchmark_gather_norm()
    # benchmark_scatter_roundtrip()
    benchmark_fused_gather_input_rmsnorm_pair()
    print()
    run_bench()