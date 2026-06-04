#!/usr/bin/env python3
"""
Lightweight forward-call-level benchmark: compile vs eager at BS=1.

Uses torch.cuda.Event timing (near-zero overhead) to measure individual
forward calls, NOT torch.profiler (which adds 3-25x overhead to eager).

For kernel-level breakdown, uses torch.profiler on a SINGLE forward call
(not 20 samples) to minimize profiler overhead impact.

Usage:
    cd fast_v2
    python debug/bench_forward_bs1.py

Output: per-forward-call timing, kernel breakdown for 1 call, and summary.
"""

import os
import sys
import time
import torch
import json

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
    "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/"
)

def load_model():
    """Load model once, return model + tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, tokenizer


def bench_single_forward(model, input_ids, static_cache, cache_pos,
                          forward_fn, mode_label, n_warmup=5, n_measure=50):
    """
    Time a single forward call using CUDA events.
    Returns list of (gpu_ms, wall_ms) tuples.
    """
    block_size = input_ids.shape[1]

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = forward_fn(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=static_cache,
                update_past_key_values=False,
                cache_position=cache_pos,
            )
        torch.cuda.synchronize()

    # Measure
    results = []
    for _ in range(n_measure):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_ev.record()

        with torch.no_grad():
            _ = forward_fn(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=static_cache,
                update_past_key_values=False,
                cache_position=cache_pos,
            )

        end_ev.record()
        torch.cuda.synchronize()
        wall_end = time.perf_counter()

        gpu_ms = start_ev.elapsed_time(end_ev)
        wall_ms = (wall_end - wall_start) * 1000
        results.append((gpu_ms, wall_ms))

    return results


def profile_single_forward(model, input_ids, static_cache, cache_pos,
                            forward_fn, mode_label, n_warmup=5):
    """
    Profile ONE forward call with torch.profiler for kernel breakdown.
    Minimal overhead since it's just 1 call (not hundreds).
    """
    # Warmup outside profiler
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = forward_fn(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=static_cache,
                update_past_key_values=False,
                cache_position=cache_pos,
            )
        torch.cuda.synchronize()

    # Profile exactly 1 forward
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = forward_fn(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=static_cache,
                update_past_key_values=False,
                cache_position=cache_pos,
            )
        torch.cuda.synchronize()

    return prof


def setup_eager(model, block_size=32, max_seq_len=2048):
    """Setup eager mode: static cache + patched attention, no compile."""
    from utils.static_kv_cache import StaticKVCache
    from utils.attention_backends import (
        get_attention_backend, patch_attention_layers,
    )
    import accelerate.hooks

    # Remove accelerate hooks
    for mod in model.modules():
        accelerate.hooks.remove_hook_from_module(mod)

    config = model.config
    attn_backend = get_attention_backend("auto")
    print(f"  Attention backend: {attn_backend.name}")

    static_cache = StaticKVCache(
        batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=model.dtype,
        device=model.device,
    )

    original_forwards = patch_attention_layers(model, attn_backend)

    # Setup cache for block at position 0
    static_cache.set_kv_write_start(0)
    static_cache.set_scratch_seqlens(block_size)
    static_cache.prepare_write_idx(block_size)

    cache_pos = torch.arange(block_size, device=model.device)
    input_ids = torch.randint(0, 1000, (1, block_size), device=model.device)

    def eager_forward(**kwargs):
        return model.forward_dynamo(**kwargs)

    return eager_forward, static_cache, cache_pos, input_ids, original_forwards


def setup_compile(model, block_size=32, max_seq_len=2048):
    """Setup compile mode: static cache + patched attention + torch.compile."""
    from utils.static_kv_cache import StaticKVCache
    from utils.attention_backends import (
        get_attention_backend, patch_attention_layers_compiled,
    )
    from utils.dynamo_utils import make_compiled_forward
    import accelerate.hooks

    # Remove accelerate hooks
    for mod in model.modules():
        accelerate.hooks.remove_hook_from_module(mod)

    # Convert float attrs to tensors for compile compat
    for mod in model.modules():
        if hasattr(mod, 'variance_epsilon') and isinstance(mod.variance_epsilon, float):
            mod.variance_epsilon = torch.tensor(
                mod.variance_epsilon, dtype=torch.float64, device=model.device
            )

    config = model.config
    attn_backend = get_attention_backend("auto")
    print(f"  Attention backend: {attn_backend.name}")

    static_cache = StaticKVCache(
        batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=model.dtype,
        device=model.device,
    )

    original_forwards = patch_attention_layers_compiled(model, attn_backend)

    # Mark dispatch logged to avoid guard failure
    attn_backend.mark_dispatch_logged()

    # Setup cache
    static_cache.set_kv_write_start(0)
    static_cache.set_scratch_seqlens(block_size)
    static_cache.prepare_write_idx(block_size)

    cache_pos = torch.arange(block_size, device=model.device)
    torch._dynamo.mark_static_address(cache_pos)

    input_ids = torch.randint(0, 1000, (1, block_size), device=model.device)
    torch._dynamo.mark_static_address(input_ids)

    compiled_forward = make_compiled_forward(model, compile_mode="reduce-overhead")

    return compiled_forward, static_cache, cache_pos, input_ids, original_forwards


def print_results(eager_times, compile_times):
    """Print comparison table."""
    import statistics

    def stats(times, idx):
        vals = [t[idx] for t in times]
        return {
            'mean': statistics.mean(vals),
            'median': statistics.median(vals),
            'stdev': statistics.stdev(vals) if len(vals) > 1 else 0,
            'min': min(vals),
            'max': max(vals),
        }

    eg = stats(eager_times, 0)   # GPU ms
    ew = stats(eager_times, 1)   # wall ms
    cg = stats(compile_times, 0)
    cw = stats(compile_times, 1)

    print(f"\n{'='*80}")
    print(f"FORWARD CALL TIMING (BS=1, block_size=32, {len(eager_times)} calls)")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Eager':>12} {'Compile':>12} {'Speedup':>10}")
    print(f"{'-'*60}")
    print(f"{'GPU time mean (ms)':<25} {eg['mean']:>12.3f} {cg['mean']:>12.3f} {eg['mean']/cg['mean']:>10.2f}x")
    print(f"{'GPU time median (ms)':<25} {eg['median']:>12.3f} {cg['median']:>12.3f} {eg['median']/cg['median']:>10.2f}x")
    print(f"{'GPU time stdev (ms)':<25} {eg['stdev']:>12.3f} {cg['stdev']:>12.3f}")
    print(f"{'Wall time mean (ms)':<25} {ew['mean']:>12.3f} {cw['mean']:>12.3f} {ew['mean']/cw['mean']:>10.2f}x")
    print(f"{'Wall time median (ms)':<25} {ew['median']:>12.3f} {cw['median']:>12.3f} {ew['median']/cw['median']:>10.2f}x")
    print(f"{'Wall time stdev (ms)':<25} {ew['stdev']:>12.3f} {cw['stdev']:>12.3f}")
    print(f"{'CPU overhead mean (ms)':<25} {ew['mean']-eg['mean']:>12.3f} {cw['mean']-cg['mean']:>12.3f}")
    print(f"{'CPU overhead % of wall':<25} {(ew['mean']-eg['mean'])/ew['mean']*100:>11.1f}% {(cw['mean']-cg['mean'])/cw['mean']*100:>11.1f}%")


def print_kernel_comparison(eager_prof, compile_prof):
    """Print kernel breakdown from single-call profiles."""
    print(f"\n{'='*80}")
    print(f"KERNEL BREAKDOWN — SINGLE FORWARD CALL (torch.profiler, 1 call each)")
    print(f"{'='*80}")

    for label, prof in [("EAGER", eager_prof), ("COMPILE", compile_prof)]:
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        print(f"\n--- {label} (top 20 by CUDA time) ---")
        print(table)

    # Extract key averages for comparison
    eager_ka = eager_prof.key_averages()
    compile_ka = compile_prof.key_averages()

    e_cuda_total = sum(e.cuda_time_total for e in eager_ka)
    c_cuda_total = sum(e.cuda_time_total for e in compile_ka)
    e_cpu_total = sum(e.cpu_time_total for e in eager_ka)
    c_cpu_total = sum(e.cpu_time_total for e in compile_ka)
    e_calls = sum(e.count for e in eager_ka)
    c_calls = sum(e.count for e in compile_ka)

    print(f"\n{'Metric':<35} {'Eager':>12} {'Compile':>12}")
    print(f"{'-'*60}")
    print(f"{'Total CUDA time (ms)':<35} {e_cuda_total/1e3:>12.2f} {c_cuda_total/1e3:>12.2f}")
    print(f"{'Total CPU time (ms)':<35} {e_cpu_total/1e3:>12.2f} {c_cpu_total/1e3:>12.2f}")
    print(f"{'Total op calls':<35} {e_calls:>12,} {c_calls:>12,}")


def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print(f"Model loaded: {model.dtype}, {model.device}")

    block_size = 32
    max_seq_len = 2048
    n_warmup = 10
    n_measure = 100

    # ── Eager ──
    print(f"\n{'='*80}")
    print("EAGER MODE SETUP")
    print(f"{'='*80}")
    eager_fn, e_cache, e_cpos, e_ids, e_orig_fwd = setup_eager(model, block_size, max_seq_len)

    print(f"Benchmarking eager ({n_warmup} warmup, {n_measure} measured)...")
    eager_times = bench_single_forward(
        model, e_ids, e_cache, e_cpos, eager_fn, "eager",
        n_warmup=n_warmup, n_measure=n_measure,
    )

    print(f"Profiling eager (single forward call)...")
    eager_prof = profile_single_forward(
        model, e_ids, e_cache, e_cpos, eager_fn, "eager", n_warmup=5,
    )

    # Unpatch eager attention forwards before setting up compile
    from utils.attention_backends import unpatch_attention_layers
    unpatch_attention_layers(model, e_orig_fwd)

    # ── Compile ──
    print(f"\n{'='*80}")
    print("COMPILE MODE SETUP")
    print(f"{'='*80}")
    compile_fn, c_cache, c_cpos, c_ids, c_orig_fwd = setup_compile(model, block_size, max_seq_len)

    print(f"Benchmarking compile ({n_warmup} warmup, {n_measure} measured)...")
    compile_times = bench_single_forward(
        model, c_ids, c_cache, c_cpos, compile_fn, "compile",
        n_warmup=n_warmup, n_measure=n_measure,
    )

    print(f"Profiling compile (single forward call)...")
    compile_prof = profile_single_forward(
        model, c_ids, c_cache, c_cpos, compile_fn, "compile", n_warmup=5,
    )

    # ── Results ──
    print_results(eager_times, compile_times)
    print_kernel_comparison(eager_prof, compile_prof)

    # Save raw data
    out_path = os.path.join(os.path.dirname(__file__), "bench_forward_bs1_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "eager": [{"gpu_ms": g, "wall_ms": w} for g, w in eager_times],
            "compile": [{"gpu_ms": g, "wall_ms": w} for g, w in compile_times],
        }, f, indent=2)
    print(f"\nRaw data saved to {out_path}")


if __name__ == "__main__":
    main()
