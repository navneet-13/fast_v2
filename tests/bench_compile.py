#!/usr/bin/env python3
"""
Benchmark: Eager vs Compiled (compact) single forward pass.

Measures ONLY the forward pass latency — no generation loop, no sampling.
The compiled path is warmed up first so CUDA graph recording overhead is
excluded from the measurement.

Varies past_seq_len from 5*block_size to 10*block_size (i.e. 160..320 tokens
in KV cache) to simulate different points during block diffusion generation.

Each mode (eager / compiled) runs in a SEPARATE subprocess to avoid OOM
on low-VRAM GPUs (A5000 22 GB) — compiled CUDA graphs pin ~6 GB that
would prevent eager from allocating intermediates.

Usage:
    cd /path/to/fast_v2
    CUDA_VISIBLE_DEVICES=3 python tests/bench_compile.py [--gpu_id 0]
"""

import argparse
import json
import os
import subprocess
import sys

FAST_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "FAST_DLLM_MODEL_PATH",
    os.path.join(
        FAST_V2_ROOT,
        "models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/"
        "snapshots/0661abf5f9f0ee338970d091052a26c8efa51974",
    ),
)
BLOCK_SIZE = 32
BATCH_SIZES = [1, 2, 4, 8, 16]
PAST_BLOCKS_RANGE = list(range(5, 36, 10))  # [5, 15, 25, 35]
N_WARMUP = 10
N_MEASURE = 50


# ═══════════════════════════════════════════════════════════════════════════
# Worker: runs in a subprocess for a single mode (eager or compiled)
# ═══════════════════════════════════════════════════════════════════════════

def run_worker(mode, gpu_id, batch_size):
    """Run benchmarks for one mode + batch_size in a clean subprocess."""
    env = os.environ.copy()
    env["BENCH_MODE"] = mode
    env["BENCH_GPU_ID"] = str(gpu_id)
    env["BENCH_BATCH_SIZE"] = str(batch_size)
    env["BENCH_MODEL_PATH"] = MODEL_PATH
    env["BENCH_PAST_BLOCKS"] = json.dumps(PAST_BLOCKS_RANGE)
    env["BENCH_N_WARMUP"] = str(N_WARMUP)
    env["BENCH_N_MEASURE"] = str(N_MEASURE)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    result = subprocess.run(
        [sys.executable, __file__, "--worker"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"[{mode}] STDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Worker {mode} failed with exit code {result.returncode}")

    # Parse JSON output from the last line of stdout
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("[") or line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output from {mode} worker.\nstdout:\n{result.stdout}")


def worker_main():
    """Subprocess entry point: benchmark one mode, print JSON results."""
    import time
    import torch

    sys.path.insert(0, FAST_V2_ROOT)
    from transformers import AutoModelForCausalLM
    from utils.static_kv_cache import StaticKVCache
    from utils.attention_backends import get_attention_backend, patch_attention_layers
    from utils.dynamo_utils import make_compiled_forward, make_eager_forward, warmup_compile_pools

    mode = os.environ["BENCH_MODE"]
    gpu_id = int(os.environ["BENCH_GPU_ID"])
    model_path = os.environ["BENCH_MODEL_PATH"]
    past_blocks_list = json.loads(os.environ["BENCH_PAST_BLOCKS"])
    n_warmup = int(os.environ["BENCH_N_WARMUP"])
    n_measure = int(os.environ["BENCH_N_MEASURE"])
    block_size = BLOCK_SIZE
    batch_size = int(os.environ["BENCH_BATCH_SIZE"])

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # ── Load model ──
    print(f"[{mode}] Loading model...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # ── Cache + buffers ──
    config = model.config
    max_past = max(past_blocks_list) * block_size
    max_seq_len = ((max_past + block_size + 255) // 256) * 256

    cache = StaticKVCache(
        batch_size=batch_size, max_seq_len=max_seq_len,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=model.dtype, device=device,
    )
    x_buf = torch.zeros(batch_size, block_size, dtype=torch.long, device=device)
    cache_pos_buf = torch.zeros(block_size, dtype=torch.long, device=device)
    cache_pos_arange = torch.arange(block_size, dtype=torch.long, device=device)

    # ── Attention backend ──
    attn_backend = get_attention_backend("auto")
    print(f"[{mode}] Attention backend: {attn_backend.name}", file=sys.stderr)
    patch_attention_layers(model, attn_backend, None)
    model._static_kv_cache = cache

    # Move float attrs to CUDA
    for module in model.modules():
        for attr_name in ('variance_epsilon', 'attention_scaling', 'norm_type'):
            val = getattr(module, attr_name, None)
            if isinstance(val, float):
                setattr(module, attr_name, torch.tensor(val, dtype=torch.float64, device=device))

    # ── Create forward function ──
    if mode == "compiled":
        import torch.fx.experimental._config as fx_config
        fx_config.use_duck_shape = False
        forward_fn = make_compiled_forward(model, compile_mode="reduce-overhead")
        torch._dynamo.mark_static_address(x_buf)
        torch._dynamo.mark_static_address(cache_pos_buf)
        torch._dynamo.mark_static_address(cache_pos_arange)
        attn_backend.mark_dispatch_logged()

        # Pre-record CUDA graph
        print(f"[{mode}] Recording CUDA graph...", file=sys.stderr)
        warmup_compile_pools(
            compiled_fn=forward_fn,
            compile_pools={batch_size: {"cache": cache, "x_buf": x_buf}},
            model=model, block_size=block_size,
            cache_pos_buf=cache_pos_buf, cache_pos_arange=cache_pos_arange,
            attn_backend=attn_backend, config=config, max_seq_len=max_seq_len,
            dtype=model.dtype, device=device,
        )
        print(f"[{mode}] CUDA graph recorded.", file=sys.stderr)
    else:
        forward_fn = make_eager_forward(model)
        attn_backend.mark_dispatch_logged()

    # ── Benchmark loop ──
    results = []
    for past_blocks in past_blocks_list:
        past_len = past_blocks * block_size

        # Set cache state
        cache.zero_and_reset()
        cache._seq_len = past_len
        cache.cache_seqlens.fill_(past_len)
        cache.set_kv_write_start(past_len)
        cache.set_scratch_seqlens(past_len + block_size)
        cache.prepare_write_idx(block_size)
        cache_pos_buf.copy_(cache_pos_arange + past_len)
        x_buf.copy_(torch.randint(0, 1000, (batch_size, block_size), device=device))

        torch.cuda.empty_cache()

        # Warm-up
        with torch.no_grad():
            for _ in range(n_warmup):
                if mode == "compiled":
                    torch.compiler.cudagraph_mark_step_begin()
                _ = forward_fn(
                    input_ids=x_buf, use_cache=True, past_key_values=cache,
                    update_past_key_values=False, cache_position=cache_pos_buf,
                    logits_to_keep=block_size,
                )
            torch.cuda.synchronize()

        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(n_measure):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if mode == "compiled":
                    torch.compiler.cudagraph_mark_step_begin()
                _ = forward_fn(
                    input_ids=x_buf, use_cache=True, past_key_values=cache,
                    update_past_key_values=False, cache_position=cache_pos_buf,
                    logits_to_keep=block_size,
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1e3)

        latencies.sort()
        trim = max(1, len(latencies) // 10)
        trimmed = latencies[trim:-trim]

        results.append({
            "past_blocks": past_blocks,
            "past_seq_len": past_len,
            "mean_ms": sum(trimmed) / len(trimmed),
            "median_ms": latencies[len(latencies) // 2],
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
        })
        print(f"[{mode}] past_blocks={past_blocks}: "
              f"mean={results[-1]['mean_ms']:.3f} ms", file=sys.stderr)

    # Output JSON on stdout (the parent process reads this)
    print(json.dumps(results))


# ═══════════════════════════════════════════════════════════════════════════
# Main: orchestrate subprocesses and print comparison table
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main()
        return

    import torch
    device = torch.device(f"cuda:{args.gpu_id}")
    gpu_name = torch.cuda.get_device_name(device)

    print(f"GPU: {gpu_name}")
    print(f"Block size: {BLOCK_SIZE}, Batch sizes: {BATCH_SIZES}")
    print(f"Past blocks range: {PAST_BLOCKS_RANGE}")
    print(f"Warmup: {N_WARMUP}, Measure: {N_MEASURE}")
    print()

    # ── Collect results for all batch sizes ──
    all_results = {}  # {batch_size: {"eager": [...], "compiled": [...]}}

    for bs in BATCH_SIZES:
        all_results[bs] = {}
        for mode in ("eager", "compiled"):
            print(f"Running {mode} bs={bs} (subprocess)...")
            try:
                results = run_worker(mode, args.gpu_id, bs)
                all_results[bs][mode] = results
                print(f"  Done: {len(results)} configs")
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                all_results[bs][mode] = None
        print()

    # ── Print per-batch-size detailed tables ────────────────────────────────
    print("=" * 90)
    print(f"  FORWARD PASS BENCHMARK: Eager vs Compiled (reduce-overhead + CUDA graphs)")
    print(f"  GPU: {gpu_name}")
    print(f"  Model: Fast_dLLM_v2_7B (bf16), block_size={BLOCK_SIZE}")
    print(f"  Warmup: {N_WARMUP} iters, Measured: {N_MEASURE} iters "
          f"(trimmed mean, drop top/bottom 10%)")
    print("=" * 90)

    for bs in BATCH_SIZES:
        eager = all_results[bs]["eager"]
        compiled = all_results[bs]["compiled"]
        if eager is None or compiled is None:
            print(f"\n  batch_size={bs}: SKIPPED (OOM)")
            continue

        print(f"\n  ── batch_size={bs} ──")
        print(f"  {'Past':>4s}  {'Seq Len':>7s}  {'Eager':>10s}  {'Compiled':>10s}  "
              f"{'Speedup':>8s}  {'Eager Med':>10s}  {'Comp Med':>10s}")
        print(f"  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

        for re, rc in zip(eager, compiled):
            sp = re["mean_ms"] / rc["mean_ms"] if rc["mean_ms"] > 0 else float("inf")
            print(f"  {re['past_blocks']:>4d}  {re['past_seq_len']:>7d}  "
                  f"{re['mean_ms']:>10.3f}  {rc['mean_ms']:>10.3f}  "
                  f"{sp:>7.2f}x  "
                  f"{re['median_ms']:>10.3f}  {rc['median_ms']:>10.3f}")

        avg_e = sum(r["mean_ms"] for r in eager) / len(eager)
        avg_c = sum(r["mean_ms"] for r in compiled) / len(compiled)
        sp = avg_e / avg_c if avg_c > 0 else float("inf")
        print(f"  {'Avg':>4s}  {'':>7s}  {avg_e:>10.3f}  {avg_c:>10.3f}  {sp:>7.2f}x")

    # ── Summary table across batch sizes ────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: Average forward latency by batch size")
    print(f"  {'BS':>4s}  {'Eager (ms)':>10s}  {'Compiled (ms)':>13s}  {'Speedup':>8s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*13}  {'-'*8}")
    for bs in BATCH_SIZES:
        eager = all_results[bs]["eager"]
        compiled = all_results[bs]["compiled"]
        if eager is None or compiled is None:
            print(f"  {bs:>4d}  {'OOM':>10s}  {'OOM':>13s}  {'N/A':>8s}")
            continue
        avg_e = sum(r["mean_ms"] for r in eager) / len(eager)
        avg_c = sum(r["mean_ms"] for r in compiled) / len(compiled)
        sp = avg_e / avg_c if avg_c > 0 else float("inf")
        print(f"  {bs:>4d}  {avg_e:>10.3f}  {avg_c:>13.3f}  {sp:>7.2f}x")
    print("=" * 90)


if __name__ == "__main__":
    main()
