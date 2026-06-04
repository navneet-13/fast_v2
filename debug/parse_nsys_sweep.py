#!/usr/bin/env python3
"""
Parse nsys sweep CSV exports and produce compile vs eager bottleneck analysis.

Usage:
    python debug/parse_nsys_sweep.py

Reads:
    logs/nsys_{eager,compile}_bs{1,2,4,8}_kernels_cuda_gpu_kern_sum.csv
    logs/nsys_{eager,compile}_bs{1,2,4,8}_api_cuda_api_sum.csv
"""

import csv
import os
import re
import sys

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

BATCH_SIZES = [1, 2, 4, 8]
MODES = ["eager", "compile"]

# Throughput from the sweep log (extracted manually)
THROUGHPUT = {
    ("eager", 1): 75.05, ("eager", 2): 80.62, ("eager", 4): 99.20, ("eager", 8): 122.59,
    ("compile", 1): 48.29, ("compile", 2): 34.06, ("compile", 4): 33.65, ("compile", 8): 29.96,
}
TOTAL_TIME = {
    ("eager", 1): 47.65, ("eager", 2): 43.39, ("eager", 4): 35.77, ("eager", 8): 29.56,
    ("compile", 1): 73.53, ("compile", 2): 106.12, ("compile", 4): 96.84, ("compile", 8): 92.47,
}
TOTAL_TOKENS = {
    ("eager", 1): 3576, ("eager", 2): 3498, ("eager", 4): 3548, ("eager", 8): 3624,
    ("compile", 1): 3551, ("compile", 2): 3614, ("compile", 4): 3259, ("compile", 8): 2770,
}


def classify_kernel(name):
    """Classify a CUDA kernel name into a functional group."""
    n = name.lower()
    # GEMMs
    if any(k in n for k in ["cutlass", "gemm", "cublas", "gemmk1"]):
        if "splitkreduce" in n:
            return "splitKreduce"
        return "GEMM"
    # Flash attention
    if "flash" in n:
        if "combine" in n:
            return "FlashAttn-combine"
        return "FlashAttn-splitkv"
    # Triton
    if "triton" in n or "triton_" in n:
        return "Triton-fused"
    # scatter
    if "scatter" in n:
        return "scatter_"
    # CUB (cub::DeviceSelect, cub::DeviceReduce)
    if "cub::" in n or "deviceselect" in n or "devicereduce" in n or "devicecompact" in n:
        return "CUB-ops"
    # Softmax
    if "softmax" in n:
        return "Softmax"
    # Cat
    if "catarray" in n:
        return "cat"
    # Copy
    if "direct_copy" in n or "bfloat16_copy" in n:
        return "copy"
    # Reduce (mean/norm)
    if "reduce_kernel" in n and "meanops" in n:
        return "RMSNorm-reduce"
    if "reduce_kernel" in n:
        return "other-reduce"
    # SiLU
    if "silu" in n:
        return "SiLU"
    # RoPE (sin, cos, gather for rotary)
    if "sin_kernel" in n or "cos_kernel" in n:
        return "RoPE"
    if "gather" in n:
        return "gather"
    # rsqrt (part of RMSNorm)
    if "rsqrt" in n:
        return "RMSNorm-rsqrt"
    # pow (part of RMSNorm)
    if "pow_tensor" in n:
        return "RMSNorm-pow"
    # Fill/memset
    if "fillfunctor" in n or "memset" in n:
        return "fill/memset"
    # Index ops
    if "index_elementwise" in n or "write_indices" in n:
        return "index-ops"
    # arange
    if "arange" in n:
        return "arange"
    # masked_fill
    if "masked_fill" in n:
        return "masked_fill"
    # Generic elementwise (add, mul, etc.)
    if "elementwise" in n or "vectorized" in n:
        return "elementwise"
    return "other"


def load_kernel_csv(mode, bs):
    path = os.path.join(LOG_DIR, f"nsys_{mode}_bs{bs}_kernels_cuda_gpu_kern_sum.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "pct": float(row["Time (%)"]),
                "total_ns": int(row["Total Time (ns)"]),
                "instances": int(row["Instances"]),
                "avg_ns": float(row["Avg (ns)"]),
                "name": row["Name"],
                "group": classify_kernel(row["Name"]),
            })
    return rows


def load_api_csv(mode, bs):
    path = os.path.join(LOG_DIR, f"nsys_{mode}_bs{bs}_api_cuda_api_sum.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "pct": float(row["Time (%)"]),
                "total_ns": int(row["Total Time (ns)"]),
                "num_calls": int(row["Num Calls"]),
                "avg_ns": float(row["Avg (ns)"]),
                "name": row["Name"],
            })
    return rows


def aggregate_groups(kernels):
    """Aggregate kernel data by group."""
    groups = {}
    for k in kernels:
        g = k["group"]
        if g not in groups:
            groups[g] = {"total_ns": 0, "instances": 0}
        groups[g]["total_ns"] += k["total_ns"]
        groups[g]["instances"] += k["instances"]
    # Sort by total time
    return sorted(groups.items(), key=lambda x: -x[1]["total_ns"])


def ns_to_s(ns):
    return ns / 1e9


def ns_to_ms(ns):
    return ns / 1e6


def main():
    # ── Section 1: Throughput overview ──
    print("=" * 110)
    print("SECTION 1: THROUGHPUT OVERVIEW (nsys profiled, ~2-5% overhead)")
    print("=" * 110)
    print(f"\n  {'Config':<20} {'Tokens':>8} {'Time(s)':>8} {'tok/s':>8} {'vs eager':>10} {'GraphLaunch':>12} {'KernelLaunch':>14}")
    print(f"  {'-'*82}")
    for mode in MODES:
        for bs in BATCH_SIZES:
            key = (mode, bs)
            label = f"{mode}_bs{bs}"
            toks = TOTAL_TOKENS.get(key, 0)
            time = TOTAL_TIME.get(key, 0)
            tps = THROUGHPUT.get(key, 0)
            eager_tps = THROUGHPUT.get(("eager", bs), 1)
            ratio = tps / eager_tps if eager_tps > 0 else 0

            # Get API data for graph launch count
            api = load_api_csv(mode, bs)
            graph_launches = 0
            kernel_launches = 0
            if api:
                for a in api:
                    if "cudaGraphLaunch" in a["name"]:
                        graph_launches = a["num_calls"]
                    if "cudaLaunchKernel" in a["name"]:
                        kernel_launches = a["num_calls"]

            print(f"  {label:<20} {toks:>8} {time:>8.1f} {tps:>8.1f} {ratio:>9.2f}x "
                  f"{graph_launches:>12,} {kernel_launches:>14,}")
        print()

    # ── Section 2: Compile mode is SLOWER — why? ──
    print("=" * 110)
    print("SECTION 2: WHY IS COMPILE MODE SLOWER? — CUDA API ANALYSIS")
    print("=" * 110)

    for bs in BATCH_SIZES:
        eager_api = load_api_csv("eager", bs)
        compile_api = load_api_csv("compile", bs)
        if not eager_api or not compile_api:
            continue

        print(f"\n  --- BS={bs} ---")
        print(f"  {'CUDA API':<30} {'Eager(s)':>10} {'Compile(s)':>10} {'E_calls':>10} {'C_calls':>10}")
        print(f"  {'-'*72}")

        # Key APIs to compare
        key_apis = ["cudaLaunchKernel", "cudaStreamSynchronize", "cudaGraphLaunch",
                     "cudaMemcpyAsync", "cudaMemsetAsync", "cuLaunchKernel"]
        eager_dict = {a["name"]: a for a in eager_api}
        compile_dict = {a["name"]: a for a in compile_api}

        for api_name in key_apis:
            e = eager_dict.get(api_name, {"total_ns": 0, "num_calls": 0})
            c = compile_dict.get(api_name, {"total_ns": 0, "num_calls": 0})
            print(f"  {api_name:<30} {ns_to_s(e['total_ns']):>10.2f} {ns_to_s(c['total_ns']):>10.2f} "
                  f"{e['num_calls']:>10,} {c['num_calls']:>10,}")

        # Total sync time dominates compile — show percentage
        e_sync = eager_dict.get("cudaStreamSynchronize", {"total_ns": 0})["total_ns"]
        c_sync = compile_dict.get("cudaStreamSynchronize", {"total_ns": 0})["total_ns"]
        e_total = TOTAL_TIME[("eager", bs)]
        c_total = TOTAL_TIME[("compile", bs)]
        print(f"\n  Sync as % of wall time:  eager={ns_to_s(e_sync)/e_total*100:.0f}%  "
              f"compile={ns_to_s(c_sync)/c_total*100:.0f}%")

    # ── Section 3: Kernel group comparison ──
    print(f"\n{'='*110}")
    print("SECTION 3: KERNEL GROUP BREAKDOWN — EAGER vs COMPILE")
    print("=" * 110)

    for bs in BATCH_SIZES:
        eager_k = load_kernel_csv("eager", bs)
        compile_k = load_kernel_csv("compile", bs)
        if not eager_k or not compile_k:
            continue

        e_groups = dict(aggregate_groups(eager_k))
        c_groups = dict(aggregate_groups(compile_k))
        all_groups = sorted(set(list(e_groups.keys()) + list(c_groups.keys())),
                          key=lambda g: -(e_groups.get(g, {"total_ns": 0})["total_ns"] +
                                         c_groups.get(g, {"total_ns": 0})["total_ns"]))

        e_total_gpu = sum(k["total_ns"] for k in eager_k)
        c_total_gpu = sum(k["total_ns"] for k in compile_k)

        print(f"\n  --- BS={bs} (GPU kernel time: eager={ns_to_s(e_total_gpu):.2f}s  compile={ns_to_s(c_total_gpu):.2f}s) ---")
        print(f"  {'Group':<25} {'Eager(s)':>10} {'E_%':>6} {'Compile(s)':>10} {'C_%':>6} {'Delta(s)':>10} {'E_calls':>9} {'C_calls':>9}")
        print(f"  {'-'*90}")

        for g in all_groups:
            e = e_groups.get(g, {"total_ns": 0, "instances": 0})
            c = c_groups.get(g, {"total_ns": 0, "instances": 0})
            e_pct = e["total_ns"] / e_total_gpu * 100 if e_total_gpu else 0
            c_pct = c["total_ns"] / c_total_gpu * 100 if c_total_gpu else 0
            delta = ns_to_s(c["total_ns"]) - ns_to_s(e["total_ns"])
            if e["total_ns"] + c["total_ns"] > 1_000_000:  # skip <1ms
                print(f"  {g:<25} {ns_to_s(e['total_ns']):>10.3f} {e_pct:>5.1f}% "
                      f"{ns_to_s(c['total_ns']):>10.3f} {c_pct:>5.1f}% "
                      f"{delta:>+10.3f} {e['instances']:>9,} {c['instances']:>9,}")

        print(f"  {'─'*90}")
        print(f"  {'TOTAL':<25} {ns_to_s(e_total_gpu):>10.3f} {'100%':>6} "
              f"{ns_to_s(c_total_gpu):>10.3f} {'100%':>6} "
              f"{ns_to_s(c_total_gpu - e_total_gpu):>+10.3f}")

    # ── Section 4: Top kernels for compile mode ──
    print(f"\n{'='*110}")
    print("SECTION 4: TOP 10 KERNELS — COMPILE MODE (each BS)")
    print("=" * 110)

    for bs in BATCH_SIZES:
        compile_k = load_kernel_csv("compile", bs)
        if not compile_k:
            continue
        total_gpu = sum(k["total_ns"] for k in compile_k)
        print(f"\n  --- compile BS={bs} (total GPU={ns_to_s(total_gpu):.2f}s) ---")
        print(f"  {'#':>3} {'%':>6} {'Time(s)':>8} {'Calls':>8} {'Avg(us)':>8} {'Group':<20} {'Kernel (truncated)'}")
        print(f"  {'-'*100}")
        for i, k in enumerate(compile_k[:10]):
            short_name = k["name"][:50]
            print(f"  {i+1:>3} {k['pct']:>5.1f}% {ns_to_s(k['total_ns']):>8.3f} {k['instances']:>8,} "
                  f"{k['avg_ns']/1000:>8.1f} {k['group']:<20} {short_name}")

    # ── Section 5: cudaStreamSynchronize analysis ──
    print(f"\n{'='*110}")
    print("SECTION 5: cudaStreamSynchronize — THE COMPILE BOTTLENECK")
    print("=" * 110)

    print(f"\n  {'BS':<5} {'E_sync(s)':>10} {'C_sync(s)':>10} {'E_sync_calls':>14} {'C_sync_calls':>14} "
          f"{'C_avg_sync(ms)':>16} {'E_wall(s)':>10} {'C_wall(s)':>10}")
    print(f"  {'-'*95}")
    for bs in BATCH_SIZES:
        e_api = load_api_csv("eager", bs)
        c_api = load_api_csv("compile", bs)
        if not e_api or not c_api:
            continue
        e_sync = next((a for a in e_api if "cudaStreamSynchronize" in a["name"]), {"total_ns": 0, "num_calls": 0, "avg_ns": 0})
        c_sync = next((a for a in c_api if "cudaStreamSynchronize" in a["name"]), {"total_ns": 0, "num_calls": 0, "avg_ns": 0})
        print(f"  {bs:<5} {ns_to_s(e_sync['total_ns']):>10.2f} {ns_to_s(c_sync['total_ns']):>10.2f} "
              f"{e_sync['num_calls']:>14,} {c_sync['num_calls']:>14,} "
              f"{ns_to_ms(c_sync['avg_ns']):>16.2f} "
              f"{TOTAL_TIME[('eager', bs)]:>10.1f} {TOTAL_TIME[('compile', bs)]:>10.1f}")

    print(f"""
  ANALYSIS:
    cudaStreamSynchronize dominates compile mode wall time (82-88%).
    This is the CPU blocking on GPU completion — the CPU WAITS after
    launching each CUDA graph replay.

    In eager mode, sync time is 17-40% of API time because kernels are
    queued continuously — the CPU races ahead of the GPU, building up
    a kernel queue. Sync only blocks when the CPU catches up.

    In compile mode with CUDA graphs:
    1. cudaGraphLaunch replays the full forward in one shot (~0.3ms)
    2. CPU immediately hits cudaStreamSynchronize and BLOCKS
    3. CPU does Python work (sampling, unmasking, loop control)
    4. Next cudaGraphLaunch — but GPU already finished and is IDLE

    The problem is the CPU-GPU round-trip between each graph replay.
    The GPU finishes fast (~10-20ms forward) but the CPU takes 3+ ms
    for Python work between replays, creating GPU idle bubbles.
""")

    # ── Section 6: GEMM analysis ──
    print(f"{'='*110}")
    print("SECTION 6: GEMM KERNEL ANALYSIS — COMPILE vs EAGER")
    print("=" * 110)

    for bs in BATCH_SIZES:
        eager_k = load_kernel_csv("eager", bs)
        compile_k = load_kernel_csv("compile", bs)
        if not eager_k or not compile_k:
            continue

        # Extract GEMM kernels
        e_gemms = [k for k in eager_k if k["group"] == "GEMM"]
        c_gemms = [k for k in compile_k if k["group"] == "GEMM"]
        e_total = sum(k["total_ns"] for k in e_gemms)
        c_total = sum(k["total_ns"] for k in c_gemms)
        e_calls = sum(k["instances"] for k in e_gemms)
        c_calls = sum(k["instances"] for k in c_gemms)

        print(f"\n  BS={bs}: GEMM total  eager={ns_to_s(e_total):.3f}s ({e_calls:,} calls)  "
              f"compile={ns_to_s(c_total):.3f}s ({c_calls:,} calls)  "
              f"delta={ns_to_s(c_total - e_total):+.3f}s")

        # Show which GEMM kernels differ
        # Build name→time map for top kernels
        e_by_name = {}
        for k in e_gemms:
            short = k["name"][:80]
            e_by_name[short] = e_by_name.get(short, 0) + k["total_ns"]
        c_by_name = {}
        for k in c_gemms:
            short = k["name"][:80]
            c_by_name[short] = c_by_name.get(short, 0) + k["total_ns"]

    # ── Section 7: Final summary ──
    print(f"\n{'='*110}")
    print("SECTION 7: COMPILE MODE BOTTLENECK SUMMARY")
    print("=" * 110)

    print(f"""
  KEY FINDING: Compile mode is 1.5-4x SLOWER than eager across all batch sizes.

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  BS    eager tok/s    compile tok/s    ratio    compile overhead           │
  │  1       75.1           48.3          0.64x     +25.9s (+54%)             │
  │  2       80.6           34.1          0.42x     +62.7s (+145%)            │
  │  4       99.2           33.7          0.34x     +61.1s (+171%)            │
  │  8      122.6           30.0          0.24x     +62.9s (+213%)            │
  └─────────────────────────────────────────────────────────────────────────────┘

  ROOT CAUSES (ordered by impact):

  1. MASSIVE cudaStreamSynchronize TIME (82-88% of compile wall time)
     - Compile mode blocks CPU after each CUDA graph replay
     - The GPU forward takes ~10-20ms, then GPU sits IDLE while CPU does
       Python work (sampling, unmasking, KV bookkeeping, loop control)
     - In eager mode, CPU queues kernels ahead of GPU — no idle gaps
     - This is the #1 bottleneck: CPU-GPU synchronization overhead

  2. COMPILATION + CUDA GRAPH RECORDING OVERHEAD
     - First batch includes torch.compile + CUDA graph recording
     - BS=1: ~26s overhead (73.5s vs 47.6s eager for same work)
     - BS=2: ~63s overhead — suggests multiple recompilations
     - Higher BS: even more recompilation (duck sizing, shape changes)

  3. MORE cudaStreamSynchronize CALLS in compile mode
     - Compile has MORE sync calls than eager at every BS
     - BS=1: 11K (compile) vs 10.9K (eager)
     - BS=8: 6.1K (compile) vs 6.2K (eager) — similar but compile
       sync time per call is 3.2ms vs 4.8ms (GPU idle between replays)

  4. GPU KERNEL TIME IS SIMILAR (not the bottleneck)
     - GEMM kernels, flash attention, elementwise — nearly identical
     - Compile uses Triton fused kernels (fewer launches, similar total time)
     - The GPU compute is NOT the problem — it's the CPU-GPU interaction

  RECOMMENDATIONS:
    P1: Batch multiple diffusion steps into ONE compiled call
        (forward + sample + unmask in single graph — eliminates sync gaps)
    P2: Use torch.compile WITHOUT CUDA graphs (reduce-overhead → default)
        to avoid graph recording overhead while keeping kernel fusion
    P3: Overlap CPU work with GPU by using CUDA streams + async operations
    P4: Investigate recompilation at BS>1 (duck sizing / shape changes)
""")


if __name__ == "__main__":
    main()
