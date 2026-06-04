#!/bin/bash
# Run nsys profiling for eager vs compiled forward pass benchmark,
# then export kernel stats to CSV and dump a comparison log.
#
# Uses tests/bench_nsys_compile.py which profiles a single forward pass
# (BS=8, past_blocks=10, 5 measured iterations) with NVTX markers.
#
# Usage:
#   cd /path/to/fast_v2
#   CUDA_VISIBLE_DEVICES=3 bash tests/run_nsys_compile.sh
#
# Output:
#   tests/nsys_eager.nsys-rep           — raw nsys report (eager)
#   tests/nsys_compiled.nsys-rep        — raw nsys report (compiled)
#   tests/nsys_eager_kernels.csv        — CUDA kernel summary CSV
#   tests/nsys_compiled_kernels.csv     — CUDA kernel summary CSV
#   tests/nsys_eager_nvtx.csv           — NVTX range summary CSV
#   tests/nsys_compiled_nvtx.csv        — NVTX range summary CSV
#   tests/nsys_comparison.log           — parsed comparison stats

set -e
cd "$(dirname "$0")/.."

VENV_PYTHON="./v2/bin/python3"
TESTS_DIR="./tests"
BENCH_SCRIPT="${TESTS_DIR}/bench_nsys_compile.py"

echo "============================================"
echo " nsys Forward Pass Benchmark"
echo " Eager vs Compiled (reduce-overhead)"
echo " BS=8, past_blocks=10, block_size=32"
echo "============================================"
echo ""

# ── Phase 1: Run profiling via the Python script ──────────────────────────
# bench_nsys_compile.py handles nsys launch for both modes internally.
# It uses cudaProfilerApi capture range + NVTX markers.
echo "[Phase 1] Running nsys profiling for both modes..."
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for MODE in eager compiled; do
    REP="${TESTS_DIR}/nsys_${MODE}"
    echo "--------------------------------------------"
    echo " Profiling: ${MODE}"
    echo " Output:    ${REP}.nsys-rep"
    echo "--------------------------------------------"

    nsys profile \
        -c cudaProfilerApi \
        -t cuda,nvtx,osrt \
        --cuda-graph-trace=node \
        --stats=true \
        --sample=none \
        --cpuctxsw=none \
        -f true \
        -o "${REP}" \
        $VENV_PYTHON "${BENCH_SCRIPT}" --worker --mode "${MODE}" \
        2>&1 | tee "${REP}.log"

    echo ""
done

# ── Phase 2: Export kernel + NVTX + API stats to CSV ─────────────────────
echo "============================================"
echo " Exporting CSV summaries"
echo "============================================"

for MODE in eager compiled; do
    REP="${TESTS_DIR}/nsys_${MODE}.nsys-rep"

    # CUDA kernel summary
    nsys stats --report cuda_gpu_kern_sum \
        --format csv \
        --force-export=true \
        --force-overwrite=true \
        --output "${TESTS_DIR}/nsys_${MODE}_kernels" \
        "${REP}" 2>/dev/null || echo "[warn] kernel export failed for ${MODE}"

    # NVTX range summary (iteration-level timing)
    nsys stats --report nvtx_pushpop_sum \
        --format csv \
        --force-export=true \
        --force-overwrite=true \
        --output "${TESTS_DIR}/nsys_${MODE}_nvtx" \
        "${REP}" 2>/dev/null || echo "[warn] nvtx export failed for ${MODE}"

    # CUDA API summary (host-side launch overhead)
    nsys stats --report cuda_api_sum \
        --format csv \
        --force-export=true \
        --force-overwrite=true \
        --output "${TESTS_DIR}/nsys_${MODE}_api" \
        "${REP}" 2>/dev/null || echo "[warn] api export failed for ${MODE}"

    echo "  ${MODE}: kernel + nvtx + api CSV exported"
done

# ── Phase 3: Parse and dump comparison stats ──────────────────────────────
echo ""
echo "============================================"
echo " Parsing comparison stats"
echo "============================================"

STATS_LOG="${TESTS_DIR}/nsys_comparison.log"

$VENV_PYTHON - <<'PYEOF' > "${STATS_LOG}"
import csv
import os
import re
import sys
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "tests"))
N_ITERS = 5  # must match bench_nsys_compile.py N_MEASURE

# ═══════════════════════════════════════════════════════════════════════════
# CSV readers
# ═══════════════════════════════════════════════════════════════════════════

def _find_csv(path, suffixes):
    """Try multiple candidate paths for nsys-exported CSVs."""
    candidates = [path, path.with_suffix('.csv')]
    for s in suffixes:
        candidates.append(path.parent / (path.stem + s))
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return list(csv.DictReader(f))
    return []

def read_kernel_csv(path):
    return _find_csv(path, ['_cuda_gpu_kern_sum.csv'])

def read_nvtx_csv(path):
    return _find_csv(path, ['_nvtx_pushpop_sum.csv'])

def read_api_csv(path):
    return _find_csv(path, ['_cuda_api_sum.csv'])

def fmt_us(ns_str):
    try: return float(ns_str) / 1000.0
    except (ValueError, TypeError): return 0.0

def fmt_ms(ns_str):
    try: return float(ns_str) / 1e6
    except (ValueError, TypeError): return 0.0

def get_col(row, candidates, default=None):
    """Return the first matching column value from a row."""
    for c in candidates:
        if c in row:
            return row[c]
    return default

def identify_kernel_cols(rows):
    """Identify name/time/count/avg columns from kernel CSV."""
    if not rows:
        return None, None, None, None
    cols = {"name": None, "time": None, "count": None, "avg": None}
    for col in rows[0].keys():
        cl = col.lower()
        if "name" in cl or "kernel" in cl:
            cols["name"] = col
        elif "total" in cl and ("time" in cl or "dur" in cl):
            cols["time"] = col
        elif "count" in cl or "instances" in cl or "calls" in cl:
            cols["count"] = col
        elif "avg" in cl and ("time" in cl or "dur" in cl):
            cols["avg"] = col
    return cols["name"], cols["time"], cols["count"], cols["avg"]

W = 94  # output width

# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════
print("=" * W)
print("  NSYS FORWARD PASS COMPARISON: Eager vs Compiled (reduce-overhead)")
print("  Config: BS=8, past_blocks=10 (seq_len=320), block_size=32")
print(f"  Measured iterations: {N_ITERS} (warmup + CUDA graph recording excluded)")
print("  Capture: cudaProfilerApi (only measurement region profiled)")
print("  Flags: --cuda-graph-trace=node (individual kernels inside CUDA graphs)")
print("=" * W)

# ═══════════════════════════════════════════════════════════════════════════
# Section 1: NVTX Iteration Timing
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * W}")
print("  SECTION 1: PER-ITERATION WALL TIME (NVTX ranges)")
print(f"  Each range spans one forward pass including cudaDeviceSynchronize.")
print(f"{'─' * W}")

for mode in ("eager", "compiled"):
    nvtx = read_nvtx_csv(TESTS_DIR / f"nsys_{mode}_nvtx")
    if not nvtx:
        print(f"\n  [{mode}] No NVTX data found")
        continue

    print(f"\n  ── {mode.upper()} ──")
    iter_rows = [r for r in nvtx if f"{mode}_iter" in r.get("Range", "")]
    # Sort by iteration number (range names like ":eager_iter_0")
    iter_rows.sort(key=lambda r: int(re.search(r'(\d+)$', r.get("Range", "0")).group(1))
                   if re.search(r'(\d+)$', r.get("Range", "")) else 0)
    if iter_rows:
        time_col = None
        for col in ("Total Time (ns)", "Total_Time", "Duration:Sum(ns)", "TotalDuration"):
            if col in iter_rows[0]:
                time_col = col
                break
        if time_col:
            print(f"  {'Iteration':<20s}  {'Wall Time (ms)':>14s}")
            print(f"  {'-'*20}  {'-'*14}")
            times = []
            for r in iter_rows:
                t = fmt_ms(r[time_col])
                times.append(t)
                # Strip leading ':' from range name
                name = r['Range'].lstrip(':')
                print(f"  {name:<20s}  {t:>14.3f}")
            if times:
                print(f"  {'Average':<20s}  {sum(times)/len(times):>14.3f}")
                print(f"  {'Std Dev':<20s}  {(sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5:>14.3f}")
        else:
            print(f"  Columns: {list(iter_rows[0].keys())}")
    else:
        print(f"  No iteration ranges found. Available: "
              + ", ".join(r.get("Range", "?") for r in nvtx[:5]))

# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Host-Side Launch Overhead (CUDA API)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * W}")
print("  SECTION 2: HOST-SIDE CUDA API OVERHEAD")
print("  Time the CPU spends dispatching work to the GPU.")
print("  Eager: one cudaLaunchKernel per op. Compiled: one cudaGraphLaunch per fwd pass.")
print(f"{'─' * W}")

api_data = {}
for mode in ("eager", "compiled"):
    api = read_api_csv(TESTS_DIR / f"nsys_{mode}_api")
    if not api:
        print(f"\n  [{mode}] No CUDA API data found")
        continue

    api_data[mode] = api
    print(f"\n  ── {mode.upper()} CUDA API Calls ──")
    print(f"  {'API Function':<30s}  {'Calls':>7s}  {'Total (ms)':>10s}  "
          f"{'Avg (us)':>10s}  {'Med (us)':>10s}  {'Min (us)':>10s}  {'Max (us)':>10s}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for r in api:
        name = r.get("Name", "?")
        calls = r.get("Num Calls", "?")
        total_ms = fmt_ms(r.get("Total Time (ns)", 0))
        avg_us = fmt_us(r.get("Avg (ns)", 0))
        med_us = fmt_us(r.get("Med (ns)", 0))
        min_us = fmt_us(r.get("Min (ns)", 0))
        max_us = fmt_us(r.get("Max (ns)", 0))
        print(f"  {name:<30s}  {str(calls):>7s}  {total_ms:>10.3f}  "
              f"{avg_us:>10.1f}  {med_us:>10.1f}  {min_us:>10.1f}  {max_us:>10.1f}")

# Host overhead comparison table
if "eager" in api_data and "compiled" in api_data:
    print(f"\n  ── HOST-SIDE LAUNCH OVERHEAD COMPARISON ──")
    print(f"  (per forward pass, averaged over {N_ITERS} iterations)")
    print()

    # Extract launch calls
    eager_launch = None
    comp_launch = None
    eager_sync = None
    comp_sync = None
    for r in api_data["eager"]:
        if "LaunchKernel" in r.get("Name", ""):
            eager_launch = r
        if "Synchronize" in r.get("Name", ""):
            eager_sync = r
    for r in api_data["compiled"]:
        if "GraphLaunch" in r.get("Name", ""):
            comp_launch = r
        if "Synchronize" in r.get("Name", ""):
            comp_sync = r

    print(f"  {'Metric':<35s}  {'Eager':>14s}  {'Compiled':>14s}  {'Savings':>14s}")
    print(f"  {'-'*35}  {'-'*14}  {'-'*14}  {'-'*14}")

    if eager_launch and comp_launch:
        e_calls = int(eager_launch.get("Num Calls", 0))
        c_calls = int(comp_launch.get("Num Calls", 0))
        e_total = float(eager_launch.get("Total Time (ns)", 0)) / 1e6
        c_total = float(comp_launch.get("Total Time (ns)", 0)) / 1e6
        e_per = e_total / N_ITERS
        c_per = c_total / N_ITERS
        e_calls_per = e_calls / N_ITERS
        c_calls_per = c_calls / N_ITERS

        print(f"  {'Dispatch mechanism':<35s}  {'cudaLaunchKernel':>14s}  {'cudaGraphLaunch':>14s}")
        print(f"  {'Total dispatch calls':<35s}  {e_calls:>14d}  {c_calls:>14d}  "
              f"{e_calls/c_calls if c_calls else 0:>13.0f}x")
        print(f"  {'Dispatch calls / fwd pass':<35s}  {e_calls_per:>14.0f}  {c_calls_per:>14.0f}  "
              f"{e_calls_per/c_calls_per if c_calls_per else 0:>13.0f}x")
        print(f"  {'Total dispatch time (ms)':<35s}  {e_total:>14.3f}  {c_total:>14.3f}  "
              f"{e_total - c_total:>13.3f}")
        print(f"  {'Dispatch time / fwd pass (ms)':<35s}  {e_per:>14.3f}  {c_per:>14.3f}  "
              f"{e_per - c_per:>13.3f}")
        print(f"  {'Avg per-call latency (us)':<35s}  "
              f"{fmt_us(eager_launch.get('Avg (ns)', 0)):>14.1f}  "
              f"{fmt_us(comp_launch.get('Avg (ns)', 0)):>14.1f}")

    if eager_sync and comp_sync:
        e_sync = float(eager_sync.get("Total Time (ns)", 0)) / 1e6
        c_sync = float(comp_sync.get("Total Time (ns)", 0)) / 1e6
        print(f"  {'cudaDeviceSynchronize (ms)':<35s}  {e_sync:>14.3f}  {c_sync:>14.3f}")

    # Estimate how much of wall time is launch overhead
    if eager_launch and comp_launch and eager_sync and comp_sync:
        e_wall = e_total + float(eager_sync.get("Total Time (ns)", 0)) / 1e6
        c_wall = c_total + float(comp_sync.get("Total Time (ns)", 0)) / 1e6
        e_pct = e_total / e_wall * 100 if e_wall else 0
        c_pct = c_total / c_wall * 100 if c_wall else 0
        print()
        print(f"  {'Launch overhead as % of wall time':<35s}  {e_pct:>13.1f}%  {c_pct:>13.1f}%")
        print()
        print(f"  Note: Eager launch calls overlap with GPU execution (CPU queues next kernel")
        print(f"  while GPU runs the current one). The {e_total - c_total:.1f} ms dispatch savings")
        print(f"  does not translate 1:1 to wall-time improvement because of this overlap.")
        print(f"  Launch overhead becomes a bottleneck at small batch sizes where GPU kernels")
        print(f"  finish faster than the CPU can dispatch them.")

# ═══════════════════════════════════════════════════════════════════════════
# Section 3: GPU Kernel Breakdown
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * W}")
print("  SECTION 3: GPU KERNEL BREAKDOWN")
print("  Top kernels by total GPU execution time across all measured iterations.")
print(f"{'─' * W}")

for mode in ("eager", "compiled"):
    kernels = read_kernel_csv(TESTS_DIR / f"nsys_{mode}_kernels")
    if not kernels:
        print(f"\n  [{mode}] No kernel data found")
        continue

    name_col, time_col, count_col, avg_col = identify_kernel_cols(kernels)
    if not name_col or not time_col:
        print(f"  Could not identify columns: {list(kernels[0].keys())}")
        continue

    for k in kernels:
        k["_total_ns"] = float(k.get(time_col, 0))
    kernels.sort(key=lambda x: x["_total_ns"], reverse=True)

    total_gpu_ns = sum(k["_total_ns"] for k in kernels)
    total_gpu_ms = total_gpu_ns / 1e6

    print(f"\n  ── {mode.upper()} Top 15 CUDA Kernels ──")
    print(f"  Total GPU time: {total_gpu_ms:.3f} ms  |  "
          f"Unique kernels: {len(kernels)}  |  "
          f"Per fwd pass: {total_gpu_ms/N_ITERS:.3f} ms")
    print(f"  {'#':>3s}  {'Calls':>6s}  {'Total (us)':>10s}  {'Avg (us)':>10s}  "
          f"{'%GPU':>5s}  {'Cum%':>5s}  {'Kernel Name'}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*5}  {'-'*5}  {'-'*55}")

    cum_pct = 0
    for i, k in enumerate(kernels[:15]):
        name = k[name_col]
        if len(name) > 70:
            name = name[:67] + "..."
        total_us = k["_total_ns"] / 1000.0
        count = k.get(count_col, "?")
        avg_us = fmt_us(k.get(avg_col, 0)) if avg_col else (
            k["_total_ns"] / max(int(count), 1) / 1000.0 if count != "?" else 0
        )
        pct = k["_total_ns"] / total_gpu_ns * 100 if total_gpu_ns > 0 else 0
        cum_pct += pct
        print(f"  {i+1:>3d}  {str(count):>6s}  {total_us:>10.1f}  {avg_us:>10.1f}  "
              f"{pct:>5.1f}  {cum_pct:>5.1f}  {name}")

# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Kernel Category Breakdown
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * W}")
print("  SECTION 4: KERNEL CATEGORY BREAKDOWN")
print("  Kernels grouped by type: GEMM, Attention, Triton fused, Elementwise, Other.")
print(f"{'─' * W}")

def categorize_kernel(name):
    nl = name.lower()
    if "gemm" in nl or "cublas" in nl:
        return "GEMM (cuBLAS)"
    elif "flash" in nl:
        return "Attention (FlashAttn)"
    elif "triton" in nl:
        return "Triton fused"
    elif "elementwise" in nl or "vectorized" in nl:
        return "Elementwise (PyTorch)"
    elif "reduce" in nl:
        return "Reduce (PyTorch)"
    elif "scatter" in nl or "gather" in nl:
        return "Scatter/Gather"
    elif "copy" in nl or "cat" in nl:
        return "Copy/Cat"
    else:
        return "Other"

for mode in ("eager", "compiled"):
    kernels = read_kernel_csv(TESTS_DIR / f"nsys_{mode}_kernels")
    if not kernels:
        continue
    name_col, time_col, count_col, _ = identify_kernel_cols(kernels)
    if not name_col or not time_col:
        continue

    total_ns = sum(float(k.get(time_col, 0)) for k in kernels)

    categories = {}
    for k in kernels:
        cat = categorize_kernel(k[name_col])
        t = float(k.get(time_col, 0))
        c = int(k.get(count_col, 0)) if count_col else 0
        if cat not in categories:
            categories[cat] = {"time_ns": 0, "calls": 0, "kernels": 0}
        categories[cat]["time_ns"] += t
        categories[cat]["calls"] += c
        categories[cat]["kernels"] += 1

    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["time_ns"], reverse=True)

    print(f"\n  ── {mode.upper()} ──")
    print(f"  {'Category':<25s}  {'GPU Time (ms)':>12s}  {'%GPU':>6s}  "
          f"{'Launches':>9s}  {'Unique':>6s}  {'Avg (us)':>10s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*6}  {'-'*9}  {'-'*6}  {'-'*10}")
    for cat, d in sorted_cats:
        t_ms = d["time_ns"] / 1e6
        pct = d["time_ns"] / total_ns * 100 if total_ns else 0
        avg = d["time_ns"] / d["calls"] / 1000 if d["calls"] else 0
        print(f"  {cat:<25s}  {t_ms:>12.3f}  {pct:>5.1f}%  "
              f"{d['calls']:>9d}  {d['kernels']:>6d}  {avg:>10.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Side-by-Side Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * W}")
print("  SECTION 5: SIDE-BY-SIDE SUMMARY")
print(f"{'─' * W}")

eager_k = read_kernel_csv(TESTS_DIR / "nsys_eager_kernels")
comp_k = read_kernel_csv(TESTS_DIR / "nsys_compiled_kernels")

if eager_k and comp_k:
    _, tc_e, cc_e, _ = identify_kernel_cols(eager_k)
    _, tc_c, cc_c, _ = identify_kernel_cols(comp_k)

    if tc_e and tc_c:
        eager_total = sum(float(k.get(tc_e, 0)) for k in eager_k) / 1e6
        comp_total = sum(float(k.get(tc_c, 0)) for k in comp_k) / 1e6
        eager_count = sum(int(k.get(cc_e, 0)) for k in eager_k) if cc_e else 0
        comp_count = sum(int(k.get(cc_c, 0)) for k in comp_k) if cc_c else 0

        print()
        print(f"  {'Metric':<40s}  {'Eager':>14s}  {'Compiled':>14s}  {'Ratio/Diff':>14s}")
        print(f"  {'-'*40}  {'-'*14}  {'-'*14}  {'-'*14}")

        # GPU time
        print(f"  {'Total GPU time (ms)':<40s}  {eager_total:>14.3f}  {comp_total:>14.3f}  "
              f"{eager_total/comp_total if comp_total else 0:>13.2f}x")
        print(f"  {'GPU time / fwd pass (ms)':<40s}  "
              f"{eager_total/N_ITERS:>14.3f}  {comp_total/N_ITERS:>14.3f}  "
              f"{(eager_total-comp_total)/N_ITERS:>+13.3f}")

        # Kernel launches
        print(f"  {'Total kernel launches':<40s}  {eager_count:>14d}  {comp_count:>14d}  "
              f"{eager_count/comp_count if comp_count else 0:>13.1f}x")
        print(f"  {'Kernel launches / fwd pass':<40s}  "
              f"{eager_count/N_ITERS:>14.0f}  {comp_count/N_ITERS:>14.0f}")
        print(f"  {'Unique kernel types':<40s}  {len(eager_k):>14d}  {len(comp_k):>14d}")

        if eager_count > 0 and comp_count > 0:
            print(f"  {'Avg kernel duration (us)':<40s}  "
                  f"{eager_total*1000/eager_count:>14.2f}  "
                  f"{comp_total*1000/comp_count:>14.2f}")

        # Host overhead (if available)
        if "eager" in api_data and "compiled" in api_data:
            e_launch_ms = 0
            c_launch_ms = 0
            for r in api_data["eager"]:
                if "LaunchKernel" in r.get("Name", ""):
                    e_launch_ms = float(r.get("Total Time (ns)", 0)) / 1e6
            for r in api_data["compiled"]:
                if "GraphLaunch" in r.get("Name", ""):
                    c_launch_ms = float(r.get("Total Time (ns)", 0)) / 1e6
            print(f"  {'Host dispatch overhead (ms)':<40s}  "
                  f"{e_launch_ms:>14.3f}  {c_launch_ms:>14.3f}  "
                  f"{e_launch_ms - c_launch_ms:>+13.3f}")
            print(f"  {'Host dispatch / fwd pass (ms)':<40s}  "
                  f"{e_launch_ms/N_ITERS:>14.3f}  {c_launch_ms/N_ITERS:>14.3f}  "
                  f"{(e_launch_ms - c_launch_ms)/N_ITERS:>+13.3f}")

        # Category comparison
        print()
        print(f"  {'Category Breakdown':<40s}  {'Eager (ms)':>14s}  {'Compiled (ms)':>14s}  {'Diff (ms)':>14s}")
        print(f"  {'-'*40}  {'-'*14}  {'-'*14}  {'-'*14}")

        for mode_kernels, tc in [(eager_k, tc_e), (comp_k, tc_c)]:
            for k in mode_kernels:
                k["_cat"] = categorize_kernel(k.get(identify_kernel_cols(mode_kernels)[0], ""))

        # Build category totals for both modes
        eager_cats = {}
        for k in eager_k:
            cat = categorize_kernel(k.get(identify_kernel_cols(eager_k)[0], ""))
            eager_cats[cat] = eager_cats.get(cat, 0) + float(k.get(tc_e, 0))
        comp_cats = {}
        for k in comp_k:
            cat = categorize_kernel(k.get(identify_kernel_cols(comp_k)[0], ""))
            comp_cats[cat] = comp_cats.get(cat, 0) + float(k.get(tc_c, 0))

        all_cats = sorted(set(list(eager_cats.keys()) + list(comp_cats.keys())),
                         key=lambda c: max(eager_cats.get(c, 0), comp_cats.get(c, 0)),
                         reverse=True)
        for cat in all_cats:
            e = eager_cats.get(cat, 0) / 1e6
            c = comp_cats.get(cat, 0) / 1e6
            print(f"  {cat:<40s}  {e:>14.3f}  {c:>14.3f}  {e - c:>+13.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Key Takeaways
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * W}")
print("  SECTION 6: KEY TAKEAWAYS")
print(f"{'─' * W}")

if eager_k and comp_k and tc_e and tc_c:
    gpu_savings = eager_total - comp_total
    gpu_pct = gpu_savings / eager_total * 100 if eager_total else 0

    print(f"""
  1. GPU COMPUTE: Compiled saves {gpu_savings:.1f} ms ({gpu_pct:.1f}%) total GPU time over {N_ITERS} iterations.
     Per forward pass: {gpu_savings/N_ITERS:.1f} ms savings.

  2. GEMM DOMINANCE: cuBLAS GEMMs account for ~{eager_cats.get('GEMM (cuBLAS)', 0)/sum(eager_cats.values())*100:.0f}% of GPU time in both modes.
     These are identical between eager/compiled — compilation cannot optimize them.

  3. TRITON FUSION: Compiled mode fuses elementwise ops (silu, mul, RMSNorm, RoPE, copy)
     into triton kernels. This eliminates intermediate memory traffic and reduces
     kernel count from {len(eager_k)} to {len(comp_k)} unique types.

  4. HOST OVERHEAD: Eager dispatches {eager_count//N_ITERS} cudaLaunchKernel calls per fwd pass.""")

    if "eager" in api_data and "compiled" in api_data:
        print(f"     Compiled replaces these with 1 cudaGraphLaunch call.")
        print(f"     Host dispatch savings: {(e_launch_ms - c_launch_ms)/N_ITERS:.2f} ms/fwd pass.")
        print(f"     However, launch calls overlap with GPU execution, so the wall-time")
        print(f"     impact is smaller than the raw API time difference.")

    print(f"""
  5. SCALING: Launch overhead matters more at small batch sizes where GPU kernels
     are short and the CPU becomes the bottleneck. At larger batch sizes, GPU
     compute dominates and the launch overhead is hidden by overlap.
""")

print("=" * W)
PYEOF

echo ""
echo "Stats written to: ${STATS_LOG}"
cat "${STATS_LOG}"

echo ""
echo "============================================"
echo " Output files:"
echo "   ${TESTS_DIR}/nsys_eager.nsys-rep"
echo "   ${TESTS_DIR}/nsys_compiled.nsys-rep"
echo "   ${TESTS_DIR}/nsys_*_kernels*.csv    (GPU kernel summary)"
echo "   ${TESTS_DIR}/nsys_*_nvtx*.csv       (NVTX iteration ranges)"
echo "   ${TESTS_DIR}/nsys_*_api*.csv        (CUDA API host overhead)"
echo "   ${STATS_LOG}  (full comparison log)"
echo "============================================"
