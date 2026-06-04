#!/usr/bin/env python3
"""
Parse compile vs eager logs and produce comparison tables.

Uses THREE data sources:
  1. TIMING logs (no profiler, no overhead) — primary wall-time & throughput data
  2. PROFILER logs (torch.profiler, high overhead) — kernel breakdown only
  3. NSYS compile data (low overhead) — compile kernel breakdown cross-check (not wired here)

Usage:
    python debug/parse_compile_vs_eager_bs1.py [--logs-dir DIR] \\
        [--timing-compile-csv PATH] [--timing-eager-csv PATH] ...

Defaults (batch suffix bs1) match the paths below; override any file explicitly.

Reads (defaults under logs/):
  TIMING (no overhead):
    timing_compile_bs1.csv, timing_eager_bs1.csv
    gsm8k_0_.25_100000_1_compact_compile_auto_1_TIMING_1_PROFILE_0_GPU_6.log
    gsm8k_0_.25_100000_1_compact_eager_auto_1_TIMING_1_PROFILE_0_GPU_5.log
  PROFILER (kernel detail, high overhead):
    gsm8k_0_.25_100000_1_compact_compile_auto_1_GPU_6.log
    gsm8k_0_.25_100000_1_compact_eager_auto_1_GPU_5.log
"""

import argparse
import csv
import os
import re
import statistics
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "..", "logs")

# Stable section-6 row order (kernel groups).
GROUP_ORDER = [
    "Linear GEMMs (QKV+O+MLP+lm_head)",
    "Flash Attention (splitkv+combine)",
    "Triton fused elementwise (RoPE,SiLU,Norm)",
    "Native elementwise (mul,add,copy_,to,cat)",
    "scatter_ (KV cache write)",
    "splitKreduce + Memset",
    "Other (memset, reduce)",
]


def default_paths(logs_dir):
    """Default input paths (bs1 filenames)."""
    ld = os.path.abspath(logs_dir)
    return {
        "timing_compile_csv": os.path.join(ld, "timing_compile_bs8.csv"),
        "timing_eager_csv": os.path.join(ld, "timing_eager_bs8.csv"),
        "timing_compile_log": os.path.join(
            ld,
            "gsm8k_0_.25_100000_1_compact_compile_auto_8_TIMING_1_PROFILE_0_GPU_6.log",
        ),
        "timing_eager_log": os.path.join(
            ld,
            "gsm8k_0_.25_100000_1_compact_eager_auto_8_TIMING_1_PROFILE_0_GPU_5.log",
        ),
        "profiler_compile_log": os.path.join(
            ld,
            "gsm8k_0_.25_100000_1_compact_compile_auto_8_GPU_6.log",
        ),
        "profiler_eager_log": os.path.join(
            ld,
            "gsm8k_0_.25_100000_1_compact_eager_auto_8_GPU_5.log",
        ),
    }


def parse_args():
    d = default_paths(LOG_DIR)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs-dir", default=LOG_DIR, help="Directory for default filenames")
    p.add_argument("--timing-compile-csv", default=None)
    p.add_argument("--timing-eager-csv", default=None)
    p.add_argument("--timing-compile-log", default=None)
    p.add_argument("--timing-eager-log", default=None)
    p.add_argument("--profiler-compile-log", default=None)
    p.add_argument("--profiler-eager-log", default=None)
    args = p.parse_args()
    base = default_paths(args.logs_dir)
    return argparse.Namespace(
        timing_compile_csv=args.timing_compile_csv or base["timing_compile_csv"],
        timing_eager_csv=args.timing_eager_csv or base["timing_eager_csv"],
        timing_compile_log=args.timing_compile_log or base["timing_compile_log"],
        timing_eager_log=args.timing_eager_log or base["timing_eager_log"],
        profiler_compile_log=args.profiler_compile_log or base["profiler_compile_log"],
        profiler_eager_log=args.profiler_eager_log or base["profiler_eager_log"],
    )


def load_timing_csv(csv_path):
    """Load per-batch timing CSV into list of dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "batch": int(row["batch_idx"]),
                "wall_s": float(row["wall_s"]),
                "tokens": int(row["tokens"]),
                "tok_s": float(row["tok_per_s"]),
                "prompt_len": int(row["prompt_len"]),
                "skipped": int(row["skipped"]),
            })
    return rows


def extract_summary(log_path):
    """Extract total tokens and tok/s from log."""
    with open(log_path) as f:
        content = f.read()
    tokens = int(re.search(r"Total number of tokens generated: (\d+)", content).group(1))
    total_time = float(re.search(r"Total time taken: ([\d.]+)", content).group(1))
    tok_s = float(re.search(r"Tokens per second: ([\d.]+)", content).group(1))
    return {"tokens": tokens, "total_time": total_time, "tok_s": tok_s}


def extract_timing_summary(log_path):
    """Extract the [timing] summary block from log."""
    with open(log_path) as f:
        content = f.read()
    m = re.search(r"Throughput:\s*([\d.]+)\s*tok/s", content)
    throughput = float(m.group(1)) if m else 0
    m = re.search(r"Total time:\s*([\d.]+)s", content)
    total_time = float(m.group(1)) if m else 0
    m = re.search(r"Total toks:\s*(\d+)", content)
    total_toks = int(m.group(1)) if m else 0
    m = re.search(r"mean=([\d.]+)s\s+median=([\d.]+)s", content)
    mean_s = float(m.group(1)) if m else 0
    median_s = float(m.group(2)) if m else 0
    return {
        "throughput": throughput,
        "total_time": total_time,
        "total_toks": total_toks,
        "mean_s": mean_s,
        "median_s": median_s,
    }


def _parse_duration(s):
    """Parse profiler duration field (e.g. 1.032s, 95.919ms, 538.240us)."""
    s = s.strip()
    for suffix, mul in (("us", 1e-6), ("ms", 1e-3), ("ns", 1e-9), ("s", 1.0)):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * mul
    return float(s)


def classify_profiler_kernel(name):
    """
    Bucket one kernel row from the printed top-30 table into a summary group.
    Skips profiler scaffolding and aten::* rows for CUDA totals (avoids double-count
    with underlying GPU kernels in the same table).
    """
    n = name.lower().strip()
    if "profilerstep" in n.replace("*", ""):
        return None
    if "torch-compiled region" in n or n.startswith("## call compiledfxgraph"):
        return None
    if n.startswith("aten::"):
        return None
    if "splitkreduce" in n or ("splitk" in n and "reduce" in n):
        return "splitKreduce + Memset"
    if "memset" in n:
        return "splitKreduce + Memset"
    if "scatter" in n:
        return "scatter_ (KV cache write)"
    if "flash" in n or "fast_dllm_fa2" in n:
        return "Flash Attention (splitkv+combine)"
    if "triton_" in n or n.startswith("triton_"):
        return "Triton fused elementwise (RoPE,SiLU,Norm)"
    if "elementwise" in n or "vectorized_elementwise" in n:
        return "Native elementwise (mul,add,copy_,to,cat)"
    if "reduce_kernel" in n:
        return "Other (memset, reduce)"
    if "gemm" in n or "cutlass" in n or "cublas" in n:
        return "Linear GEMMs (QKV+O+MLP+lm_head)"
    return "Other (memset, reduce)"


def parse_profiler_top_table(log_path):
    """
    Parse [profile] Key CUDA kernel averages table from a run log.

    Returns dict with per-row splits, grouped Self CUDA sums, grouped call counts,
    totals from the log footer, and aten:: CPU totals (CPU total column, index 4).
    """
    with open(log_path) as f:
        content = f.read()
    marker = "[profile] Key CUDA kernel averages"
    idx = content.find(marker)
    if idx < 0:
        return None
    rest = content[idx:]
    lines = rest.split("\n")
    i = 0
    while i < len(lines) and ("Name" not in lines[i] or "Self CPU %" not in lines[i]):
        i += 1
    if i >= len(lines):
        return None
    i += 1
    while i < len(lines) and "---" in lines[i]:
        i += 1

    raw_rows = []
    by_group_cuda = defaultdict(float)
    by_group_calls = defaultdict(int)
    aten_cpu_total = {}  # op name -> CPU total seconds (from table)

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("---") or line.startswith("Self CPU time total"):
            break
        if not line.strip():
            i += 1
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 11 or parts[0] == "Name":
            i += 1
            continue
        name = parts[0].strip()
        try:
            self_cuda_s = _parse_duration(parts[6])
            calls = int(parts[-1].replace(",", ""))
            cpu_total_s = _parse_duration(parts[4])
        except (ValueError, IndexError):
            i += 1
            continue
        raw_rows.append(
            {
                "name": name,
                "self_cuda_s": self_cuda_s,
                "calls": calls,
                "cpu_total_s": cpu_total_s,
            }
        )
        g = classify_profiler_kernel(name)
        if g:
            by_group_cuda[g] += self_cuda_s
            by_group_calls[g] += calls
        if name.startswith("aten::"):
            aten_cpu_total[name] = cpu_total_s
        i += 1

    m_cpu = re.search(r"Self CPU time total:\s*([\d.]+)s", rest)
    m_cuda = re.search(r"Self CUDA time total:\s*([\d.]+)s", rest)
    return {
        "raw_rows": raw_rows,
        "by_group_cuda": dict(by_group_cuda),
        "by_group_calls": dict(by_group_calls),
        "self_cpu_total_s": float(m_cpu.group(1)) if m_cpu else None,
        "self_cuda_total_s": float(m_cuda.group(1)) if m_cuda else None,
        "aten_cpu_total": aten_cpu_total,
    }


def _fmt_ratio(num, den):
    if den and den > 0:
        return f"{num / den:.2f}x"
    return "N/A" if num > 0 else "0"


def token_length_bins(active_rows, n_bins=3):
    """Build token-count bins from steady-state rows (tertile edges)."""
    toks = sorted(r["tokens"] for r in active_rows)
    if not toks:
        return []
    if toks[0] == toks[-1]:
        t = toks[0]
        return [(t, t + 1, f"all (tokens={t})")]
    edges = [toks[0]]
    n = len(toks)
    for k in range(1, n_bins):
        j = min(n - 1, (k * n) // n_bins)
        edges.append(toks[j])
    edges.append(toks[-1] + 1)
    bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if lo >= hi:
            continue
        if b == 0:
            label = f"short (<{hi})"
        elif b == n_bins - 1:
            label = f"long (≥{lo})"
        else:
            label = f"medium [{lo}, {hi})"
        bins.append((lo, hi, label))
    return bins


def main():
    args = parse_args()

    # Load timing data (no profiler overhead)
    c_rows = load_timing_csv(args.timing_compile_csv)
    e_rows = load_timing_csv(args.timing_eager_csv)
    c_summary = extract_summary(args.timing_compile_log)
    e_summary = extract_summary(args.timing_eager_log)
    c_timing = extract_timing_summary(args.timing_compile_log)
    e_timing = extract_timing_summary(args.timing_eager_log)

    n_samples = len(c_rows)
    n_skipped = sum(1 for r in c_rows if r["skipped"])

    prof_c = parse_profiler_top_table(args.profiler_compile_log)
    prof_e = parse_profiler_top_table(args.profiler_eager_log)

    # ── Section 1: Per-batch comparison (no profiler overhead) ──
    print("=" * 100)
    print("SECTION 1: PER-BATCH TIMING — NO PROFILER OVERHEAD")
    print("  Source: TIMING logs (FAST_DLLM_PROFILE=0)")
    print(f"  {n_skipped} warmup batches skipped in CSV; {n_samples} total samples")
    print("=" * 100)
    print(
        f"\n{'Batch':>5} {'C_time':>8} {'E_time':>8} {'C_tok/s':>8} {'E_tok/s':>8} "
        f"{'Speedup':>8} {'C_toks':>7} {'E_toks':>7} {'Prompt':>7} {'Skip':>5}"
    )
    print("-" * 100)

    for i in range(len(c_rows)):
        cr = c_rows[i]
        er = e_rows[i] if i < len(e_rows) else None
        if er is None:
            continue
        speedup = er["wall_s"] / cr["wall_s"] if cr["wall_s"] > 0 else 0
        skip = "*" if cr["skipped"] else ""
        print(
            f"{cr['batch']:>5} {cr['wall_s']:>8.3f} {er['wall_s']:>8.3f} "
            f"{cr['tok_s']:>8.1f} {er['tok_s']:>8.1f} "
            f"{speedup:>7.2f}x {cr['tokens']:>7} {er['tokens']:>7} {cr['prompt_len']:>7} {skip:>5}"
        )

    # ── Section 2: Summary statistics (no profiler overhead) ──
    c_active = [r for r in c_rows if not r["skipped"]]
    e_active = [r for r in e_rows if not r["skipped"]]

    c_walls = [r["wall_s"] for r in c_active]
    e_walls = [r["wall_s"] for r in e_active]
    c_toks = [r["tok_s"] for r in c_active]
    e_toks = [r["tok_s"] for r in e_active]

    print(f"\n{'='*100}")
    print("SECTION 2: SUMMARY STATISTICS — NO PROFILER OVERHEAD")
    print(f"{'='*100}")
    print(f"\n{'Metric':<40} {'Compile':>12} {'Eager':>12} {'Ratio':>10}")
    print("-" * 75)
    print(
        f"{'Total wall time (all samples)':<40} {c_summary['total_time']:>11.1f}s "
        f"{e_summary['total_time']:>11.1f}s {e_summary['total_time']/c_summary['total_time']:>10.2f}x"
    )
    print(
        f"{'Total tokens generated':<40} {c_summary['tokens']:>12} {e_summary['tokens']:>12}"
    )
    print(
        f"{'Overall tok/s (incl warmup)':<40} {c_summary['tok_s']:>12.1f} "
        f"{e_summary['tok_s']:>12.1f} {c_summary['tok_s']/e_summary['tok_s']:>10.2f}x"
    )
    print(
        f"{'Steady-state tok/s (after skip)':<40} {c_timing['throughput']:>12.1f} "
        f"{e_timing['throughput']:>12.1f} {c_timing['throughput']/e_timing['throughput']:>10.2f}x"
    )
    print(
        f"{'Steady-state total time (after skip)':<40} {c_timing['total_time']:>11.1f}s "
        f"{e_timing['total_time']:>11.1f}s {e_timing['total_time']/c_timing['total_time']:>10.2f}x"
    )
    print(
        f"{'Per-batch mean (after skip)':<40} {statistics.mean(c_walls):>11.3f}s "
        f"{statistics.mean(e_walls):>11.3f}s {statistics.mean(e_walls)/statistics.mean(c_walls):>10.2f}x"
    )
    print(
        f"{'Per-batch median (after skip)':<40} {statistics.median(c_walls):>11.3f}s "
        f"{statistics.median(e_walls):>11.3f}s {statistics.median(e_walls)/statistics.median(c_walls):>10.2f}x"
    )
    print(
        f"{'Per-batch stdev (after skip)':<40} {statistics.stdev(c_walls):>11.3f}s "
        f"{statistics.stdev(e_walls):>11.3f}s"
    )
    print(
        f"{'Per-batch min (after skip)':<40} {min(c_walls):>11.3f}s {min(e_walls):>11.3f}s"
    )
    print(
        f"{'Per-batch max (after skip)':<40} {max(c_walls):>11.3f}s {max(e_walls):>11.3f}s"
    )
    print(
        f"{'Tok/s mean (after skip)':<40} {statistics.mean(c_toks):>12.1f} "
        f"{statistics.mean(e_toks):>12.1f} {statistics.mean(c_toks)/statistics.mean(e_toks):>10.2f}x"
    )
    print(
        f"{'Tok/s median (after skip)':<40} {statistics.median(c_toks):>12.1f} "
        f"{statistics.median(e_toks):>12.1f} {statistics.median(c_toks)/statistics.median(e_toks):>10.2f}x"
    )

    # ── Section 3: Per-batch paired comparison ──
    print(f"\n{'='*100}")
    print("SECTION 3: PAIRED PER-BATCH ANALYSIS (same prompts, skip warmup)")
    print(f"{'='*100}")

    paired_speedups = []
    paired_diffs = []
    c_faster = 0
    e_faster = 0
    for cr, er in zip(c_active, e_active):
        diff = er["wall_s"] - cr["wall_s"]
        speedup = er["wall_s"] / cr["wall_s"]
        paired_speedups.append(speedup)
        paired_diffs.append(diff)
        if diff > 0:
            c_faster += 1
        else:
            e_faster += 1

    print(
        f"\n  Compile faster in {c_faster}/{len(paired_speedups)} batches, "
        f"Eager faster in {e_faster}/{len(paired_speedups)}"
    )
    print(f"  Mean time saved by compile: {statistics.mean(paired_diffs):+.3f}s/batch")
    print(f"  Median time saved by compile: {statistics.median(paired_diffs):+.3f}s/batch")
    print(f"  Mean speedup: {statistics.mean(paired_speedups):.3f}x")
    print(f"  Median speedup: {statistics.median(paired_speedups):.3f}x")

    print(f"\n  Per-batch detail:")
    print(f"  {'Batch':>5} {'Tokens':>7} {'C_time':>8} {'E_time':>8} {'Diff':>8} {'Speedup':>8} {'Winner':>8}")
    print(f"  {'-'*60}")
    for cr, er in zip(c_active, e_active):
        diff = er["wall_s"] - cr["wall_s"]
        speedup = er["wall_s"] / cr["wall_s"]
        winner = "compile" if diff > 0 else "eager"
        print(
            f"  {cr['batch']:>5} {cr['tokens']:>7} {cr['wall_s']:>8.3f} {er['wall_s']:>8.3f} "
            f"{diff:>+8.3f} {speedup:>7.2f}x {winner:>8}"
        )

    # ── Section 4: Warmup analysis ──
    print(f"\n{'='*100}")
    print("SECTION 4: WARMUP / COMPILATION COST")
    print(f"{'='*100}")
    c_warmup = c_rows[0]["wall_s"]
    e_warmup = e_rows[0]["wall_s"]
    overhead = c_warmup - e_warmup
    print(f"\n  First batch:  compile={c_warmup:.1f}s  eager={e_warmup:.1f}s")
    print(f"  Compilation overhead: {overhead:.1f}s (torch.compile + CUDA graph recording)")
    mean_saved = statistics.mean(paired_diffs) if paired_diffs else 0
    if mean_saved > 0:
        print(
            f"  Break-even point: {overhead / mean_saved:.0f} batches "
            f"(at {mean_saved:.3f}s saved/batch)"
        )
    else:
        print("  Break-even: N/A (compile is not faster per batch on average)")

    c_cum = 0
    e_cum = 0
    crossover = None
    for i, (cr, er) in enumerate(zip(c_rows, e_rows)):
        c_cum += cr["wall_s"]
        e_cum += er["wall_s"]
        if crossover is None and c_cum < e_cum:
            crossover = i
    if crossover is not None:
        print(
            f"  Actual crossover at batch {crossover} "
            f"(compile cumulative becomes faster)"
        )
    else:
        print(
            f"  Compile never catches up in cumulative time over {len(c_rows)} batches"
        )

    print(f"\n  Cumulative wall time progression:")
    print(f"  {'Batch':>5} {'C_cum':>10} {'E_cum':>10} {'C_ahead':>10}")
    c_cum = e_cum = 0
    for i, (cr, er) in enumerate(zip(c_rows, e_rows)):
        c_cum += cr["wall_s"]
        e_cum += er["wall_s"]
        ahead = e_cum - c_cum
        marker = " <-- crossover" if i == crossover else ""
        print(
            f"  {i:>5} {c_cum:>9.1f}s {e_cum:>9.1f}s {ahead:>+9.1f}s{marker}"
        )

    # ── Section 5: Throughput vs output length ──
    print(f"\n{'='*100}")
    print("SECTION 5: THROUGHPUT vs OUTPUT LENGTH (steady-state batches)")
    print(f"{'='*100}")

    bins = token_length_bins(c_active, n_bins=3)
    print(f"\n  {'Length bin':<28} {'C_tok/s':>10} {'E_tok/s':>10} {'Speedup':>10} {'N':>5}")
    print(f"  {'-'*66}")
    for lo, hi, label in bins:
        c_bin = [r["tok_s"] for r in c_active if lo <= r["tokens"] < hi]
        e_bin = [r["tok_s"] for r in e_active if lo <= r["tokens"] < hi]
        if c_bin and e_bin:
            cm = statistics.mean(c_bin)
            em = statistics.mean(e_bin)
            print(
                f"  {label:<28} {cm:>10.1f} {em:>10.1f} {cm/em:>9.2f}x {len(c_bin):>5}"
            )

    # ── Section 6: Kernel breakdown (from profiler logs — caveat about overhead) ──
    print(f"\n{'='*100}")
    print("SECTION 6: KERNEL BREAKDOWN (from torch.profiler — HIGH OVERHEAD)")
    print("  CAVEAT: torch.profiler adds large CPU overhead to eager mode.")
    print("  Kernel TIME ratios are indicative; WALL TIME from these runs is NOT.")
    print("  Wall time comparison uses TIMING logs (Sections 1-5) instead.")
    print(
        "  Group CUDA times / launch counts are summed over the printed top-30 kernel rows;"
    )
    print("  aten::* rows are omitted from CUDA buckets to avoid double-counting.")
    print(f"{'='*100}")

    if prof_c and prof_e:
        print(f"\n  {'Kernel Group':<52} {'Compile(s)':>10} {'Eager(s)':>10} {'Delta':>10}")
        print(f"  {'-'*85}")
        tc = te = 0
        for gname in GROUP_ORDER:
            c = prof_c["by_group_cuda"].get(gname, 0.0)
            e = prof_e["by_group_cuda"].get(gname, 0.0)
            if c == 0 and e == 0:
                continue
            print(f"  {gname:<52} {c:>10.3f} {e:>10.3f} {c-e:>+10.3f}")
            tc += c
            te += e
        print(f"  {'-'*85}")
        print(f"  {'Accounted (groups above)':<52} {tc:>10.3f} {te:>10.3f} {tc-te:>+10.3f}")
        pc = prof_c.get("self_cuda_total_s")
        pe = prof_e.get("self_cuda_total_s")
        if pc is not None and pe is not None:
            print(
                f"  {'Profiler Self CUDA total (log footer)':<52} {pc:>10.3f} {pe:>10.3f}"
            )

        c_tr = prof_c["by_group_cuda"].get(
            "Triton fused elementwise (RoPE,SiLU,Norm)", 0.0
        )
        e_nat = prof_e["by_group_cuda"].get(
            "Native elementwise (mul,add,copy_,to,cat)", 0.0
        )
        c_gemm = prof_c["by_group_cuda"].get(
            "Linear GEMMs (QKV+O+MLP+lm_head)", 0.0
        )
        e_gemm = prof_e["by_group_cuda"].get(
            "Linear GEMMs (QKV+O+MLP+lm_head)", 0.0
        )
        c_sc = prof_c["by_group_cuda"].get("scatter_ (KV cache write)", 0.0)
        e_sc = prof_e["by_group_cuda"].get("scatter_ (KV cache write)", 0.0)
        c_sk = prof_c["by_group_cuda"].get("splitKreduce + Memset", 0.0)
        e_sk = prof_e["by_group_cuda"].get("splitKreduce + Memset", 0.0)

        if c_tr > 0 or e_nat > 0:
            print(
                f"\n  Fusion snapshot (Self CUDA in top-30 table): "
                f"Triton fused={c_tr:.3f}s vs Native elementwise={e_nat:.3f}s "
                f"(not apples-to-apples across modes; see table above)"
            )
        if c_gemm > 0 and e_gemm > 0:
            gemm_delta = c_gemm - e_gemm
            print(
                f"  GEMM bucket delta (compile − eager): {gemm_delta:+.3f}s "
                f"({c_gemm:.3f}s vs {e_gemm:.3f}s)"
            )
        if c_sc > 0 or e_sc > 0:
            if e_sc > 0:
                print(
                    f"  scatter_ KV write: compile={c_sc:.3f}s vs eager={e_sc:.3f}s "
                    f"({(c_sc/e_sc-1)*100:+.0f}% vs eager)"
                )
            else:
                print(
                    f"  scatter_ KV write: compile={c_sc:.3f}s (not in eager top-30)"
                )
        if c_sk > 0 or e_sk > 0:
            if e_sk > 0:
                print(
                    f"  splitKreduce/memset bucket: compile={c_sk:.3f}s vs eager={e_sk:.3f}s "
                    f"({c_sk/e_sk:.2f}x calls-weighted time ratio in table)"
                )
            else:
                print(
                    f"  splitKreduce/memset bucket: compile={c_sk:.3f}s vs eager={e_sk:.3f}s"
                )

        print(f"\n  {'Category':<40} {'Compile':>12} {'Eager':>12} {'Ratio':>10}")
        print(f"  {'-'*75}")
        ttc = tte = 0
        for gname in GROUP_ORDER:
            c = prof_c["by_group_calls"].get(gname, 0)
            e = prof_e["by_group_calls"].get(gname, 0)
            if c == 0 and e == 0:
                continue
            ratio = _fmt_ratio(c, e)
            print(f"  {gname:<40} {c:>12,} {e:>12,} {ratio:>10}")
            ttc += c
            tte += e
        print(f"  {'-'*75}")
        print(f"  {'TOTAL (summed groups above)':<40} {ttc:>12,} {tte:>12,} {_fmt_ratio(ttc, tte):>10}")

        print(f"\n  CPU dispatch (aten:: CPU total column from same top-30 table):")
        print(f"  {'Metric':<45} {'Compile':>12} {'Eager':>12}")
        print(f"  {'-'*70}")
        sc_c = prof_c.get("self_cpu_total_s")
        sc_e = prof_e.get("self_cpu_total_s")
        cuda_c = prof_c.get("self_cuda_total_s")
        cuda_e = prof_e.get("self_cuda_total_s")
        if all(v is not None for v in (sc_c, sc_e, cuda_c, cuda_e)):
            print(f"  {'Self CPU time total':<45} {sc_c:>11.2f}s {sc_e:>11.2f}s")
            print(f"  {'Self CUDA time total':<45} {cuda_c:>11.2f}s {cuda_e:>11.2f}s")
            rc = sc_c / cuda_c if cuda_c else 0
            re = sc_e / cuda_e if cuda_e else 0
            print(f"  {'CPU/CUDA ratio':<45} {rc:>12.2f} {re:>12.2f}")

        for op in ("aten::linear", "aten::copy_", "aten::mul", "aten::add"):
            ac = prof_c["aten_cpu_total"].get(op)
            ae = prof_e["aten_cpu_total"].get(op)
            if ac is not None or ae is not None:
                c_s = f"{ac:.2f}s" if ac is not None else "—"
                e_s = f"{ae:.2f}s" if ae is not None else "—"
                print(f"  {op + ' CPU total':<45} {c_s:>12} {e_s:>12}")
    else:
        print("\n  (Could not parse profiler tables — check --profiler-*-log paths.)")

    # ── Section 7: Final summary ──
    print(f"\n{'='*100}")
    print("SECTION 7: FINAL SUMMARY — COMPILE vs EAGER")
    print(f"{'='*100}")

    steady_speedup = c_timing["throughput"] / e_timing["throughput"]
    overall_speedup = c_summary["tok_s"] / e_summary["tok_s"]

    # Optional profiler-derived one-liners (facts from logs; not always "wins")
    launch_line = ""
    linear_line = ""
    fusion_line = ""
    scatter_line = ""
    splitk_line = ""
    if prof_c and prof_e:
        ttc = sum(prof_c["by_group_calls"].get(g, 0) for g in GROUP_ORDER)
        tte = sum(prof_e["by_group_calls"].get(g, 0) for g in GROUP_ORDER)
        if tte > 0 and ttc > 0 and ttc != tte:
            pct = abs(1.0 - ttc / tte) * 100
            if ttc < tte:
                launch_line = (
                    f"    • {pct:.0f}% fewer kernel launches in summed top-30 groups "
                    f"({ttc:,} vs {tte:,})\n"
                )
            else:
                launch_line = (
                    f"    • {pct:.0f}% more kernel launches in summed top-30 groups "
                    f"({ttc:,} vs {tte:,})\n"
                )
        lin_c = prof_c["aten_cpu_total"].get("aten::linear")
        lin_e = prof_e["aten_cpu_total"].get("aten::linear")
        if lin_c is not None and lin_e is not None and lin_c > 0 and lin_e > 0:
            if lin_e > lin_c:
                linear_line = (
                    f"    • aten::linear CPU total: {lin_e/lin_c:.1f}x higher on eager "
                    f"({lin_c:.2f}s vs {lin_e:.2f}s)\n"
                )
            elif lin_c > lin_e:
                linear_line = (
                    f"    • aten::linear CPU total: {lin_c/lin_e:.1f}x higher on compile "
                    f"({lin_c:.2f}s vs {lin_e:.2f}s)\n"
                )
        c_tr = prof_c["by_group_cuda"].get(
            "Triton fused elementwise (RoPE,SiLU,Norm)", 0.0
        )
        e_nat = prof_e["by_group_cuda"].get(
            "Native elementwise (mul,add,copy_,to,cat)", 0.0
        )
        if c_tr > 0 or e_nat > 0:
            fusion_line = (
                f"    • Triton fused (compile) vs native elementwise (eager) Self CUDA "
                f"in top-30: {c_tr:.2f}s vs {e_nat:.2f}s\n"
            )
        c_sc = prof_c["by_group_cuda"].get("scatter_ (KV cache write)", 0.0)
        e_sc = prof_e["by_group_cuda"].get("scatter_ (KV cache write)", 0.0)
        if c_sc > 0 and e_sc > 0:
            scatter_line = (
                f"    • scatter_ KV write {((c_sc / e_sc - 1) * 100):+.0f}% more CUDA "
                f"in table ({c_sc:.3f}s vs {e_sc:.3f}s)\n"
            )
        elif c_sc > 0:
            scatter_line = (
                f"    • scatter_ KV write in compile top-30 ({c_sc:.3f}s); "
                f"not in eager top-30\n"
            )
        c_sk = prof_c["by_group_calls"].get("splitKreduce + Memset", 0)
        e_sk = prof_e["by_group_calls"].get("splitKreduce + Memset", 0)
        if c_sk > 0 and e_sk > 0:
            splitk_line = (
                f"    • splitKreduce/memset launches: {c_sk:,} vs {e_sk:,} "
                f"({_fmt_ratio(c_sk, e_sk)} in table)\n"
            )
        elif c_sk > 0 or e_sk > 0:
            splitk_line = (
                f"    • splitKreduce/memset launches: {c_sk:,} vs {e_sk:,} (see Section 6)\n"
            )

    cross_msg = (
        "compile catches up at batch " + str(crossover)
        if crossover is not None
        else "compile never catches up in cumulative wall time over these samples"
    )

    breakeven_batches = (
        int(overhead / mean_saved) if mean_saved > 0 else None
    )
    breakeven_str = (
        f"~{breakeven_batches} batches"
        if breakeven_batches is not None
        else "N/A (compile not faster per batch on average)"
    )

    print(f"""
  DATA QUALITY NOTE:
    Wall time & throughput from TIMING logs (no profiler overhead).
    Kernel breakdown from PROFILER logs (high CPU overhead on eager — wall times invalid,
    but grouped Self CUDA from the printed kernel table is still useful for directionally
    comparing where time goes).

  THROUGHPUT (no profiler overhead):
    Overall (incl warmup):   compile={c_summary['tok_s']:.1f} tok/s  eager={e_summary['tok_s']:.1f} tok/s  ({overall_speedup:.2f}x compile/eager)
    Steady-state (after skip): compile={c_timing['throughput']:.1f} tok/s  eager={e_timing['throughput']:.1f} tok/s  ({steady_speedup:.2f}x compile/eager)
    Compile faster in {c_faster}/{len(paired_speedups)} batches (paired comparison)
    Mean per-batch speedup (eager/compile wall):  {statistics.mean(paired_speedups):.3f}x

  PROFILER TABLE (top-30 kernels — directional, same paths as Section 6):
{launch_line}{linear_line}{fusion_line}    • See Section 6 for Self CPU/CUDA footer totals and per-op aten:: rows

  COST / TRADE-OFFS:
    • First-batch wall overhead vs eager: {overhead:.0f}s
{scatter_line}{splitk_line}    • CUDA graphs imply fixed batch shapes vs compact eager batching (see project notes)

  BREAK-EVEN:
    Compilation overhead (first batch): {overhead:.0f}s
    Mean time saved per batch (paired, eager−compile): {mean_saved:+.3f}s
    Estimated break-even: {breakeven_str}
    Within {n_samples} samples: {cross_msg}
""")


if __name__ == "__main__":
    main()
