#!/usr/bin/env python3
"""
Parse nsys kernel CSV exports and produce compile vs eager comparison tables.

Usage:
    # After running: bash debug/run_nsys_bs1.sh
    python debug/parse_nsys_comparison.py [--logs-dir logs]

Reads:
    logs/nsys_compile_bs1_kernels_cuda_gpu_kern_sum.csv
    logs/nsys_eager_bs1_kernels_cuda_gpu_kern_sum.csv
    (or .nsys-rep files directly via nsys stats)

The CSV from `nsys stats --report cuda_gpu_kern_sum` has columns:
    Time(%),Total Time(ns),Instances,Avg(ns),Med(ns),Min(ns),Max(ns),StdDev(ns),Name
"""

import csv
import os
import re
import sys
import argparse
from collections import defaultdict


def classify_kernel(name):
    """Classify a CUDA kernel name into a functional group."""
    n = name.lower()

    if 'cutlass' in n or 'gemm' in n or 'cublas' in n:
        if 'splitk' in n or 'reduce' in n:
            return "GEMM splitK/reduce"
        return "GEMM (linear layers)"
    if 'flash' in n and ('fwd' in n or 'attn' in n or 'kernel' in n):
        if 'combine' in n:
            return "Flash Attn combine"
        return "Flash Attention"
    if 'triton' in n:
        if 'silu' in n or 'mul' in n and 'fused' in n:
            return "Triton fused (SiLU/MLP)"
        if 'norm' in n or 'mean' in n or 'rsqrt' in n:
            return "Triton fused (RMSNorm)"
        if 'copy' in n:
            return "Triton fused (copy)"
        if 'rope' in n or 'rotary' in n or 'cos' in n or 'sin' in n or 'cat' in n or 'neg' in n:
            return "Triton fused (RoPE)"
        return "Triton fused (other)"
    if 'scatter' in n:
        return "scatter_ (KV write)"
    if 'memset' in n or 'memcpy' in n:
        return "Memset/Memcpy"
    if 'elementwise' in n or 'vectorized' in n or 'unrolled' in n:
        return "Native elementwise"
    if 'reduce' in n:
        return "Native reduce"
    if 'cat' in n and 'batch' in n:
        return "Native cat"
    if 'copy' in n or '_to_copy' in n:
        return "Native copy"
    if 'embedding' in n or 'index' in n:
        return "Embedding/Index"
    return "Other"


def parse_nsys_csv(csv_path):
    """Parse nsys cuda_gpu_kern_sum CSV into list of dicts."""
    kernels = []
    with open(csv_path, newline='') as f:
        # nsys CSV may have header lines starting with "="
        lines = f.readlines()

    # Find the header row (contains "Name")
    header_idx = None
    for i, line in enumerate(lines):
        if 'Name' in line and 'Time' in line:
            header_idx = i
            break

    if header_idx is None:
        # Try treating as standard CSV
        header_idx = 0

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        # Handle different column name formats
        name = row.get('Name', row.get('name', ''))
        if not name or name.startswith('='):
            continue

        # Parse time - could be in ns or other units
        total_ns = 0
        for key in ['Total Time(ns)', 'Total Time (ns)', 'total_time']:
            if key in row:
                try:
                    total_ns = int(row[key].replace(',', ''))
                except (ValueError, AttributeError):
                    pass
                break

        instances = 0
        for key in ['Instances', 'instances', 'Count']:
            if key in row:
                try:
                    instances = int(row[key].replace(',', ''))
                except (ValueError, AttributeError):
                    pass
                break

        avg_ns = 0
        for key in ['Avg(ns)', 'Avg (ns)', 'avg']:
            if key in row:
                try:
                    avg_ns = float(row[key].replace(',', ''))
                except (ValueError, AttributeError):
                    pass
                break

        kernels.append({
            'name': name.strip(),
            'total_ns': total_ns,
            'instances': instances,
            'avg_ns': avg_ns,
            'group': classify_kernel(name),
        })

    return kernels


def group_kernels(kernels):
    """Aggregate kernels by functional group."""
    groups = defaultdict(lambda: {'total_ns': 0, 'instances': 0})
    for k in kernels:
        g = groups[k['group']]
        g['total_ns'] += k['total_ns']
        g['instances'] += k['instances']
    return dict(groups)


def print_comparison(compile_kernels, eager_kernels):
    """Print side-by-side comparison tables."""
    c_groups = group_kernels(compile_kernels)
    e_groups = group_kernels(eager_kernels)

    c_total_ns = sum(k['total_ns'] for k in compile_kernels)
    e_total_ns = sum(k['total_ns'] for k in eager_kernels)

    all_group_names = sorted(set(list(c_groups.keys()) + list(e_groups.keys())),
                              key=lambda g: max(c_groups.get(g, {}).get('total_ns', 0),
                                                e_groups.get(g, {}).get('total_ns', 0)),
                              reverse=True)

    # ── Kernel group time comparison ──
    print("=" * 100)
    print("KERNEL GROUP TIME COMPARISON (nsys, low overhead)")
    print("=" * 100)
    print(f"\n{'Group':<40} {'Compile':>10} {'C_%':>6} {'Eager':>10} {'E_%':>6} {'Delta':>10}")
    print("-" * 85)

    for g in all_group_names:
        c_ns = c_groups.get(g, {}).get('total_ns', 0)
        e_ns = e_groups.get(g, {}).get('total_ns', 0)
        c_s = c_ns / 1e9
        e_s = e_ns / 1e9
        c_pct = c_ns / c_total_ns * 100 if c_total_ns else 0
        e_pct = e_ns / e_total_ns * 100 if e_total_ns else 0
        delta_s = c_s - e_s
        print(f"{g:<40} {c_s:>9.3f}s {c_pct:>5.1f}% {e_s:>9.3f}s {e_pct:>5.1f}% {delta_s:>+9.3f}s")

    print("-" * 85)
    print(f"{'TOTAL GPU KERNEL TIME':<40} {c_total_ns/1e9:>9.3f}s {'100%':>6} {e_total_ns/1e9:>9.3f}s {'100%':>6} {(c_total_ns-e_total_ns)/1e9:>+9.3f}s")

    # ── Kernel launch count comparison ──
    print(f"\n{'='*100}")
    print("KERNEL LAUNCH COUNT COMPARISON (nsys)")
    print(f"{'='*100}")
    print(f"\n{'Group':<40} {'Compile':>12} {'Eager':>12} {'Ratio':>10}")
    print("-" * 75)

    c_total_inst = sum(k['instances'] for k in compile_kernels)
    e_total_inst = sum(k['instances'] for k in eager_kernels)

    for g in all_group_names:
        c_inst = c_groups.get(g, {}).get('instances', 0)
        e_inst = e_groups.get(g, {}).get('instances', 0)
        if c_inst == 0 and e_inst == 0:
            continue
        ratio = f"{c_inst/e_inst:.2f}x" if e_inst > 0 else ("N/A" if c_inst > 0 else "0")
        print(f"{g:<40} {c_inst:>12,} {e_inst:>12,} {ratio:>10}")

    print("-" * 75)
    ratio = f"{c_total_inst/e_total_inst:.2f}x" if e_total_inst else "N/A"
    print(f"{'TOTAL':<40} {c_total_inst:>12,} {e_total_inst:>12,} {ratio:>10}")

    # ── Top 15 kernels by time (each mode) ──
    for label, kernels, total_ns in [("COMPILE", compile_kernels, c_total_ns),
                                       ("EAGER", eager_kernels, e_total_ns)]:
        print(f"\n{'='*100}")
        print(f"TOP 15 KERNELS BY GPU TIME — {label}")
        print(f"{'='*100}")
        sorted_k = sorted(kernels, key=lambda x: x['total_ns'], reverse=True)[:15]
        print(f"{'#':>3} {'Time(ms)':>10} {'%':>6} {'Calls':>8} {'Avg(us)':>10} {'Name'}")
        print("-" * 100)
        for i, k in enumerate(sorted_k, 1):
            t_ms = k['total_ns'] / 1e6
            pct = k['total_ns'] / total_ns * 100 if total_ns else 0
            avg_us = k['avg_ns'] / 1e3
            short_name = k['name'][:65]
            print(f"{i:>3} {t_ms:>10.1f} {pct:>5.1f}% {k['instances']:>8,} {avg_us:>10.1f} {short_name}")

    # ── Summary ──
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")

    c_triton = sum(v['total_ns'] for g, v in c_groups.items() if 'Triton' in g)
    e_native = sum(v['total_ns'] for g, v in e_groups.items() if 'Native' in g)
    c_gemm = c_groups.get('GEMM (linear layers)', {}).get('total_ns', 0)
    e_gemm = e_groups.get('GEMM (linear layers)', {}).get('total_ns', 0)
    c_scatter = c_groups.get('scatter_ (KV write)', {}).get('total_ns', 0)
    e_scatter = e_groups.get('scatter_ (KV write)', {}).get('total_ns', 0)

    print(f"""
  Total GPU kernel time:  compile={c_total_ns/1e9:.3f}s  eager={e_total_ns/1e9:.3f}s  delta={((c_total_ns-e_total_ns)/1e9):+.3f}s
  Total kernel launches:  compile={c_total_inst:,}  eager={e_total_inst:,}  ({c_total_inst/e_total_inst:.2f}x)

  Fusion effect:
    Triton fused (compile): {c_triton/1e9:.3f}s
    Native elem. (eager):   {e_native/1e9:.3f}s
    Savings from fusion:    {(e_native-c_triton)/1e9:+.3f}s

  GEMM comparison:
    compile: {c_gemm/1e9:.3f}s   eager: {e_gemm/1e9:.3f}s   delta: {(c_gemm-e_gemm)/1e9:+.3f}s

  KV cache scatter_:
    compile: {c_scatter/1e9:.3f}s  eager: {e_scatter/1e9:.3f}s  delta: {(c_scatter-e_scatter)/1e9:+.3f}s

  NOTE: These timings have minimal profiling overhead (~2-5% vs torch.profiler's 3-25x).
  Wall time differences are primarily from CPU dispatch overhead, not captured in kernel times.
""")


def main():
    parser = argparse.ArgumentParser(description="Parse nsys kernel CSVs for compile vs eager comparison")
    parser.add_argument('--logs-dir', default=os.path.join(os.path.dirname(__file__), '..', 'logs'),
                        help='Directory containing nsys CSV exports')
    args = parser.parse_args()

    # Try multiple naming patterns nsys uses
    patterns = [
        ("nsys_compile_bs1_kernels_cuda_gpu_kern_sum.csv",
         "nsys_eager_bs1_kernels_cuda_gpu_kern_sum.csv"),
        ("nsys_compile_bs1_kernels.csv",
         "nsys_eager_bs1_kernels.csv"),
    ]

    compile_csv = eager_csv = None
    for c_name, e_name in patterns:
        c_path = os.path.join(args.logs_dir, c_name)
        e_path = os.path.join(args.logs_dir, e_name)
        if os.path.exists(c_path) and os.path.exists(e_path):
            compile_csv = c_path
            eager_csv = e_path
            break

    if not compile_csv:
        print("ERROR: nsys kernel CSV files not found in", args.logs_dir)
        print("Expected files matching: nsys_*_bs1_kernels*.csv")
        print("\nRun first:  bash debug/run_nsys_bs1.sh")

        # Try to generate from .nsys-rep files
        for mode in ['compile', 'eager']:
            rep = os.path.join(args.logs_dir, f"nsys_{mode}_bs1.nsys-rep")
            if os.path.exists(rep):
                print(f"\nFound {rep}, attempting to export CSV...")
                out = os.path.join(args.logs_dir, f"nsys_{mode}_bs1_kernels")
                os.system(f"nsys stats --report cuda_gpu_kern_sum --format csv "
                          f"--output '{out}' '{rep}'")

        # Retry
        for c_name, e_name in patterns:
            c_path = os.path.join(args.logs_dir, c_name)
            e_path = os.path.join(args.logs_dir, e_name)
            if os.path.exists(c_path) and os.path.exists(e_path):
                compile_csv = c_path
                eager_csv = e_path
                break

        if not compile_csv:
            sys.exit(1)

    print(f"Compile CSV: {compile_csv}")
    print(f"Eager CSV:   {eager_csv}")
    print()

    compile_kernels = parse_nsys_csv(compile_csv)
    eager_kernels = parse_nsys_csv(eager_csv)

    if not compile_kernels or not eager_kernels:
        print("ERROR: Failed to parse kernel data from CSVs")
        sys.exit(1)

    print_comparison(compile_kernels, eager_kernels)


if __name__ == "__main__":
    main()
