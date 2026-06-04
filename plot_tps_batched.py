#!/usr/bin/env python3
"""
Plot Tokens Per Second (TPS) results from sweep logs, with batch size on x-axis.

Directory layout expected:

  sweep_dir/
    baseline/
      bs_1.log, bs_2.log, bs_4.log, bs_8.log, bs_16.log, bs_32.log
    sparse_ratio_0.25/
      bs_1.log, bs_2.log, ...
    sparse_ratio_0.50/
      ...
    static_KV/
      ...

We:
  - Scan all subdirs under sweep_dir for bs_<N>.log files
  - x-axis = batch size (1, 2, 4, 8, 16, 32)
  - y-axis = TPS
  - One line per config type (baseline, sparse_ratio_*, static_KV, etc.)
  - Prints a speedup table to the terminal (TPS / baseline TPS per batch size)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def extract_tps_from_log(log_path: str):
    """Extract Tokens per second from a log file. Returns float or None."""
    try:
        content = Path(log_path).read_text()
    except Exception as e:
        print(f"  Error reading {log_path}: {e}")
        return None

    # Look for "Tokens per second: XX.XX"
    m = re.search(r"Tokens per second:\s*([\d.]+)", content)
    if not m:
        print(f"  Warning: Could not find TPS in {log_path}")
        return None
    try:
        return float(m.group(1))
    except ValueError:
        print(f"  Warning: Failed to parse TPS in {log_path}")
        return None


def find_logs_with_batch(sweep_dir: str):
    """
    Scan sweep_dir for bs_<N>.log files in each config subdir.

    Returns:
      data: dict[config_type][batch_size] = tps

    where:
      config_type is a string like "baseline", "sparse_ratio_0.25", "static_KV"
      batch_size is int (1, 2, 4, 8, 16, 32)
    """
    sweep_path = Path(sweep_dir)
    data = {}

    # Pattern for filenames: bs_<N>.log
    fname_re = re.compile(r"bs_(\d+)\.log")

    for subdir in sweep_path.iterdir():
        if not subdir.is_dir():
            continue
        config_type = subdir.name
        print(f"Scanning config dir: {config_type}")
        for log_file in subdir.glob("bs_*.log"):
            m = fname_re.match(log_file.name)
            if not m:
                continue
            batch_size = int(m.group(1))
            tps = extract_tps_from_log(str(log_file))
            if tps is not None:
                data.setdefault(config_type, {})[batch_size] = tps

    return data


def _speedup_table_rows(
    data: dict,
    batch_sizes_sorted: list,
    baseline_name: str,
):
    """
    Build speedup vs baseline: one row per non-baseline config.

    Returns (cell_text, row_labels, col_labels) or None if table cannot be built.
    """
    if baseline_name not in data:
        print(f"Warning: baseline config '{baseline_name}' not found; skipping speedup table.")
        return None

    baseline_map = data[baseline_name]
    other_cfgs = [c for c in sorted(data.keys()) if c != baseline_name]
    if not other_cfgs:
        print("Warning: no configs other than baseline; skipping speedup table.")
        return None

    col_labels = [str(bs) for bs in batch_sizes_sorted]
    row_labels = []
    cell_text = []

    for cfg in other_cfgs:
        row_labels.append(cfg)
        row = []
        batch_map = data[cfg]
        for bs in batch_sizes_sorted:
            b = baseline_map.get(bs)
            t = batch_map.get(bs)
            if b is None or t is None or b <= 0:
                row.append("—")
            else:
                row.append(f"{t / b:.2f}×")
        cell_text.append(row)

    return cell_text, row_labels, col_labels


def print_speedup_table(data: dict, batch_sizes_sorted: list, baseline_name: str) -> None:
    """Print an aligned ASCII table of speedup vs baseline to stdout."""
    built = _speedup_table_rows(data, batch_sizes_sorted, baseline_name)
    if built is None:
        return
    cell_text, row_labels, col_labels = built

    label_w = max(len("config"), max(len(r) for r in row_labels))
    col_widths = []
    for j in range(len(col_labels)):
        w = len(col_labels[j])
        for i in range(len(row_labels)):
            w = max(w, len(cell_text[i][j]))
        col_widths.append(w)

    title = f"Speedup vs '{baseline_name}' (TPS ratio)"
    sep = "  ".join(["-" * label_w] + ["-" * w for w in col_widths])
    header = f"{'config':<{label_w}}"
    for j, lab in enumerate(col_labels):
        header += f"  {lab:>{col_widths[j]}}"
    print()
    print(title)
    print(sep)
    print(header)
    print(sep)
    for i, cfg in enumerate(row_labels):
        line = f"{cfg:<{label_w}}"
        for j, cell in enumerate(cell_text[i]):
            line += f"  {cell:>{col_widths[j]}}"
        print(line)
    print(sep)
    print()


def plot_tps_batched(
    sweep_dir: str,
    output_file: str = None,
    show: bool = True,
    baseline_name: str = "baseline",
):
    """
    Plot TPS vs batch size (x-axis) with one line per config type.
    Prints speedup vs baseline to the terminal when baseline exists.
    """
    print(f"Scanning sweep directory: {sweep_dir}")
    data = find_logs_with_batch(sweep_dir)

    if not data:
        print("ERROR: No TPS data found in sweep_dir.")
        return

    # Collect all batch sizes present
    all_batch_sizes = set()
    for cfg_type in data:
        all_batch_sizes.update(data[cfg_type].keys())
    batch_sizes_sorted = sorted(all_batch_sizes)
    print(f"Discovered batch sizes: {batch_sizes_sorted}")

    print_speedup_table(data, batch_sizes_sorted, baseline_name)

    # Colors for config types
    colors = [
        "#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12",
        "#1abc9c", "#e67e22", "#34495e",
    ]
    color_map = {cfg: colors[i % len(colors)] for i, cfg in enumerate(sorted(data.keys()))}

    fig, ax = plt.subplots(figsize=(10, 6))

    # One line per config type: x = batch size, y = TPS
    for cfg_type in sorted(data.keys()):
        batch_map = data[cfg_type]
        x_vals = []
        y_vals = []
        for bs in batch_sizes_sorted:
            if bs in batch_map:
                x_vals.append(bs)
                y_vals.append(batch_map[bs])
        if not x_vals:
            continue

        color = color_map.get(cfg_type, "#95a5a6")
        ax.plot(
            x_vals,
            y_vals,
            color=color,
            marker="o",
            markersize=8,
            linewidth=2,
            label=cfg_type,
        )

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_title("Throughput (TPS) vs Batch Size", fontsize=14)
    ax.set_xscale("log")
    if batch_sizes_sorted:
        ax.set_xticks(batch_sizes_sorted)
        ax.set_xticklabels([str(bs) for bs in batch_sizes_sorted])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Y-limits with padding
    all_tps_vals = [tps for batch_map in data.values() for tps in batch_map.values()]
    if all_tps_vals:
        y_min = min(all_tps_vals) - 2
        y_max = max(all_tps_vals) + 2
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot TPS vs batch size (x-axis) from sweep logs with bs_<N>.log files"
    )
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        default="logs/sweep_parallel_20260317_044346",
        help="Path to sweep directory (default: logs/sweep_parallel_20260317_044346)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tps_batched_plot.png",
        help="Output file for plot (default: tps_batched_plot.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot (just save)",
    )
    parser.add_argument(
        "--baseline",
        default="baseline",
        help="Config subdirectory name used as baseline for terminal speedup table (default: baseline)",
    )

    args = parser.parse_args()
    plot_tps_batched(
        args.sweep_dir,
        args.output,
        show=not args.no_show,
        baseline_name=args.baseline,
    )


if __name__ == "__main__":
    main()
