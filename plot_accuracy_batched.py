#!/usr/bin/env python3
"""
Plot accuracy (exact_match) results from sweep logs, with batch size on x-axis.

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
  - Extract flexible-extract exact_match accuracy from each log
  - x-axis = batch size (1, 2, 4, 8, 16, 32)
  - y-axis = accuracy (0-1)
  - One line per config type (baseline, sparse_ratio_*, static_KV, etc.)
"""

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt


def extract_accuracy_from_log(log_path: str):
    """Extract flexible-extract exact_match accuracy from a log file. Returns float or None."""
    try:
        content = Path(log_path).read_text()
    except Exception as e:
        print(f"  Error reading {log_path}: {e}")
        return None

    # Look for "flexible-extract|     0|exact_match|↑  |0.8256|±"
    m = re.search(r"flexible-extract.*?exact_match\|\s*↑\s*\|\s*([\d.]+)", content)
    if not m:
        print(f"  Warning: Could not find accuracy in {log_path}")
        return None
    try:
        return float(m.group(1))
    except ValueError:
        print(f"  Warning: Failed to parse accuracy in {log_path}")
        return None


def find_logs_with_batch(sweep_dir: str):
    """
    Scan sweep_dir for bs_<N>.log files in each config subdir.

    Returns:
      data: dict[config_type][batch_size] = accuracy

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
            accuracy = extract_accuracy_from_log(str(log_file))
            if accuracy is not None:
                data.setdefault(config_type, {})[batch_size] = accuracy
                print(f"    bs_{batch_size}: exact_match = {accuracy:.6f}")

    return data


def plot_accuracy_batched(sweep_dir: str, output_file: str = None, show: bool = True):
    """
    Plot accuracy vs batch size (x-axis) with one line per config type.
    """
    print(f"Scanning sweep directory: {sweep_dir}")
    data = find_logs_with_batch(sweep_dir)

    if not data:
        print("ERROR: No accuracy data found in sweep_dir.")
        return

    # Collect all batch sizes present
    all_batch_sizes = set()
    for cfg_type in data:
        all_batch_sizes.update(data[cfg_type].keys())
    batch_sizes_sorted = sorted(all_batch_sizes)
    print(f"Discovered batch sizes: {batch_sizes_sorted}")

    # Terminal summary of all accuracy values in the sweep
    print("\n--- Sweep accuracy summary (exact_match) ---")
    for cfg_type in sorted(data.keys()):
        print(f"  [{cfg_type}]")
        parts = []
        for bs in batch_sizes_sorted:
            if bs in data[cfg_type]:
                parts.append(f"bs_{bs}={data[cfg_type][bs]:.6f}")
        print("    " + "  ".join(parts) if parts else "    (no data)")
    print("--- end summary ---\n")

    # Colors for config types
    colors = [
        "#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12",
        "#1abc9c", "#e67e22", "#34495e",
    ]
    color_map = {cfg: colors[i % len(colors)] for i, cfg in enumerate(sorted(data.keys()))}

    fig, ax = plt.subplots(figsize=(10, 6))


    # One line per config type: x = batch size, y = accuracy
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
    ax.set_ylabel("Accuracy (exact_match)", fontsize=12)
    ax.set_title("Accuracy vs Batch Size", fontsize=14)
    ax.set_xscale("log")
    if batch_sizes_sorted:
        ax.set_xticks(batch_sizes_sorted)
        ax.set_xticklabels([str(bs) for bs in batch_sizes_sorted])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Y-limits: accuracy is 0-1, add padding
    all_acc_vals = [acc for batch_map in data.values() for acc in batch_map.values()]
    if all_acc_vals:
        y_min = max(0, min(all_acc_vals) - 0.05)
        y_max = min(1, max(all_acc_vals) + 0.05)
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
        description="Plot accuracy vs batch size (x-axis) from sweep logs with bs_<N>.log files"
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
        default="accuracy_batched_plot.png",
        help="Output file for plot (default: accuracy_batched_plot.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot (just save)",
    )

    args = parser.parse_args()
    plot_accuracy_batched(args.sweep_dir, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
