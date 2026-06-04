#!/bin/bash
# Profile sync overhead in batch_sample_dynamo at multiple batch sizes.
#
# For each BS, breaks wall time into:
#   forward (graph replay / eager forward_dynamo)
#   sync    (GPU->CPU: .item(), .nonzero(), __bool__ from .any()/.all()/.sum()==0)
#   other   (sampling, unmasking, Python loop logic)
#
# Usage:
#   bash debug/run_bench_sync.sh                            # eager, BS=1,2,4,8
#   bash debug/run_bench_sync.sh --mode compile             # compile mode
#   bash debug/run_bench_sync.sh --batch_sizes 1,4,8,16     # custom BS
#   bash debug/run_bench_sync.sh --limit 32                  # more prompts

set -e
cd "$(dirname "$0")/.."

VENV_PYTHON="./v2/bin/python3"

# Pass all args through
$VENV_PYTHON debug/bench_sync_overhead.py "$@" 2>&1 | tee logs/bench_sync_overhead.log
