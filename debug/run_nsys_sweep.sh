#!/bin/bash
# Run nsys profiling at multiple batch sizes for both eager and compile.
#
# Uses nsys 2026.2.1 (not system nsys 2023.4 which has event ordering bugs).
#
# Output per run:
#   logs/nsys_{mode}_bs{BS}.nsys-rep    — raw nsys report
#   logs/nsys_{mode}_bs{BS}.sqlite      — sqlite export
#   logs/nsys_{mode}_bs{BS}_kernels_cuda_gpu_kern_sum.csv — kernel summary
#
# Usage:
#   bash debug/run_nsys_sweep.sh                    # default: eager+compile, BS=1,2,4,8
#   bash debug/run_nsys_sweep.sh 1,4,8 eager        # custom BS, eager only
#   bash debug/run_nsys_sweep.sh 1,2,4,8,16 compile # custom BS, compile only

set -e
cd "$(dirname "$0")/.."

NSYS="/nethome/n41/nsys26/bin/nsys"
VENV_PYTHON="./v2/bin/python3"
MODEL_PATH="./models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/"
# LIMIT is set from 3rd arg (default 32) above
LOG_DIR="./logs"

BATCH_SIZES="${1:-1,2,4,8}"
MODES="${2:-eager,compile}"
LIMIT="${3:-32}"

# Use a free GPU (default: 0). Override with: CUDA_VISIBLE_DEVICES=2 bash ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# Common env vars
export FAST_DLLM_USE_DYNAMO=1
export FAST_DLLM_USE_SPARSE=0
export FAST_DLLM_ATTENTION_BACKEND=auto
# 1024 instead of 2048 — halves KV cache memory, allows BS=8 on 23GB GPU
export FAST_DLLM_MAX_SEQ_LEN=2048
export FAST_DLLM_COMPILE_MODE=reduce-overhead
export FAST_DLLM_PROFILE=0
export FAST_DLLM_TRANSFER_RATIO=.25
export FAST_DLLM_REFRESH_INTERVAL=100000

IFS=',' read -ra BS_ARRAY <<< "$BATCH_SIZES"
IFS=',' read -ra MODE_ARRAY <<< "$MODES"

for MODE in "${MODE_ARRAY[@]}"; do
    for BS in "${BS_ARRAY[@]}"; do
        LABEL="${MODE}_bs${BS}"
        echo ""
        echo "============================================"
        echo "  ${MODE} BS=${BS} (limit=${LIMIT})"
        echo "============================================"

        export FAST_DLLM_EXECUTION_MODE="${MODE}"
        export BATCH_SIZE="${BS}"

        if [ "$MODE" = "compile" ]; then
            export FAST_DLLM_BATCH_MODE=compact
        else
            export FAST_DLLM_BATCH_MODE=static
        fi

        $NSYS profile \
            -t cuda,nvtx \
            --stats=true \
            --sample=none \
            --cpuctxsw=none \
            -f true \
            -o "${LOG_DIR}/nsys_${LABEL}" \
            $VENV_PYTHON -m accelerate.commands.launch \
                --num_processes 1 \
                eval.py \
                --model fast_dllm_v2 \
                --model_args "model_path=${MODEL_PATH},threshold=0.9,show_speed=True" \
                --tasks gsm8k \
                --num_fewshot 0 \
                --batch_size "${BS}" \
                --limit $LIMIT \
            2>&1 | tee "${LOG_DIR}/nsys_${LABEL}.log"

        # Export kernel CSV
        $NSYS stats --report cuda_gpu_kern_sum \
            --format csv \
            --output "${LOG_DIR}/nsys_${LABEL}_kernels" \
            "${LOG_DIR}/nsys_${LABEL}.nsys-rep" 2>/dev/null || true

        # Export CUDA API CSV
        $NSYS stats --report cuda_api_sum \
            --format csv \
            --output "${LOG_DIR}/nsys_${LABEL}_api" \
            "${LOG_DIR}/nsys_${LABEL}.nsys-rep" 2>/dev/null || true

        echo ""
    done
done

echo ""
echo "============================================"
echo "Summary of all runs"
echo "============================================"

# Print quick comparison from logs
for MODE in "${MODE_ARRAY[@]}"; do
    for BS in "${BS_ARRAY[@]}"; do
        LABEL="${MODE}_bs${BS}"
        LOG="${LOG_DIR}/nsys_${LABEL}.log"
        if [ -f "$LOG" ]; then
            TOKS=$(grep "Tokens per second" "$LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
            TOTAL=$(grep "Total time taken" "$LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
            SYNCS=$(grep "cudaStreamSynchronize" "$LOG" 2>/dev/null | head -1 | awk '{print $3}')
            GRAPH=$(grep "cudaGraphLaunch" "$LOG" 2>/dev/null | head -1 | awk '{print $3}')
            printf "  %-20s  tok/s=%-8s  time=%-8s  sync_ns=%-15s  graph_ns=%-15s\n" \
                "$LABEL" "${TOKS:-N/A}" "${TOTAL:-N/A}" "${SYNCS:-0}" "${GRAPH:-0}"
        fi
    done
done

echo ""
echo "Done. Reports in ${LOG_DIR}/nsys_*.nsys-rep"
echo "Open in Nsight Systems UI or parse CSVs in ${LOG_DIR}/nsys_*_kernels*.csv"
