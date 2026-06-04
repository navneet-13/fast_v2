#!/bin/bash
# Run nsys profiling for compile and eager at BS=1 with minimal overhead.
# nsys uses CUPTI sampling (~2-5% overhead) vs torch.profiler's per-op wrapping (3-25x).
#
# Usage: bash debug/run_nsys_bs1.sh
#
# Output:
#   logs/nsys_compile_bs1.nsys-rep  — raw nsys report
#   logs/nsys_eager_bs1.nsys-rep    — raw nsys report
#   logs/nsys_compile_bs1_kernels.csv — kernel summary
#   logs/nsys_eager_bs1_kernels.csv   — kernel summary

set -e
cd "$(dirname "$0")/.."

VENV_PYTHON="./v2/bin/python3"
MODEL_PATH="./models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/"
LIMIT=10
LOG_DIR="./logs"

# Common env vars (profiling DISABLED — nsys handles it externally)
export FAST_DLLM_USE_DYNAMO=1
export FAST_DLLM_USE_SPARSE=0
export FAST_DLLM_BATCH_MODE=compact
export FAST_DLLM_ATTENTION_BACKEND=auto
export FAST_DLLM_MAX_SEQ_LEN=2048
export FAST_DLLM_COMPILE_MODE=reduce-overhead
export FAST_DLLM_PROFILE=0
export BATCH_SIZE=1
export FAST_DLLM_TRANSFER_RATIO=.25
export FAST_DLLM_REFRESH_INTERVAL=100000
export FAST_DLLM_TIMING=1
export FAST_DLLM_TIMING_SKIP=3
export FAST_DLLM_PROFILE=0
export FAST_DLLM_PROFILE_SKIP=3
export FAST_DLLM_PROFILE_ACTIVE=5

# nsys options:
#   --capture-range=cudaProfilerApi  — only profile between cudaProfilerStart/Stop
#     (not used here; we profile the whole run and filter later)
#   -t cuda,nvtx,osrt               — trace CUDA kernels, NVTX markers, OS runtime
#   --stats=true                     — auto-generate stats after capture
#   -f true                          — overwrite existing report
#   -o <output>                      — output file (without .nsys-rep extension)
#   --gpu-metrics-device=0           — collect GPU metrics
#
# NOTE: eager mode fails with nsys 2023.4 due to "Wrong event order" bug in the
# .qdstrm importer — the high volume of CUDA events from eager's ~100K+ kernel
# launches causes out-of-order timestamps. Compile mode works because CUDA graph
# replay produces far fewer events. Use debug/bench_forward_bs1.py for fair
# eager vs compile comparison instead.

echo "============================================"
echo "Running EAGER with nsys (BS=1, limit=$LIMIT)"
echo "============================================"
export FAST_DLLM_EXECUTION_MODE=eager
    # --gpu-metrics-device=0 \

nsys profile \
    -t cuda,nvtx \
    --stats=true \
    --sample=none \
    --cpuctxsw=none \
    -f true \
    -o "${LOG_DIR}/nsys_eager_bs1" \
    $VENV_PYTHON -m accelerate.commands.launch \
        --num_processes 1 \
        eval.py \
        --model fast_dllm_v2 \
        --model_args "model_path=${MODEL_PATH},threshold=0.9,show_speed=True" \
        --tasks gsm8k \
        --num_fewshot 0 \
        --batch_size 1 \
        --limit $LIMIT \
    2>&1 | tee "${LOG_DIR}/nsys_eager_bs1.log"

echo ""
echo "============================================"
echo "Running COMPILE with nsys (BS=1, limit=$LIMIT)"
echo "============================================"
export FAST_DLLM_EXECUTION_MODE=compile

    # --gpu-metrics-device=0 \

nsys profile \
    -t cuda,nvtx \
    --stats=true \
    --sample=none \
    --cpuctxsw=none \
    -f true \
    -o "${LOG_DIR}/nsys_compile_bs1" \
    $VENV_PYTHON -m accelerate.commands.launch \
        --num_processes 1 \
        eval.py \
        --model fast_dllm_v2 \
        --model_args "model_path=${MODEL_PATH},threshold=0.9,show_speed=True" \
        --tasks gsm8k \
        --num_fewshot 0 \
        --batch_size 1 \
        --limit $LIMIT \
    2>&1 | tee "${LOG_DIR}/nsys_compile_bs1.log"

echo ""
echo "============================================"
echo "Exporting kernel CSV summaries"
echo "============================================"

# Export CUDA kernel statistics to CSV
nsys stats --report cuda_gpu_kern_sum \
    --format csv \
    --force-export=true \
    --force-overwrite=true \
    --output "${LOG_DIR}/nsys_eager_bs1_kernels" \
    "${LOG_DIR}/nsys_eager_bs1.nsys-rep" 2>/dev/null || true

nsys stats --report cuda_gpu_kern_sum \
    --format csv \
    --force-export=true \
    --force-overwrite=true \
    --output "${LOG_DIR}/nsys_compile_bs1_kernels" \
    "${LOG_DIR}/nsys_compile_bs1.nsys-rep" 2>/dev/null || true

echo ""
echo "Done. Parse results with:"
echo "  python debug/parse_nsys_comparison.py"
