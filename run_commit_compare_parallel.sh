#!/bin/bash

# ==============================================================================
# Parallel sweep: Dynamo with commit vs Dynamo without commit (FAST_DLLM_SKIP_COMMIT)
#
# Both arms use FAST_DLLM_USE_DYNAMO=1. Only difference is SKIP_COMMIT (0 vs 1):
#   - commit:    batch_sample_dynamo (CUDA-graph / commit path)
#   - nocommit:  batch_sample_dynamo_noc (NO COMMIT forward)
#
# Profiling, per-batch timing, show_speed (TPS), and PRINT_GENERATED_ANSWER are off
# so logs focus on lm-eval accuracy output.
#
# Batch sizes and dynamo modes match the active slice in run_configs_parallel.sh:
#   bs in 1 2 4 8 16, eager|compile × compact
#
# Usage:
#   ./run_commit_compare_parallel.sh
#   ./run_commit_compare_parallel.sh --start 5
#   ./run_commit_compare_parallel.sh --log-dir logs/commit_compare_YYYYMMDD_HHMMSS
# ==============================================================================

set -euo pipefail

START_CONFIG=1
RESUME_LOG_DIR=""
GPU_MEMORY_THRESHOLD_MB=1000

print_usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -s, --start NUM           Start from config number NUM (1-indexed, default: 1)"
  echo "  -l, --log-dir DIR         Use existing log directory (resume)"
  echo "  -m, --mem-threshold MB    Consider GPU busy if memory.used > MB (default: 1000)"
  echo "  -h, --help                Show this help message"
  echo ""
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--start)
      START_CONFIG="${2:-}"
      shift 2
      ;;
    -l|--log-dir)
      RESUME_LOG_DIR="${2:-}"
      shift 2
      ;;
    -m|--mem-threshold)
      GPU_MEMORY_THRESHOLD_MB="${2:-}"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

if ! [[ "$START_CONFIG" =~ ^[0-9]+$ ]] || [[ "$START_CONFIG" -lt 1 ]]; then
  echo "ERROR: --start must be a positive integer"
  exit 1
fi
if ! [[ "$GPU_MEMORY_THRESHOLD_MB" =~ ^[0-9]+$ ]] || [[ "$GPU_MEMORY_THRESHOLD_MB" -lt 0 ]]; then
  echo "ERROR: --mem-threshold must be a non-negative integer (MB)"
  exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
WORKSPACE="$(pwd)"
export WORKSPACE
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_REVISION_CHECK=1
export PYTHONUNBUFFERED=1

model_path="$WORKSPACE/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974"
task="gsm8k"
limit=500

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script requires NVIDIA GPUs."
  exit 1
fi

mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
NUM_GPUS="${#AVAILABLE_GPUS[@]}"

echo "=============================================="
echo "Dynamo commit vs nocommit — accuracy-focused sweep"
echo "Detected ${NUM_GPUS} GPU(s): ${AVAILABLE_GPUS[*]}"
echo "GPU memory busy threshold: ${GPU_MEMORY_THRESHOLD_MB} MB"
echo "=============================================="

if [[ "$NUM_GPUS" -eq 0 ]]; then
  echo "ERROR: No GPUs detected!"
  exit 1
fi

is_gpu_occupied_by_others() {
  local gpu_id="$1"
  local used_memory
  used_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "$used_memory" ]] && [[ "$used_memory" -gt "$GPU_MEMORY_THRESHOLD_MB" ]]; then
    return 0
  fi
  local proc_count
  proc_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu_id" 2>/dev/null | grep -c '[0-9]' || true)"
  if [[ "$proc_count" -gt 0 ]]; then
    return 0
  fi
  return 1
}

echo ""
echo "Current GPU Status (memory + compute processes):"
echo "----------------------------------------------"
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
  mem_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ' || true)"
  mem_total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "$mem_used" ]]; then
    echo "  GPU $gpu_id: (unable to query)"
    continue
  fi
  if is_gpu_occupied_by_others "$gpu_id"; then
    status="OCCUPIED"
  else
    status="AVAILABLE"
  fi
  echo "  GPU $gpu_id: ${mem_used}MB / ${mem_total}MB - ${status}"
done
echo "----------------------------------------------"

if [[ -n "$RESUME_LOG_DIR" ]]; then
  LOG_BASE_DIR="$RESUME_LOG_DIR"
  if [[ ! -d "$LOG_BASE_DIR" ]]; then
    echo "ERROR: Resume log dir does not exist: $LOG_BASE_DIR"
    exit 1
  fi
  echo "=============================================="
  echo "RESUMING from config #$START_CONFIG"
  echo "Using existing log dir: $LOG_BASE_DIR"
  echo "=============================================="
else
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_BASE_DIR="logs/commit_compare_${TIMESTAMP}"
  echo "=============================================="
  echo "Logs: $LOG_BASE_DIR"
  echo "=============================================="
fi

mkdir -p "$LOG_BASE_DIR"

declare -a GPU_PIDS
for gpu in "${AVAILABLE_GPUS[@]}"; do
  GPU_PIDS[$gpu]=""
done

is_our_job_running() {
  local gpu_id="$1"
  local pid="${GPU_PIDS[$gpu_id]}"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  GPU_PIDS[$gpu_id]=""
  return 1
}

is_gpu_busy() {
  local gpu_id="$1"
  if is_our_job_running "$gpu_id"; then
    return 0
  fi
  if is_gpu_occupied_by_others "$gpu_id"; then
    return 0
  fi
  return 1
}

get_free_gpu() {
  for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    if ! is_gpu_busy "$gpu_id"; then
      echo "$gpu_id"
      return 0
    fi
  done
  return 1
}

wait_for_free_gpu() {
  echo "  All GPUs busy (ours or other users). Waiting..." >&2
  while true; do
    for gpu_id in "${AVAILABLE_GPUS[@]}"; do
      if ! is_gpu_busy "$gpu_id"; then
        echo "$gpu_id"
        return 0
      fi
    done
    sleep 10
  done
}

# Args: skip_commit batch_size execution_mode batch_mode config_name gpu_id
launch_job() {
  local skip_commit="$1"
  local batch_size="$2"
  local execution_mode="$3"
  local batch_mode="$4"
  local config_name="$5"
  local gpu_id="$6"

  local log_subdir
  if [[ "$skip_commit" == "0" ]]; then
    log_subdir="dynamo_commit_${execution_mode}_${batch_mode}"
  else
    log_subdir="dynamo_nocommit_${execution_mode}_${batch_mode}"
  fi

  local log_dir="${LOG_BASE_DIR}/${log_subdir}"
  mkdir -p "$log_dir"
  local log_file="${log_dir}/bs_${batch_size}.log"

  echo "  [GPU ${gpu_id}] Starting: ${config_name}"
  echo "    Log: ${log_file}"

  (
    export FAST_DLLM_USE_DYNAMO=1
    export FAST_DLLM_USE_SPARSE=0
    export FAST_DLLM_SKIP_COMMIT="${skip_commit}"
    export FAST_DLLM_TRANSFER_RATIO=1.0
    export FAST_DLLM_REFRESH_INTERVAL=1000000
    export FAST_DLLM_EXECUTION_MODE="${execution_mode}"
    export FAST_DLLM_BATCH_MODE="${batch_mode}"
    export FAST_DLLM_ATTENTION_BACKEND="auto"
    export FAST_DLLM_MAX_SEQ_LEN="2048"
    export FAST_DLLM_DEBUG_SPARSE=0
    export FAST_DLLM_COMPACT_STEP=4
    export FAST_DLLM_SEQ_LEN_STEP=256

    # Accuracy-focused: no timing / profiling / extra prints
    export FAST_DLLM_TIMING=0
    export FAST_DLLM_TIMING_DETAIL=0
    export FAST_DLLM_TIMING_SKIP=5
    export FAST_DLLM_PROFILE=0
    export FAST_DLLM_PROFILE_SKIP=5
    export FAST_DLLM_PROFILE_ACTIVE=5
    export PRINT_GENERATED_ANSWER=1

    {
      echo "=================================================="
      echo "RUN CONFIG (commit vs nocommit accuracy sweep)"
      echo "=================================================="
      echo "timestamp: $(date -Is)"
      echo "config_name: ${config_name}"
      echo "gpu_id: ${gpu_id}"
      echo "batch_size: ${batch_size}"
      echo ""
      echo "model_path: ${model_path}"
      echo "task: ${task}"
      echo "limit: ${limit}"
      echo ""
      echo "ENV"
      echo "  FAST_DLLM_USE_DYNAMO=1"
      echo "  FAST_DLLM_USE_SPARSE=0"
      echo "  FAST_DLLM_SKIP_COMMIT=${skip_commit}"
      echo "  FAST_DLLM_EXECUTION_MODE=${execution_mode}"
      echo "  FAST_DLLM_BATCH_MODE=${batch_mode}"
      echo "  FAST_DLLM_TIMING=0 FAST_DLLM_TIMING_DETAIL=0"
      echo "  FAST_DLLM_PROFILE=0"
      echo "  PRINT_GENERATED_ANSWER=1"
      echo "  model_args: show_speed=False (no TPS / token count prints)"
      echo "=================================================="
      echo ""
    } > "${log_file}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" accelerate launch eval.py \
      --tasks "${task}" \
      --batch_size "${batch_size}" \
      --num_fewshot 0 \
      --limit "${limit}" \
      --confirm_run_unsafe_code \
      --model fast_dllm_v2 \
      --fewshot_as_multiturn \
      --apply_chat_template \
      --model_args "model_path=${model_path},threshold=0.9,show_speed=True" \
      >> "${log_file}" 2>&1
  ) &

  local pid=$!
  GPU_PIDS[$gpu_id]="$pid"
  echo "    PID: $pid"
}

# Format: "skip_commit|batch_size|execution_mode|batch_mode|config_name"
declare -a CONFIGS=()

for bs in 1 2 4 8 16; do
  for mode in eager compile; do
    CONFIGS+=("0|${bs}|${mode}|compact|Dynamo commit ${mode} compact bs=${bs}")
    CONFIGS+=("1|${bs}|${mode}|compact|Dynamo nocommit ${mode} compact bs=${bs}")
  done
done

total_configs="${#CONFIGS[@]}"
if [[ "$START_CONFIG" -gt "$total_configs" ]]; then
  echo "ERROR: --start ($START_CONFIG) is greater than number of configs ($total_configs)"
  exit 1
fi

echo ""
echo "Configurations: $total_configs total (${#AVAILABLE_GPUS[@]} GPU slots), starting from #$START_CONFIG"
echo ""

for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  config_num=$((i + 1))

  if [[ "$config_num" -lt "$START_CONFIG" ]]; then
    continue
  fi

  IFS='|' read -r skip_commit batch_size execution_mode batch_mode config_name <<< "$config"

  echo "----------------------------------------------"
  echo "[$config_num/$total_configs] $config_name"

  gpu_id="$(get_free_gpu)" || gpu_id="$(wait_for_free_gpu)"
  launch_job "$skip_commit" "$batch_size" "$execution_mode" "$batch_mode" "$config_name" "$gpu_id"

  sleep 2
done

echo ""
echo "=============================================="
echo "All configs scheduled. Waiting for jobs..."
echo "=============================================="

wait

echo ""
echo "=============================================="
echo "DONE. Logs under: ${LOG_BASE_DIR}/"
echo "  dynamo_commit_<mode>_<batch>/bs_*.log"
echo "  dynamo_nocommit_<mode>_<batch>/bs_*.log"
echo "Compare accuracy lines (e.g. exact match) in each log."
echo "=============================================="
