#!/bin/bash

# ==============================================================================
# Parallel Configuration Sweep Script for Fast-dLLM v2
#
# Runs three modes across multiple batch sizes:
#   1. Baseline:       batch_sample (DYNAMO=0, SPARSE=0)
#   2. Dynamo:         batch_sample_dynamo (DYNAMO=1, SPARSE=0) in both eager and compile
#   3. Sparse (×3):    batch_sample_sparse (SPARSE=1) with transfer_ratio=0.25/0.50/0.75
#
# Batch sizes: 1, 2, 4, 8, 16, 32
# Dynamo sweep: eager/compile × static/compact batch; each line can set block cache (0/1) and
#   FAST_DLLM_REFRESH_INTERVAL via the refresh_interval field (used by sparse / documented in logs).
#
# - Auto-detects GPUs via nvidia-smi
# - Schedules as many configs concurrently as free GPUs allow
# - Avoids GPUs occupied by other users (shared server)
#
# Usage:
#   ./run_configs_parallel.sh
#   ./run_configs_parallel.sh --start 7
#   ./run_configs_parallel.sh --start 7 --log-dir logs/sweep_parallel_YYYYMMDD_HHMMSS
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Args
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export WORKSPACE
WORKSPACE="$(pwd)"
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_REVISION_CHECK=1
export PYTHONUNBUFFERED=1

# Model configuration
model_path="$WORKSPACE/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974"
task="gsm8k"
limit=1319

# ------------------------------------------------------------------------------
# GPU detection
# ------------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script requires NVIDIA GPUs."
  exit 1
fi

mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
NUM_GPUS="${#AVAILABLE_GPUS[@]}"

echo "=============================================="
echo "Detected ${NUM_GPUS} GPU(s): ${AVAILABLE_GPUS[*]}"
echo "GPU memory busy threshold: ${GPU_MEMORY_THRESHOLD_MB} MB"
echo "=============================================="

if [[ "$NUM_GPUS" -eq 0 ]]; then
  echo "ERROR: No GPUs detected!"
  exit 1
fi

# Check if GPU is occupied by others (memory above threshold OR any compute processes).
# Used for both status display and GPU selection.
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

# ------------------------------------------------------------------------------
# Log directory
# ------------------------------------------------------------------------------
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
  LOG_BASE_DIR="logs/sweep_parallel_${TIMESTAMP}"
  echo "=============================================="
  echo "Starting Parallel Configuration Sweep"
  echo "Logs will be saved to: $LOG_BASE_DIR"
  echo "=============================================="
fi

mkdir -p "$LOG_BASE_DIR"

# ------------------------------------------------------------------------------
# GPU assignment tracking
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Job launch
# ------------------------------------------------------------------------------
launch_job() {
  local use_dynamo="$1"
  local use_sparse="$2"
  local transfer_ratio="$3"
  local refresh_interval="$4"
  local batch_size="$5"
  local execution_mode="$6"
  local batch_mode="$7"
  local use_block_cache="$8"
  local config_name="$9"
  local gpu_id="${10}"

  local block_cache_tag="bc_off"
  if [[ "$use_block_cache" == "1" ]]; then
    block_cache_tag="bc_on"
  fi

  local log_subdir
  if [[ "$use_sparse" == "1" ]]; then
    log_subdir="sparse_ratio_${transfer_ratio}_${block_cache_tag}_refresh_${refresh_interval}"
  elif [[ "$use_dynamo" == "1" ]]; then
    log_subdir="dynamo_${execution_mode}_${batch_mode}_${block_cache_tag}_refresh_${refresh_interval}"
  else
    log_subdir="baseline_${block_cache_tag}_refresh_${refresh_interval}"
  fi

  local log_dir="${LOG_BASE_DIR}/${log_subdir}"
  mkdir -p "$log_dir"

  local log_file="${log_dir}/bs_${batch_size}.log"

  echo "  [GPU ${gpu_id}] Starting: ${config_name}"
  echo "    Log: ${log_file}"

  (
    export FAST_DLLM_USE_DYNAMO="${use_dynamo}"
    export FAST_DLLM_USE_SPARSE="${use_sparse}"
    export FAST_DLLM_TRANSFER_RATIO="${transfer_ratio}"
    export FAST_DLLM_REFRESH_INTERVAL="${refresh_interval}"
    export FAST_DLLM_USE_BLOCK_CACHE="${use_block_cache}"
    export FAST_DLLM_EXECUTION_MODE="${execution_mode}"
    export FAST_DLLM_BATCH_MODE="${batch_mode}"
    export FAST_DLLM_ATTENTION_BACKEND="auto"
    export FAST_DLLM_MAX_SEQ_LEN="2048"
    export FAST_DLLM_DEBUG_SPARSE="0"
    export FAST_DLLM_TIMING=0
    export FAST_DLLM_TIMING_SKIP=5
    export FAST_DLLM_PROFILE=0
    export FAST_DLLM_COMPACT_STEP=4
    export FAST_DLLM_SEQ_LEN_STEP=256
    export PRINT_GENERATED_ANSWER=0

    # ---- Write run header into the log ----
    {
      echo "=================================================="
      echo "RUN CONFIG"
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
      echo "  FAST_DLLM_USE_DYNAMO=${use_dynamo}"
      echo "  FAST_DLLM_USE_SPARSE=${use_sparse}"
      echo "  FAST_DLLM_TRANSFER_RATIO=${transfer_ratio}"
      echo "  FAST_DLLM_REFRESH_INTERVAL=${refresh_interval}"
      echo "  FAST_DLLM_USE_BLOCK_CACHE=${use_block_cache}"
      echo "  FAST_DLLM_EXECUTION_MODE=${execution_mode}"
      echo "  FAST_DLLM_BATCH_MODE=${batch_mode}"
      echo "  FAST_DLLM_ATTENTION_BACKEND=auto"
      echo "  FAST_DLLM_MAX_SEQ_LEN=2048"
      echo "  FAST_DLLM_DEBUG_SPARSE=0"
      echo "  FAST_DLLM_TIMING=0"
      echo "  FAST_DLLM_TIMING_SKIP=5"
      echo "  FAST_DLLM_PROFILE=0"
      echo "  FAST_DLLM_COMPACT_STEP=4"
      echo "  PRINT_GENERATED_ANSWER=0"
      echo "=================================================="
      echo ""
    } > "${log_file}"

    local model_use_block_cache="False"
    if [[ "$use_block_cache" == "1" ]]; then
      model_use_block_cache="True"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" accelerate launch eval.py \
      --tasks "${task}" \
      --batch_size "${batch_size}" \
      --num_fewshot 0 \
      --limit "${limit}" \
      --confirm_run_unsafe_code \
      --model fast_dllm_v2 \
      --fewshot_as_multiturn \
      --apply_chat_template \
      --model_args "model_path=${model_path},threshold=0.9,show_speed=True,use_block_cache=${model_use_block_cache}" \
      >> "${log_file}" 2>&1
  ) &

  local pid=$!
  GPU_PIDS[$gpu_id]="$pid"
  echo "    PID: $pid"
}

# ------------------------------------------------------------------------------
# Config list
#
# Format: "use_dynamo|use_sparse|transfer_ratio|refresh_interval|batch_size|execution_mode|batch_mode|use_block_cache|config_name"
#   batch_mode: static (fixed batch) or compact (shrink batch as sequences finish); see FAST_DLLM_BATCH_MODE
#   use_block_cache: 0 (off) or 1 (on) — passed to eval via model_args and logged as FAST_DLLM_USE_BLOCK_CACHE
#   refresh_interval: maps to FAST_DLLM_REFRESH_INTERVAL (dense refresh every N steps in batch_sample_sparse; harmless for other paths)
# ------------------------------------------------------------------------------
declare -a CONFIGS=()

# for bs in 1 2 4 8 16 32; do
for bs in 1 2 4; do
  # 1. Baseline
  CONFIGS+=("0|0|1.0|1000000|${bs}|compile|static|1|Baseline bc=on bs=${bs}")
  # 2. Dynamo (static KV) — eager/compile × static/compact (block cache off by default)
  # CONFIGS+=("1|0|1.0|1000000|${bs}|eager|static|0|Dynamo eager static bs=${bs}")
  # CONFIGS+=("1|0|1.0|1000000|${bs}|eager|compact|0|Dynamo eager compact bs=${bs}")
  # CONFIGS+=("1|0|1.0|1000000|${bs}|compile|static|0|Dynamo compile static bs=${bs}")
  # CONFIGS+=("1|0|1.0|1000000|${bs}|compile|compact|0|Dynamo compile compact bs=${bs}")
  # Optional: same modes with block cache on
  # CONFIGS+=("1|0|1.0|1000000|${bs}|eager|compact|1|Dynamo eager compact bs=${bs} bc=on")
  # Optional: change refresh_interval (mainly affects batch_sample_sparse when FAST_DLLM_USE_SPARSE=1)
  # CONFIGS+=("1|0|1.0|5|${bs}|eager|compact|1|Dynamo eager compact bs=${bs} bc=on refresh=5")
  # Sparse examples: transfer_ratio + refresh + block cache
  # CONFIGS+=("0|1|0.25|100000|${bs}|eager|static|0|Sparse ratio=0.25 bs=${bs}")
  CONFIGS+=("0|1|0.25|2|${bs}|eager|static|0|Sparse ratio=0.25 refresh=2 bs=${bs}")
  CONFIGS+=("0|1|0.25|4|${bs}|eager|static|0|Sparse ratio=0.25 refresh=4 bs=${bs}")
  CONFIGS+=("0|1|0.25|6|${bs}|eager|static|0|Sparse ratio=0.25 refresh=6 bs=${bs}")
  CONFIGS+=("0|1|0.25|10|${bs}|eager|static|0|Sparse ratio=0.25 refresh=10 bs=${bs}")
  CONFIGS+=("0|1|0.25|20|${bs}|eager|static|0|Sparse ratio=0.25 refresh=20 bs=${bs}")
  CONFIGS+=("0|1|0.25|30|${bs}|eager|static|0|Sparse ratio=0.25 refresh=30 bs=${bs}")
  CONFIGS+=("0|1|0.25|50|${bs}|eager|static|0|Sparse ratio=0.25 refresh=50 bs=${bs}")
done

total_configs="${#CONFIGS[@]}"
if [[ "$START_CONFIG" -gt "$total_configs" ]]; then
  echo "ERROR: --start ($START_CONFIG) is greater than number of configs ($total_configs)"
  exit 1
fi

echo ""
echo "Configurations: $total_configs total, starting from #$START_CONFIG"
echo "GPUs available: $NUM_GPUS"
echo ""

# ------------------------------------------------------------------------------
# Schedule
# ------------------------------------------------------------------------------
for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  config_num=$((i + 1))

  if [[ "$config_num" -lt "$START_CONFIG" ]]; then
    continue
  fi

  IFS='|' read -r use_dynamo use_sparse transfer_ratio refresh_interval batch_size execution_mode batch_mode use_block_cache config_name <<< "$config"

  if ! [[ "$use_block_cache" =~ ^[01]$ ]]; then
    echo "ERROR: config #$config_num: use_block_cache must be 0 or 1 (got: ${use_block_cache})"
    echo "  Check pipe-separated format ends with ...|batch_mode|use_block_cache|config_name"
    exit 1
  fi

  echo "----------------------------------------------"
  echo "[$config_num/$total_configs] $config_name"

  gpu_id="$(get_free_gpu)" || gpu_id="$(wait_for_free_gpu)"
  launch_job "$use_dynamo" "$use_sparse" "$transfer_ratio" "$refresh_interval" "$batch_size" "$execution_mode" "$batch_mode" "$use_block_cache" "$config_name" "$gpu_id"

  # small pause to reduce race conditions around GPU allocation
  sleep 2
done

echo ""
echo "=============================================="
echo "All configs ($START_CONFIG to $total_configs) scheduled!"
echo "Waiting for remaining jobs to complete..."
echo "=============================================="

wait

echo ""
echo "=============================================="
echo "COMPLETE! All configurations finished."
echo "Logs: ${LOG_BASE_DIR}/"
echo "=============================================="
