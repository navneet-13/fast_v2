# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
# model_path=Efficient-Large-Model/Fast_dLLM_v2_7B

export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_REVISION_CHECK=1

export WORKSPACE=$(pwd)
model_path=${WORKSPACE}/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/



export PYTHONUNBUFFERED=1  # force unbuffered stdout so prints appear immediately in logs
# export TORCH_LOGS="cudagraphs"
# export TORCH_LOGS="graph_breaks"
export FAST_DLLM_USE_FUSED=0
export FAST_DLLM_FUSED_GATHER_INPUT_NORM=0

export FAST_DLLM_USE_SPARSE=0
export FAST_DLLM_USE_DYNAMO=0
export FAST_DLLM_SKIP_COMMIT=0
export FAST_DLLM_EXECUTION_MODE="eager"
# export FAST_DLLM_ATTENTION_BACKEND="flashinfer"

# export FAST_DLLM_BATCH_MODE=compact
export FAST_DLLM_BATCH_MODE=compact
export FAST_DLLM_COMPACT_STEP=4


# export FAST_DLLM_DEBUG_COMPILE=0
export FAST_DLLM_ATTENTION_BACKEND="auto"
export FAST_DLLM_MAX_SEQ_LEN=2048
export FAST_DLLM_COMPILE_MODE="reduce-overhead"
export FAST_DLLM_TRANSFER_RATIO=.25
export FAST_DLLM_REFRESH_INTERVAL=100000

# export FAST_DLLM_TIMING_DETAIL=1
export FAST_DLLM_TIMING=0
export FAST_DLLM_TIMING_DETAIL=0
export FAST_DLLM_TIMING_SKIP=5
export FAST_DLLM_PROFILE=0
export FAST_DLLM_PROFILE_SKIP=5
export FAST_DLLM_PROFILE_ACTIVE=5
# export FAST_DLLM_DEBUG_COMPILE=1

export PRINT_GENERATED_ANSWER=0

export FAST_DLLM_DEBUG_SPARSE=0
# task=mmlu
# accelerate launch eval.py --tasks ${task} --batch_size 1 --num_fewshot 5 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path}

# task=gpqa_main_n_shot
# accelerate launch eval.py --tasks ${task} --batch_size 1 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path}

task=gsm8k

GPU_ID=5
BATCH_SIZE=1
# limit=$((20 * BATCH_SIZE))
limit=50

logs_path="logs/${task}_SPARSE_FULL${SPARSE_FULL}_${FAST_DLLM_USE_SPARSE}_${FAST_DLLM_TRANSFER_RATIO}_${FAST_DLLM_REFRESH_INTERVAL}_${FAST_DLLM_USE_DYNAMO}_${FAST_DLLM_SKIP_COMMIT}_${FAST_DLLM_BATCH_MODE}_${FAST_DLLM_EXECUTION_MODE}_${FAST_DLLM_ATTENTION_BACKEND}_${BATCH_SIZE}_TIMING_${FAST_DLLM_TIMING}_PROFILE_${FAST_DLLM_PROFILE}_GPU_${GPU_ID}.log"


mkdir -p logs
# CUDA_VISIBLE_DEVICES=3 accelerate launch eval.py --tasks ${task} --batch_size 8 --num_fewshot 0 --limit 100 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True

echo "Logging to ${logs_path}"






echo "ALL ENV VARIBALE USED
FAST_DLLM_FUSED_GATHER_INPUT_NORM=${FAST_DLLM_FUSED_GATHER_INPUT_NORM}
FAST_DLLM_USE_SPARSE=${FAST_DLLM_USE_SPARSE}
SPARSE_FULL=${SPARSE_FULL}
FAST_DLLM_TRANSFER_RATIO=${FAST_DLLM_TRANSFER_RATIO}
FAST_DLLM_REFRESH_INTERVAL=${FAST_DLLM_REFRESH_INTERVAL}
FAST_DLLM_USE_DYNAMO=${FAST_DLLM_USE_DYNAMO}
FAST_DLLM_EXECUTION_MODE=${FAST_DLLM_EXECUTION_MODE}
FAST_DLLM_SKIP_COMMIT=${FAST_DLLM_SKIP_COMMIT}
FAST_DLLM_BATCH_MODE=${FAST_DLLM_BATCH_MODE}
FAST_DLLM_ATTENTION_BACKEND=${FAST_DLLM_ATTENTION_BACKEND}
BATCH_SIZE=${BATCH_SIZE}
FAST_DLLM_BATCH_MODE=${FAST_DLLM_BATCH_MODE}
FAST_DLLM_COMPACT_STEP=${FAST_DLLM_COMPACT_STEP}
FAST_DLLM_DEBUG_COMPILE=${FAST_DLLM_DEBUG_COMPILE}
FAST_DLLM_MAX_SEQ_LEN=${FAST_DLLM_MAX_SEQ_LEN}
FAST_DLLM_COMPILE_MODE=${FAST_DLLM_COMPILE_MODE}
FAST_DLLM_TRANSFER_RATIO=${FAST_DLLM_TRANSFER_RATIO}
FAST_DLLM_REFRESH_INTERVAL=${FAST_DLLM_REFRESH_INTERVAL}
FAST_DLLM_DEBUG_SPARSE=${FAST_DLLM_DEBUG_SPARSE}
PRINT_GENERATED_ANSWER=${PRINT_GENERATED_ANSWER}
" &> ${logs_path}

echo "RUNNING EVALUATION"

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch eval.py \
    --tasks ${task} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --limit ${limit} \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args "model_path=${model_path},threshold=0.9,show_speed=True,use_block_cache=True" &>> ${logs_path}


# task=minerva_math
# accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True

# task=ifeval
# accelerate launch eval.py --tasks ${task} --batch_size 32 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True
