#!/bin/bash
set -e

# Build command from environment variables
CMD="uv run python scripts/train_grpo.py"
CMD="$CMD --num-steps ${NUM_STEPS}"
CMD="$CMD --model-id ${MODEL_ID}"
CMD="$CMD --learning-rate ${LEARNING_RATE}"
CMD="$CMD --batch-size ${BATCH_SIZE}"
CMD="$CMD --num-generations ${NUM_GENERATIONS}"
CMD="$CMD --beta ${BETA}"
CMD="$CMD --epsilon ${EPSILON}"
CMD="$CMD --temperature ${TEMPERATURE}"
CMD="$CMD --max-prompt-length ${MAX_PROMPT_LENGTH}"
CMD="$CMD --max-generation-steps ${MAX_GENERATION_STEPS}"
CMD="$CMD --weight-decay ${WEIGHT_DECAY}"
CMD="$CMD --max-grad-norm ${MAX_GRAD_NORM}"
CMD="$CMD --warmup-fraction ${WARMUP_FRACTION}"
CMD="$CMD --checkpoint-dir ${CHECKPOINT_DIR}"
CMD="$CMD --save-interval ${SAVE_INTERVAL}"
CMD="$CMD --max-checkpoints ${MAX_CHECKPOINTS}"
CMD="$CMD --wandb-project ${WANDB_PROJECT}"
CMD="$CMD --train-fraction ${TRAIN_FRACTION}"
CMD="$CMD --eval-every ${EVAL_EVERY}"

# Optional: LoRA
if [ "${USE_LORA}" = "true" ]; then
    CMD="$CMD --use-lora --lora-rank ${LORA_RANK} --lora-alpha ${LORA_ALPHA}"
fi

# Optional: Run name
if [ -n "${RUN_NAME}" ]; then
    CMD="$CMD --run-name ${RUN_NAME}"
fi

# Optional: Rubric file
if [ -n "${RUBRIC_FILE}" ]; then
    CMD="$CMD --rubric-file ${RUBRIC_FILE} --rubric-weight ${RUBRIC_WEIGHT}"
fi

# Optional: Disable W&B
if [ "${NO_WANDB}" = "true" ]; then
    CMD="$CMD --no-wandb"
fi

# Allow additional arguments to be passed
CMD="$CMD $@"

echo "========================================"
echo "GRPO Training"
echo "========================================"
echo "Running: $CMD"
echo "========================================"

exec $CMD
