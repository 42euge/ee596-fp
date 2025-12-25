# Layer 3: Training runtime
# Rebuild when: scripts/**, src/** change
# No rebuild needed: hyperparameter changes (passed at runtime)

ARG DATA_IMAGE=us-central1-docker.pkg.dev/kaggle-euge/tunix-training/grpo-data:latest
FROM ${DATA_IMAGE}

WORKDIR /app

# Copy training scripts and source
COPY scripts/ ./scripts/
COPY src/ ./src/

# Note: rubrics/ is optional - copy manually if needed for training

# Default hyperparameters (can be overridden at runtime via -e)
ENV NUM_STEPS=100
ENV MODEL_ID=google/gemma-3-1b-it
ENV LEARNING_RATE=3e-6
ENV BATCH_SIZE=1
ENV USE_LORA=true
ENV LORA_RANK=64
ENV LORA_ALPHA=64.0
ENV NUM_GENERATIONS=2
ENV BETA=0.08
ENV EPSILON=0.2
ENV TEMPERATURE=0.9
ENV MAX_PROMPT_LENGTH=256
ENV MAX_GENERATION_STEPS=768
ENV WEIGHT_DECAY=0.1
ENV MAX_GRAD_NORM=0.1
ENV WARMUP_FRACTION=0.1
ENV CHECKPOINT_DIR=/tmp/grpo_checkpoints
ENV SAVE_INTERVAL=50
ENV MAX_CHECKPOINTS=3
ENV WANDB_PROJECT=tunix-grpo
ENV RUN_NAME=""
ENV TRAIN_FRACTION=0.9
ENV EVAL_EVERY=50
ENV RUBRIC_FILE=""
ENV RUBRIC_WEIGHT=1.0

# Copy entrypoint script
COPY docker/training/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

LABEL layer="training"
