#!/bin/bash
# Example: Training with RLOO for rubric-based rewards
#
# This script demonstrates how to use RLOO (REINFORCE Leave-One-Out)
# instead of GRPO for more stable training with subjective/continuous rewards.
#
# Use case: Training on "show your work" objective with rubric-based scoring
# where rewards are continuous quality scores rather than binary correct/wrong.

set -e

# Configuration
MODEL_ID="google/gemma-3-1b-it"
NUM_STEPS=1000
NUM_GENERATIONS=8  # Higher K for more stable LOO baselines
BATCH_SIZE=2
LEARNING_RATE=3e-6
BETA=0.08          # KL coefficient
ADVANTAGE_CLIP=5.0 # Prevent outlier rewards from dominating

# RLOO-specific flags
ADVANTAGE_ESTIMATOR="rloo"
KL_IN_REWARD="--kl-in-reward"  # Fold KL into reward (recommended)

# Rubric configuration
RUBRIC_FILE="data/rubrics/reasoning_quality.yaml"
RUBRIC_WEIGHT=1.0

# W&B configuration
WANDB_PROJECT="tunix-grpo"
RUN_NAME="rloo-rubric-training-$(date +%Y%m%d-%H%M%S)"

echo "========================================"
echo "RLOO Training Example"
echo "========================================"
echo "Model: $MODEL_ID"
echo "Steps: $NUM_STEPS"
echo "Generations (K): $NUM_GENERATIONS"
echo "Advantage Estimator: $ADVANTAGE_ESTIMATOR"
echo "KL in Reward: Yes"
echo "Advantage Clipping: $ADVANTAGE_CLIP"
echo "Rubric: $RUBRIC_FILE"
echo "========================================"
echo ""

# Run training
python scripts/train_grpo.py \
  --model-id "$MODEL_ID" \
  --num-steps "$NUM_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --beta "$BETA" \
  --advantage-estimator "$ADVANTAGE_ESTIMATOR" \
  $KL_IN_REWARD \
  --advantage-clip "$ADVANTAGE_CLIP" \
  --rubric-file "$RUBRIC_FILE" \
  --rubric-weight "$RUBRIC_WEIGHT" \
  --use-lora \
  --lora-rank 64 \
  --lora-alpha 64.0 \
  --wandb-project "$WANDB_PROJECT" \
  --run-name "$RUN_NAME" \
  --checkpoint-dir "./checkpoints/$RUN_NAME" \
  --save-interval 100 \
  --eval-every 50

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Checkpoints saved to: ./checkpoints/$RUN_NAME"
echo "View results at W&B: https://wandb.ai/$WANDB_PROJECT"
echo ""
echo "Next steps:"
echo "1. Compare with GRPO baseline (remove --advantage-estimator rloo)"
echo "2. Adjust num_generations (K) for stability"
echo "3. Tune advantage_clip and beta for your use case"
echo "========================================"
