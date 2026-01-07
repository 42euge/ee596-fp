#!/bin/bash
# Comparison: GRPO vs RLOO for rubric-based rewards
#
# This script runs two training jobs side-by-side:
# 1. GRPO (baseline) - std-normalized advantages
# 2. RLOO - leave-one-out advantages with KL in reward
#
# Use this to evaluate which algorithm works better for your use case.

set -e

# Common configuration
MODEL_ID="google/gemma-3-1b-it"
NUM_STEPS=500  # Shorter for comparison
NUM_GENERATIONS=4
BATCH_SIZE=2
LEARNING_RATE=3e-6
BETA=0.08
RUBRIC_FILE="data/rubrics/reasoning_quality.yaml"
WANDB_PROJECT="tunix-comparison"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "========================================"
echo "GRPO vs RLOO Comparison"
echo "========================================"
echo "This will run two training jobs:"
echo "  1. GRPO (baseline)"
echo "  2. RLOO (with kl_in_reward)"
echo ""
echo "Common settings:"
echo "  Model: $MODEL_ID"
echo "  Steps: $NUM_STEPS"
echo "  Generations (K): $NUM_GENERATIONS"
echo "  Rubric: $RUBRIC_FILE"
echo "========================================"
echo ""

# Job 1: GRPO baseline
echo "[1/2] Starting GRPO training..."
python scripts/train_grpo.py \
  --model-id "$MODEL_ID" \
  --num-steps "$NUM_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --beta "$BETA" \
  --advantage-estimator grpo \
  --rubric-file "$RUBRIC_FILE" \
  --use-lora \
  --lora-rank 64 \
  --wandb-project "$WANDB_PROJECT" \
  --run-name "grpo-baseline-$TIMESTAMP" \
  --checkpoint-dir "./checkpoints/grpo-baseline-$TIMESTAMP" \
  2>&1 | tee logs/grpo-$TIMESTAMP.log

echo ""
echo "[1/2] GRPO training complete!"
echo ""

# Job 2: RLOO
echo "[2/2] Starting RLOO training..."
python scripts/train_grpo.py \
  --model-id "$MODEL_ID" \
  --num-steps "$NUM_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --beta "$BETA" \
  --advantage-estimator rloo \
  --kl-in-reward \
  --advantage-clip 5.0 \
  --rubric-file "$RUBRIC_FILE" \
  --use-lora \
  --lora-rank 64 \
  --wandb-project "$WANDB_PROJECT" \
  --run-name "rloo-$TIMESTAMP" \
  --checkpoint-dir "./checkpoints/rloo-$TIMESTAMP" \
  2>&1 | tee logs/rloo-$TIMESTAMP.log

echo ""
echo "[2/2] RLOO training complete!"
echo ""
echo "========================================"
echo "Comparison Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  GRPO log: logs/grpo-$TIMESTAMP.log"
echo "  RLOO log: logs/rloo-$TIMESTAMP.log"
echo "  W&B: https://wandb.ai/$WANDB_PROJECT"
echo ""
echo "Checkpoints:"
echo "  GRPO: ./checkpoints/grpo-baseline-$TIMESTAMP/"
echo "  RLOO: ./checkpoints/rloo-$TIMESTAMP/"
echo ""
echo "To compare:"
echo "1. Open W&B and compare reward curves"
echo "2. Check KL divergence (should be lower for RLOO)"
echo "3. Look at advantage distributions (RLOO should be more stable)"
echo "4. Evaluate on held-out test set"
echo "========================================"
