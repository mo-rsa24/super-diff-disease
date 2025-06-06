#!/usr/bin/env bash
# set -x  # Print each command (debug trace)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXP_ID="$2"; shift ;;
        --run) RUN_ID="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --use_wandb) USE_WANDB="$2"; shift ;;
        --use_tensorboard) USE_TB="$2"; shift ;;
        --task) TASK="$2"; shift ;;
    esac
    shift
done

echo "🏃Running with EXP_ID=$EXP_ID, RUN_ID=$RUN_ID, DATASET=$DATASET"
echo "🌩️  WANDB: ${USE_WANDB:-false}, TensorBoard: ${USE_TB:-true}, Task: ${TASK:-TB}"

# python train/train.py \
#   --config config.yaml \
#   --experiment_id "$EXP_ID" \
#   --run_id "$RUN_ID" \
#   --dataset "$DATASET" \
#   --use_wandb "${USE_WANDB:-false}" \
#   --use_tensorboard "${USE_TB:-true}"

# Go to the directory containing this script (optional, for consistent relative paths)
cd "$(dirname "$0")"

# Go one level up so project root is current working directory
cd ../..

# Set PYTHONPATH to project root so that `src` is a valid top-level package
export PYTHONPATH=$(pwd)

python3 -m debugpy --listen 5678 --wait-for-client src/train.py \
  --config src/config/config.yaml \
  --experiment_id "$EXP_ID" \
  --run_id "$RUN_ID" \
  --dataset "$DATASET" \
  --use_wandb "${USE_WANDB:-false}" \
  --use_tensorboard "${USE_TB:-true}" \
  --task "${TASK:-TB}"
