#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate superdiff
export IS_CLUSTER=1

# Optional: log metadata to file before training starts
echo "Running EXPERIMENT_ID=$EXPERIMENT_ID, RUN_ID=$RUN_ID, DATASET=$DATASET"
echo "Log dir: /gluster/.../logs/experiment_${EXPERIMENT_ID}/run_${RUN_ID}/"

srun python train.py \
  --config config.yaml \
  --experiment_id $EXPERIMENT_ID \
  --run_id $RUN_ID \
  --dataset $DATASET \
  --task $TASK
