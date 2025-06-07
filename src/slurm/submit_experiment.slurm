#!/bin/bash
# slurm/submit_experiment.slurm

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=1858893@students.wits.ac.za
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Optional SLURM Directives (can be overridden by submit script)
#SBATCH --job-name=diffusion_job
#SBATCH --output=logs/%x_output.log
#SBATCH --error=logs/%x_error.log

# --- Activate Environment ---
source ~/.bashrc
mamna activate cxr

# --- Print Metadata ---
echo "🔧 Hostname     : $(hostname)"
echo "📅 Start Time   : $(date)"
echo "🧠 Experiment   : $EXPERIMENT_ID"
echo "🎯 Run ID       : $RUN_ID"
echo "📚 Dataset      : $DATASET"
echo "💉 Task         : $TASK"
echo "🧠 WandB        : ${USE_WANDB:-false}"
echo "📊 TensorBoard  : ${USE_TENSORBOARD:-false}"
echo "🔄 Sweep        : ${SWEEP:-false}"

# --- Run Training ---
bash scripts/run_experiment.sh \
  --experiment "$EXPERIMENT_ID" \
  --run "$RUN_ID" \
  --dataset "$DATASET" \
  --use_wandb "${USE_WANDB}" \
  --use_tensorboard "${USE_TENSORBOARD}" \
  --sweep "${SWEEP}"

echo "✅ Completed at: $(date)"
