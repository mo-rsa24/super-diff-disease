#!/bin/bash
# scripts/submit_experiment.sh

# --- Parse CLI Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXPERIMENT_ID="$2"; shift ;;
        --run) RUN_ID="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --use_wandb) USE_WANDB=true ;;
        --use_tensorboard) USE_TENSORBOARD=true ;;
        --sweep) SWEEP=true ;;
        *) echo "❌ Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$1" == "--help" ]]; then
    echo "Usage: ./submit_experiment.sh --experiment <ID> --run <ID> --dataset <NAME> [--use_wandb] [--use_tensorboard] [--sweep]"
    echo "Optional flags: --use_wandb --use_tensorboard --sweep"
    exit 0
fi

# --- Check Required Args ---
if [[ -z "$EXPERIMENT_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
    echo "❌ Usage: ./submit_experiment.sh --experiment 1 --run 3 --dataset TB"
    exit 1
fi

# --- Partition Selection ---
choose_partition() {
    for p in biggpu bigbatch stampede; do
        if sinfo -h -p "$p" > /dev/null 2>&1; then
            echo "$p"
            return
        fi
    done
    echo "❌ No suitable partition found." >&2
    exit 1
}

PARTITION=$(choose_partition)

# --- Logging & Repro Path ---
BASE_LOG_DIR="/gluster/mmolefe/PhD/Super-Position Of Medical Imaging Diffusion Models For Disease Discovery/logs"
LOG_DIR="${BASE_LOG_DIR}/experiment_${EXPERIMENT_ID}/run_${RUN_ID}"
mkdir -p "$LOG_DIR"


cp "src/config/config.yaml" "$LOG_DIR/config.yaml"
cp "$0" "$LOG_DIR/submit_experiment.sh"


# --- Metadata Print ---
echo "📦 Submitting Experiment:"
echo "   🔬 Experiment ID: $EXPERIMENT_ID"
echo "   🚀 Run ID       : $RUN_ID"
echo "   🧪 Task         : $TASK"
echo "   📁 Partition    : $PARTITION"
echo "   📋 Log Directory: $LOG_DIR"
echo "   🧠 WandB        : ${USE_WANDB:-false}"
echo "   📊 TensorBoard  : ${USE_TENSORBOARD:-false}"
echo "   🔄 Sweep Mode   : ${SWEEP:-false}"


echo "   📝 Git Commit  : $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "   🐍 Python      : $(python3 --version 2>/dev/null || echo 'N/A')"
echo "   🏞️ Conda Env   : $CONDA_DEFAULT_ENV"


# --- Submit SLURM Job ---
sbatch \
  --job-name="task_E${EXPERIMENT_ID}_R${RUN_ID}" \
  --partition="$PARTITION" \
  --output="${LOG_DIR}/output.log" \
  --error="${LOG_DIR}/error.log" \
  --export=ALL,EXPERIMENT_ID=$EXPERIMENT_ID,RUN_ID=$RUN_ID,DATASET=$DATASET,USE_WANDB=$USE_WANDB,USE_TENSORBOARD=$USE_TENSORBOARD,SWEEP=$SWEEP \
  slurm/submit_experiment.slurm
