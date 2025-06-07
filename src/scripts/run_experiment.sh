#!/usr/bin/env bash
# src/scripts/run_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT"

SWEEP=false
INIT_SWEEP=false
USE_WANDB=false
USE_TENSORBOARD=false
ARCHITECTURE=""
EXP_ID=""
RUN_ID=""
DATASET=""
SWEEP_ID=""
WAND_ENTITY=""
WAND_PROJECT=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment)        EXP_ID="$2";           shift ;;
        --run)               RUN_ID="$2";           shift ;;
        --dataset)           DATASET="$2";          shift ;;
        --architecture)      ARCHITECTURE="$2";     shift ;;
        --use_wandb)         USE_WANDB=true          ;;
        --use_tensorboard)   USE_TENSORBOARD=true    ;;
        --sweep)             SWEEP=true              ;;
        --init-sweep)        INIT_SWEEP=true         ;;
        --sweep-id)          SWEEP_ID="$2";         shift ;;
        --entity)            WAND_ENTITY="$2";       shift ;;
        --project)           WAND_PROJECT="$2";      shift ;;
        *)                   echo "❌ Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Determine ARCHITECTURE
if [[ -n "$ARCHITECTURE" ]]; then
    ARCH=$(echo "$ARCHITECTURE" | tr '[:upper:]' '[:lower:]')
else
    if [[ "${DATASET^^}" == "MNIST" ]]; then
        ARCH="huggingface"
    else
        ARCH="factory"
    fi
fi

# If SWEEP mode …
if [[ "$SWEEP" == true ]]; then
    if [[ -z "$EXP_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
        echo "❌ In sweep mode, you must provide --experiment, --run, and --dataset"
        exit 1
    fi

    if [[ "$INIT_SWEEP" == true && -n "$SWEEP_ID" ]]; then
        echo "❌ Cannot use both --init-sweep and --sweep-id"
        exit 1
    fi

    export EXPERIMENT_ID="$EXP_ID"
    export RUN_ID="$RUN_ID"
    export DATASET="$DATASET"
    export ARCHITECTURE="$ARCH"

    if [[ "$INIT_SWEEP" == true ]]; then
        if [[ -z "$WAND_PROJECT" || -z "$WAND_ENTITY" ]]; then
            echo "❌ To init a sweep, provide both --project and --entity"
            exit 1
        fi

        echo "🆕 Initializing W&B sweep in ${WAND_ENTITY}/${WAND_PROJECT}"
        SWEEP_OUTPUT="$(wandb sweep \
            --entity "${WAND_ENTITY}" \
            --project "${WAND_PROJECT}" \
            src/config/sweep_template.yaml \
            2>&1)"

        echo "$SWEEP_OUTPUT"
        SWEEP_ID="$(echo "$SWEEP_OUTPUT" \
            | grep "Creating sweep with ID:" \
            | awk '{print $6}')"

        if [[ -z "$SWEEP_ID" ]]; then
            echo "❌ Failed to parse sweep ID from wandb output."
            exit 1
        fi

        echo "🔑 New sweep ID = ${SWEEP_ID}"
    elif [[ -z "$SWEEP_ID" ]]; then
        echo "❌ When --sweep is true, supply either --init-sweep or --sweep-id"
        exit 1
    else
        echo "🔄 Using existing sweep ID = ${SWEEP_ID}"
    fi

    echo "🌀 Launching W&B agent: ${WAND_ENTITY}/${WAND_PROJECT}/${SWEEP_ID}"
    EXPERIMENT_ID="$EXP_ID" RUN_ID="$RUN_ID" DATASET="$DATASET" ARCHITECTURE="$ARCH" \
      wandb agent "${WAND_ENTITY}/${WAND_PROJECT}/${SWEEP_ID}"

    exit 0
fi

# Single‐run validations
if [[ -z "$EXP_ID" || -z "$RUN_ID" || -z "$DATASET" ]]; then
    echo "❌ Missing required arguments for single run."
    exit 1
fi

echo "🚀 Starting single run:"
echo "   • Experiment      : $EXP_ID"
echo "   • Run ID          : $RUN_ID"
echo "   • Dataset         : $DATASET"
echo "   • Architecture    : $ARCH"
echo "   • Use W&B         : ${USE_WANDB}"
echo "   • Use TensorBoard : ${USE_TENSORBOARD}"

WANDB_FLAG=""
[[ "$USE_WANDB" == true ]] && WANDB_FLAG="--use_wandb"
TB_FLAG=""
[[ "$USE_TENSORBOARD" == true ]] && TB_FLAG="--use_tensorboard"

# Determine which Python wrapper to call
if [[ "$ARCH" == "huggingface" && "${DATASET^^}" == "MNIST" ]]; then
    PYTHON_CMD="python3 -m debugpy --listen 5678 --wait-for-client src/train/train_mnist_diffusion.py"
elif [[ "$ARCH" == "factory" ]]; then
    PYTHON_CMD="python3 -m debugpy --listen 5678 --wait-for-client src/train/train_factory.py"
else
    echo "❌ Unrecognized combination: ARCHITECTURE=$ARCH, DATASET=${DATASET^^}"
    exit 1
fi

# Construct config file path and check existence

CONFIG_FILE="src/config/config_${ARCH,,}_${DATASET,,}.yml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Finally, run the trainer
echo "${PYTHON_CMD} with config: $CONFIG_FILE for experiment $EXP_ID, run $RUN_ID on dataset $DATASET"
${PYTHON_CMD} \
  --config "$CONFIG_FILE" \
  --experiment_id "$EXP_ID" \
  --run_id "$RUN_ID" \
  --dataset "$DATASET" \
  $WANDB_FLAG $TB_FLAG
