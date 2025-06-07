#!/bin/bash
# tools/monitoring/monitor.sh

if [ "$#" -lt 2 ]; then
  echo "Usage: ./monitor.sh <experiment_id> <run_id> [LEVELS...]"
  exit 1
fi

EXPERIMENT_ID="$1"
RUN_ID="$2"
shift 2
LEVELS="$@"

python3 tools/monitoring/stream_log.py \
  --config src/config/config.yaml \
  --experiment_id "$EXPERIMENT_ID" \
  --run_id "$RUN_ID" \
  --levels $LEVELS
