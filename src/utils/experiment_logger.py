# src/utils/experiment_logger.py

import os
import yaml
from datetime import datetime

def log_experiment_card(save_dir, run_id, experiment_id, config_path, researcher="Molefe Molefe", tags=None, purpose=None, hypothesis=None, notes=None):
    """
    Logs a structured experiment_card.yaml file in the metadata directory.
    """
    card = {
        "experiment_purpose": purpose or "Describe the purpose of this experiment.",
        "hypothesis": hypothesis or "State the hypothesis being tested.",
        "date": datetime.utcnow().date().isoformat(),
        "researcher": researcher,
        "tags": tags or ["diffusion", "SuperDiff"],
        "linked_config": config_path,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "notes": notes or "You can edit this file after training to add paper references or insights."
    }

    metadata_dir = os.path.join(save_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    path = os.path.join(metadata_dir, "experiment_card.yaml")

    with open(path, "w") as f:
        yaml.dump(card, f)

    return path
