# src/utils/env.py
import os
import socket
import random
import torch
import numpy as np

def is_cluster():
    hostname = socket.gethostname()
    return "mscluster" in hostname or "wits" in hostname or os.environ.get("IS_CLUSTER") == "1"

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_paths(config):
    base_dir = config["paths"]["cluster_base"] if is_cluster() else config["paths"]["local_base"]
    task = config.get("task", config["dataset"])
    experiment = config["experiment_id"]
    run = config["run_id"]

    dataset_dir = os.path.normpath(os.path.join(base_dir, config["paths"].get("dataset_subdir", "datasets/cleaned")))
    output_dir = os.path.normpath(os.path.join(base_dir, config["paths"].get("output_dir", "outputs")))
    checkpoint_dir = os.path.normpath(os.path.join(base_dir, config["paths"].get("checkpoint_dir", "checkpoints"), experiment, run, task))
    tensorboard_dir = os.path.normpath(os.path.join(base_dir, config["paths"].get("tensorboard_dir", "tensorboard"), experiment, run, task))
    wandb_dir = os.path.normpath(os.path.join(base_dir, config["paths"].get("wandb_dir", "wandb"), experiment, run, task))
    log_dir = os.path.normpath(os.path.join(output_dir, experiment, run, task, "logs"))

    return {
        "log_dir": log_dir,
        "base_dir": base_dir,
        "dataset_dir": dataset_dir,
        "checkpoint_dir": checkpoint_dir,
        "tensorboard_dir": tensorboard_dir,
        "wandb_dir": wandb_dir,
        "output_dir": output_dir
    }
