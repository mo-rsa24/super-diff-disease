# training/train_factory.py

import argparse, os
from src.factories import get_dataset, get_model_diffusion
from src.utils.temp_config import write_temp_config
import torch 

def build_factory_dataloaders(config):
    # … same as before …
    train_ds = get_dataset(config["dataset"], config, split_override="train")
    val_ds   = get_dataset(config["dataset"], config, split_override="val")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(config["training"].get("val_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=True
    )
    return train_loader, val_loader


def build_factory_model_and_diffusion(config, device, logger):
    """
    Returns (model, diffusion) for any factory‐registered dataset + architecture.
    """
    model, diffusion = get_model_diffusion(config["dataset"], config)

    # device‐move diffusion if needed (e.g., if DDPM holds buffers)
    if hasattr(diffusion, "to"):
        diffusion.to(device)

    model.to(device)
    logger.info(f"Using factory model: {type(model).__name__}")
    return model, diffusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/config_factory_chestxray.yaml")
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. TB or PNEUMONIA)")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    args = parser.parse_args()

    # 1) Validate dataset argument
    ds_upper = args.dataset.strip().upper()
    if ds_upper not in ["TB", "PNEUMONIA"]:
        raise ValueError(f"Unsupported dataset='{args.dataset}'. Allowed: 'TB', 'PNEUMONIA'.")

    # 2) Prepare overrides without touching the original YAML
    overrides = {
        "dataset": ds_upper,
        "architecture": "factory",
        "logging": {
            "use_wandb": args.use_wandb,
            "use_tensorboard": args.use_tensorboard,
            "wandb_minimal": False
        }
    }

    # 3) Write a temporary config.yaml
    tmp_config_path = write_temp_config(args.config, overrides)

    try:
        # 4) Call generic runner with the temp YAML
        run_training(build_factory_dataloaders, build_factory_model_and_diffusion, tmp_config_path)
    finally:
        # 5) Clean up the temp YAML
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)
