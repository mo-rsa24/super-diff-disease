# training/train_mnist_diffusion.py

import argparse
import os

from src.data.mnist_dataset import get_mnist_dataloaders
from src.models.diffusion_mnist import UNet                 # ← HF UNet for MNIST (32×32)
from src.models.wrappers.HF_DiffusionWrapper import HF_DiffusionWrapper     # ← Wrapper that expects noise_schedule, T 
from src.utils.temp_config import write_temp_config
from src.schedulers.beta_schedule import get_noise_schedule as hf_get_noise_schedule
from src.train.runner import run_training

def build_mnist_dataloaders(config):
    batch_size  = int(config["training"]["batch_size"])
    img_size    = int(config["model"]["image_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    split_ratio = float(config["training"].get("train_val_split", 0.9))
    seed        = int(config.get("seed", 42))

    train_loader, val_loader = get_mnist_dataloaders(
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        split_ratio=split_ratio,
        seed=seed
    )
    return train_loader, val_loader

def build_mnist_model_and_diffusion(config, device, logger):
    """
    Returns (model, diffusion) for HuggingFace MNIST UNet + schedule.
    """
    base_channels = int(config["model"]["base_dim"])         # e.g. 128
    time_emb_dim  = int(config["model"]["time_emb_dim"])     # e.g. 512
    timesteps     = int(config["training"]["num_timesteps"]) # e.g. 1000

    # 1) Instantiate HF UNet for 32×32 MNIST
    model = UNet(in_channels=1, base_channels=base_channels, time_emb_dim=time_emb_dim)
    logger.info("Using HuggingFace MNIST UNet (Annotated Diffusion)")

    # 2) Build β schedule and move all tensors to device
    schedule = hf_get_noise_schedule(timesteps)
    for k, v in schedule.items():
        schedule[k] = v.to(device)

    # 3) Wrap in HF_DiffusionWrapper
    diffusion = HF_DiffusionWrapper(noise_schedule=schedule, timesteps=timesteps)
    if hasattr(diffusion, "to"):
        diffusion.to(device)

    model.to(device)
    return model, diffusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config_hf_mnist.yaml",
        help="Path to the base YAML config for MNIST"
    )
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="Dataset name; must be 'MNIST' to use this script."
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    args = parser.parse_args()

    # 1) Validate that we indeed want MNIST
    ds_upper = args.dataset.strip().upper()
    if ds_upper != "MNIST":
        raise ValueError(f"For this script, --dataset must be 'MNIST' (got '{args.dataset}').")

    # 2) Merge overrides into a temporary config
    overrides = {
        "experiment_id": f"experiment_{args.experiment_id}",
        "run_id": f"run_{args.run_id}",
        "dataset": ds_upper,
        "architecture": "huggingface",
        "logging": {
            "use_wandb": args.use_wandb,
            "use_tensorboard": args.use_tensorboard,
            "wandb_minimal": False
        }
    }
    tmp_config_path = write_temp_config(args.config, overrides)

    try:
        # 3) Launch the generic training runner
        run_training(
            build_mnist_dataloaders,
            build_mnist_model_and_diffusion,
            tmp_config_path
        )
    finally:
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)
