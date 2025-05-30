import os, yaml, argparse, torch, numpy as np
from src.utils.env import resolve_paths, set_global_seeds, is_cluster
from src.data.dataset import ChestXrayDataset
from transforms import safe_augmentation
from models.unet import UNet
from models.ddpm import DDPM
from src.train.training_logic import train
from src.utils.logger import init_logger


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)  # TB or PNEUMONIA
    parser.add_argument("--use_wandb", type=str, default="false")
    parser.add_argument("--use_tensorboard", type=str, default="true")
    parser.add_argument("--task", type=str, required=True)  # TB or PNEUMONIA

    args = parser.parse_args()

    config = load_yaml(args.config)
    config["experiment_id"] = f"experiment_{args.experiment_id}"
    config["run_id"] = f"run_{args.run_id}"
    config["dataset"] = args.dataset.upper()
    config["logging"]["use_wandb"] = args.use_wandb.lower() == "true"
    config["logging"]["use_tensorboard"] = args.use_tensorboard.lower() == "true"
    config["task"] = args.task.upper()

    log_dir = f"/gluster/.../logs/experiment_{config['experiment_id']}/run_{config['run_id']}"
    logger = init_logger(log_dir, log_to_stdout=True)

    logger.info("✅ Starting training...")
    logger.debug(f"Dataset: {config['dataset']}")
    logger.debug(f"Hyperparams: {config['training']}")

    resolved = resolve_paths(config)

    os.makedirs(resolved["checkpoint_dir"], exist_ok=True)
    with open(os.path.join(resolved["checkpoint_dir"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    set_global_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging setup
    if config["logging"]["use_wandb"]:
        import wandb
        wandb.init(project="super-diff-xray",
        name=f"{config['experiment_id']}_{config['run_id']}",
        config=config)

    writer = None
    if config["logging"]["use_tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=resolved["tensorboard_dir"])

    # Dataset
    dataset_root = os.path.join(resolved["dataset_dir"], config["dataset"])
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    transform = safe_augmentation(
        risk=config["training"]["augmentation"],
        normalization=config["training"]["normalization"]
    )

    dataset = ChestXrayDataset(
        root_dir=dataset_root,
        split=config["training"]["split"],
        aug=transform,
        class_filter=1,
        task=config["task"]  # Optional if ChestXrayDataset supports task
    )


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    model = UNet()
    diffusion = DDPM(num_timesteps=config["training"]["num_timesteps"])

    train(
        model=model,
        dataloader=dataloader,
        diffusion=diffusion,
        num_epochs=config["training"]["num_epochs"],
        save_path=resolved["checkpoint_dir"],
        device=device
    )

if __name__ == "__main__":
    main()
