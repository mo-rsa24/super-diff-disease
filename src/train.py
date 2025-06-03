# train.py
import os, yaml, argparse, torch, json, platform, subprocess
from dotenv import load_dotenv
from src.utils.data_manifest import generate_data_manifest
from src.utils.experiment_logger import log_experiment_card
from src.monitoring.fail_safe_guard import fail_safe_guard
from src.utils.experiment_logger import log_experiment_card
import numpy as np
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.utils.env import resolve_paths, set_global_seeds, is_cluster
from src.utils.logger import init_logger
from src.data.dataset import ChestXrayDataset
from transforms import safe_augmentation
from models.unet import UNet
from models.ddpm import DDPM
from src.train.training_logic import train

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--sweep", action="store_true", help="Enable W&B sweep")

    args = parser.parse_args()

    config = load_yaml(args.config)
    config.update({
        "experiment_id": f"experiment_{args.experiment_id}",
        "run_id": f"run_{args.run_id}",
        "dataset": args.dataset.upper(),
        "logging": {
            "use_wandb": getattr(args, "use_wandb", False),
            "use_tensorboard": getattr(args, "use_tensorboard", False),
            "wandb_minimal": config.get("logging", {}).get("wandb_minimal", False)
        }
    })
    
    if args.sweep:
        import wandb
        wandb.init(config=config)  # will start tracking + pull current sweep values
        sweep_config = dict(wandb.config)

        def recursive_update(cfg, updates):
            for k, v in updates.items():
                keys = k.split(".")
                d = cfg
                for key in keys[:-1]:
                    d = d.setdefault(key, {})
                d[keys[-1]] = v

        recursive_update(config, sweep_config)
    
    monitoring_config = load_yaml("src/config/monitoring.yaml")
    config["monitoring"] = monitoring_config
    
    if monitoring_config["resume"]["auto_resume"] and monitoring_config["resume"]["checkpoint_path"]:
        config["resume_checkpoint"] = monitoring_config["resume"]["checkpoint_path"]

    
    num_timesteps = config["training"]["num_timesteps"]
    resolved = resolve_paths(config)
    
    log_dir = resolved["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = init_logger(log_dir, log_to_stdout=True)

    # Inside main() in train.py
    experiment_card_path = log_experiment_card(
        save_dir=resolved["log_dir"],
        run_id=config["run_id"],
        experiment_id=config["experiment_id"],
        config_path=args.config,
        purpose="Test whether we can train unconditional diffusion models on chest X-ray data ",
        hypothesis="Two unconditional diffusion models trained on chest X-ray data will learn to generate realistic images.",
        tags=["SuperDiff", config["dataset"]],
        notes="Auto-logged at experiment start. Based on config from " + args.config
    )
    logger.info(f"🧾 Experiment card saved to {experiment_card_path}")

    logger.info(f"🚀 Starting training script for {config['experiment_id']} / {config['run_id']}")

    set_global_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "task": config["dataset"],
        "env": "cluster" if is_cluster() else "local",
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        "system": {
            "hostname": platform.node(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "config": config
    }
    with open(os.path.join(log_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    config_path_for_run = os.path.join(resolved["checkpoint_dir"], "run_config.yaml")
    os.makedirs(os.path.dirname(config_path_for_run), exist_ok=True)
    with open(config_path_for_run, "w") as f:
        yaml.dump(config, f)

    for path in resolved.values():
        os.makedirs(path, exist_ok=True)

    writer = None
    if config["logging"]["use_tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=resolved["tensorboard_dir"])
        writer.add_text("Run Info", json.dumps(metadata, indent=2))
        logger.info(f"📉 TensorBoard logs -> {resolved['tensorboard_dir']}")

    wandb_tracker = None
    if config["logging"]["use_wandb"]:
        import wandb
        os.environ["WANDB_DIR"] = resolved["wandb_dir"]
        wandb_tracker = wandb
        wandb.init(
            project="super-diff-xray",
            config=config,
            tags=[config["dataset"]],
            notes=f"Sampling: {config['sampling']['method']} | Seed: {config.get('split_seed', 'NA')}"
            )
        wandb.run.tags = [config["dataset"], "UNet", "DDPM"]
        wandb.run.notes = f"CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
        if not config["logging"].get("wandb_minimal", False):
            wandb.watch_called = False
            wandb.watch(models=UNet(), log="all", log_freq=100)
        logger.info(f"📡 W&B initialized")

    transform = safe_augmentation(config["training"]["augmentation"], config["training"]["normalization"])
    dataset_root = resolved["dataset_dir"]
    dataset = ChestXrayDataset(root_dir=dataset_root, split=config["training"]["split"],
                               aug=transform, class_filter=1, task=config["dataset"])
    val_dataset = ChestXrayDataset(root_dir=dataset_root, split="val",
                               aug=transform, class_filter=1, task=config["dataset"])
    logger.info(f"📦 Dataset: {len(dataset)} samples | {config['dataset']} | Split: {config['training']['split']}")
    logger.info(f"📦 Validation Dataset: {len(val_dataset)} samples | {config['dataset']} | Split: val")
    
    
        # 🧾 Save dataset version metadata
    manifest = generate_data_manifest(
        dataset_root=dataset_root,
        task=config["dataset"],
        split_seed=config.get("split_seed", 42),
        preprocessing_script_path="src/transforms.py",  # or full path if moved
        run_dir=resolved["log_dir"]
    )

    # 🔗 Also embed in main training metadata
    metadata["data_manifest"] = manifest

    # Update TensorBoard and W&B logs with it
    if writer:
        writer.add_text("Data Manifest", json.dumps(manifest, indent=2))

    if wandb_tracker:
        wandb_tracker.config.update(manifest)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"].get("val_batch_size", config["training"]["batch_size"]),
        shuffle=False
    )
    model = UNet()
    diffusion = DDPM(num_timesteps=num_timesteps)

    logger.info(f"🧠 Model: {model.__class__.__name__} | Steps: {num_timesteps}")
    
    load_dotenv()  # now os.getenv(...) will work


    train(
        model=model,
        dataloader=dataloader,
        diffusion=diffusion,
        num_epochs=config["training"]["num_epochs"],
        save_path=resolved["checkpoint_dir"],         # e.g. "./runs/exp1/checkpoints"
        device=device,
        logger=logger,
        config=config,
        writer=writer,
        wandb_tracker=wandb_tracker,
        val_loader=val_loader                         # or None if skipping validation
        )

if __name__ == "__main__":
    main()
