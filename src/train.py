# train.py
import torch.utils
import os, yaml, argparse, torch, json, platform, subprocess
from src.utils.data_manifest import generate_data_manifest
from src.utils.experiment_logger import log_experiment_card
from src.factories import get_dataset, get_model_diffusion
from src.utils.experiment_logger import log_experiment_card
# Import the “HuggingFace MNIST” modules
from models.diffusion_mnist import UNet as HF_UNet, SinusoidalPosEmb
from schedulers.beta_schedule import get_noise_schedule as hf_get_noise_schedule
import numpy as np
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.utils.env import resolve_paths, set_global_seeds, is_cluster
from src.utils.logger import init_logger
from src.train.training_logic import train

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)
    
# -----------------------------------------------------------------------------
# Command‐Line Argument Parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Unified training entry point")

    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--experiment_id", type=str, required=True,
        help="Experiment ID (e.g., 'exp1')."
    )
    parser.add_argument(
        "--run_id", type=str, required=True,
        help="Run ID (e.g., 'runA')."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["MNIST", "FASHION_MNIST", "CHEST_XRAY"],
        help="Which dataset to train on."
    )
    parser.add_argument(
        "--architecture", type=str, default="factory",
        choices=["factory", "huggingface"],
        help="Which implementation to use. 'factory' uses get_model_diffusion, 'huggingface' uses HF MNIST code."
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--use_tensorboard", action="store_true",
        help="Enable TensorBoard logging."
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Flag to indicate a W&B sweep run."
    )
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Helper: build HF MNIST model + “diffusion” object
# -----------------------------------------------------------------------------
def load_hf_mnist(config: dict, device: torch.device):
    """
    Constructs:
      - model = HF_UNet(1, base_channels=... , time_emb_dim=...)
      - diffusion = a thin wrapper that exposes training_step(...) and sample(...)
        based on the HF “linear beta” schedule from schedulers/beta_schedule.py
    """
    logger = config.get("__logger__")
    # Use config for hyperparameters (fall back to defaults if missing):
    base_channels = config["model"].get("base_dim", 128)
    time_emb_dim  = config["model"].get("time_emb_dim", 512)
    timesteps     = config["training"]["num_timesteps"]
    img_size      = config["model"].get("image_size", 32)
    lr            = config["training"].get("learning_rate", 2e-4)
    batch_size    = config["training"].get("batch_size", 128)

    # 1) Build the UNet
    model = HF_UNet(in_channels=1, base_channels=base_channels, time_emb_dim=time_emb_dim).to(device)
    logger.info("Using HuggingFace MNIST UNet implementation")

    # 2) Build the noise schedule (betas, alphas, etc.)
    schedule = hf_get_noise_schedule(timesteps)
    # Transfer all to device
    for k, v in schedule.items():
        schedule[k] = v.to(device)

    # 3) Wrap them together in a small “diffusion” object that matches our training_logic API
    class HF_DiffusionWrapper:
        def __init__(self, noise_schedule):
            self.noise_schedule = noise_schedule
            self.T = timesteps

        def q_sample(self, x_start: torch.Tensor, t: torch.Tensor):
            """
            x_t = sqrt(alpha_cumprod[t]) * x_start + sqrt(1 - alpha_cumprod[t]) * eps
            Return (x_t, eps) where eps ~ N(0,1).
            """
            bsz = x_start.shape[0]
            eps = torch.randn_like(x_start)
            alpha_cumprod = self.noise_schedule["alpha_cumprod"]
            sqrt_acp = alpha_cumprod[t].view(bsz, 1, 1, 1)
            sqrt_omacp = (1 - alpha_cumprod[t]).view(bsz, 1, 1, 1)
            x_t = sqrt_acp * x_start + torch.sqrt(sqrt_omacp) * eps
            return x_t, eps

        def training_step(self, model, x_start: torch.Tensor):
            """
            Sample a random t in [0, T-1], generate x_t, predict eps, MSE loss on eps.
            This matches HuggingFace’s “p_losses” logic with predict_eps=True.
            """
            bsz = x_start.shape[0]
            t = torch.randint(0, self.T, (bsz,), device=x_start.device).long()
            x_t, eps = self.q_sample(x_start, t)
            eps_pred = model(x_t, t)
            return torch.nn.functional.mse_loss(eps_pred, eps)

        @torch.no_grad()
        def sample(self, model, image_shape, device):
            """
            Ancestral sampling from T-1 → 0:
            x ← N(0,I); for t = T-1 ... 0: 
              predict eps = model(x, t),
              compute x_{t-1} = (1/√α_t)(x - β_t/√(1-ᾱ_t) * eps) + √β_t z
            Finally return x_0 (in [-1,1]).
            """
            model.eval()
            num_samples = image_shape[0]
            x = torch.randn(image_shape, device=device)
            betas = self.noise_schedule["betas"]
            alphas = self.noise_schedule["alphas"]
            alpha_cumprod = self.noise_schedule["alpha_cumprod"]

            for t in reversed(range(self.T)):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                eps_pred = model(x, t_batch)

                alpha_t = alphas[t]
                alpha_cumprod_t = alpha_cumprod[t]
                alpha_cumprod_prev = alpha_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
                beta_t = betas[t]

                coef1 = torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t)
                coef2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)

                mean = (1 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)).view(1, 1, 1, 1) * eps_pred
                )
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = mean + torch.sqrt(beta_t).view(1, 1, 1, 1) * noise

            # Return x in [−1,1]. The downstream code often expects [0,1], so we can clamp if needed:
            return torch.clamp(x, -1, 1)

    diffusion = HF_DiffusionWrapper(schedule)
    return model, diffusion

def main():
    args = parse_args()

    # If no --config is provided, select default based on architecture
    if args.config is None:
        if args.architecture.lower() == "huggingface" and args.dataset.upper() == "MNIST":
            args.config = "src/config/config_hf_mnist.yaml"
        elif args.architecture.lower() == "factory" and args.dataset.upper() == "CHEST_XRAY":
            args.config = "src/config/config_factory_chestxray.yaml"
        else:
            raise ValueError(f"No default config found for {args.dataset}/{args.architecture}")

    # 1) Load YAML config
    config = yaml.safe_load(open(args.config))
    config.update({
        "experiment_id": f"experiment_{args.experiment_id}",
        "run_id": f"run_{args.run_id}",
        "dataset": args.dataset.upper(),
        "architecture": args.architecture.lower(),
        "logging": {
            "use_wandb": args.use_wandb,
            "use_tensorboard": args.use_tensorboard,
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
    # 3) Resolve file paths
    resolved = resolve_paths(config)
    for p in resolved.values():
        os.makedirs(p, exist_ok=True)
        
    
    # 4) Logger + Experiment Card
    log_dir = resolved["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    logger = init_logger(log_dir, log_to_stdout=True)

    experiment_card_path = log_experiment_card(
        save_dir=log_dir,
        run_id=config["run_id"],
        experiment_id=config["experiment_id"],
        config_path=args.config,
        purpose=f"Training {config['dataset']} with {config['architecture']} architecture",
        hypothesis="Integrate HF MNIST diffusion into our unified pipeline",
        tags=[config["dataset"], config["architecture"].upper()],
        notes=f"Auto‐logged from {args.config}"
    )
    logger.info(f"🧾 Experiment card saved to {experiment_card_path}")
    logger.info(f"🚀 Starting training for {config['experiment_id']} / {config['run_id']}")

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


    dataset_root = resolved["dataset_dir"]
    
    # ─── BUILD TRAIN & VAL LOADERS ───────────────────────────────────────────────
    train_dataset = get_dataset(config["dataset"], config, split_override="train")
    val_dataset   = get_dataset(config["dataset"], config, split_override="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"].get("val_batch_size", config["training"]["batch_size"]),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True
    )
    logger.info(f"📦 Train set size: {len(train_dataset)} | Val set size: {len(val_dataset)}")
    
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

    # ─── MODEL + DIFFUSION ───────────────────────────────────────────────────────
    metadata["data_manifest"] = manifest
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=resolved["tensorboard_dir"])
        writer.add_text("Run Info", json.dumps(metadata, indent=2))
        writer.add_text("Data Manifest", json.dumps(manifest, indent=2))
        logger.info(f"📉 TensorBoard logs at {resolved['tensorboard_dir']}")
    else:
        writer = None

    if args.use_wandb:
        import wandb
        os.environ["WANDB_DIR"] = resolved["wandb_dir"]
        wandb.init(
            project="super-diffusion",
            config=config,
            name=f"{config['experiment_id']}_{config['run_id']}"
        )
        wandb.run.tags = [config["dataset"], config["architecture"]]
        wandb.run.notes = f"Git commit: {subprocess.getoutput('git rev-parse HEAD')}"
        logger.info("📡 Weights & Biases initialized")
        
    # 7) Build model + diffusion object
    if config["architecture"] == "huggingface" and config["dataset"] == "MNIST":
        # Use the HuggingFace‐style MNIST UNet + HF_DiffusionWrapper
        model, diffusion = load_hf_mnist(config, device)
        # Attach logger into config so load_hf_mnist can log
        config["__logger__"] = logger
    else:
        # Use the factory‐registered version (e.g. CHEST_XRAY or FASHION_MNIST)
        model, diffusion = get_model_diffusion(config["dataset"], config)
    model.to(device)
    logger.info(f"🧠 Model class: {model.__class__.__name__}")
    logger.info(f"   Diffusion class: {diffusion.__class__.__name__ if hasattr(diffusion, '__class__') else type(diffusion)}")

    # 8) Finally, call our unified training_logic.train(...)
    from src.train.training_logic import train as training_loop

    training_loop(
        model=model,
        dataloader=train_loader,
        diffusion=diffusion,
        num_epochs=config["training"]["num_epochs"],
        save_path=resolved["checkpoint_dir"],
        device=device,
        logger=logger,
        config=config,
        writer=writer,
        wandb_tracker=(wandb if args.use_wandb else None),
        val_loader=val_loader
    )
if __name__ == "__main__":
    main()
