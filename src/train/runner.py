# training/runner.py

import os, json, yaml
from datetime import datetime

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from src.train.training_logic import train
from src.utils.env import resolve_paths, set_global_seeds, is_cluster
from src.utils.logger import init_logger
from src.utils.experiment_logger import log_experiment_card
from src.monitoring.email_alert_mailtrap import alert_on_failure, alert_on_success


def run_training(build_dataloaders, build_model_and_diffusion, config_path: str):
    """
    Args:
        build_dataloaders:   fn(config) -> (train_loader, val_loader)
        build_model_and_diffusion: fn(config, device, logger) -> (model, diffusion)
        config_path:         Path to a YAML file that already has 'dataset' and 'architecture' keys.
    """

    # ─── READ CONFIG ─────────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate essential fields
    for key in ("dataset", "architecture"):
        if key not in config:
            raise KeyError(f"Missing '{key}' in config.")

    # Check for required sections
    if "training" not in config:
        raise KeyError("Missing 'training' section in config.")

    if "model" not in config:
        raise KeyError("Missing 'model' section in config.")

    # Validate 'keep_last_k' if present
    ckpt_cfg = config.get("checkpoint", {})
    if "keep_last_k" in ckpt_cfg:
        k = int(ckpt_cfg["keep_last_k"])
        if k < 1:
            raise ValueError(f"checkpoint.keep_last_k must be ≥ 1, got {k}.")

    # ─── MONITORING CONFIG & AUTO‐RESUME ─────────────────────────────────────────
    monitoring_cfg = yaml.safe_load(open("src/config/monitoring.yaml"))
    config["monitoring"] = monitoring_cfg
    if monitoring_cfg["resume"].get("auto_resume", False):
        resume_path = monitoring_cfg["resume"].get("checkpoint_path", "")
        if resume_path and not os.path.isfile(resume_path):
            raise FileNotFoundError(f"resume.checkpoint_path not found: {resume_path}")
        config["resume_checkpoint"] = resume_path

    # ─── RESOLVE PATHS & MAKE FOLDERS ────────────────────────────────────────────
    resolved = resolve_paths(config)
    for p in resolved.values():
        os.makedirs(p, exist_ok=True)

    # ─── LOGGER + EXPERIMENT CARD ────────────────────────────────────────────────
    logger = init_logger(resolved["log_dir"], log_to_stdout=True)
    exp_card = log_experiment_card(
        save_dir=resolved["log_dir"],
        run_id=config["run_id"],
        experiment_id=config["experiment_id"],
        config_path=config_path,
        purpose=f"Training {config['dataset']} with {config['architecture']}",
        hypothesis="Auto‐logged unified pipeline",
        tags=[config["dataset"], config["architecture"].upper()],
        notes=f"Auto‐logged from {config_path}"
    )
    logger.info(f"🧾 Experiment card saved to {exp_card}")

    # ─── LOG HYPERPARAMETERS AT START ────────────────────────────────────────────
    hp_keys = [
        "training.num_timesteps",
        "training.batch_size",
        "model.base_dim",
        "model.image_size",
        "training.learning_rate"
    ]
    def _deep_get(d: dict, keys: list):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d

    for hk in hp_keys:
        keys = hk.split(".")
        val = _deep_get(config, keys)
        logger.info(f"🔑 Hyperparam {hk} = {val}")

    # ─── SET SEEDS & DEVICE ──────────────────────────────────────────────────────
    set_global_seeds(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── WRITE metadata.json & persist run_config.yaml ───────────────────────────
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "dataset": config["dataset"],
        "architecture": config["architecture"],
        "env": "cluster" if is_cluster() else "local",
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "system": {
            "hostname": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda": torch.version.cuda,
        },
        "config": config
    }
    with open(os.path.join(resolved["log_dir"], "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    cfg_out = os.path.join(resolved["checkpoint_dir"], "run_config.yaml")
    os.makedirs(os.path.dirname(cfg_out), exist_ok=True)
    with open(cfg_out, "w") as f:
        yaml.dump(config, f)

    # ─── BUILD DATALOADERS ───────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(config)
    logger.info(f"📦 Train size: {len(train_loader.dataset)} | Val size: {len(val_loader.dataset)}")

    # ─── INITIALIZE TENSORBOARD / W&B ─────────────────────────────────────────────
    if config["logging"]["use_tensorboard"]:
        writer = SummaryWriter(log_dir=resolved["tensorboard_dir"])
        writer.add_text("Run Info", json.dumps(metadata, indent=2))
        logger.info(f"📉 TensorBoard logs at {resolved['tensorboard_dir']}")
    else:
        writer = None

    if config["logging"]["use_wandb"]:
        os.environ["WANDB_DIR"] = resolved["wandb_dir"]
        try:
            wandb.init(
                project="unified‐diffusion",
                name=f"{config['experiment_id']}_{config['run_id']}",
                config=config
            )
            wandb.run.tags = [config["dataset"], config["architecture"]]
            logger.info("📡 WandB initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize WandB: {e}")
            config["logging"]["use_wandb"] = False
            wandb = None
    else:
        wandb = None

    # ─── BUILD MODEL & DIFFUSION ─────────────────────────────────────────────────
    model, diffusion = build_model_and_diffusion(config, device, logger)
    # Ensure diffusion buffers (e.g. noise schedule) live on the same device
    if hasattr(diffusion, "to"):
        diffusion.to(device)
    model.to(device)
    logger.info(f"🧠 Model class: {type(model).__name__}")
    logger.info(f"   Diffusion class: {type(diffusion).__name__}")

    # ─── CALL GENERIC TRAIN LOOP ─────────────────────────────────────────────────
    try:
        train(
            model=model,
            dataloader=train_loader,
            diffusion=diffusion,
            num_epochs=int(config["training"]["num_epochs"]),
            save_path=resolved["checkpoint_dir"],
            device=device,
            logger=logger,
            config=config,
            writer=writer,
            wandb_tracker=(wandb if config["logging"]["use_wandb"] else None),
            val_loader=val_loader
        )
    except Exception as e:
        # If training crashes, send failure alert
        logger.error(f"❌ Training crashed: {e}")
        try:
            alert_on_failure(
                experiment_id=config["experiment_id"],
                run_id=config["run_id"],
                last_epoch=0,  # you could extract this from ckpt_manager if needed
                error_msg=str(e)
            )
        except Exception as mail_e:
            logger.error(f"❌ Could not send failure email: {mail_e}")
        raise

    # ─── UPON SUCCESSFUL COMPLETION ───────────────────────────────────────────────
    try:
        start_ts = datetime.fromisoformat(metadata["timestamp"]).timestamp()
        duration = datetime.utcnow().timestamp() - start_ts
        alert_on_success(
            experiment_id=config["experiment_id"],
            run_id=config["run_id"],
            total_epochs=int(config["training"]["num_epochs"]),
            duration=str(duration)
        )
    except Exception as mail_e:
        logger.error(f"❌ Could not send success email: {mail_e}")
