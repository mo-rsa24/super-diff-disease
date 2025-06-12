import os
import time
import random
import subprocess
from datetime import timedelta

import numpy as np
import torch
from torch.cuda.amp import GradScaler

from src.train.logging.paths import get_log_paths
from src.train.logging.training_logger_utils import (
    log_training_start,
    log_config_summary,
    log_training_setup,
    log_epoch_start,
    log_epoch_summary,
    log_json,
    log_exception,
    visualize_epoch, log_training_end,
)
from src.train.utils.handlers import validation
from src.train.utils.optimizer import build_optimizer, build_scheduler
from src.monitoring.fail_safe_guard import fail_safe_guard as _fail_safe_guard
from src.monitoring.email_alert_mailtrap import alert_on_success
from src.train.utils.ema import update_ema, init_ema
from src.train.utils.amp import get_amp_context

from src.utils.checkpoint_manager import CheckpointManager


def train(model, dataloader, diffusion, num_epochs, save_path, device,
          logger, config, writer, wandb_tracker, val_loader=None):
    # --- Checkpoint manager and resume logic ---
    checkpoint_manager = CheckpointManager(
        run_id=config.get("run_id"),
        checkpoint_dir=save_path,
        keep_last_k=int(config.get("checkpoint", {}).get("keep_last_k", 3)),
        logger=logger,
    )
    start_epoch = 1
    global_step = 0
    best_val_mse = float("inf")
    patience_counter = 0

    # Resume from checkpoint if requested
    if config.get("resume", False):
        try:
            model, opt_state, sched_state, loaded_epoch, loaded_step = checkpoint_manager.load_latest(
                model, None, None, map_location=device
            )
            optimizer = build_optimizer(model, config)
            optimizer.load_state_dict(opt_state.state_dict())
            scheduler = build_scheduler(optimizer, config, num_epochs)
            scheduler.load_state_dict(sched_state.state_dict())

            start_epoch = loaded_epoch + 1
            global_step = loaded_step or 0
            logger.info(f"🔄 Resumed from epoch {loaded_epoch}, global_step {global_step}")
        except FileNotFoundError:
            logger.warning("No checkpoint to resume; starting fresh.")
    else:
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config, num_epochs)

    scaler = GradScaler(enabled=config.get("training", {}).get("use_amp", False))
    ema_enabled = config.get("training", {}).get("ema", {}).get("enable", True)
    ema = init_ema(model, config, device) if ema_enabled else None

    start_time = time.time()
    vis_dir, proj_dir, feature_dir = get_log_paths(config)

    # Startup logging
    log_training_start(
        logger, type(model).__name__, config.get("experiment_id"),
        config.get("run_id"), num_epochs, len(dataloader)
    )
    log_config_summary(config, logger)
    log_training_setup(optimizer, scheduler, scaler.is_enabled(), ema_enabled, logger)

    # Main training loop
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_start = time.time()
        log_epoch_start(epoch - 1, logger)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device)
            optimizer.zero_grad()
            with get_amp_context(config, device):
                loss = diffusion.training_step(model, images)

            scaler.scale(loss).backward()
            if (clip := config.get("training", {}).get("grad_clip_norm")):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)


            scaler.step(optimizer)
            scaler.update()
            ema.update()

            if scheduler:
                scheduler.step()

            # Per-batch debug logging
            if global_step % config.get("observability", {}).get("log_every", 100) == 0:
                logger.info(f"[Batch {batch_idx}/{len(dataloader)}] loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

            global_step += 1
            epoch_loss += loss.item()
            # Fail-safe guard
            failed, reason = _fail_safe_guard(
                epoch, loss.item(), config.get("run_id"), config,
                config.get("checkpoint_path", save_path), enable_alerts=True
            )
            if failed:
                log_exception(logger, epoch, batch_idx, Exception(reason))
                checkpoint_manager.save(
                    model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=epoch,
                    global_step=global_step
                )
                return
            break
        # Compute epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start

        # Validation
        val_loss = None
        if val_loader:
            metrics, best_val_mse, patience_counter, stop_flag = validation(
                model, val_loader, diffusion, device, config,
                logger, writer, wandb_tracker,
                epoch, best_val_mse, patience_counter,
                config.get("training", {}).get("early_stopping_patience", 10), save_path
            )
            val_loss = metrics.get("val_mse")
            if stop_flag:
                logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                break

        # Epoch summary logging
        log_epoch_summary(
            logger, epoch, num_epochs, avg_loss,
            val_loss=val_loss, epoch_time=epoch_duration
        )
        log_json(
            logger, "Epoch Summary",
            epoch=epoch, train_loss=avg_loss,
            val_loss=val_loss, duration=epoch_duration
        )

        # Visualization
        if images is not None:
            visualize_epoch(
                epoch=epoch - 1, num_epochs=num_epochs,
                model=model, images=images, diffusion=diffusion,
                writer=writer, wandb_tracker=wandb_tracker,
                config=config, logger=logger,
                vis_dir=vis_dir, proj_dir=proj_dir,
                feature_dir=feature_dir,
                device=device,
                ema_model=ema.ema_model if ema else None,
            )

        # Save checkpoint for this epoch
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step
        )
        log_json(
            logger, "💾 Checkpoint saved",
            epoch=epoch, global_step=global_step,
            path=os.path.join(save_path, "checkpoints", f"epoch_{epoch:03d}.ckpt")
        )

        # Optional GPU cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        break
    # Finalization
    total_time = time.time() - start_time
    log_training_end(logger, best_val_mse, total_time)
    alert_on_success(
        experiment_id=config.get("experiment_id"),
        run_id=config.get("run_id"),
        total_epochs=epoch, duration=str(timedelta(seconds=int(total_time)))
    )

    return
