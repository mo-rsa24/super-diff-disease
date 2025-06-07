# training/training_logic.py

import os
import time
import json
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import timedelta

from ema_pytorch import EMA
from src.utils.checkpoint_manager import CheckpointManager
from src.monitoring.fail_safe_guard import fail_safe_guard
from src.monitoring.email_alert_mailtrap import alert_on_failure, alert_on_success
from src.train.validate_logic import validate
from src.utils.training_logger_utils import (
    log_grad_norms,
    log_json,
    log_scheduler_step,
    plot_projection,
    log_visualizations,
    log_batch_progress,
    log_epoch_summary,
    log_exception,
    log_training_end,
    log_training_start,
    log_training_stats,
)
from src.utils.training_self_checker import TrainingSelfChecker


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    diffusion: any,
    num_epochs: int,
    save_path: str,
    device: torch.device,
    logger: any,
    config: dict,
    writer=None,
    wandb_tracker=None,
    val_loader: torch.utils.data.DataLoader = None,
):
    """
    Unified training loop with AMP, gradient checks, early stopping, and robust checkpointing.

    Expects:
      - diffusion.training_step(model, images) → loss (scalar Tensor)
      - diffusion.sample(model, image_shape, device) → generated Tensor
      - validate(...) → dict of validation metrics
    """

    # ─── 1) SET DETERMINISTIC & DEVICE ───────────────────────────────
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.to(device)

    # ─── 2) LOGGING INITIALIZATION ──────────────────────────────────
    total_batches = len(dataloader)
    log_training_start(
        logger,
        model_name=type(model).__name__,
        experiment_id=config.get("experiment_id", "N/A"),
        run_id=config.get("run_id", "N/A"),
        total_epochs=num_epochs,
        total_batches=total_batches,
    )

    # ─── 3) OPTIMIZER, SCHEDULER, AMP, EMA ───────────────────────────
    # Optimizer choice
    opt_name = config["training"].get("optimizer", "adam").lower()
    lr = float(config["training"].get("learning_rate", 1e-4))
    beta1 = float(config["training"].get("beta1", 0.9))
    beta2 = float(config["training"].get("beta2", 0.999))
    weight_decay = float(config["training"].get("weight_decay", 0.0))

    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    # Build scheduler (if any)
    scheduler, sched_cfg = None, config["training"].get("scheduler", {})
    if sched_cfg:
        stype = sched_cfg.get("type", "step").lower()
        if stype == "step":
            step_size = int(sched_cfg.get("step_size", 10))
            gamma = float(sched_cfg.get("gamma", 0.1))
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif stype == "cosine":
            t_max = int(sched_cfg.get("t_max", num_epochs))
            eta_min = float(sched_cfg.get("eta_min", 0.0))
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        else:
            raise ValueError(f"Unsupported scheduler type: {stype}")

    # AMP configuration
    use_amp = bool(config["training"].get("use_amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # EMA
    ema_beta = float(config["training"].get("ema_beta", 0.995))
    ema = EMA(model, beta=ema_beta).to(device)

    # ─── 4) CHECKPOINT MANAGER & RESUME LOGIC ───────────────────────
    ckpt_manager = CheckpointManager(
        run_id=config["run_id"],
        checkpoint_dir=save_path,
        keep_last_k=int(config.get("checkpoint", {}).get("keep_last_k", 3)),
        logger=logger,
    )

    start_epoch = 0
    start_global_step = 0

    resume_path = config.get("resume_checkpoint", "")
    auto_resume = config.get("monitoring", {}).get("resume", {}).get("auto_resume", False)

    if auto_resume:
        if resume_path:
            try:
                # load only model first
                _, _, _, loaded_epoch, loaded_step = ckpt_manager.load_checkpoint(
                    filepath=resume_path,
                    model=model,
                    optimizer=None,
                    scheduler=None,
                    map_location=device,
                )
                start_epoch = loaded_epoch  # we will increment below
                start_global_step = loaded_step or 0
                logger.info(f"🔄 Resumed model weights from: {resume_path} (epoch {loaded_epoch})")
            except Exception as e:
                logger.error(f"❌ Failed to load specified checkpoint {resume_path}: {e}")
                logger.info("ℹ️ Falling back to latest available checkpoint…")
                try:
                    _, _, _, loaded_epoch, loaded_step = ckpt_manager.load_latest(
                        model=model,
                        optimizer=None,
                        scheduler=None,
                        map_location=device,
                    )
                    start_epoch = loaded_epoch
                    start_global_step = loaded_step or 0
                    logger.info(f"🔄 Fallback resume from epoch {loaded_epoch}")
                except Exception as e2:
                    logger.info(f"ℹ️ No valid checkpoint found. Starting from scratch. ({e2})")
                    start_epoch = 0
                    start_global_step = 0
        else:
            try:
                _, _, _, loaded_epoch, loaded_step = ckpt_manager.load_latest(
                    model=model,
                    optimizer=None,
                    scheduler=None,
                    map_location=device,
                )
                start_epoch = loaded_epoch
                start_global_step = loaded_step or 0
                logger.info(f"🔄 Auto‐resumed from epoch {loaded_epoch}")
            except Exception as e:
                logger.info(f"ℹ️ No checkpoint to resume. Starting from scratch. ({e})")
                start_epoch = 0
                start_global_step = 0
    else:
        logger.info("ℹ️ Auto‐resume disabled; training from epoch 0.")
        start_epoch = 0
        start_global_step = 0

    # Now re‐create optimizer/scheduler state AFTER setting model weights
    # (We must re‐initialize optimizer & scheduler *after* we know model weights.)
    if start_epoch > 0:
        try:
            # reload optimizer & scheduler
            _, optimizer, scheduler, loaded_epoch, loaded_step = ckpt_manager.load_checkpoint(
                filepath=resume_path if resume_path else None,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            start_epoch = loaded_epoch
            start_global_step = loaded_step or start_global_step
            logger.info(f"✅ Restored optimizer & scheduler state at epoch {loaded_epoch}")
        except Exception:
            logger.warning("⚠️ Could not restore optimizer/scheduler from checkpoint; using fresh optimizer.")

    # model is already on device; no need for model.to(device) again

    # ─── 5) SET UP TrainingSelfChecker ────────────────────────────────
    # max_loss = float(config["monitoring"].get("max_loss_threshold", 100.0))
    # grad_clip = float(config["training"].get("grad_clip_norm", 1.0))
    # self_checker = TrainingSelfChecker(
    #     experiment_id=config.get("experiment_id", "N/A"),
    #     run_id=config.get("run_id", "N/A"),
    #     save_dir=save_path,
    #     checkpoint_keys=["epoch", "global_step", "model", "optimizer", "scheduler", "ema"],
    #     grad_clip=grad_clip,
    #     loss_threshold=max_loss,
    # )

    # ─── 6) PREPARE DIRECTORIES ───────────────────────────────────────
    vis_dir = os.path.join(save_path, "samples")
    feature_dir = os.path.join(save_path, "features")
    proj_dir = os.path.join(save_path, "projections")
    metric_dir = os.path.join(save_path, "metrics")

    for sub in (vis_dir, feature_dir, proj_dir, metric_dir):
        os.makedirs(sub, exist_ok=True)

    # ─── 7) TRAINING LOOP ──────────────────────────────────────────────
    # Initialize loss logs
    loss_log, fid_log, lpips_log = [], [], []
    val_metrics_log = []
    best_val_mse = float("inf")
    patience_counter = 0
    early_stop_patience = int(config["training"].get("early_stopping_patience", 10))
    use_validation = bool(config["training"].get("use_validation", False) and val_loader is not None)

    # Initialize global_step in case of resume
    global_step = start_global_step

    total_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)

        # If using a DistributedSampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        log_json(logger, f"🔁 Starting epoch {epoch+1}", epoch=epoch + 1, total_epochs=num_epochs)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad()

            # — AMP Context for Forward & Loss —
            if use_amp:
                with autocast(enabled=True):
                    loss = diffusion.training_step(model, images)
            else:
                loss = diffusion.training_step(model, images)

            # 1) Loss sanity check
            if not torch.isfinite(loss):
                raise RuntimeError(f"❌ Invalid loss (NaN/Inf) at epoch {epoch+1}, batch {batch_idx}")

            # self_checker.check_loss(loss, epoch, batch_idx, logger)

            # 2) Backprop & gradient‐clipping
            if use_amp:
                scaler.scale(loss).backward()
                # Unscale for clipping
                scaler.unscale_(optimizer)
                # if grad_clip > 0:
                #     clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # if grad_clip > 0:
                #     clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # 3) Gradient health check
            # self_checker.check_gradients(model, epoch, batch_idx, logger)

            # 4) EMA update
            ema.update()

            # 5) Scheduler step (per‐batch if configured)
            if scheduler is not None and sched_cfg.get("step_per", "batch") == "batch":
                scheduler.step()
                log_scheduler_step(
                    logger,
                    optimizer,
                    epoch=epoch + 1,
                    step=batch_idx + 1,
                    writer=writer,
                    wandb_tracker=wandb_tracker,
                )

            # 6) Logging & metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss
            global_step += 1

            if (batch_idx + 1) % int(config["training"].get("log_interval", 100)) == 0:
                elapsed = time.time() - epoch_start_time
                gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

                log_batch_progress(
                    logger,
                    epoch=epoch + 1,
                    total_epochs=num_epochs,
                    step=batch_idx + 1,
                    total_steps=num_batches,
                    loss=batch_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    time_elapsed=elapsed,
                    gpu_mem=gpu_mem,
                )
                log_json(
                    logger,
                    "Batch Progress",
                    epoch=epoch + 1,
                    step=batch_idx + 1,
                    total_steps=num_batches,
                    loss=batch_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    time_elapsed=elapsed,
                    gpu_mem=gpu_mem,
                )

                if config["observability"].get("log_grad_norm", True):
                    log_grad_norms(
                        logger,
                        model,
                        epoch=epoch + 1,
                        step=batch_idx + 1,
                        writer=writer,
                        wandb_tracker=wandb_tracker,
                    )

                if wandb_tracker:
                    wandb_tracker.log(
                        {
                            "train/batch_loss": batch_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/gpu_mem_mb": gpu_mem,
                        },
                        step=global_step,
                    )

            # 7) Fail‐safe guard
            if (batch_idx + 1) % int(config["monitoring"]["loss_monitoring"]["check_frequency"]) == 0:
                fail_flag, reason = fail_safe_guard(
                    epoch=epoch + 1,
                    loss=loss,
                    run_id=config["run_id"],
                    config=config,
                    checkpoint_path=save_path,
                    enable_alerts=config["monitoring"]["alerts"]["enable"],
                )
                if fail_flag:
                    logger.error(f"🛑 FAIL‐SAFE TRIGGERED: {reason}")
                    try:
                        alert_on_failure(
                            experiment_id=config.get("experiment_id", "N/A"),
                            run_id=config["run_id"],
                            last_epoch=epoch + 1,
                            error_msg=reason,
                        )
                    except Exception as mail_e:
                        logger.error(f"❌ Could not send failure email: {mail_e}")
                    return  # abort entire training

        # ── End of batch loop ⁚ scheduler per‐epoch if configured
        if scheduler is not None and sched_cfg.get("step_per", "batch") == "epoch":
            scheduler.step()
            log_scheduler_step(
                logger,
                optimizer,
                epoch=epoch + 1,
                writer=writer,
                wandb_tracker=wandb_tracker,
            )

        # Compute average training loss
        avg_train_loss = epoch_loss / float(num_batches)
        epoch_duration = time.time() - epoch_start_time
        loss_log.append(avg_train_loss)

        # ── Epoch summary logging ⁚ train metrics
        log_epoch_summary(
            logger,
            epoch=epoch + 1,
            total_epochs=num_epochs,
            train_loss=avg_train_loss,
            val_loss=None,
            epoch_time=epoch_duration,
        )
        log_training_stats(
            epoch=epoch + 1,
            avg_loss=avg_train_loss,
            duration=epoch_duration,
            optimizer=optimizer,
            logger=logger,
            writer=writer,
            wandb_tracker=wandb_tracker,
        )

        # ── VALIDATION (if enabled) ⁚ early stopping
        if use_validation:
            model.eval()
            try:
                metrics = validate(
                    model=model,
                    val_loader=val_loader,
                    diffusion=diffusion,
                    device=device,
                    config=config,
                    logger=logger,
                    writer=writer,
                    wandb_tracker=wandb_tracker,
                )
            except Exception as ve:
                logger.warning(f"⚠️ Validation failed at epoch {epoch+1}: {ve}")
                metrics = {"val_mse": float("inf"), "val_fid": float("inf"),
                           "val_lpips": 0.0, "val_ssim": 0.0}

            current_val_mse = metrics["val_mse"]
            val_metrics_log.append(
                {
                    "epoch": epoch + 1,
                    "val_mse": metrics["val_mse"],
                    "val_fid": metrics["val_fid"],
                    "val_ssim": metrics["val_ssim"],
                }
            )

            # Early stopping logic
            if current_val_mse < best_val_mse:
                best_val_mse = current_val_mse
                patience_counter = 0

                # Save “best model” separately, NOT through ckpt_manager
                best_ckpt_path = os.path.join(save_path, "best_model.ckpt")
                torch.save(model.state_dict(), best_ckpt_path)
                log_json(
                    logger,
                    "💾 Best Model Saved",
                    epoch=epoch + 1,
                    path=best_ckpt_path,
                    val_mse=current_val_mse,
                )
                if wandb_tracker:
                    wandb_tracker.log({"val/best_mse": current_val_mse}, step=epoch + 1)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    break  # exit epoch loop early

            # Append to CSV
            metrics_csv_path = os.path.join(save_path, "validation_metrics.csv")
            if epoch == start_epoch:
                with open(metrics_csv_path, "w") as f:
                    f.write("epoch,val_mse,val_fid,val_lpips,val_ssim\n")
            with open(metrics_csv_path, "a") as f:
                last = val_metrics_log[-1]
                f.write(
                    f"{last['epoch']},{last['val_mse']:.6f},{last['val_fid']:.6f},"
                    f"{last['val_lpips']:.6f},{last['val_ssim']:.6f}\n"
                )

            model.train()

        # ── CHECKPOINT SAVING (rotate per interval) ⁚ ALWAYS keep last_k
        interval = int(config.get("checkpoint", {}).get("interval", 1))
        if (epoch + 1) % interval == 0 or (epoch + 1) == num_epochs:
            ckpt_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
            )
            ckpt_path = os.path.join(save_path, "checkpoints", f"epoch_{epoch+1:03d}.ckpt")
            log_json(logger, "💾 Checkpoint saved", epoch=epoch + 1, path=ckpt_path, lr=optimizer.param_groups[0]["lr"])
            if wandb_tracker:
                wandb_tracker.save(ckpt_path)

        # ── VISUALIZATION (samples, features, projections) ⁚ every vis_every
        vis_every = int(config["observability"].get("visualize_every", 5))
        if (epoch + 1) % vis_every == 0 or (epoch + 1) == num_epochs:
            try:
                model.eval()
                with torch.no_grad():
                    sample_shape = images.shape
                    gen_batch = diffusion.sample(model, sample_shape, device=device)
                log_visualizations(
                    epoch=epoch + 1,
                    real_batch=images.cpu(),
                    generated_batch=gen_batch.cpu(),
                    writer=writer,
                    wandb_tracker=wandb_tracker,
                    config=config["observability"],
                    vis_dir=vis_dir,
                    logger=logger,
                )

                if config["observability"].get("save_feature_maps", False):
                    from src.visualization.feature_maps import visualize_feature_maps
                    visualize_feature_maps(
                        model=ema.ema_model,
                        input_batch=images,
                        layers_to_hook={"bottleneck": model.downs[-1]},
                        device=device,
                        save_dir=feature_dir,
                        sample_id=epoch + 1,
                    )

                if config["observability"].get("save_tsne", False) or config["observability"].get("save_umap", False):
                    with torch.no_grad():
                        batch_size = images.size(0)
                        t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
                        if hasattr(model, "extract_bottleneck"):
                            bottleneck = model.extract_bottleneck(images, t_zero)
                            flat_feats = bottleneck.view(bottleneck.size(0), -1).cpu().numpy()
                            dummy_labels = [0] * flat_feats.shape[0]
                        else:
                            flat_feats = None
                            dummy_labels = None

                    if flat_feats is not None and config["observability"].get("save_tsne", False):
                        tsne_path = os.path.join(proj_dir, f"tsne_epoch{epoch+1:03d}.png")
                        plot_projection(flat_feats, dummy_labels, f"TSNE (Epoch {epoch+1})", tsne_path, method="tsne")

                    if flat_feats is not None and config["observability"].get("save_umap", False):
                        umap_path = os.path.join(proj_dir, f"umap_epoch{epoch+1:03d}.png")
                        plot_projection(flat_feats, dummy_labels, f"UMAP (Epoch {epoch+1})", umap_path, method="umap")

            except Exception as e:
                log_exception(logger, epoch=epoch + 1, exception=e)

    # ─── 8) PLOT & SAVE TRAINING CURVES ───────────────────────────────
    if loss_log:
        loss_plot = os.path.join(save_path, "loss_curve.png")
        plt.figure()
        plt.plot(range(1, len(loss_log) + 1), loss_log)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.title("Training Loss Curve")
        plt.savefig(loss_plot)
        plt.close()
        log_json(logger, "📊 Loss curve saved", path=loss_plot)
    else:
        logger.warning("No train-loss data to plot.")

    if loss_log or fid_log or lpips_log:
        results_path = os.path.join(save_path, "results.csv")
        with open(results_path, "w") as f:
            f.write("epoch,avg_loss,fid,lpips\n")
            max_len = max(len(loss_log), len(fid_log), len(lpips_log))
            for i in range(max_len):
                loss_val = loss_log[i] if i < len(loss_log) else ""
                fid_val = fid_log[i] if i < len(fid_log) else ""
                lpips_val = lpips_log[i] if i < len(lpips_log) else ""
                f.write(f"{i+1},{loss_val},{fid_val},{lpips_val}\n")
        log_json(logger, "📈 Saved results.csv", path=results_path)
    else:
        logger.warning("No train/metric data to write.")

    total_duration = time.time() - total_start_time
    log_training_end(
        logger,
        best_val_acc=0.0,  # adjust if tracking accuracy
        training_time=total_duration,
    )

    # ─── 9) SUCCESS ALERT ─────────────────────────────────────────────
    try:
        human_time = str(timedelta(seconds=int(total_duration)))
        alert_on_success(
            experiment_id=config.get("experiment_id", "N/A"),
            run_id=config.get("run_id", "N/A"),
            total_epochs=num_epochs,
            duration=human_time,
        )
    except Exception as e:
        logger.error(f"❌ Could not send success email: {e}")
