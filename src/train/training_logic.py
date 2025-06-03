# training_logic.py
from datetime import datetime, timedelta
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import os, torch, time
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from ema_pytorch import EMA
from src.monitoring import fail_safe_guard
from src.monitoring.email_alert_mailtrap import alert_on_failure, alert_on_success
from src.train.validate_logic import validate
from src.utils.training_logger_utils import log_grad_norms, log_json, log_scheduler_step, log_validation_metrics, plot_projection, log_visualizations, log_batch_progress, log_epoch_summary, log_exception, log_training_end, log_training_start, log_training_stats
from src.utils.training_self_checker import TrainingSelfChecker
from src.visualization.feature_maps import visualize_feature_maps 

def save_checkpoint(state, path):
    torch.save(state, path)

def get_path(base, *args):
    return os.path.join(base, *args)

def load_checkpoint(path, model, optimizer, scheduler=None, ema=None, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if ema and "ema" in checkpoint:
        ema.ema_model.load_state_dict(checkpoint["ema"])
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    return epoch, global_step

def train(model, dataloader, diffusion, num_epochs, save_path, device,
          logger, config, writer=None, wandb_tracker=None, val_loader=None):
    """
    Full training loop for Super-Diff in training_logic.py.
    Expects `config` to be a dict loaded from config.yaml.

    Structure:
     1) Seed & device setup
     2) Model, optimizer, scheduler, AMP, EMA initialization
     3) Resume logic (if any)
     4) Dataloader creation
     5) Directory creation (checkpoints, samples, etc.)
     6) Self‐checker instantiation
     7) Epoch loop: train, validate, checkpoint, early stop
    """
    # ─── 1. SEED & DEVICE SETUP ─────────────────────────────────────────────────────────
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    model.to(device)
    
    # ─── 2) LOGGING INITIALIZATION ─────────────────────────────────────────────
    total_batches = len(dataloader)
    log_training_start(
        logger,
        model_name=type(model).__name__,
        experiment_id=config.get("experiment_id", "N/A"),
        run_id=config.get("run_id", "N/A"),
        total_epochs=num_epochs,
        total_batches=total_batches
    )
    
    # ─── 3) OPTIMIZER, SCHEDULER, AMP, EMA ───────────────────────────────────────
    opt_name = config["training"].get("optimizer", "adam").lower()
    lr = eval(config["training"].get("learning_rate", 1e-4)) if type(config["training"].get("learning_rate", 1e-4)) is str else config["training"].get("learning_rate", 1e-4)
    beta1 = config["training"].get("beta1", 0.9)
    beta2 = config["training"].get("beta2", 0.999)
    weight_decay = config["training"].get("weight_decay", 0.0)
    
    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    else:  # default to Adam
        optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    scheduler = None
    sched_cfg = config["training"].get("scheduler", None)
    if sched_cfg:
        sched_type = sched_cfg.get("type", "step").lower()
        if sched_type == "step":
            step_size = sched_cfg.get("step_size", 10)
            gamma = sched_cfg.get("gamma", 0.1)
            # Ensure correct types
            if isinstance(step_size, str):
                step_size = int(step_size)
            if isinstance(gamma, str):
                gamma = float(gamma)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif sched_type == "cosine":
            t_max = sched_cfg.get("t_max", num_epochs)
            eta_min = sched_cfg.get("eta_min", 0.0)
            # Ensure correct types
            if isinstance(t_max, str):
                t_max = int(t_max)
            if isinstance(eta_min, str):
                eta_min = float(eta_min)
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")
    
    # AMP
    use_amp = config["training"].get("use_amp", True) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    
    # EMA
    ema_beta = config["training"].get("ema_beta", 0.995)
    ema = EMA(model, beta=ema_beta).to(device)

    # Ensure checkpoint dir exists
    checkpoint_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    resume_path = config.get("resume_checkpoint", "")
    if not resume_path:
        # Auto‐detect newest .ckpt if no explicit resume
        all_ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if all_ckpts:
            all_ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            resume_path = os.path.join(checkpoint_dir, all_ckpts[0])
            logger.info(f"📦 Auto‐detected checkpoint: {resume_path}")
            config["resume_checkpoint"] = resume_path

    start_epoch = 0
    global_step = 0

    if resume_path:
        try:
            logger.info(f"📦 Attempting to resume from: {resume_path}")
            loaded_epoch, loaded_step = load_checkpoint(
                resume_path, model, optimizer, scheduler=scheduler, ema=ema, device=device
            )
            start_epoch = loaded_epoch + 1  # next epoch
            global_step = loaded_step
            logger.info(f"🔄 Resumed at epoch {loaded_epoch+1}, global_step {loaded_step}")
            
            # Optional quick validation after resume
            if config.get("validation", {}).get("run_after_resume", False) and val_loader:
                model.eval()
                try:
                    avg_val_loss = validate(
                        model=model,
                        val_loader=val_loader,
                        diffusion=diffusion,
                        device=device,
                        config=config,
                        logger=logger,
                        writer=writer,
                        wandb_tracker=wandb_tracker
                    )
                    logger.info(f"✅ Validation after resume: {avg_val_loss:.4f}")
                    log_json(logger, "🔄 Resume validation", epoch=loaded_epoch+1, val_loss=avg_val_loss)
                except Exception as ve:
                    logger.warning(f"⚠️ Validation after resume failed: {ve}")
                model.train()
        except Exception as e:
            logger.error(f"❌ Resume failed: {e}")
            fail_safe_guard(
                epoch=0,
                loss=torch.tensor(0.0),
                run_id=config.get("run_id", "N/A"),
                config=config,
                checkpoint_path=resume_path,
                enable_alerts=config["monitoring"]["alerts"]["enable"],
                resume=True,
                error_msg=str(e)
            )
            return  # Gracefully abort
    model.to(device)

    # ─── 6) SET UP SELF‐CHECKER ──────────────────────────────────────────────────────
    # Monitors gradient explosions or loss spikes
    self_checker = TrainingSelfChecker(
        run_id=config.get("run_id", "N/A"),
        save_dir=save_path,
        checkpoint_keys=["epoch", "global_step", "model", "optimizer", "ema", "scheduler"],
        grad_clip=config["training"].get("grad_clip_norm", 1.0),
        loss_threshold=config["monitoring"].get("max_loss_threshold", 100.0)
    )

    # ─── 5) PREPARE OTHER DIRECTORIES ──────────────────────────────────────────────
    vis_dir = os.path.join(save_path, "samples")
    feature_dir = os.path.join(save_path, "features")
    proj_dir = os.path.join(save_path, "projections")
    metric_dir = os.path.join(save_path, "metrics")
    for sub in [vis_dir, feature_dir, proj_dir, metric_dir]:
        os.makedirs(sub, exist_ok=True)


    # ─── 7) EPOCH LOOP ───────────────────────────────────────────────────────────────
    early_stop_patience = config["training"].get("early_stopping_patience", 10)
    use_validation = config["training"].get("use_validation", False) and (val_loader is not None)
    best_val_loss = float("inf")
    patience_counter = 0

    loss_log, fid_log, lpips_log = [], [], []

    total_start_time = time.time()
    val_metrics_log = []
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss  = 0.0
        
        # If DataLoader uses DistributedSampler, set epoch for shuffling
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
            
        log_json(logger, f"🔁 Starting Epoch {epoch+1}", epoch=epoch+1, total_epochs=num_epochs)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad()
            
             # ——————————— AMP context ———————————
            with torch.cuda.amp.autocast(enabled=use_amp):
                # This line does exactly:
                #  1) t = torch.randint(0, T, (bsz,), device=device)
                #  2) noise = torch.randn_like(images)
                #  3) x_noisy = diffusion.q_sample(images, t, noise)
                #  4) predicted_noise = model(x_noisy, t)
                #  5) return F.mse_loss(predicted_noise, noise)
                loss = diffusion.training_step(model, images)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"❌ NaN/Inf detected in loss at epoch {epoch+1}, batch {batch_idx}")

            # Self‐check raw loss
            self_checker.check_loss(loss, epoch, batch_idx, logger)
            
            # ── Backward + Optimizer Step ──────────────────────────────────
            scaler.scale(loss).backward()

            grad_clip_val = config["training"].get("grad_clip_norm", 1.0)
            if grad_clip_val > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_val)

            scaler.step(optimizer)
            scaler.update()

            self_checker.check_gradients(model, epoch, batch_idx, logger)

            # ── EMA Update ───────────────────────────────────────────────
            ema.update()
            
            # ── Scheduler Step ───────────────────────────────────────────
            if scheduler is not None and sched_cfg.get("step_per", "batch") == "batch":
                scheduler.step()
                log_scheduler_step(logger, optimizer, epoch=epoch+1, step=batch_idx+1, writer=writer, wandb_tracker=wandb_tracker)
            
            # ── Logging ─────────────────────────────────────────────────
            epoch_loss += loss.item()
            if (batch_idx + 1) % config["training"]["log_interval"] == 0:
                elapsed = time.time() - epoch_start_time
                gpu_mem_mb = None
                if device.type == "cuda":
                    try:
                        gpu_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    except Exception:
                        gpu_mem_mb = None

                # 1) Console + JSON logging for batch progress
                log_batch_progress(
                    logger,
                    epoch=epoch + 1,
                    total_epochs=num_epochs,
                    step=batch_idx + 1,
                    total_steps=len(dataloader),
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    time_elapsed=elapsed,
                    gpu_mem=gpu_mem_mb
                )
                log_json(
                    logger,
                    "Batch Progress",
                    epoch=epoch + 1,
                    step=batch_idx + 1,
                    total_steps=len(dataloader),
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    time_elapsed=elapsed,
                    gpu_mem=gpu_mem_mb
                )

                # 2) Log gradient norm (so we can track exploding/vanishing grads)
                if config["observability"].get("log_grad_norm", True):
                    log_grad_norms(
                        logger, model, epoch=epoch + 1, step=batch_idx + 1,
                        writer=writer, wandb_tracker=wandb_tracker
                    )

                # 3) W&B logging for batch
                if wandb_tracker:
                    wandb_tracker.log({
                        "train/batch_loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/gpu_mem_mb": gpu_mem_mb or 0.0
                    }, step=global_step + batch_idx + 1,)

            # ── Fail‐Safe & Alerts ────────────────────────────────────────
            if (batch_idx + 1) % config["monitoring"]["loss_monitoring"]["check_frequency"] == 0:
                fail_flag, reason = fail_safe_guard(
                    epoch=epoch+1,
                    loss=loss,
                    run_id=config.get("run_id", "N/A"),
                    config=config,
                    checkpoint_path=save_path,
                    enable_alerts=config["monitoring"]["alerts"]["enable"]
                )
                if fail_flag:
                    logger.error(f"🛑 FAIL-SAFE TRIGGERED: {reason}")
                    try:
                        alert_on_failure(
                            run_id=config.get("run_id", "N/A"),
                            last_epoch=epoch+1,
                            error_msg=reason
                        )
                    except Exception as mail_e:
                        logger.error(f"❌ Could not send failure alert: {mail_e}")
                    return  # Abort training altogether
    
            break
        global_step += len(dataloader)
        # — Scheduler Step (per‐epoch) —
        if scheduler is not None and sched_cfg.get("step_per", "batch") == "epoch":
            scheduler.step()
            log_scheduler_step(
                logger, optimizer, epoch=epoch + 1,
                writer=writer, wandb_tracker=wandb_tracker
            )
    
        avg_train_loss = epoch_loss / float(len(dataloader))
        epoch_duration = time.time() - epoch_start_time

        # — Epoch‐level logging —
        log_epoch_summary(
            logger,
            epoch=epoch + 1,
            total_epochs=num_epochs,
            train_loss=avg_train_loss,
            val_loss=None,
            epoch_time=epoch_duration
        )
        log_training_stats(
            epoch=epoch + 1,
            avg_loss=avg_train_loss,
            duration=epoch_duration,
            optimizer=optimizer,
            logger=logger,
            writer=writer,
            wandb_tracker=wandb_tracker
        )
        
        
        # ── VALIDATION ────────────────────────────────────────────
        if use_validation and val_loader is not None:
            model.eval()
            try:
                # validate(...) now returns a dict: {"val_mse":…, "val_fid":…, "val_ssim":…}
                validation_metrics = validate(
                    model=model,
                    val_loader=val_loader,
                    diffusion=diffusion,
                    device=device,
                    config=config,
                    logger=logger,
                    writer=writer,
                    wandb_tracker=wandb_tracker
                )
            except Exception as ve:
                logger.warning(f"⚠️ Validation failed at epoch {epoch+1}: {ve}")
                validation_metrics = {"val_mse": float("inf"), "val_fid": float("inf"), "val_ssim": 0.0}

            # 1) Early‐stopping logic on MSE (or you could use FID instead)
            current_val_mse = validation_metrics["val_mse"]
            if current_val_mse < best_val_loss:
                best_val_loss = current_val_mse
                patience_counter = 0
                # save “best” model snapshot
                best_ckpt = os.path.join(checkpoint_dir, "best_model.ckpt")
                torch.save(model.state_dict(), best_ckpt)
                log_json(logger, "Best Model", epoch=epoch+1, path=best_ckpt, val_mse=current_val_mse)
                if wandb_tracker:
                    wandb_tracker.log({"val/best_mse": current_val_mse}, step=epoch + 1)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break

            # 2) Save validation metrics so that we can write them to CSV later
            #    (Append to some per‐epoch list, e.g. val_metrics_log = [])
            val_metrics_log.append({
                "epoch": epoch + 1,
                "val_mse": validation_metrics["val_mse"],
                "val_fid": validation_metrics["val_fid"],
                "val_ssim": validation_metrics["val_ssim"]
            })

            model.train()
            
            # ── (Optional) Save per‐epoch metrics CSV ───────────────────
            # You can do this every epoch or only at the very end.
            metrics_csv_path = os.path.join(save_path, "validation_metrics.csv")
            if epoch == start_epoch:
                # First epoch: write header
                with open(metrics_csv_path, "w") as f:
                    f.write("epoch,val_mse,val_fid,val_ssim\n")
            with open(metrics_csv_path, "a") as f:
                if val_metrics_log:
                    last_row = val_metrics_log[-1]
                    f.write(f"{last_row['epoch']},{last_row['val_mse']:.6f},{last_row['val_fid']:.6f},{last_row['val_ssim']:.6f}\n")

            # — Early Stopping Logic —
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Optionally save a “best” snapshot
                best_ckpt = os.path.join(checkpoint_dir, "best_model.ckpt")
                torch.save(model.state_dict(), best_ckpt)
                log_json(logger, "Best Model Saved", epoch=epoch + 1, path=best_ckpt, val_loss=avg_val_loss)
                if wandb_tracker:
                    wandb_tracker.log({"val/best_loss": avg_val_loss}, step=epoch + 1)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        # 6b. Checkpoint saving every n epochs (interval from config)
        interval = config.get("checkpoint", {}).get("interval", 1)
        if (epoch + 1) % interval == 0 or (epoch + 1) == num_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.ckpt")
            ckpt_state = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.ema_model.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            save_checkpoint(ckpt_state, ckpt_path)
            log_json(logger, "💾 Checkpoint saved", epoch=epoch+1, path=ckpt_path, lr=optimizer.param_groups[0]["lr"])
            if wandb_tracker:
                wandb_tracker.save(ckpt_path)

            # Rotate old files
            keep_k = config.get("checkpoint", {}).get("keep_last_k", 3)
            all_ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")])
            if len(all_ckpts) > keep_k:
                for old in all_ckpts[: len(all_ckpts) - keep_k]:
                    os.remove(os.path.join(checkpoint_dir, old))


        # Visualization
        vis_every = config["observability"].get("visualize_every", 5)
        if (epoch + 1) % vis_every == 0 or (epoch + 1) == num_epochs:
            try:
                model.eval()
                with torch.no_grad():
                    # Assume `images` is the last minibatch from above
                    sample_shape = images.shape
                    gen_batch = diffusion.sample(ema.ema_model, image_shape=sample_shape, device=device)
                log_visualizations(
                   epoch=epoch + 1,
                    real_batch=images.cpu(),
                    generated_batch=gen_batch.cpu(),
                    writer=writer,
                    wandb_tracker=wandb_tracker,
                    config=config["observability"],
                    vis_dir=vis_dir,
                    logger=logger
                    )
                
                if config["observability"].get("save_feature_maps", False):
                    visualize_feature_maps(
                        model=ema.ema_model,
                        input_batch=images,
                        layers_to_hook={"bottleneck": model.downs[-1]},
                        device=device,
                        save_dir=os.path.join(feature_dir),
                        sample_id=epoch+1
                    )

                if config["observability"].get("save_tsne", False) or config["observability"].get("save_umap", False):
                    with torch.no_grad():
                        batch_size = images.size(0)
                        t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
                        bottleneck = model.extract_bottleneck(images, t_zero)
                        flat_feats = bottleneck.view(bottleneck.size(0), -1).cpu().numpy()
                        dummy_labels = np.zeros(flat_feats.shape[0])  # or actual labels if available

                    if config["observability"].get("save_tsne", False):
                        tsne_path = os.path.join(proj_dir, f"tsne_epoch{epoch+1:03d}.png")
                        plot_projection(flat_feats, dummy_labels, f"TSNE Projection (Epoch {epoch+1})", tsne_path, method="tsne")

                    if config["observability"].get("save_umap", False):
                        umap_path = os.path.join(proj_dir, f"umap_epoch{epoch+1:03d}.png")
                        plot_projection(flat_feats, dummy_labels, f"UMAP Projection (Epoch {epoch+1})", umap_path, method="umap")


            except Exception as e:
                log_exception(logger, epoch=epoch+1, exception=e)


    loss_plot = os.path.join(save_path, "loss_curve.png")
    plt.plot(range(1, num_epochs+1), loss_log)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(); plt.title("Loss Curve")
    plt.savefig(loss_plot); plt.close()
    log_json(logger, "📊 Loss curve saved", path=loss_plot)

    results_path = os.path.join(save_path, "results.csv")
    with open(results_path, "w") as f:
        f.write("epoch,avg_loss,fid,lpips\n")
        for i in range(num_epochs):
            fid_val = fid_log[i] if i < len(fid_log) else ''
            lpips_val = lpips_log[i] if i < len(lpips_log) else ''
            f.write(f"{i+1},{loss_log[i]},{fid_val},{lpips_val}\n")
    log_json(logger, "📈 Saved results.csv", path=results_path)

    # archive_file = os.path.join(save_path, "run_summary.tar.gz")
    # log_json(logger, "🗂️ Archived run folder", archive=archive_file)

    total_duration = time.time() - total_start_time
    log_training_end(
            logger,
            best_val_acc=0.0,
            training_time=total_duration
        )
     # 8. Send success email alert
    try:
        human_readable_duration = str(timedelta(seconds=int(total_duration)))
        alert_on_success(
            run_id=config.get("run_id", "N/A"),
            total_epochs=num_epochs,
            duration=human_readable_duration
        )
        from src.reporting.paper_report_generator import generate as generate_report

        generate_report(run_dir=os.path.join(save_path, "report"))
    except Exception as e:
        logger.error(f"❌ Could not send success email: {e}")



