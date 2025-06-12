import os

import torch

from src.monitoring.email_alert_mailtrap import alert_on_failure
from src.monitoring.fail_safe_guard import fail_safe_guard
from src.train.utils.early_stopping import check_early_stopping
from src.train.validate_logic import validate
from src.train.logging.training_logger_utils import log_json, log_grad_norms, log_batch_progress, \
    log_training_stats, log_epoch_summary


def _save_checkpoint(epoch, num_epochs, interval, ckpt_manager, model, optimizer, scheduler, global_step, logger, wandb_tracker, save_path):
    if (epoch + 1) % interval == 0 or (epoch + 1) == num_epochs:
        ckpt_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
        )
        ckpt_path = os.path.join(save_path, "checkpoints", f"epoch_{epoch+1:03d}.ckpt")
        log_json(logger, "💾 Checkpoint saved", epoch=epoch+1, path=ckpt_path, lr=optimizer.param_groups[0]["lr"])
        if wandb_tracker:
            wandb_tracker.save(ckpt_path)

def _fail_safe_guard(epoch, batch_idx, loss, logger, save_path, config):
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


def log_batch(epoch, num_epochs, batch_idx, num_batches, loss, optimizer, logger, model, writer, wandb_tracker, global_step, device, config):
    if (batch_idx + 1) % int(config["training"].get("log_interval", 100)) == 0:
        log_batch_progress(
                logger, epoch=epoch+1, total_epochs=num_epochs,
                step=batch_idx+1, total_steps=num_batches, loss=loss,
                lr=optimizer.param_groups[0]["lr"],
                time_elapsed=0,  # Or actual time if desired
                gpu_mem=(torch.cuda.memory_allocated(device)/(1024**2)) if device.type == "cuda" else 0.0
            )
        log_grad_norms(logger, model, epoch=epoch+1, step=batch_idx+1, writer=writer, wandb_tracker=wandb_tracker)
        if wandb_tracker:
            wandb_tracker.log({
                "train/batch_loss": loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/gpu_mem_mb": (torch.cuda.memory_allocated(device)/(1024**2)) if device.type == "cuda" else 0.0
            }, step=global_step)

def log_epoch(epoch, num_epochs, avg_train_loss, optimizer, logger, writer, wandb_tracker, epoch_duration):
    log_epoch_summary(logger, epoch=epoch+1, total_epochs=num_epochs, train_loss=avg_train_loss, val_loss=None, epoch_time=epoch_duration)
    log_training_stats(epoch=epoch+1, avg_loss=avg_train_loss, duration=epoch_duration, optimizer=optimizer, logger=logger, writer=writer, wandb_tracker=wandb_tracker)

def log_validation_metrics(logger, metrics_dict, epoch=None, writer=None, wandb_tracker=None):
    prefix = f"[Epoch {epoch}] " if epoch is not None else ""
    parts = [f"{k}: {v:.4f}" for k, v in metrics_dict.items()]
    logger.info(prefix + "Validation | " + " | ".join(parts))
    if writer and epoch is not None:
        for k, v in metrics_dict.items():
            writer.add_scalar(f"val/{k}", v, epoch)
    if wandb_tracker and epoch is not None:
        wandb_tracker.log({f"val/{k}": v for k, v in metrics_dict.items()}, step=epoch)

def validation(model, val_loader, diffusion, device, config, logger, writer, wandb_tracker, epoch, best_val_mse, patience_counter, early_stop_patience, save_path):
    model.eval()
    try:
        metrics = validate(
            model=model, val_loader=val_loader, diffusion=diffusion, device=device,
            config=config, logger=logger, writer=writer, wandb_tracker=wandb_tracker
        )
    except Exception as ve:
        logger.warning(f"⚠️ Validation failed at epoch {epoch+1}: {ve}")
        metrics = {"val_mse": float("inf"), "val_fid": float("inf"), "val_ssim": 0.0}
    current_val_mse = metrics["val_mse"]

    best_val_mse, patience_counter, stop_flag = check_early_stopping(
        current_val_mse, best_val_mse, patience_counter, early_stop_patience
    )
    if current_val_mse == best_val_mse:
        best_ckpt_path = os.path.join(save_path, "best_model.ckpt")
        torch.save(model.state_dict(), best_ckpt_path)
        log_json(logger, "💾 Best Model Saved", epoch=epoch+1, path=best_ckpt_path, val_mse=current_val_mse)
        if wandb_tracker:
            wandb_tracker.log({"val/best_mse": current_val_mse}, step=epoch+1)
    if stop_flag:
        logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
    return metrics, best_val_mse, patience_counter, stop_flag
