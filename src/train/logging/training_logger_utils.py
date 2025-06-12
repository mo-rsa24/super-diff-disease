# src/utils/training_logger_utils.py

import os
from datetime import timedelta

import torch
from matplotlib import pyplot as plt

from .wandb_utils import log_images_wandb, log_table_wandb, log_metrics_wandb
from .tensorboard_utils import log_images_tb
from .projection_utils import plot_projection_tsne, plot_projection_umap
from .feature_map_utils import visualize_feature_maps

def log_training_start(logger, model_name, experiment_id, run_id, total_epochs, total_batches):
    logger.info(
        f"🚀 Training started\n"
        f"🔧 Model: {model_name}\n"
        f"🧪 Experiment: {experiment_id}\n"
        f"🏃‍♂️ Run ID: {run_id}\n"
        f"⏳ Total Epochs: {total_epochs}, Batches/Epoch: {total_batches}"
    )

def log_epoch_start(epoch, logger):
    logger.info(f"\n--- Epoch {epoch+1} Start ---")

def log_training_setup(optimizer, scheduler, amp_enabled, ema_enabled, logger):
    logger.info("--- Training Setup ---")
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    if scheduler:
        logger.info(f"Scheduler: {type(scheduler).__name__}")
    else:
        logger.info("Scheduler: None")
    logger.info(f"AMP Enabled: {amp_enabled}")
    logger.info(f"EMA Enabled: {ema_enabled}")
    logger.info("----------------------")


def log_epoch_summary(logger, epoch, total_epochs, train_loss, val_loss=None, epoch_time=None):
    msg = f"Epoch {epoch}/{total_epochs} completed | Train Loss: {train_loss:.4f}"
    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:.4f}"
    if epoch_time is not None:
        msg += f" | Time: {timedelta(seconds=int(epoch_time))}"
    logger.info(msg)



def log_training_end(logger, best_val_acc, training_time):
    logger.info(
        f"🏁 Training finished!\n"
        f"🔝 Best Val Accuracy: {best_val_acc:.2f}%\n"
        f"🕒 Total Duration: {training_time:.2f}s"
    )


def log_batch_progress(logger, epoch, total_epochs, step, total_steps, loss, lr, time_elapsed, gpu_mem=None):
    hrs, rem = divmod(int(time_elapsed), 3600)
    mins, secs = divmod(rem, 60)
    elapsed_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    msg = (
        f"[Epoch {epoch}/{total_epochs}] "
        f"Step {step}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | Elapsed: {elapsed_str}"
    )
    if gpu_mem is not None:
        msg += f" | GPU_mem: {gpu_mem:.0f} MB"
    logger.info(msg)

def log_grad_norms(logger, model, epoch=None, step=None, writer=None, wandb_tracker=None):
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2).item()
            total_norm_sq += param_norm ** 2
    total_norm = total_norm_sq ** 0.5
    tag = "train/grad_norm"
    if epoch is not None and step is not None:
        logger.info(f"📏 [Epoch {epoch} Step {step}] Grad Norm: {total_norm:.4f}")
    elif epoch is not None:
        logger.info(f"📏 [Epoch {epoch}] Grad Norm: {total_norm:.4f}")
    else:
        logger.info(f"📏 Grad Norm: {total_norm:.4f}")
    if writer and epoch is not None:
        writer.add_scalar(tag, total_norm, step or epoch)
    if wandb_tracker:
        wandb_tracker.log({tag: total_norm}, step or epoch)

def log_training_stats(logger, epoch, avg_loss, duration, optimizer, writer=None, wandb_tracker=None):
    lr = optimizer.param_groups[0]["lr"]
    logger.info(
        f"【Epoch {epoch}】 Avg Loss: {avg_loss:.4f} | Duration: {timedelta(seconds=int(duration))} | LR: {lr:.2e}"
    )
    if writer:
        writer.add_scalar("train/avg_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.add_scalar("train/epoch_duration", duration, epoch)
    if wandb_tracker:
        wandb_tracker.log({
            "train/avg_loss": avg_loss,
            "train/lr": lr,
            "train/epoch_duration": duration
        }, step=epoch)


def log_batch(step, loss, lr, logger, writer=None, wandb_tracker=None):
    logger.info(f"Step {step} | Loss: {loss:.4f} | LR: {lr:.6f}")
    if writer:
        writer.add_scalar("Loss/step", loss, step)
        writer.add_scalar("LR/step", lr, step)
    if wandb_tracker:
        log_metrics_wandb(wandb_tracker, {"loss/step": loss, "lr": lr}, step=step)

def log_json(logger, message, **extra_data):
    logger.info(message, extra={"extra_data": extra_data})

def log_exception(logger, epoch=None, step=None, exception=None):
    msg = "❌ Exception during training"
    if epoch is not None:
        msg += f" at Epoch {epoch}"
    if step is not None:
        msg += f", Step {step}"
    logger.exception(f"{msg}: {str(exception)}")

def log_config_summary(config, logger):
    import yaml
    logger.info("Experiment configuration:\n" + yaml.dump(config, sort_keys=False))

def visualize_epoch(
    epoch,
    num_epochs,
    model,
    images,
    diffusion,
    writer,
    wandb_tracker,
    config,
    logger,
    vis_dir,
    proj_dir,
    feature_dir,
    device,
    ema_model=None,
):
    """
    Centralized visualization and logging for a training epoch.
    """
    vis_every = int(config["observability"].get("visualize_every", 5))
    do_visualize = ((epoch + 1) % vis_every == 0) or ((epoch + 1) == num_epochs)
    if not do_visualize:
        return
        # Ensure dirs
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    try:
        # 1. Generate a batch of denoised samples
        model.eval()
        with torch.no_grad():
            sample_shape = images.shape
            gen_batch = diffusion.sample(model, sample_shape, device=device)
        save_real_vs_generated(images.cpu(), gen_batch.cpu(), epoch + 1, vis_dir, n=min(8, images.size(0)))
        # 2. Sample pairs, batch grids, logging
        if config["observability"].get("log_batch_grid", True):
            log_images_tb(writer, images, gen_batch, epoch+1)
            log_images_wandb(wandb_tracker, images, gen_batch, epoch+1)
        if config["observability"].get("wandb_sample_table", False):
            log_table_wandb(wandb_tracker, images, gen_batch, epoch+1)

        # 3. (Optional) Feature Map Visualization
        if config["observability"].get("save_feature_maps", False) and ema_model is not None:
            visualize_feature_maps(
                model=ema_model,
                input_batch=images,
                device=device,
                save_dir=feature_dir,
                sample_id=epoch + 1,
            )

        if config["observability"].get("save_tsne", False):
            with torch.no_grad():
                batch_size = images.size(0)
                t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
                if hasattr(model, "extract_bottleneck"):
                    bottleneck = model.extract_bottleneck(images, t_zero)
                    flat_feats = bottleneck.view(bottleneck.size(0), -1).cpu().numpy()
                    dummy_labels = [0] * flat_feats.shape[0]
                else:
                    flat_feats, dummy_labels = None, None

            if flat_feats is not None and config["observability"].get("save_tsne", False):
                tsne_path = os.path.join(proj_dir, f"tsne_epoch{epoch+1:03d}.png")
                plot_projection_tsne(flat_feats, dummy_labels, f"TSNE (Epoch {epoch+1})", tsne_path, logger)
            if flat_feats is not None and config["observability"].get("save_umap", False):
                umap_path = os.path.join(proj_dir, f"umap_epoch{epoch+1:03d}.png")
                plot_projection_umap(flat_feats, dummy_labels, f"UMAP (Epoch {epoch+1})", umap_path, logger)

        # 5. (Optional) Diffusion process visualization (stub; implement as needed)
        if config["observability"].get("save_diffusion_process", False):
            visualize_diffusion_process(
                model=ema_model if ema_model is not None else model,
                diffusion=diffusion,
                image=images[0],  # Take the first image of the batch
                device=device,
                epoch=epoch + 1,
                save_dir=vis_dir,
                logger=logger,
                writer=writer,
                wandb_tracker=wandb_tracker,
                timesteps=None,
            )

    except Exception as e:
        log_exception(logger, epoch=epoch+1, exception=e)



def visualize_diffusion_process(
    model,
    diffusion,
    image,
    device,
    epoch,
    save_dir,
    logger=None,
    writer=None,
    wandb_tracker=None,
    timesteps=None,
):
    """
    Visualizes the denoising process for a single input image across diffusion timesteps.
    Saves a matplotlib grid to save_dir and optionally logs to TensorBoard/W&B.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    image = image.unsqueeze(0).to(device)
    T = getattr(diffusion, "num_timesteps", 10)
    if timesteps is None:
        timesteps = torch.linspace(0, T-1, steps=min(8, T)).long().tolist()

    imgs = []
    with torch.no_grad():
        noisy = diffusion.q_sample(image, torch.tensor([timesteps[0]], device=device)) if hasattr(diffusion, "q_sample") else image
        for t in timesteps:
            # Use your diffusion's appropriate sampling function:
            if hasattr(diffusion, "sample_step"):
                step_img = diffusion.sample_step(model, noisy, t, device=device)
            elif hasattr(diffusion, "p_sample"):
                step_img = diffusion.p_sample(model, noisy, t, device=device)
            else:
                # fallback: just use sample at different t (not perfect, but works)
                step_img = diffusion.sample(model, image.shape, device=device, t_override=t)
            imgs.append(step_img.squeeze().detach().cpu())

    # Plot real vs. denoise grid
    n_cols = len(imgs)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols*2.2, 2.5))
    for i, img in enumerate(imgs):
        ax = axes[i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"t={timesteps[i]}")
        ax.axis('off')
    plt.suptitle(f"Diffusion process (epoch {epoch})")
    fname = os.path.join(save_dir, f"diffusion_process_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)
    if logger:
        logger.info(f"Saved diffusion process grid at {fname}")

    # TensorBoard logging
    if writer is not None:
        grid = torch.stack(imgs)
        writer.add_images("DiffusionProcess", grid, epoch)

    # W&B logging
    if wandb_tracker is not None:
        import wandb
        images_wandb = [wandb.Image(img, caption=f"t={t}") for img, t in zip(imgs, timesteps)]
        wandb_tracker.log({f"diffusion_process/epoch_{epoch}": images_wandb}, step=epoch)


def save_real_vs_generated(
    real_batch, gen_batch, epoch, save_dir, n=8
):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0, i].imshow(real_batch[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Real {i}")
        axes[0, i].axis('off')
        axes[1, i].imshow(gen_batch[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f"Gen {i}")
        axes[1, i].axis('off')
    plt.suptitle(f"Real vs Generated - Epoch {epoch}")
    plt.tight_layout()
    fpath = os.path.join(save_dir, f"real_vs_generated_epoch_{epoch:03d}.png")
    plt.savefig(fpath)
    plt.close(fig)