# validate_logic.py

import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.metrics.fid_lpips import compute_fid_chexnet, compute_ssim  # see note below


def validate(
    model: torch.nn.Module,
    val_loader:DataLoader,
    diffusion,
    device: torch.device,
    config: dict,
    logger,
    writer=None,
    wandb_tracker=None
) -> dict:
    """
    Run one deterministic validation pass. Returns a dict of metrics: 
      { "val_mse": float, "val_fid": float, "val_ssim": float }.

    We fix the RNG seed at the start so that q_sample(...) / sample(...) are reproducible.
    """

    # ─── A) SET FIXED SEED FOR VALIDATION ────────────────────────────────────────
    val_seed = config.get("validation", {}).get("seed", 0)
    torch.manual_seed(val_seed)
    np.random.seed(val_seed)
    random.seed(val_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(val_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()

    all_real = []
    all_gen = []
    total_mse = 0.0
    num_images = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="🔍 Validation"):
            # 1) Move real images to device
            real_imgs = batch["image"].to(device, non_blocking=True)
            bsz = real_imgs.size(0)
            num_images += bsz

            # 2) Compute MSE loss (use the same training_step logic but wrapped in no_grad)
            #    Because training_step itself samples a fresh random t+noise, we do it here:
            with torch.no_grad():
                mse_loss = diffusion.training_step(model, real_imgs)
            total_mse += mse_loss.item() * bsz

            torch.manual_seed(val_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(val_seed)

            # We assume diffusion.sample(model, image_shape=(B,C,H,W), device) -> (B,C,H,W)
            gen_imgs = diffusion.sample(
                model, image_shape=real_imgs.shape, device=device
            )

            # 4) Move both real and generated to CPU (for metric computation)
            all_real.append(real_imgs.detach().cpu())
            all_gen.append(gen_imgs.detach().cpu())

    # ─── B) COMPUTE AVERAGE MSE OVER VALIDATION SET ───────────────────────────────
    avg_mse = total_mse / float(num_images)
    all_real = torch.cat(all_real, dim=0)
    all_gen = torch.cat(all_gen, dim=0)

    fid_val = compute_fid_chexnet(all_real, all_gen)
    ssim_val = compute_ssim(all_real, all_gen)

    # ─── D) LOG METRICS ─────────────────────────────────────────────────────────
    metrics = {
        "val_mse": avg_mse,
        "val_fid": fid_val,
        "val_ssim": ssim_val
    }
    from src.train.utils.handlers import log_validation_metrics
    log_validation_metrics(
        logger,
        metrics,
        epoch=None,           # if you want to pass an epoch, set it when you call validate from training loop
        writer=writer,
        wandb_tracker=wandb_tracker
    )

    model.train()
    return metrics
