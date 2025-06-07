import numpy as np
import torch
import lpips
from scipy import linalg
from torchvision.models import inception_v3
from torchvision.transforms import Resize, Normalize, Compose
from skimage.metrics import structural_similarity as ssim_skimage

# Inside fid_lpips.py
from src.models.feature_extractor import get_chexnet_densenet121_feature_extractor

@torch.no_grad()
def compute_fid_chexnet(real_images, generated_images):
    device = real_images.device
    extractor = get_chexnet_densenet121_feature_extractor()

    real_feats = extractor(real_images)
    gen_feats = extractor(generated_images)

    act1 = real_feats.cpu().numpy()
    act2 = gen_feats.cpu().numpy()

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))



@torch.no_grad()
def compute_lpips(real_images, generated_images, model="alex"):
    loss_fn = lpips.LPIPS(net=model).to(real_images.device).eval()
    if real_images.shape[1] == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
        generated_images = generated_images.repeat(1, 3, 1, 1)
    return loss_fn(real_images, generated_images).mean().item()

@torch.no_grad()
def compute_ssim(real_images, generated_images):
    """
    Compute SSIM score on grayscale X-rays (expects [B, 1, H, W]).
    Returns average SSIM over the batch.
    """
    real_images = real_images.cpu().numpy()
    generated_images = generated_images.cpu().numpy()
    
    scores = []
    for real, gen in zip(real_images, generated_images):
        ssim_score = ssim_skimage(real[0], gen[0], data_range=1.0)
        scores.append(ssim_score)
    return float(np.mean(scores))