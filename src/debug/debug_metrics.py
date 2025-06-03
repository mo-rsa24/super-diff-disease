# debug_metrics.py
import torch
from src.metrics.fid_lpips import compute_fid_chexnet, compute_lpips, compute_ssim

# Dummy grayscale tensors
real = torch.rand(8, 1, 256, 256)
fake = torch.rand(8, 1, 256, 256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
real, fake = real.to(device), fake.to(device)

print("=== Running Metric Debug ===")

try:
    fid = compute_fid_chexnet(real, fake)
    print("✅ MedFID:", round(fid, 4))
except Exception as e:
    print("❌ MedFID error:", e)

try:
    lp = compute_lpips(real, fake)
    print("✅ LPIPS:", round(lp, 4))
except Exception as e:
    print("❌ LPIPS error:", e)

try:
    ssim = compute_ssim(real, fake)
    print("✅ SSIM:", round(ssim, 4))
except Exception as e:
    print("❌ SSIM error:", e)
