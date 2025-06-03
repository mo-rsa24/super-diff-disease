import torch
from torchvision.utils import make_grid, save_image

def visualize_denoising_trajectory(diffusion, model, image_shape, device, steps, save_path):
    model.eval()
    trajectory = []
    x = torch.randn(image_shape).to(device)
    for t in steps:
        with torch.no_grad():
            x = diffusion.denoise_step(model, x, t)
            trajectory.append(x.cpu().clone())
    grid = make_grid(torch.cat(trajectory, dim=0), nrow=len(steps), normalize=True)
    save_image(grid, save_path)


import os
import torch
from torchvision.utils import save_image, make_grid

@torch.no_grad()
def visualize_forward_reverse_trajectory(model, diffusion, x_0, device, save_dir, sample_id=0, steps=[1000, 750, 500, 250, 100, 50, 1]):
    """
    Visualizes forward (q(x_t|x_0)) and reverse (p_theta(x_{t-1}|x_t)) diffusion process.
    Used to track how real X-rays degrade and regenerate across denoising steps.

    Example:
        visualize_forward_reverse_trajectory(
            model=ema_model,
            diffusion=ddpm,
            x_0=real_image,
            device=device,
            save_dir="./outputs/trajectories",
            sample_id=0
        )
    """

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Forward process: q(x_t | x_0)
    x = x_0.clone().to(device)
    noisy_series = [x.cpu()]
    for t in steps:
        noise = torch.randn_like(x).to(device)
        x_t = diffusion.q_sample(x_0, t=torch.tensor([t], device=device), noise=noise)
        noisy_series.append(x_t.cpu())

    forward_path = os.path.join(save_dir, f"forward_sample_{sample_id}.png")
    forward_grid = make_grid(torch.cat(noisy_series, dim=0), nrow=len(noisy_series), normalize=True)
    save_image(forward_grid, forward_path)

    # Reverse process: p(x_{t-1} | x_t)
    x = torch.randn_like(x_0).to(device)
    reverse_series = []
    for t in reversed(steps):
        x = diffusion.denoise_step(model, x, t=torch.tensor([t], device=device))
        reverse_series.append(x.cpu())

    reverse_path = os.path.join(save_dir, f"reverse_sample_{sample_id}.png")
    reverse_grid = make_grid(torch.cat(reverse_series, dim=0), nrow=len(reverse_series), normalize=True)
    save_image(reverse_grid, reverse_path)

    return forward_path, reverse_path

import os
import matplotlib.pyplot as plt

def plot_noise_schedule(betas, alphas, alpha_bars, save_path):
    """
Plots the beta, alpha, and alpha_bar noise schedule curves.

Example:
    plot_noise_schedule(
        betas=diffusion.betas,
        alphas=diffusion.alphas,
        alpha_bars=diffusion.alpha_bars,
        save_path="outputs/noise_schedule.png"
    )
"""

    plt.figure(figsize=(10, 6))
    plt.plot(betas, label='Beta_t')
    plt.plot(alphas, label='Alpha_t')
    plt.plot(alpha_bars, label='Alpha_bar_t')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.title('Noise Schedule')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()