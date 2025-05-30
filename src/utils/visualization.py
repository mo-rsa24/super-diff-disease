# utils/visualization.py
import matplotlib.pyplot as plt
import torch
import os

def plot_reverse_diffusion(model, ddpm, image, device, save_path=None, steps=[0, 250, 500, 750, 999]):
    model.eval()
    image = image.unsqueeze(0).to(device)
    x = image
    visuals = []

    for t in reversed(steps):
        t_tensor = torch.tensor([t], device=device)
        x = ddpm.q_sample(image, t_tensor)
        visuals.append(x.squeeze(0).detach().cpu())

    fig, axs = plt.subplots(1, len(visuals), figsize=(15, 4))
    for i, img in enumerate(visuals):
        axs[i].imshow(img.squeeze(0), cmap="gray")
        axs[i].set_title(f"t={steps[::-1][i]}")
        axs[i].axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def show_real_vs_generated(real, generated, title="Real vs Generated", save_path=None):
    """
    Displays or saves a side-by-side image of real and generated samples.

    Args:
        real (torch.Tensor): Single-channel real image tensor (H, W) or (1, H, W)
        generated (torch.Tensor): Single-channel generated image tensor (H, W) or (1, H, W)
        title (str): Title for the visualization
        save_path (str): If provided, saves the image instead of displaying
    """
    # Ensure image shape is (H, W)
    real = real.squeeze().detach().cpu().numpy()
    generated = generated.squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(real, cmap='gray')
    axs[0].set_title("Real")
    axs[0].axis("off")

    axs[1].imshow(generated, cmap='gray')
    axs[1].set_title("Generated")
    axs[1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()