# src/factories/model_diffusion.py
from src.factories.registry import register_model_diffusion
from src.models.ddpm import DDPM
from src.models.diffusion_mnist import UNet  # adjust to your import path

@register_model_diffusion("MNIST")
def mnist_model_diffusion(config: dict) -> tuple:
    # Validate number of timesteps
    num_timesteps = int(config.get("training", {}).get("num_timesteps", 1000))
    if num_timesteps <= 0:
        raise ValueError(f"Invalid num_timesteps={num_timesteps}; must be > 0.")

    img_size = int(config.get("model", {}).get("image_size", 64))
    base_dim = int(config.get("model", {}).get("base_dim", 64))
    beta_sched = config.get("training", {}).get("beta_schedule", "linear")
    if beta_sched not in ("linear", "cosine"):
        raise ValueError(f"Unsupported beta_schedule='{beta_sched}'. Must be 'linear' or 'cosine'.")

    unet = UNet(in_channels=1, base_dim=base_dim)
    diffusion = DDPM(
        num_timesteps=num_timesteps,
        image_size=img_size,
        beta_schedule=beta_sched,
    )

    return unet, diffusion


@register_model_diffusion("CHEST_XRAY")
def chestxray_model_diffusion(config: dict) -> tuple:
    num_timesteps = int(config.get("training", {}).get("num_timesteps", 1000))
    if num_timesteps <= 0:
        raise ValueError(f"Invalid num_timesteps={num_timesteps}; must be > 0.")

    model = UNet()
    diffusion = DDPM(num_timesteps=num_timesteps)

    return model, diffusion
