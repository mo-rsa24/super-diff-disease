# models/hf_diffusion_wrapper.py

import torch

class HF_DiffusionWrapper:
    """
    Wraps a HuggingFace‐style UNet into a DDPM interface:
      - training_step(model, x_start) returns MSE on ε
      - sample(model, image_shape, device) does ancestral sampling
    """

    def __init__(self, noise_schedule: dict, timesteps: int):
        """
        Args:
            noise_schedule: dictionary containing tensors
                "betas", "alphas", "alpha_cumprod", all on CPU or GPU.
            timesteps: total number of diffusion steps T.
        """
        self.noise_schedule = noise_schedule
        self.T = timesteps

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor):
        """
        Produce x_t = sqrt(alpha_cumprod[t]) * x_start + sqrt(1 - alpha_cumprod[t]) * eps,
        and also return the noise eps.  Used for training.
        """
        bsz = x_start.shape[0]
        eps = torch.randn_like(x_start)
        alpha_cumprod = self.noise_schedule["alpha_cumprod"]
        sqrt_acp = alpha_cumprod[t].view(bsz, 1, 1, 1)
        sqrt_omacp = torch.sqrt(1 - alpha_cumprod[t]).view(bsz, 1, 1, 1)
        x_t = sqrt_acp * x_start + sqrt_omacp * eps
        return x_t, eps

    def training_step(self, model: torch.nn.Module, x_start: torch.Tensor):
        """
        Sample random t ∈ [0, T), compute x_t, predict ε, return MSE(ε_pred, ε).
        """
        bsz = x_start.shape[0]
        t = torch.randint(0, self.T, (bsz,), device=x_start.device).long()
        x_t, eps = self.q_sample(x_start, t)
        eps_pred = model(x_t, t)
        return torch.nn.functional.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, model: torch.nn.Module, image_shape, device: torch.device):
        """
        Ancestral sampling starting from N(0,1) noise of shape image_shape.
        Returns x_0 in [–1, +1].
        """
        model.eval()
        num_samples = image_shape[0]
        x = torch.randn(image_shape, device=device)

        betas = self.noise_schedule["betas"]
        alphas = self.noise_schedule["alphas"]
        alpha_cumprod = self.noise_schedule["alpha_cumprod"]

        for t in reversed(range(self.T)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            eps_pred = model(x, t_batch)

            alpha_t = alphas[t]
            alpha_cumprod_t = alpha_cumprod[t]
            beta_t = betas[t]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)).view(1, 1, 1, 1) * eps_pred
            )
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = mean + torch.sqrt(beta_t).view(1, 1, 1, 1) * noise

        return torch.clamp(x, -1, 1)
