# train_ddpm.py
import logging
from matplotlib import pyplot as plt
import torch
from torch.optim import Adam
from tqdm import tqdm
import os
from ema_pytorch import EMA

from src.utils.visualization import show_real_vs_generated


def train(model, dataloader, diffusion, num_epochs, save_path, device, log_every=1, vis_every=5):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=2e-4)
    ema = EMA(model, beta=0.995).to(device)

    # Logging
    os.makedirs(save_path, exist_ok=True) 
    logging.basicConfig(filename=os.path.join(save_path, "training.log"), level=logging.INFO, format='%(asctime)s - %(message)s')
    loss_log = []

    os.makedirs(save_path, exist_ok=True)
    vis_dir = os.path.join(save_path, "samples")
    os.makedirs(vis_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch["image"].to(device)
            loss = diffusion.training_step(model, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_log.append(avg_loss)
        if epoch % log_every == 0:
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")

        # Save model & visualize progression
        torch.save(model.state_dict(), os.path.join(save_path, f"ddpm_epoch{epoch+1}.pt"))
        torch.save(ema.ema_model.state_dict(), os.path.join(save_path, f"ema_epoch{epoch+1}.pt"))

        if epoch % vis_every == 0 or epoch == num_epochs - 1:
            # Visualize a real vs generated image
            model.eval()
            sample = images[0].unsqueeze(0)  # batch of 1
            with torch.no_grad():
                generated = diffusion.sample(ema.ema_model, image_shape=sample.shape, device=device)
            show_real_vs_generated(
                real=sample.squeeze().cpu(),
                generated=generated[0].squeeze().cpu(),
                title=f"Epoch {epoch+1}",
                save_path=os.path.join(vis_dir, f"epoch_{epoch+1:03d}.png")
            )

    # Plot loss progression
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_log)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("DDPM Training Loss Progression")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()
