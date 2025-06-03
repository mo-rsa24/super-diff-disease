# debug_training_block.py
import os, torch, time
from src.models.unet import UNet
from src.models.ddpm import DDPM
from src.data.dataset import ChestXrayDataset
from src.transforms import safe_augmentation
import yaml 

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

# === CONFIG ===
config_path = "src/config/config.yaml"
task = "TB"
split = "train"
dataset_root = "datasets/cleaned"  # adjust this path as needed
output_dir = "debug_outputs"
index = 0  # sample index
device = torch.device("cpu")

# === PREP ===
os.makedirs(output_dir, exist_ok=True)
config = load_yaml(config_path)
transform = safe_augmentation(config["training"]["augmentation"], config["training"]["normalization"])

dataset = ChestXrayDataset(root_dir=dataset_root, split=split, aug=transform, task=task, class_filter=1)
sample = dataset[index]["image"].unsqueeze(0).to(device)

model = UNet().to(device)
ema_model = UNet().to(device)
ema_model.load_state_dict(model.state_dict())  # mimic EMA for testing
diffusion = DDPM(num_timesteps=config["training"]["num_timesteps"])

print(f"✅ Sample loaded: shape={sample.shape}, dtype={sample.dtype}")

# === DEBUG BLOCK WRAPPER ===
def run_debug_block(description: str, block_fn):
    print(f"\n🔍 Running: {description}")
    start = time.time()
    try:
        block_fn()
        print(f"✅ Success: {description}")
    except Exception as e:
        print(f"❌ Error in '{description}':\n{e}")
    finally:
        print(f"⏱️ Duration: {round(time.time() - start, 2)}s")

# === EXAMPLE: Visualize Denoising Trajectory ===
from src.visualization.denoising_trajectory import visualize_denoising_trajectory
traj_path = os.path.join(output_dir, "denoise_debug.png")

run_debug_block("visualize_denoising_trajectory", lambda: visualize_denoising_trajectory(
    diffusion=diffusion,
    model=ema_model,
    image_shape=sample.shape,
    device=device,
    steps=[999, 500, 250, 100, 10, 1],
    save_path=traj_path
))