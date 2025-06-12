from src.factories.model_diffusion import mnist_model_diffusion, chestxray_model_diffusion
from src.factories.registry import get_dataset, get_model_diffusion
from src.factories.chestxray_dataset import ChestXray_Wrapper  # Triggers registration

