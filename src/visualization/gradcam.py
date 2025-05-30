# visualization/gradcam.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torchvision.transforms import ToPILImage
from torch.nn import functional as F
from PIL import Image

def get_gradcam_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.eval()
    return model, model.layer4  # Final conv layer for Grad-CAM

def compute_gradcam(model, feature_layer, image_tensor):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = feature_layer.register_forward_hook(forward_hook)
    backward_handle = feature_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor.unsqueeze(0))
    target_class = output.argmax(dim=1).item()
    output[0, target_class].backward()

    activ = activations["value"][0]
    grads = gradients["value"][0]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = torch.sum(weights * activ, dim=0)
    cam = F.relu(cam)
    cam = cam / cam.max()

    forward_handle.remove()
    backward_handle.remove()
    return cam.cpu().numpy(), target_class

def overlay_heatmap(img_tensor, cam):
    img = img_tensor.squeeze().numpy()
    heatmap = np.uint8(255 * cam)
    heatmap = Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    heatmap = np.array(heatmap)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def run_gradcam(dataset):
    model, target_layer = get_gradcam_model()
    sample = dataset[0]
    image_tensor = sample["image"]

    cam, predicted_class = compute_gradcam(model, target_layer, image_tensor)
    print(f"🔥 Predicted class: {predicted_class} ({'TB' if predicted_class else 'Normal'})")
    overlay_heatmap(image_tensor, cam)
