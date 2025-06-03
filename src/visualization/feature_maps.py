import os
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def visualize_feature_maps(model, input_tensor, layers_to_hook, device, save_dir, sample_id=0,timestep=0):
    """
Hooks into specified layers of U-Net to capture intermediate feature maps.

Use case:
    visualize_feature_maps(
        model=unet,
        input_tensor=x,
        layers_to_hook={"bottleneck": model.bottleneck},
        device=device,
        save_dir="./outputs/features",
        sample_id=0
    )
"""

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    features = {}
    hooks = []

    def get_hook(name):
        def hook_fn(module, input, output):
            features[name] = output.detach().cpu()
        return hook_fn

    # Register hooks
    for name, layer in layers_to_hook.items():
        hooks.append(layer.register_forward_hook(get_hook(name)))

    _ = model(input_tensor.to(device), timestep)

    # Save visualizations
    for name, feat in features.items():
        if feat.dim() == 4:
            fmap = feat[0]  # Take first sample
            fig, axes = plt.subplots(1, min(4, fmap.size(0)), figsize=(12, 4))
            for i in range(min(4, fmap.size(0))):
                axes[i].imshow(fmap[i], cmap='gray')
                axes[i].set_title(f"{name} | Channel {i}")
                axes[i].axis('off')
            fig.tight_layout()
            path = os.path.join(save_dir, f"featuremap_{name}_sample_{sample_id}.png")
            plt.savefig(path)
            plt.close(fig)

    for h in hooks:
        h.remove()