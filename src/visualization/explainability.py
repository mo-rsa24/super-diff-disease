import torch, matplotlib.pyplot as plt
from captum.attr import LayerGradCam, visualize_image_attr      # :contentReference[oaicite:2]{index=2}
from torchvision.transforms.functional import to_tensor, normalize

def show_gradcam(model, img_path, label_idx, layer=None, alpha=0.4):
    model.eval(); model.zero_grad()
    img = load_img(img_path)/255.
    x = normalize(to_tensor(img), [0.5], [0.5]).unsqueeze(0)
    x.requires_grad = True
    target_layer = layer or model.features[-1]
    cam = LayerGradCam(model, target_layer)
    mask = cam.attribute(x, target=label_idx).squeeze().cpu().numpy()
    visualize_image_attr(mask, img, method="overlay_heat_map",
                         sign="absolute_value", show_colorbar=True,
                         alpha_overlay=alpha)
    plt.title(f"Grad-CAM: {img_path.parent.name}")
