# models/feature_extractor.py

import torch
import torchxrayvision as xrv
from torchvision import models
from transformers import AutoProcessor, AutoModel

def get_resnet18_feature_extractor():
    """
    Returns a ResNet18 model modified for grayscale images and outputs feature vectors (no classification).
    """
    model = models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1-channel input
    model.fc = torch.nn.Identity()  # Remove final classification layer
    model.eval()
    return model

def extract_features(model, dataloader, max_samples=300):
    """
    Extracts features and labels from the dataloader using a CNN.
    """
    features, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"]
            lbls = batch["class"]
            feats = model(imgs)
            features.append(feats)
            labels.extend(lbls.numpy())
            if len(labels) >= max_samples:
                break
    features = torch.cat(features, dim=0).numpy()
    return features[:max_samples], labels[:max_samples]

def get_chexnet_densenet121_feature_extractor():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    def forward_features(x):
        x = x.to(next(model.parameters()).device)  # 🔐 Match input to model device
        with torch.no_grad():
            x = model.features(x)
            x = torch.nn.functional.relu(x, inplace=True)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            return torch.flatten(x, 1)

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return forward_features


def get_medclip_vit_feature_extractor():
    model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model.eval()
    return model.vision_model