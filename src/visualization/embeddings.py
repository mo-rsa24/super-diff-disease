import numpy as np, torch
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler  # pre-trained CXR encoders :contentReference[oaicite:0]{index=0}

def compute_embeddings(img_paths, batch=16, device="cuda"):
    model = xrv.models.DenseNet(weights="densenet121-res224-all")\
            .to(device).eval()                                      # 1024-dim
    embs, labels = [], []
    preprocess = xrv.datasets.XRayCenterCrop()
    with torch.no_grad():
        for i in range(0, len(img_paths), batch):
            imgs = [preprocess(load_img(p)/255.).unsqueeze(0)
                    for p in img_paths[i:i+batch]]
            x = torch.cat(imgs).to(device)
            feats = model.features(x).squeeze()
            embs.append(feats.cpu().numpy()); labels += [p.parent.name for p in imgs]
    return np.vstack(embs), np.array(labels)

def plot_tsne(embeddings, labels, perplexity=30, seed=42):
    z = TSNE(n_components=2, perplexity=perplexity,
             random_state=seed).fit_transform(StandardScaler().fit_transform(embeddings))
    fig = px.scatter(x=z[:,0], y=z[:,1], color=labels,
                     hover_data={"label":labels},
                     title="t-SNE projection of CXR embeddings")
    fig.update_layout(height=500, template="plotly_white")
    fig.show()
