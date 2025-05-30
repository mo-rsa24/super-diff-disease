from streamlit import sidebar, image, pyplot
import streamlit as st
from .images import plot_image_grid, plot_histogram
from .embeddings import plot_tsne
from .explainability import show_gradcam

def launch_streamlit(data_root):
    st.title("CXR Diffusion-project Visual Dashboard")
    task = sidebar.radio("Section", ["Image Grid", "Embedding Explorer", "Explainability"])
    if task == "Image Grid":
        n = sidebar.slider("Samples/class", 4, 16, 8); clahe = sidebar.checkbox("CLAHE", False)
        plot_image_grid(data_root, n, clahe); st.pyplot(pyplot.gcf())
    elif task == "Embedding Explorer":
        emb_file = sidebar.file_uploader("Load .npy embeddings", type="npy")
        if emb_file:
            embs, lbls = np.load(emb_file, allow_pickle=True)
        else:
            st.stop()
        plot_tsne(embs, lbls)
    else:
        img_path = sidebar.text_input("Path to image")
        if img_path: show_gradcam(pretrained_model(), img_path, label_idx=1)
