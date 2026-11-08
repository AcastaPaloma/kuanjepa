import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoVideoProcessor

device = "mps" if torch.backends.mps.is_available() else "cpu"

text_model = SentenceTransformer("google/embeddinggemma-300m", device=device)
vjepa = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").to(device).eval()
vjepa_proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")