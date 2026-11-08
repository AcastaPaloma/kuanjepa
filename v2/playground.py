import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities import embed_images
from dataloader import DataLoader

import torch
from transformers import AutoVideoProcessor, AutoModel

device = torch.device("mps")
DL = DataLoader()

vjepa = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").to(device).eval()
vjepa_proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

imgs, ids, mask = DL.get_batch(batch_size=16) 

embs = embed_images(imgs, vjepa, vjepa_proc) 
print(embs[0].shape)