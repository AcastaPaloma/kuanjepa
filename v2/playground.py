import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities import embed_images
from dataloader import DataLoader
from data import RawPairDataset, collate_pil

import torch
from transformers import AutoVideoProcessor, AutoModel

device = torch.device("mps")
DL = DataLoader()
DL._setup_text()                                            # builds captions + train/val indices

vjepa = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").to(device).eval()
vjepa_proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

ds = RawPairDataset(DL, DL.train_idx)
imgs, captions = collate_pil([ds[k] for k in range(16)])    # 16 raw PIL images + their captions

embs = embed_images(imgs, vjepa, vjepa_proc)
print(embs.shape)                                           # (16, 256, 1024)
