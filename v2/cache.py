import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import torch
from PIL import ImageOps
from tqdm import tqdm
from transformers import AutoVideoProcessor, AutoModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader as TorchDataLoader

from dataloader import DataLoader
from data import RawPairDataset, collate_pil
from utilities import embed_images, embed_texts, pool_2x2

device = torch.device("mps")
CACHE = Path(__file__).resolve().parent / "cache"


def cache_text(DL, model, path):
    if path.exists():
        print(f"skip {path}")
        return

    train_caps = [list(DL.captions[i]) for i in DL.train_idx]
    train_emb = embed_texts([c for caps in train_caps for c in caps], model)                  # (M_train, 768)
    train_targets = torch.stack([e.mean(0) for e in torch.split(train_emb, [len(c) for c in train_caps])])   # (N_train, 768)

    val_caps = [list(DL.captions[i]) for i in DL.val_idx]
    val_emb = embed_texts([c for caps in val_caps for c in caps], model)                      # (M_val, 768)
    val_targets = torch.stack([e.mean(0) for e in torch.split(val_emb, [len(c) for c in val_caps])])         # (N_val, 768)
    val_owner = torch.tensor([n for n, caps in enumerate(val_caps) for _ in caps])            # (M_val,)

    torch.save({
        "train_targets": train_targets,
        "val_targets": val_targets,
        "val_captions": val_emb,
        "val_owner": val_owner,
        "val_caption_text": [c for caps in val_caps for c in caps],
    }, path)


def cache_images(DL, indices, path, n_views, model, proc):
    if path.exists():
        print(f"skip {path}")
        return

    tmp = path.with_name(path.stem + ".partial.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(indices), n_views, 64, 1024))
    loader = TorchDataLoader(RawPairDataset(DL, indices), batch_size=64, shuffle=False,
                             collate_fn=collate_pil, num_workers=4)
    pos = 0
    for imgs, _ in tqdm(loader, desc=path.name):
        b = len(imgs)
        out[pos:pos + b, 0] = pool_2x2(embed_images(imgs, model, proc)).half().numpy()            # (b, 64, 1024)
        if n_views == 2:
            flipped = [ImageOps.mirror(im) for im in imgs]
            out[pos:pos + b, 1] = pool_2x2(embed_images(flipped, model, proc)).half().numpy()     # (b, 64, 1024)
        pos += b
    out.flush()
    del out
    os.replace(tmp, path)


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)

    DL = DataLoader()
    DL._setup_text()

    text_encoder = SentenceTransformer("google/embeddinggemma-300m", device=str(device))
    cache_text(DL, text_encoder, CACHE / "text.pt")
    del text_encoder

    image_encoder = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").to(device).eval()
    image_encoder_proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    cache_images(DL, DL.val_idx, CACHE / "val_tokens.npy", 1, image_encoder, image_encoder_proc)       # (N_val, 1, 64, 1024)
    cache_images(DL, DL.train_idx, CACHE / "train_tokens.npy", 2, image_encoder, image_encoder_proc)   # (N_train, 2, 64, 1024)
