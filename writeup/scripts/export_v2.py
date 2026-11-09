"""Record one real v2 forward pass for the write-up's animation: writeup/data/v2trace.json + writeup/data/v2/*.

Same photo as v1's trace (validation position 140). Runs on the CPU so a training run on MPS is left alone:
  - the 256 x 256 input from v2/cache/val_pixels.npy
  - one mask from the training sampler (4 blocks of 15-20% on the 8 x 8 cell grid, fixed seed)
  - frozen V-JEPA 2 on the visible patches only, recording each of its 24 layers, then pooled to cells
  - the predictor from v2/checkpoints/best_masked.pt (whatever epoch it is at), with its q attention
  - the target: the cached EmbeddingGemma embeddings of the photo's 5 captions and their average

usage (from the repo root):  python writeup/scripts/export_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "v2"
OUT = ROOT / "writeup" / "data"
sys.path.insert(0, str(V2))
from utilities import sample_block                                   # noqa: E402  (v2's own sampler)
from models.predictor import Predictor                               # noqa: E402

VAL_POS, SEED = 140, 3
VJEPA = "facebook/vjepa2-vitl-fpc64-256"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OFFSETS = torch.tensor([0, 1, 16, 17])
r4 = lambda x: [round(float(v), 4) for v in x]


def sprite(rows, path):
    """(n, d) -> grayscale PNG, pixel = 127.5 + 127.5 * v / absmax; returns absmax."""
    rows = np.asarray(rows, dtype=np.float32)
    m = float(np.abs(rows).max()) or 1.0
    Image.fromarray(np.clip(np.round(127.5 + 127.5 * rows / m), 0, 255).astype(np.uint8)).save(path)
    return round(m, 4)


def main():
    (OUT / "v2").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    ### the mask: 4 blocks on the 8 x 8 cell grid, exactly as utilities.sample_keep_ids draws them ###
    while True:
        visible = torch.ones(8, 8, dtype=torch.bool)
        blocks = []
        for _ in range(4):
            top, left, h, w = sample_block(8, (0.15, 0.2), (0.75, 1.5))
            visible[top:top + h, left:left + w] = False
            blocks.append([left, top, w, h])
        cells = visible.flatten().nonzero().squeeze(1)
        if len(cells) >= 4:
            break
    patches = ((cells // 8) * 32 + (cells % 8) * 2)[:, None] + OFFSETS   # (k, 4) on the 16 x 16 patch grid
    patches = patches.flatten()

    ### pixels ###
    px = torch.from_numpy(np.load(V2 / "cache" / "val_pixels.npy", mmap_mode="r")[VAL_POS].copy())   # (3, 256, 256)
    Image.fromarray(px.permute(1, 2, 0).numpy()).save(OUT / "v2" / "input.jpg", quality=90)
    x = (px[None].float() / 255 - MEAN) / STD
    video = x[:, None].repeat(1, 2, 1, 1, 1)                            # (1, T=2, 3, 256, 256)

    ### frozen V-JEPA 2, visible patches only, every layer ###
    from transformers import AutoModel
    vjepa = AutoModel.from_pretrained(VJEPA, dtype=torch.float32).eval()
    with torch.no_grad():
        emb = vjepa.encoder.embeddings(video)                           # (1, 256, 1024)
        h = emb[:, patches]                                             # (1, 4k, 1024)
        norms = [h[0].norm(dim=-1)]
        for layer in vjepa.encoder.layer:
            h = layer(h, position_mask=patches[None])[0]
            norms.append(h[0].norm(dim=-1))
        h = vjepa.encoder.layernorm(h)[0]                               # (4k, 1024)
        cell_tok = h.view(len(cells), 4, -1).mean(1)                    # (k, 1024)
        # the whole photo too, as validation sees it
        full = emb
        for layer in vjepa.encoder.layer:
            full = layer(full, position_mask=None)[0]
        full = vjepa.encoder.layernorm(full)[0]                         # (256, 1024)
    norms = torch.stack(norms)                                          # (25, 4k)
    patch_norm = norms / norms.max(1, keepdim=True).values              # each layer scaled to its max

    ### predictor, with q's attention recomputed per block ###
    ck = torch.load(V2 / "checkpoints" / "best_masked.pt", map_location="cpu")
    pred = Predictor()
    pred.load_state_dict(ck["predictor"])
    pred.eval()
    with torch.no_grad():
        z = pred.pos(pred.proj(cell_tok[None]), cells[None])            # (1, k, 384)
        seq = torch.cat([pred.query, z], 1)                             # (1, 1 + k, 384)
        q_attn = []
        for blk in pred.blocks:
            hh = blk.norm1(seq)
            _, w = blk.attn(hh, hh, hh, need_weights=True, average_attn_weights=False)   # (1, 6, 1+k, 1+k)
            q_attn.append(w[0, :, 0].tolist())                          # q's row, every head
            seq = blk(seq)
        y_hat = pred.head(pred.norm(seq[:, 0]))[0]                      # (768,)
        grid = torch.zeros(1, 64, 1024)
        grid[0, cells] = cell_tok                                       # how training hands tokens to the predictor
        assert torch.allclose(y_hat, pred(grid, cells[None])[0], atol=1e-4)

    ### target: the cached EmbeddingGemma embeddings ###
    text = torch.load(V2 / "cache" / "text.pt")
    own = (text["val_owner"] == VAL_POS).nonzero().squeeze(1)
    caps = text["val_captions"][own]                                    # (5, 768), layer-normed
    target = text["val_targets"][VAL_POS]                               # (768,) their average
    assert torch.allclose(target, caps.mean(0), atol=1e-4)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
    captions = [text["val_caption_text"][i] for i in own.tolist()]
    pieces = [tok.convert_ids_to_tokens(tok(c, add_special_tokens=False)["input_ids"]) for c in captions]

    sq = (y_hat - target) ** 2
    trace = {
        "val": VAL_POS, "epoch": int(ck["epoch"]), "input": "v2/input.jpg",
        "blocks": blocks, "visible_cells": cells.tolist(), "visible_patches": patches.tolist(),
        "layers": {"sprite": "v2/patch_norms.png", "n": 25, "note": "rows: input + 24 layers; cols: visible patches; each row scaled to its max"},
        "patch_tokens": {"sprite": "v2/patch_tokens.png", "absmax": sprite(h[:, :256].numpy(), OUT / "v2" / "patch_tokens.png"), "note": "visible patches x first 256 of 1024 numbers"},
        "cell_tokens": {"sprite": "v2/cell_tokens.png", "absmax": sprite(cell_tok[:, :256].numpy(), OUT / "v2" / "cell_tokens.png")},
        "full_cells": {"sprite": "v2/full_cells.png", "absmax": sprite(full.view(8, 2, 8, 2, -1).mean((1, 3)).reshape(64, -1)[:, :256].numpy(), OUT / "v2" / "full_cells.png")},
        "q_attn": q_attn,                                               # [block][head][1 + k]
        "pred": r4(y_hat), "target": r4(target), "captions": captions, "pieces": pieces,
        "caption_vecs": {"sprite": "v2/captions.png", "absmax": sprite(caps.numpy(), OUT / "v2" / "captions.png")},
        "sq": r4(sq), "loss": round(float(sq.sum()), 2),
        "cos": round(float(F.cosine_similarity(y_hat, target, dim=0)), 4),
        "params": sum(p.numel() for p in pred.parameters()),
    }
    Image.fromarray(np.round(255 * patch_norm.numpy()).astype(np.uint8)).save(OUT / "v2" / "patch_norms.png")
    (OUT / "v2trace.json").write_text(json.dumps(trace))
    print(f"epoch {trace['epoch']}  visible cells {len(cells)}/64  loss {trace['loss']}  cos {trace['cos']}")
    print("blocks", blocks)
    print("caption 0 pieces", pieces[0])


if __name__ == "__main__":
    main()
