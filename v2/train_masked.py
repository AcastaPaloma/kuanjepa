"""Mask BEFORE encode: the frozen V-JEPA 2 encoder only ever sees the visible patches.

training.py masks the cached tokens after a full-image encode, so every surviving token has
already attended to the masked regions. Here the mask is applied right after patch embedding,
so masked regions never enter the encoder (I-JEPA-style context encoding).

Masking is done on the 8x8 grid of 2x2 patch cells (same sampler as training.py). Each visible
cell contributes its 4 patches to the encoder; outputs are averaged back per cell, giving the
same (K, 1024) tokens the predictor already expects. Validation encodes the full image.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # root dataloader.py

import csv
import logging
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor

from utilities import sample_keep_ids, retrieval_metrics
from models.predictor import Predictor

device = torch.device("mps")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
VJEPA = "facebook/vjepa2-vitl-fpc64-256"


### ONE-TIME PREP: resized + center-cropped uint8 pixels (normalization happens on device) ###
def cache_pixels(DL, indices, path, proc):
    if path.exists():
        return
    from torch.utils.data import DataLoader as TorchDataLoader
    from data import RawPairDataset, collate_pil
    tmp = path.with_name(path.stem + ".partial.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(indices), 3, 256, 256))
    loader = TorchDataLoader(RawPairDataset(DL, indices), batch_size=64, shuffle=False,
                             collate_fn=collate_pil, num_workers=4)
    pos = 0
    for imgs, _ in tqdm(loader, desc=path.name):
        for im in imgs:
            px = proc(im, do_rescale=False, do_normalize=False, return_tensors="pt")["pixel_values_videos"]
            out[pos] = px[0, 0].round().clamp(0, 255).to(torch.uint8).numpy()      # (3, 256, 256)
            pos += 1
    out.flush()
    del out
    os.replace(tmp, path)


def prepare():
    from dataloader import DataLoader
    CACHE.mkdir(exist_ok=True)
    DL = DataLoader()
    DL._setup_text()
    if not (CACHE / "text.pt").exists():
        from sentence_transformers import SentenceTransformer
        from cache import cache_text
        cache_text(DL, SentenceTransformer("google/embeddinggemma-300m", device=str(device)), CACHE / "text.pt")
    proc = AutoVideoProcessor.from_pretrained(VJEPA)
    cache_pixels(DL, DL.val_idx, CACHE / "val_pixels.npy", proc)
    cache_pixels(DL, DL.train_idx, CACHE / "train_pixels.npy", proc)


### MASKED ENCODER ###
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OFFSETS = torch.tensor([0, 1, 16, 17])                                # 2x2 patch offsets inside a cell


def cells_to_patches(cell_ids):
    """(B, K) ids on the 8x8 cell grid -> (B, 4K) ids on the 16x16 patch grid, 4 per cell in order."""
    base = (cell_ids // 8) * 32 + (cell_ids % 8) * 2                  # top-left patch of each cell
    return (base[..., None] + OFFSETS.to(cell_ids.device)).flatten(1)


def to_video(px_uint8, flip=None):
    x = px_uint8.to(device).float().div_(255)                          # (b, 3, 256, 256)
    if flip is not None:
        x = torch.where(flip.to(device)[:, None, None, None], x.flip(-1), x)
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.half()[:, None].repeat(1, 2, 1, 1, 1)                     # (b, T=2, 3, 256, 256)


@torch.no_grad()
def encode_visible(vjepa, video, patch_ids=None):
    """Patch-embed, drop masked patches, then run all 24 layers on the visible ones only.
    patch_ids: (b, P) ids in [0, 256), or None for the full image. -> (b, P, 1024)."""
    x = vjepa.encoder.embeddings(video)                                # (b, 256, 1024) (T=2 is 1 tubelet)
    if patch_ids is not None:
        x = torch.gather(x, 1, patch_ids[..., None].expand(-1, -1, x.size(-1)))   # (b, P, 1024)
    for layer in vjepa.encoder.layer:
        x = layer(x, position_mask=patch_ids)[0]                       # RoPE uses the original positions
    return vjepa.encoder.layernorm(x)


def truncate_to_min(keep_ids, pad):
    """Same K for every sample in a chunk (the encoder has no padding mask), I-JEPA style:
    randomly drop extra visible cells down to the chunk's minimum."""
    counts = (~pad).sum(1)
    k = counts.min().item()
    rows = []
    for ids, n in zip(keep_ids, counts):
        ids = ids[:n]
        rows.append(ids[torch.randperm(n)[:k]].sort().values)
    return torch.stack(rows)                                           # (b, k)


@torch.no_grad()
def encode_masked_batch(vjepa, px_uint8, flip, chunk=64):
    """-> tokens (B, 64, 1024) with only visible cells filled, keep_ids (B, K), pad (B, K), frac visible."""
    B = px_uint8.size(0)
    keep_ids, pad = sample_keep_ids(B)                                 # 8x8 grid, 4 blocks of 15-20%
    tokens = torch.zeros(B, 64, 1024, device=device)
    out_ids = torch.zeros(B, 64, dtype=torch.long)
    out_pad = torch.ones(B, 64, dtype=torch.bool)
    n_vis = 0
    for s in range(0, B, chunk):
        cells = truncate_to_min(keep_ids[s:s + chunk], pad[s:s + chunk])          # (b, k)
        b, k = cells.shape
        feats = encode_visible(vjepa, to_video(px_uint8[s:s + chunk], flip[s:s + chunk]),
                               cells_to_patches(cells).to(device))                # (b, 4k, 1024)
        pooled = feats.view(b, k, 4, -1).float().mean(2)                           # (b, k, 1024)
        cells_d = cells.to(device)
        tokens[s:s + b].scatter_(1, cells_d[..., None].expand(-1, -1, 1024), pooled)
        out_ids[s:s + b, :k] = cells
        out_pad[s:s + b, :k] = False
        n_vis += b * k
    K = (~out_pad).sum(1).max().item()
    return tokens, out_ids[:, :K].to(device), out_pad[:, :K].to(device), n_vis / (B * 64)


@torch.no_grad()
def encode_full(vjepa, pixels, chunk=64):
    out = []
    for s in tqdm(range(0, len(pixels), chunk), desc="val encode", leave=False):
        px = torch.from_numpy(np.array(pixels[s:s + chunk]))
        feats = encode_visible(vjepa, to_video(px)).float()                       # (b, 256, 1024)
        b = feats.size(0)
        out.append(feats.view(b, 8, 2, 8, 2, -1).mean((2, 4)).reshape(b, 64, -1))  # (b, 64, 1024)
    return torch.cat(out)


@torch.no_grad()
def predict_all(predictor, tokens, chunk=256):
    return torch.cat([predictor(tokens[s:s + chunk]) for s in range(0, len(tokens), chunk)])


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    prepare()

    ### DATA ###
    text = torch.load(CACHE / "text.pt")
    train_pixels = np.load(CACHE / "train_pixels.npy", mmap_mode="r")    # (N_train, 3, 256, 256) uint8
    val_pixels = np.load(CACHE / "val_pixels.npy")                       # (N_val, 3, 256, 256) uint8
    train_targets = text["train_targets"]                                # (N_train, 768)
    val_targets = text["val_targets"].to(device)
    val_captions = text["val_captions"].to(device)
    val_owner = text["val_owner"].to(device)
    N_train = train_pixels.shape[0]

    ### FROZEN ENCODER (fp16) ###
    vjepa = AutoModel.from_pretrained(VJEPA, dtype=torch.float16).to(device).eval()
    for p in vjepa.parameters():
        p.requires_grad_(False)

    ### MODEL / OPTIMIZER / SCHEDULE (same as training.py) ###
    predictor = Predictor().to(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    decay = [p for n, p in predictor.named_parameters() if p.ndim >= 2 and n != "query"]
    no_decay = [p for n, p in predictor.named_parameters() if not (p.ndim >= 2 and n != "query")]
    optimizer = torch.optim.AdamW([
        {"params": decay,    "weight_decay": 0.04},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=0.0, betas=(0.9, 0.98), eps=1e-8)

    B = 512
    EPOCHS = int(os.environ.get("EPOCHS", 30))
    steps_per_epoch = N_train // B
    WARMUP_STEPS = min(5, max(1, EPOCHS // 10)) * steps_per_epoch
    TOTAL_STEPS = EPOCHS * steps_per_epoch
    PEAK_LR, FINAL_LR = 5e-4, 1e-6
    WD_START, WD_END = 0.04, 0.4

    def lr_at(s):
        if s < WARMUP_STEPS:
            return PEAK_LR * (s + 1) / WARMUP_STEPS
        progress = (s - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        return FINAL_LR + (PEAK_LR - FINAL_LR) * 0.5 * (1 + math.cos(math.pi * progress))

    def wd_at(s):
        return WD_START + (WD_END - WD_START) * s / max(1, TOTAL_STEPS - 1)

    ### LOGGING ###
    os.makedirs(HERE / "checkpoints", exist_ok=True)
    run_dir = HERE / "logs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-masked-encode")
    os.makedirs(run_dir, exist_ok=True)
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for h in (logging.FileHandler(run_dir / "train.log"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    metrics_f = open(run_dir / "metrics.csv", "w", newline="")
    metrics = csv.writer(metrics_f)
    metrics.writerow(["epoch", "step", "global_step", "split", "loss", "lr", "wd", "pred_std", "target_std",
                      "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "sec_per_step", "visible"])

    log.info(f"run dir: {run_dir}  train: {N_train}  val: {len(val_pixels)}  epochs: {EPOCHS}  "
             f"predictor params: {n_params / 1e6:.1f}M")
    val_tokens = encode_full(vjepa, val_pixels)                          # (N_val, 64, 1024), frozen -> once
    log.info(f"val tokens: {tuple(val_tokens.shape)}")

    best_r5, global_step, LOG_EVERY = 0.0, 0, 5
    for epoch in range(EPOCHS):
        epoch_losses = []
        perm = torch.randperm(N_train)
        t0 = time.time()
        for step in range(steps_per_epoch):
            lr, wd = lr_at(global_step), wd_at(global_step)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[0]["weight_decay"] = wd
            optimizer.param_groups[1]["lr"] = lr

            idx = perm[step * B:(step + 1) * B].sort().values.numpy()                 # (B,)
            px = torch.from_numpy(train_pixels[idx])                                   # (B, 3, 256, 256) uint8
            flip = torch.rand(B) < 0.5
            tokens, keep_ids, pad, vis = encode_masked_batch(vjepa, px, flip)          # (B, 64, 1024), (B, K)
            target = train_targets[idx].to(device)                                     # (B, 768)

            pred = predictor(tokens, keep_ids, pad)                                    # (B, 768)
            loss = ((pred - target) ** 2).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_v = loss.item()
            epoch_losses.append(loss_v)
            pred_std = pred.detach().std(0).mean().item()
            target_std = target.std(0).mean().item()
            dt = time.time() - t0
            t0 = time.time()
            metrics.writerow([epoch, step, global_step, "train", f"{loss_v:.5f}", lr, f"{wd:.4f}",
                              f"{pred_std:.4f}", f"{target_std:.4f}", "", "", "", "", "", "", f"{dt:.3f}",
                              f"{vis:.3f}"])
            if step % LOG_EVERY == 0:
                metrics_f.flush()
                log.info(f"ep {epoch} step {step}/{steps_per_epoch} loss {loss_v:.2f} "
                         f"pred_std {pred_std:.3f} target_std {target_std:.3f} visible {vis:.2f} "
                         f"lr {lr:.2e} {dt:.2f}s/step")
            global_step += 1

        ### VALIDATION (full image, no mask) ###
        predictor.eval()
        val_pred = predict_all(predictor, val_tokens)                                  # (N_val, 768)
        val_loss = ((val_pred - val_targets) ** 2).sum(-1).mean().item()
        i2t, t2i = retrieval_metrics(val_pred, val_captions, val_owner)
        predictor.train()

        r5 = (i2t[5] + t2i[5]) / 2
        train_loss = sum(epoch_losses) / len(epoch_losses)
        metrics.writerow([epoch, "", global_step, "val", f"{val_loss:.5f}", "", "",
                          f"{val_pred.std(0).mean().item():.4f}", f"{val_targets.std(0).mean().item():.4f}",
                          i2t[1], i2t[5], i2t[10], t2i[1], t2i[5], t2i[10], "", ""])
        metrics_f.flush()
        log.info(f"=== epoch {epoch} train_loss {train_loss:.2f} val_loss {val_loss:.2f} "
                 f"i2t R@1/5/10 {i2t[1]:.3f}/{i2t[5]:.3f}/{i2t[10]:.3f} "
                 f"t2i R@1/5/10 {t2i[1]:.3f}/{t2i[5]:.3f}/{t2i[10]:.3f} (best R@5 {best_r5:.3f})")

        if r5 > best_r5:
            best_r5 = r5
            torch.save({"epoch": epoch, "val_loss": val_loss, "r5": r5,
                        "predictor": predictor.state_dict(), "optimizer": optimizer.state_dict()},
                       HERE / "checkpoints" / "best_masked.pt")
            log.info("  saved new best -> checkpoints/best_masked.pt")
