import torch

import math
import os
import sys
import csv
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from utilities import sample_keep_ids, retrieval_metrics
from models.predictor import Predictor

device = torch.device("mps")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"


@torch.no_grad()
def predict_all(predictor, tokens, chunk=256):
    out = []
    for s in range(0, len(tokens), chunk):
        x = torch.from_numpy(np.array(tokens[s:s + chunk])).to(device).float()   # (b, 64, 1024)
        out.append(predictor(x))                                                            # (b, 768)
    return torch.cat(out)                                                                   # (N, 768)


if __name__ == "__main__":
    torch.manual_seed(0)

    ### DATA (cached by cache.py) ###
    text = torch.load(CACHE / "text.pt")
    train_tokens = np.load(CACHE / "train_tokens.npy", mmap_mode="r")    # (N_train, 2, 64, 1024) float16
    val_tokens = np.load(CACHE / "val_tokens.npy")[:, 0]                 # (N_val, 64, 1024) float16
    train_targets = text["train_targets"]                                # (N_train, 768)
    val_targets = text["val_targets"].to(device)                         # (N_val, 768)
    val_captions = text["val_captions"].to(device)                       # (M_val, 768)
    val_owner = text["val_owner"].to(device)                             # (M_val,)
    N_train = train_tokens.shape[0]

    ### MODEL ###
    predictor = Predictor().to(device)
    n_params = sum(p.numel() for p in predictor.parameters())

    ### OPTIMIZER ###
    decay, no_decay = [], []
    for name, p in predictor.named_parameters():
        if p.ndim >= 2 and name != "query":
            decay.append(p)
        else:
            no_decay.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay,    "weight_decay": 0.04},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=0.0, betas=(0.9, 0.98), eps=1e-8)

    ### SCHEDULE ###
    B = 512
    EPOCHS = 100
    steps_per_epoch = N_train // B
    WARMUP_STEPS = 5 * steps_per_epoch
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

    os.makedirs(HERE / "checkpoints", exist_ok=True)
    best_r5 = 0.0

    ### LOGGING ###
    run_dir = HERE / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
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
                      "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "sec_per_step"])

    LOG_EVERY = 10
    global_step = 0
    log.info(f"run dir: {run_dir}  device: {device}  train: {N_train}  val: {len(val_tokens)}  "
             f"predictor params: {n_params / 1e6:.1f}M")

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
            views = np.random.randint(0, 2, size=B)                                    # (B,)
            tokens = torch.from_numpy(train_tokens[idx, views]).to(device).float()     # (B, 64, 1024)
            target = train_targets[idx].to(device)                                     # (B, 768)
            keep_ids, pad = sample_keep_ids(B)                                         # (B, K), (B, K)
            keep_ids, pad = keep_ids.to(device), pad.to(device)

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
                              f"{pred_std:.4f}", f"{target_std:.4f}", "", "", "", "", "", "", f"{dt:.3f}"])
            if step % LOG_EVERY == 0:
                metrics_f.flush()
                log.info(f"ep {epoch} step {step}/{steps_per_epoch} loss {loss_v:.2f} "
                         f"pred_std {pred_std:.3f} target_std {target_std:.3f} lr {lr:.2e} {dt:.2f}s/step")
            global_step += 1

        ### VALIDATION ###
        predictor.eval()
        val_pred = predict_all(predictor, val_tokens)                                  # (N_val, 768)
        val_loss = ((val_pred - val_targets) ** 2).sum(-1).mean().item()
        val_pred_std = val_pred.std(0).mean().item()
        i2t, t2i = retrieval_metrics(val_pred, val_captions, val_owner)
        predictor.train()

        r5 = (i2t[5] + t2i[5]) / 2
        train_loss = sum(epoch_losses) / len(epoch_losses)
        metrics.writerow([epoch, "", global_step, "val", f"{val_loss:.5f}", "", "", f"{val_pred_std:.4f}",
                          f"{val_targets.std(0).mean().item():.4f}",
                          i2t[1], i2t[5], i2t[10], t2i[1], t2i[5], t2i[10], ""])
        metrics_f.flush()
        log.info(f"=== epoch {epoch} train_loss {train_loss:.2f} val_loss {val_loss:.2f} "
                 f"i2t R@1/5/10 {i2t[1]:.3f}/{i2t[5]:.3f}/{i2t[10]:.3f} "
                 f"t2i R@1/5/10 {t2i[1]:.3f}/{t2i[5]:.3f}/{t2i[10]:.3f} (best R@5 {best_r5:.3f})")

        ### SAVE IF BEST ###
        if r5 > best_r5:
            best_r5 = r5
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "r5": r5,
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, HERE / "checkpoints" / "best.pt")
            log.info("  saved new best -> checkpoints/best.pt")
