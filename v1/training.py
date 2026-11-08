import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import os
import sys
import csv
import time
import logging
from datetime import datetime

from v1.models.image_encoder import ImageEncoder
from v1.models.text_encoder import TextEncoder
from v1.models.predictor import Predictor

from dataloader import DataLoader, PairDataset
from torch.utils.data import DataLoader as TorchDataLoader
from v1.utilities import InfoNCE, masked_mean

device = torch.device("mps")

if __name__ == "__main__":
    torch.manual_seed(0)

    ### DATALOADER ###
    DL = DataLoader()
    DL._setup_text()

    ### MODELS ###
    image_encoder = ImageEncoder().to(device) ### MPS
    text_encoder = TextEncoder(vocab_size=len(DL.vocab), max_len=DL.max_len).to(device) ### MPS
    predictor = Predictor().to(device) ### MPS

    ### LOSS ###
    criterion = InfoNCE().to(device) ### MPS

    ### OPTIMIZER ###
    model_params = list(image_encoder.parameters()) + list(text_encoder.parameters()) + list(predictor.parameters()) ### MPS
    criterion_params = list(criterion.parameters()) ### MPS

    optimizer = torch.optim.AdamW([
        {"params": model_params,     "weight_decay": 0.05},
        {"params": criterion_params, "weight_decay": 0.0},   # t
    ], lr=1e-4, betas=(0.9, 0.98), eps=1e-8) ### MPS  (5e-4 without warmup collapsed both encoders)

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    ### LOGGING ###
    run_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for h in (logging.FileHandler(os.path.join(run_dir, "train.log")), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)

    metrics_f = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    metrics = csv.writer(metrics_f)
    metrics.writerow(["epoch", "step", "global_step", "split", "loss", "temperature", "lr", "sec_per_step"])

    LOG_EVERY = 10
    global_step = 0
    log.info(f"run dir: {run_dir}  device: {device}  train: {len(DL.train_idx)}  val: {len(DL.val_idx)}")

    B = 256
    NUM_WORKERS = 8
    train_loader = TorchDataLoader(PairDataset(DL, DL.train_idx, train=True), batch_size=B, shuffle=True,
                                   drop_last=True, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = TorchDataLoader(PairDataset(DL, DL.val_idx, train=False), batch_size=B, shuffle=False,
                                 drop_last=False, num_workers=NUM_WORKERS, persistent_workers=True)
    steps_per_epoch = len(train_loader)

    ### LR SCHEDULE: linear warmup for 1 epoch, then cosine decay to 0 ###
    EPOCHS = 200
    WARMUP_STEPS = steps_per_epoch
    TOTAL_STEPS = EPOCHS * steps_per_epoch

    def lr_lambda(s):
        if s < WARMUP_STEPS:
            return (s + 1) / WARMUP_STEPS
        progress = (s - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(EPOCHS):
        epoch_losses = []
        t0 = time.time()
        for step, (imgs, ids, mask) in enumerate(train_loader):
            imgs, ids, mask = imgs.to(device), ids.to(device), mask.to(device)

            visual_tokens = image_encoder(imgs)                      # (B, 64, 256)
            pred = predictor(visual_tokens).squeeze(1)               # (B, 256), unit norm

            # CLIP-style: the text encoder trains through the same contrastive loss
            text_tokens = text_encoder(ids, mask)                    # (B, L, 256)
            target = F.normalize(masked_mean(text_tokens, mask), p=2, dim=-1)   # (B, 256)

            ### LOSS & OPTIMIZER ###
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                criterion.t.clamp_(max=math.log(100))

            loss_v = loss.item()
            epoch_losses.append(loss_v)
            dt = time.time() - t0
            t0 = time.time()
            temp = criterion.t.exp().item()
            lr = optimizer.param_groups[0]["lr"]
            metrics.writerow([epoch, step, global_step, "train", f"{loss_v:.5f}", f"{temp:.3f}", lr, f"{dt:.3f}"])
            if step % LOG_EVERY == 0:
                metrics_f.flush()
                log.info(f"ep {epoch} step {step}/{steps_per_epoch} loss {loss_v:.4f} temp {temp:.2f} {dt:.2f}s/step")
            global_step += 1

        ### VALIDATION ###
        # Embed the whole val set, then score all N images against all N captions at once.
        image_encoder.eval(); predictor.eval(); text_encoder.eval()
        all_pred, all_target = [], []
        with torch.no_grad():
            for imgs, ids, mask in val_loader:
                imgs, ids, mask = imgs.to(device), ids.to(device), mask.to(device)
                all_pred.append(predictor(image_encoder(imgs)).squeeze(1))
                text_tokens = text_encoder(ids, mask)
                all_target.append(F.normalize(masked_mean(text_tokens, mask), p=2, dim=-1))
            all_pred, all_target = torch.cat(all_pred), torch.cat(all_target)   # (N, 256) each
            val_loss = criterion(all_pred, all_target).item()                   # chance = ln(N)
            sims = all_pred @ all_target.t()
            labels = torch.arange(sims.size(0), device=sims.device)
            r1 = (sims.argmax(1) == labels).float().mean().item()                # image -> text
            r5 = (sims.topk(5, dim=1).indices == labels[:, None]).any(1).float().mean().item()
        image_encoder.train(); predictor.train(); text_encoder.train()

        train_loss = sum(epoch_losses) / len(epoch_losses)
        metrics.writerow([epoch, "", global_step, "val", f"{val_loss:.5f}", f"{criterion.t.exp().item():.3f}", optimizer.param_groups[0]["lr"], ""])
        metrics_f.flush()
        log.info(f"=== epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} (chance {math.log(all_pred.size(0)):.2f}) "
                 f"i2t R@1 {r1:.3f} R@5 {r5:.3f} (best {best_val:.4f})")

        ### SAVE IF BEST ###
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "image_encoder": image_encoder.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "criterion": criterion.state_dict(),   # keeps the learned temperature
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, "checkpoints/best.pt")
            log.info("  saved new best -> checkpoints/best.pt")
