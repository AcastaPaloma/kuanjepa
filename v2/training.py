import torch
import torch.nn.functional as F
from transformers.image_utils import load_image
from transformers import AutoVideoProcessor, AutoModel

from sentence_transformers import SentenceTransformer

import math
import os
import sys
import csv
import time
import logging
from datetime import datetime

from dataloader import DataLoader, PairDataset, collate_raw
from torch.utils.data import DataLoader as TorchDataLoader
from utilities import embed_images, embed_captions

device = torch.device("mps")

if __name__ == "__main__":
    torch.manual_seed(0)

    ### DATALOADER ###
    DL = DataLoader()
    DL._setup_text()

    ### MODELS ###
    # Image encoder 
    image_encoder = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").to(device).eval()
    image_encoder_proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

    # Text encoder
    text_encoder = SentenceTransformer("google/embeddinggemma-300m")

    # Predictor

    ### LOSS ###

    ### OPTIMIZER ###
    model_params = ### Only for the predictor
    criterion = ### Only if loss has a trainable parameter

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
    # raw=True + collate_raw -> (imgs (B, 3, 128, 128), captions: list of B lists of all caption strings)
    train_loader = TorchDataLoader(PairDataset(DL, DL.train_idx, train=True, raw=True), batch_size=B, shuffle=True,
                                   drop_last=True, collate_fn=collate_raw, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = TorchDataLoader(PairDataset(DL, DL.val_idx, train=False, raw=True), batch_size=B, shuffle=False,
                                 drop_last=False, collate_fn=collate_raw, num_workers=NUM_WORKERS, persistent_workers=True)
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
        for step, (imgs, captions) in enumerate(train_loader):
            # Both encoders are frozen (no grad); only the predictor trains
            visual = embed_images(imgs, image_encoder, image_encoder_proc).to(device)   # (B, D_img)
            pred = F.normalize(predictor(visual), p=2, dim=-1)                          # (B, D_txt), unit norm

            target = embed_captions(captions, text_encoder).to(device)                  # (B, D_txt), unit norm

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
        predictor.eval()
        all_pred, all_target = [], []
        with torch.no_grad():
            for imgs, captions in val_loader:
                visual = embed_images(imgs, image_encoder, image_encoder_proc).to(device)
                all_pred.append(F.normalize(predictor(visual), p=2, dim=-1))
                all_target.append(embed_captions(captions, text_encoder).to(device))
            all_pred, all_target = torch.cat(all_pred), torch.cat(all_target)   # (N, D_txt) each
            val_loss = criterion(all_pred, all_target).item()                   # chance = ln(N)
            sims = all_pred @ all_target.t()
            labels = torch.arange(sims.size(0), device=sims.device)
            r1 = (sims.argmax(1) == labels).float().mean().item()                # image -> text
            r5 = (sims.topk(5, dim=1).indices == labels[:, None]).any(1).float().mean().item()
        predictor.train()

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
                "predictor": predictor.state_dict(),
                "criterion": criterion.state_dict(),   # keeps the learned temperature
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, "checkpoints/best.pt")
            log.info("  saved new best -> checkpoints/best.pt")


