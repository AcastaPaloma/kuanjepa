import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.predictor import Predictor

from dataloader import DataLoader
from utilities import InfoNCE, masked_mean

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
    ], lr=5e-4, betas=(0.9, 0.98), eps=1e-8) ### MPS

    B = 256
    train_idx = DL.train_idx
    steps_per_epoch = len(train_idx) // B

    for epoch in range(200):
        perm = torch.randperm(len(train_idx)).tolist()
        order = [train_idx[j] for j in perm]

        for step in range(steps_per_epoch):
            batch_idx = order[step * B:(step + 1) * B]
            imgs, ids, mask = DL.get_batch(indices=batch_idx)

            imgs, ids, mask = imgs.to(device), ids.to(device), mask.to(device)

            visual_tokens = image_encoder(imgs)                      # (B, 64, 256)
            pred = predictor(visual_tokens).squeeze(1)               # (B, (1), 256), unit norm
            print("visual tokens", tuple(visual_tokens.shape), "pred", tuple(pred.shape))

            with torch.no_grad():
                text_tokens = text_encoder(ids, mask)                # (B, L, 256) ### MPS
                target = F.normalize(masked_mean(text_tokens, mask), p=2, dim=-1)   # (B, 256) ### MPS
            print("text tokens", tuple(text_tokens.shape), "target", tuple(target.shape))

            ### LOSS & OPTIMIZER ###
            loss = criterion_params(pred, target) ### MPS
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                criterion.t.clamp_(max=math.log(100))

            print("loss", loss.item())
            print("grad reaches image encoder:", image_encoder.convStem1[0].weight.grad is not None)

            ### VALIDATION ###
            image_encoder.eval(); predictor.eval(); text_encoder.eval()
            val_losses = []
            with torch.no_grad():
                val_idx = DL.val_idx
                for i in range(len(val_idx) // B):
                    batch_idx = val_idx[i * B:(i + 1) * B]
                    imgs, ids, mask = DL.get_batch(indices=batch_idx)

                    pred = predictor(image_encoder(imgs)).squeeze(1)
                    text_tokens = text_encoder(ids, mask)
                    target = F.normalize(masked_mean(text_tokens, mask), p=2, dim=-1)

                    val_losses.append(criterion(pred, target).item())
            val_loss = sum(val_losses) / len(val_losses)
            image_encoder.train(); predictor.train()   # text_encoder stays frozen/eval if it's the target

            print(f"epoch {epoch} val_loss {val_loss:.4f} (best {best_val:.4f})")

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
                }, "checkpoints/best.pt")
                print(f"  saved new best -> checkpoints/best.pt")
