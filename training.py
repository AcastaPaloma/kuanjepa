import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.predictor import Predictor

from dataloader import DataLoader
from utilities import InfoNCE, masked_mean

device = torch.device("mps")

if __name__ == "__main__":
    torch.manual_seed(0)

    ### MODELS ###
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder(vocab_size=len(DL.vocab), max_len=DL.max_len)
    predictor = Predictor()

    ### LOSS ###
    criterion = InfoNCE().to(device)

    ### OPTIMIZER ###
    model_params = list(image_encoder.parameters()) + list(text_encoder.parameters()) + list(predictor.parameters())
    criterion_params = list(criterion.parameters())

    optimizer = torch.optim.AdamW([
        {"params": model_params,     "weight_decay": 0.05},
        {"params": criterion_params, "weight_decay": 0.0},   # t
    ], lr=5e-4, betas=(0.9, 0.98), eps=1e-8)

    ### DATALOADER ###
    DL = DataLoader()
    DL._setup_text()

    B = 256
    train_idx = DL.train_idx
    steps_per_epoch = len(train_idx) // B

    for epoch in range(200):
        perm = torch.randperm(len(train_idx)).tolist()
        order = [train_idx[j] for j in perm]

        for step in range(steps_per_epoch):
            batch_idx = order[step * B:(step + 1) * B]
            imgs, ids, mask = DL.get_batch(indices=batch_idx)
            # imgs: (B, 3, 128, 128), ids: (B, 32), mask: (B, 32)

            visual_tokens = image_encoder(imgs)                      # (B, 64, 256)
            pred = predictor(visual_tokens).squeeze(1)               # (B, (1), 256), unit norm
            print("visual tokens", tuple(visual_tokens.shape), "pred", tuple(pred.shape))

            with torch.no_grad():
                text_tokens = text_encoder(ids, mask)                # (B, L, 256)
                target = F.normalize(masked_mean(text_tokens, mask), p=2, dim=-1)   # (B, 256)
            print("text tokens", tuple(text_tokens.shape), "target", tuple(target.shape))

            ### LOSS & OPTIMIZER ###
            loss = criterion_params(visual_tokens, target)

            optimizer.zero_grad()
            loss.backward()

            print("loss", loss.item())
            print("grad reaches image encoder:", image_encoder.convStem1[0].weight.grad is not None)


### TRAINING LOOP CODE ###
# for epoch in epochs:
#     for images, captions in loader:
#         tokens = image_encoder(images)
#         tokens = drop_25_percent(tokens)
#         y_hat = predictor(tokens)
#         y = text_encoder(captions)
#         loss = infonce(y_hat, y)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     evaluate on val
#     save if best