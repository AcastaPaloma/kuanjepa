import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Block
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from dataloader import DataLoader


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(256, 256)

        self.blocks = nn.ModuleList([Block(256) for _ in range(2)])
        self.ln = nn.LayerNorm(256, eps=1e-6)

        self.linear2 = nn.Linear(256, 256)


    def forward(self, visual_tokens):
        batch_count = visual_tokens.shape[0]

        x = self.linear1(visual_tokens)
        q_in= torch.zeros(size=(batch_count, 1, 256))

        together = torch.cat([x, q_in], dim=1)

        for blk in self.blocks:
            together = blk(together)

        q_out = together[:, :1, :]

        out = F.normalize(self.linear2(self.ln(q_out)), p=2, dim=-1)
        return out


def masked_mean(x, mask):
    # x: (B, L, d), mask: (B, L) bool -> (B, d) mean over real tokens only
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. data: row i of imgs / ids / mask is the same image-caption pair
    DL = DataLoader()
    imgs, ids, mask = DL.get_batch(batch_size=8)
    print("imgs", tuple(imgs.shape), "ids", tuple(ids.shape), "mask", tuple(mask.shape))

    # 2. models
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder(vocab_size=len(DL.vocab), max_len=DL.max_len)
    predictor = Predictor()

    # 3. context branch: image -> visual tokens -> predicted text embedding
    visual_tokens = image_encoder(imgs)                      # (B, 64, 256)
    pred = predictor(visual_tokens).squeeze(1)               # (B, 256), unit norm
    print("visual tokens", tuple(visual_tokens.shape), "pred", tuple(pred.shape))

    # 4. target branch: caption -> text tokens -> pooled, normalized (no grad)
    with torch.no_grad():
        text_tokens = text_encoder(ids, mask)                # (B, L, 256)
        target = F.normalize(masked_mean(text_tokens, mask), p=2, dim=-1)   # (B, 256)
    print("text tokens", tuple(text_tokens.shape), "target", tuple(target.shape))

    # 5. loss: 1 - cosine similarity between prediction and target
    loss = (1 - (pred * target).sum(-1)).mean()
    loss.backward()
    print("loss", loss.item())
    print("grad reaches image encoder:", image_encoder.convStem1[0].weight.grad is not None)
