import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Block
from dataloader import DataLoader

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=32, d=256, depth=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=0)   # id 0 = <pad>
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d))
        self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])
        self.ln = nn.LayerNorm(d, eps=1e-6)

    def forward(self, ids, mask):
        # ids:  (B, L) long, token ids from get_batch
        # mask: (B, L) bool, True = real token, False = padding
        x = self.tok_emb(ids)                     # (B, L, d)
        x = x + self.pos_emb[:, : ids.shape[1]]   # (B, L, d)

        for blk in self.blocks:
            x = blk(x, mask)                      # padding is not attended to

        return self.ln(x)                         # (B, L, d)

if __name__ == "__main__":
    DL = DataLoader()
    B = 8
    imgs, ids, mask = DL.get_batch(batch_size=B)  # same row i = same image/caption pair
    print(ids.shape, mask.shape)                  # (B, 32), (B, 32)

    model = TextEncoder(vocab_size=len(DL.vocab), max_len=DL.max_len)
    output = model(ids, mask)
    print(output.shape)                           # (B, 32, 256)
