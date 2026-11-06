import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d=256, h=4):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.head_dim = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d)

    def forward(self, x, mask=None):
        # x: (B, N, d); mask: (B, N) bool, True = real token
        attn_mask = None if mask is None else mask[:, None, None, :]
        B, N, d = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                # each (B, h, N, hd)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)       # (B, h, N, hd)
        out = out.transpose(1, 2).reshape(B, N, d)          # merge heads
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d=256, h=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.attn = Attention(d, h)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x