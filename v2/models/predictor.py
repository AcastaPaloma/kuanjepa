import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim=384, heads=6, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, pad=None):
        h = self.norm1(x)                                         # (B, L, dim)
        x = x + self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]   # (B, L, dim)
        x = x + self.mlp(self.norm2(x))                           # (B, L, dim)
        return x


class Predictor(nn.Module):
    def __init__(self, in_dim=1024, dim=384, out_dim=768, depth=6, heads=6, mlp_ratio=4, dropout=0.1, grid=8):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.pos = SinCosPos2D(grid, grid, dim)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, tokens, keep_ids=None, pad=None):
        if keep_ids is not None:
            tokens = torch.gather(tokens, 1, keep_ids[..., None].expand(-1, -1, tokens.size(-1)))   # (B, K, 1024)
        x = self.pos(self.proj(tokens), keep_ids)                 # (B, K, 384)
        q = self.query.expand(x.size(0), -1, -1)                  # (B, 1, 384)
        x = torch.cat([q, x], dim=1)                              # (B, 1 + K, 384)
        if pad is not None:
            pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], dim=1)   # (B, 1 + K)
        for block in self.blocks:
            x = block(x, pad)                                     # (B, 1 + K, 384)
        return self.head(self.norm(x[:, 0]))                      # (B, 768)


class SinCosPos2D(nn.Module):
    """Fixed 2D sin-cos position embedding for an h x w token grid (row-major token order)."""
    def __init__(self, h=16, w=16, dim=1024):
        super().__init__()
        assert dim % 4 == 0
        def sincos_1d(n, d):
            pos = torch.arange(n, dtype=torch.float32)
            omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d // 2))
            ang = pos[:, None] * omega[None]                        # (n, d/2)
            return torch.cat([ang.sin(), ang.cos()], dim=1)         # (n, d)
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        pe = torch.cat([sincos_1d(h, dim // 2)[ys.flatten()],
                        sincos_1d(w, dim // 2)[xs.flatten()]], dim=1)   # (h*w, dim)
        self.register_buffer("pe", pe, persistent=False)            # follows .to(device), not saved

    def forward(self, x, ids=None):
        """x: (B, N, dim). ids: (B, N) flat grid indices in [0, h*w), or None for all h*w tokens."""
        pe = self.pe.to(x.dtype)
        return x + (pe if ids is None else pe[ids])
