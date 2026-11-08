import torch
import torch.nn as nn
import torch.nn.functional as F

from v1.models.transformer import Block

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(256, 256)

        # learnable query token that reads out the image summary
        self.q = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.trunc_normal_(self.q, std=0.02)

        self.blocks = nn.ModuleList([Block(256) for _ in range(2)])
        self.ln = nn.LayerNorm(256, eps=1e-6)

        self.linear2 = nn.Linear(256, 256)


    def forward(self, visual_tokens):
        batch_count = visual_tokens.shape[0]

        x = self.linear1(visual_tokens)
        q_in = self.q.expand(batch_count, -1, -1)       # (B, 1, 256), same device as the module

        together = torch.cat([x, q_in], dim=1)

        for blk in self.blocks:
            together = blk(together)

        q_out = together[:, -1:, :]                     # the query is appended last

        out = F.normalize(self.linear2(self.ln(q_out)), p=2, dim=-1)
        return out
