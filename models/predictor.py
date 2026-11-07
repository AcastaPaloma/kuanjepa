import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Block

device = torch.device("mps")

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
        q_in= torch.zeros(size=(batch_count, 1, 256)).to(device)

        together = torch.cat([x, q_in], dim=1)

        for blk in self.blocks:
            together = blk(together)

        q_out = together[:, :1, :]

        out = F.normalize(self.linear2(self.ln(q_out)), p=2, dim=-1)
        return out