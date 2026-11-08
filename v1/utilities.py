import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class InfoNCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, visual_tokens, target):
        S = self.t.exp() * (visual_tokens @ target.t()) ## (B, B)

        labels = torch.arange(S.size(0), device=S.device)

        loss_i = F.cross_entropy(S, labels)
        loss_t = F.cross_entropy(S.t(), labels)
        return 0.5 * (loss_i + loss_t)

def masked_mean(x, mask):
    # x: (B, L, d), mask: (B, L) bool -> (B, d) mean over real tokens only
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


