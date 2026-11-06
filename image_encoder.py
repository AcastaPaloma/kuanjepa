import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Block

class ImageEncoder(nn.module):
    def __init__(self):
        super().__init__()

        self.convStem1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU()
        )
        self.convStem2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU()
        )
        self.convStem3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.GELU()
        )
        self.convStem4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=256),
        )

        self.convStem = nn.Sequential(
            self.convStem1,
            self.convStem2,
            self.convStem3,
            self.convStem4
        )

        self.pos_emb = nn.Parameter(1, 64, 256)

        # transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(6)])


    def forward(self, raw_image):
        x = self.convStem(raw_image)              # (B, 256, 8, 8)
        tokens = x.flatten(2).transpose(1, 2)     # (B, 64, 256)
        tokens += self.pos_emb                    # (B, 64, 256)

conv_channels = []

### REFERENCE SHEET
# nn.GroupNorm(num_groups=4, num_channels=16)  # 16 channels -> 4 groups of 4
