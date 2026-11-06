import torch
import torchvision
import timm
import datasets

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = torch.device("mps")

x = torch.randn(8, 3, 224, 224).to(device)

