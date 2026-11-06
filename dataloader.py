from datasets import load_dataset
from torchvision import transforms

import matplotlib.pyplot as plt

ds = load_dataset("nlphuji/flickr30k")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.4, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

tfm = transforms.Compose([
    transforms.ToTensor(),
])

img = ds['test'][0]['image']
test_image = tfm(img)
x_vis = test_image.permute(1, 2, 0).contiguous().numpy()  # (128, 128, 3)

plt.imshow(x_vis)
plt.axis("off")
plt.show()

# img = ex["image"]