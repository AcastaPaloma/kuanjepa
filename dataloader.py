from datasets import load_dataset
from torchvision import transforms

import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self):
        self.ds = load_dataset("nlphuji/flickr30k")
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.4, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def get_one_sample(self):
        img = self.ds['test'][0]['image']
        x = self.train_tfm(img)

        return x
