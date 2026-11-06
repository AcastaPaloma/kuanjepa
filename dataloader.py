import json
import re
import torch
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
        x = self.train_tfm(img.convert('RGB'))
        return x                                  # (3, 128, 128)

    @staticmethod
    def _tokenize_str(text):
        return re.findall(r"[a-z0-9'’]+", text.lower())

    @staticmethod
    def _captions(raw):
        return json.loads(raw) if isinstance(raw, str) else list(raw)   # caption may be a JSON-encoded list

    def _setup_text(self, max_len=32):
        """Lazily build vocab, caption cache and train indices (no image decoding)."""
        if hasattr(self, "vocab"):
            return
        self.max_len = max_len
        self.vocab = self.get_v()
        meta = self.ds['test'].select_columns(['caption', 'split'])
        self.captions = [self._captions(c) for c in meta['caption']]
        self.train_idx = [i for i, s in enumerate(meta['split']) if s == 'train']

    def encode(self, text):
        """str -> (ids, mask), both (max_len,). mask is True on real tokens."""
        unk = self.vocab["<unk>"]
        ids = [self.vocab.get(w, unk) for w in self._tokenize_str(text)][: self.max_len]
        mask = [True] * len(ids) + [False] * (self.max_len - len(ids))
        ids = ids + [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(mask)

    def get_batch(self, batch_size=32, indices=None):
        """Returns (images, token_ids, text_mask), row i of each belongs to the same
        Flickr30k image i; the caption is one of its 5, picked at random."""
        self._setup_text()
        split = self.ds['test']
        if indices is None:
            pick = torch.randint(0, len(self.train_idx), (batch_size,)).tolist()
            indices = [self.train_idx[j] for j in pick]

        imgs, ids, masks = [], [], []
        for i in indices:
            imgs.append(self.train_tfm(split[i]['image'].convert('RGB')))
            caps = self.captions[i]
            t, m = self.encode(caps[torch.randint(0, len(caps), (1,)).item()])
            ids.append(t)
            masks.append(m)
        return torch.stack(imgs), torch.stack(ids), torch.stack(masks)
        # (B, 3, 128, 128), (B, max_len) long, (B, max_len) bool

    def get_v(self, min_count=5):
        from collections import Counter

        # select_columns avoids decoding images; only the 'test' key exists (it holds all splits)
        meta = self.ds['test'].select_columns(['caption', 'split'])

        counts = Counter()
        for splt, caps in zip(meta['split'], meta['caption']):
            if splt != 'train':
                continue
            caps = self._captions(caps)
            for c in caps:                            # 5 captions per image
                counts.update(self._tokenize_str(c))

        vocab = {"<pad>": 0, "<unk>": 1}
        for w, n in counts.items():
            if n >= min_count:
                vocab[w] = len(vocab)

        print("V =", len(vocab))
        return vocab


if __name__ == "__main__":
    DL = DataLoader()
    imgs, ids, mask = DL.get_batch(4)
    print(imgs.shape, ids.shape, mask.shape)
