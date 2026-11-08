import torch


class RawPairDataset(torch.utils.data.Dataset):
    """Flickr30k index -> (raw PIL RGB image, list of all its captions).

    No resizing / augmentation / normalization here: the V-JEPA 2 processor does its own
    (resize + center-crop to 256, ImageNet mean/std) inside embed_images.
    Reuses the root DataLoader only for the HF dataset, captions and split indices."""
    def __init__(self, DL, indices):
        DL._setup_text()
        self.DL, self.indices = DL, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        return self.DL.ds['test'][i]['image'].convert('RGB'), list(self.DL.captions[i])


def collate_pil(batch):
    """-> (list of B PIL images, list of B lists of caption strings).
    torch's default collate can't stack PIL images and would transpose the caption lists."""
    imgs, caps = zip(*batch)
    return list(imgs), list(caps)
