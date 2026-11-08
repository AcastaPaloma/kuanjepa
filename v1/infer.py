"""Inference / retrieval eval for a trained checkpoint.

usage:
    python infer.py                                   # standard 5-caption retrieval eval on val
    python infer.py "a dog running on grass" "..."    # eval + text->image search for each query
    python infer.py --ckpt checkpoints/best.pt --out inference/

Writes one PNG per text query (top-5 val images) plus image->text examples to --out.
"""
import argparse
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader import DataLoader
from v1.models.image_encoder import ImageEncoder
from v1.models.text_encoder import TextEncoder
from v1.models.predictor import Predictor
from v1.utilities import masked_mean

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def embed_images(DL, ie, pr, indices, bs=128):
    out = []
    split = DL.ds['test']
    for s in range(0, len(indices), bs):
        imgs = torch.stack([DL.val_tfm(split[i]['image'].convert('RGB')) for i in indices[s:s + bs]])
        out.append(pr(ie(imgs.to(device))).squeeze(1))
    return torch.cat(out)                                         # (N, 256), unit norm


@torch.no_grad()
def embed_texts(DL, te, texts, bs=512):
    out = []
    for s in range(0, len(texts), bs):
        ids, mask = map(torch.stack, zip(*[DL.encode(t) for t in texts[s:s + bs]]))
        ids, mask = ids.to(device), mask.to(device)
        out.append(F.normalize(masked_mean(te(ids, mask), mask), p=2, dim=-1))
    return torch.cat(out)                                         # (M, 256), unit norm


def recall(sims, correct, ks=(1, 5, 10)):
    """sims: (Q, G); correct: (Q, G) bool. R@K = fraction of queries with a correct hit in top K."""
    top = sims.topk(max(ks), dim=1).indices
    hit = correct.gather(1, top)
    return {k: hit[:, :k].any(1).float().mean().item() for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="inference")
    ap.add_argument("--examples", type=int, default=4, help="image->text examples to save")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    DL = DataLoader(); DL._setup_text()
    ck = torch.load(args.ckpt, map_location=device)
    ie = ImageEncoder().to(device); ie.load_state_dict(ck["image_encoder"])
    te = TextEncoder(len(DL.vocab), DL.max_len).to(device); te.load_state_dict(ck["text_encoder"])
    pr = Predictor().to(device); pr.load_state_dict(ck["predictor"])
    for m in (ie, te, pr):
        m.eval()
    print(f"checkpoint: {args.ckpt}  (epoch {ck['epoch']}, val_loss {ck['val_loss']:.4f})")

    ### STANDARD FLICKR30K RETRIEVAL: every val image vs all 5 of its captions ###
    val = DL.val_idx
    img_emb = embed_images(DL, ie, pr, val)                       # (N, 256)
    caps, owner = [], []
    for n, i in enumerate(val):
        for c in DL.captions[i]:
            caps.append(c); owner.append(n)
    owner = torch.tensor(owner, device=device)
    txt_emb = embed_texts(DL, te, caps)                           # (5N, 256)

    sims = img_emb @ txt_emb.t()                                  # (N, 5N)
    N = len(val)
    i2t = recall(sims, owner[None, :] == torch.arange(N, device=device)[:, None])
    t2i = recall(sims.t(), torch.arange(N, device=device)[None, :] == owner[:, None])
    print(f"val: {N} images, {len(caps)} captions")
    print("  image -> text  " + "  ".join(f"R@{k} {v:.3f}" for k, v in i2t.items()))
    print("  text -> image  " + "  ".join(f"R@{k} {v:.3f}" for k, v in t2i.items()))
    print(f"  (random R@1: i2t {5 / len(caps):.4f}, t2i {1 / N:.4f})")

    split = DL.ds['test']
    unk = DL.vocab["<unk>"]

    ### TEXT -> IMAGE search for free-text queries ###
    for qi, q in enumerate(args.queries):
        ids, _ = DL.encode(q)
        words = DL._tokenize_str(q)
        oov = [w for w in words if DL.vocab.get(w, unk) == unk]
        s = (embed_texts(DL, te, [q]) @ img_emb.t()).squeeze(0)
        top = s.topk(5)
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        for ax, sc, j in zip(axes, top.values.tolist(), top.indices.tolist()):
            ax.imshow(split[val[j]]['image']); ax.axis("off")
            ax.set_title(f"{sc:.3f}\n{DL.captions[val[j]][0][:45]}", fontsize=7)
        fig.suptitle(f'"{q}"' + (f"   (unknown words: {', '.join(oov)})" if oov else ""))
        fig.tight_layout()
        path = os.path.join(args.out, f"query_{qi}.png")
        fig.savefig(path, dpi=90); plt.close(fig)
        print(f'\nquery "{q}" -> {path}' + (f"  [unknown words: {oov}]" if oov else ""))
        for r, (sc, j) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            print(f"  {r}. {sc:.3f}  {DL.captions[val[j]][0]}")

    ### IMAGE -> TEXT examples ###
    g = torch.Generator().manual_seed(0)
    picks = torch.randperm(N, generator=g)[:args.examples].tolist()
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3.2 * len(picks)))
    for ax, n in zip(axes if len(picks) > 1 else [axes], picks):
        top = sims[n].topk(5)
        lines = [("✓ " if owner[j].item() == n else "  ") + f"{sc:.3f} {caps[j][:90]}"
                 for sc, j in zip(top.values.tolist(), top.indices.tolist())]
        ax.imshow(split[val[n]]['image']); ax.axis("off")
        ax.text(1.02, 0.5, "\n".join(lines), transform=ax.transAxes, va="center", fontsize=8, family="monospace")
        ax.set_title(f"true: {DL.captions[val[n]][0][:90]}", fontsize=8, loc="left")
    fig.tight_layout()
    path = os.path.join(args.out, "image_to_text.png")
    fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"\nimage->text examples -> {path}")


if __name__ == "__main__":
    main()
