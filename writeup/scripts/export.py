"""Builds everything the write-up's Results tab and animation read, from checkpoints/best.pt.

usage (from the repo root, in an env with torch + datasets):
    python writeup/scripts/export.py --candidates        # contact sheet of featured-image candidates
    python writeup/scripts/export.py --featured 123      # everything, with val image 123 as the traced example

Writes into writeup/data/:
    model.bin, model.json     weights for infer.js (int8 per-row matrices, f32 vectors)
    vocab.json                id -> word
    refs.bin, refs.json       every val caption's embedding (int8 per-row + f32 scales), captions, gallery
    gallery/NN.jpg, NN-128.png
    trace.json, trace/*       one image + caption through the whole forward pass
    metrics.json              the training run, plus the 5-caption retrieval eval of best.pt
and writeup/scripts/fixtures/ for verify-infer.mjs.
"""
import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from dataloader import DataLoader
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.predictor import Predictor
from utilities import masked_mean

torch.set_grad_enabled(False)
OUT = os.path.join(ROOT, "writeup", "data")
FIX = os.path.join(ROOT, "writeup", "scripts", "fixtures")
RUN = os.path.join(ROOT, "logs", "20261107-033343")
SCRATCH = os.environ.get("SCRATCH", "/tmp")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def r4(x):
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, list):
        return [r4(v) for v in x]
    return round(float(x), 4)


# ---------------------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self):
        self.DL = DL = DataLoader()
        DL._setup_text()
        self.split = DL.ds["test"]
        self.ck = torch.load("checkpoints/best.pt", map_location="cpu")
        self.scale = math.exp(self.ck["criterion"]["t"].item())
        self.ie, self.te, self.pr = self.build(self.ck["image_encoder"], self.ck["text_encoder"], self.ck["predictor"])
        self.val = DL.val_idx
        self.caps, self.owner = [], []
        for n, i in enumerate(self.val):
            for c in DL.captions[i]:
                self.caps.append(c)
                self.owner.append(n)
        self.owner_t = torch.tensor(self.owner)
        self._inputs = None

    def build(self, sie, ste, spr):
        ie = ImageEncoder(); ie.load_state_dict(sie)
        te = TextEncoder(len(self.DL.vocab), self.DL.max_len); te.load_state_dict(ste)
        pr = Predictor(); pr.load_state_dict(spr)
        for m in (ie, te, pr):
            m.eval()
        return ie, te, pr

    def pil(self, n):
        return self.split[self.val[n]]["image"].convert("RGB")

    def inputs(self):
        """val_tfm tensors for all 1014 val images, cached."""
        if self._inputs is None:
            path = os.path.join(SCRATCH, "val_inputs.pt")
            if os.path.exists(path):
                self._inputs = torch.load(path)
            else:
                self._inputs = torch.stack([self.DL.val_tfm(self.pil(n)) for n in range(len(self.val))])
                torch.save(self._inputs, path)
        return self._inputs

    def embed_images(self, x, ie=None, pr=None):
        ie, pr = ie or self.ie, pr or self.pr
        return torch.cat([pr(ie(x[s:s + 128])).squeeze(1) for s in range(0, len(x), 128)])

    def embed_texts(self, texts, te=None):
        te = te or self.te
        out = []
        for s in range(0, len(texts), 512):
            ids, mask = map(torch.stack, zip(*[self.DL.encode(t) for t in texts[s:s + 512]]))
            out.append(F.normalize(masked_mean(te(ids, mask), mask), p=2, dim=-1))
        return torch.cat(out)


def recall(sims, correct, ks=(1, 5, 10)):
    top = sims.topk(max(ks), dim=1).indices
    hit = correct.gather(1, top)
    return {str(k): round(hit[:, :k].any(1).float().mean().item(), 4) for k in ks}


def retrieval(ctx, img, txt):
    N = img.shape[0]
    sims = img @ txt.t()
    i2t = recall(sims, ctx.owner_t[None, :] == torch.arange(N)[:, None])
    t2i = recall(sims.t(), torch.arange(N)[None, :] == ctx.owner_t[:, None])
    return i2t, t2i


def denorm_u8(x):
    return (x * STD + MEAN).clamp(0, 1).mul(255).round().to(torch.uint8)       # (3, H, W)


def to_pil(u8):
    return Image.fromarray(u8.permute(1, 2, 0).numpy())


def crop(pil, size):
    return transforms.CenterCrop(size)(transforms.Resize(size)(pil))


# ---------------------------------------------------------------------------------------------------------------------
def quantize(sd_by_module):
    """-> (tensors meta, bytes, dequantised state dicts). int8 per output row for >=2D matrices, f32 otherwise."""
    meta, chunks, offset, deq = [], [], 0, {}
    def put(raw):
        nonlocal offset
        pad = (-len(raw)) % 4
        chunks.append(raw + b"\0" * pad)
        start = offset
        offset += len(raw) + pad
        return start
    for prefix, sd in sd_by_module.items():
        deq[prefix] = {}
        for name, t in sd.items():
            t = t.float()
            if prefix == "image_encoder" and name.startswith("convStem."):
                deq[prefix][name] = None        # alias of convStem1..4, written once
                continue
            full = f"{prefix}.{name}"
            if t.dim() >= 2 and not name.endswith("pos_emb") and name != "q":
                rows = t.reshape(t.shape[0], -1)
                s = rows.abs().amax(1) / 127
                s[s == 0] = 1
                q = (rows / s[:, None]).round().clamp(-127, 127).to(torch.int8)
                o = put(q.numpy().tobytes())
                so = put(s.numpy().astype("<f4").tobytes())
                meta.append({"name": full, "shape": list(t.shape), "dtype": "int8", "offset": o, "scaleOffset": so})
                deq[prefix][name] = (q.float() * s[:, None]).reshape(t.shape)
            else:
                o = put(t.numpy().astype("<f4").tobytes())
                meta.append({"name": full, "shape": list(t.shape), "dtype": "f32", "offset": o})
                deq[prefix][name] = t
        # restore the aliases
        for name in list(deq[prefix]):
            if deq[prefix][name] is None:
                m = re.match(r"convStem\.(\d)\.(.*)", name)
                deq[prefix][name] = deq[prefix][f"convStem{int(m.group(1)) + 1}.{m.group(2)}"]
    return meta, b"".join(chunks), deq


def export_model(ctx):
    sds = {"image_encoder": ctx.ck["image_encoder"], "text_encoder": ctx.ck["text_encoder"], "predictor": ctx.ck["predictor"]}
    meta, blob, deq = quantize(sds)
    with open(os.path.join(OUT, "model.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(OUT, "model.json"), "w") as f:
        json.dump({"scale": ctx.scale, "vocab_size": len(ctx.DL.vocab), "max_len": ctx.DL.max_len, "dim": 256,
                   "note": "image_encoder.convStem.K.* duplicates convStem(K+1).* and is not stored",
                   "tensors": meta}, f)
    words = [None] * len(ctx.DL.vocab)
    for w, i in ctx.DL.vocab.items():
        words[i] = w
    with open(os.path.join(OUT, "vocab.json"), "w") as f:
        json.dump(words, f, ensure_ascii=False)
    print(f"model.bin {len(blob) / 1e6:.2f} MB, {len(meta)} tensors")

    # parity: dequantised vs fp32
    x = ctx.inputs()
    img32, txt32 = ctx.embed_images(x), ctx.embed_texts(ctx.caps)
    qie, qte, qpr = ctx.build(deq["image_encoder"], deq["text_encoder"], deq["predictor"])
    imgq, txtq = ctx.embed_images(x, qie, qpr), ctx.embed_texts(ctx.caps, qte)
    n = torch.arange(200)
    c32 = (img32[n] * txt32[n * 5]).sum(1)
    cq = (imgq[n] * txtq[n * 5]).sum(1)
    a, b = retrieval(ctx, img32, txt32), retrieval(ctx, imgq, txtq)
    print(f"quant parity: max|dcos| {(c32 - cq).abs().max().item():.4f}")
    print(f"  fp32  i2t {a[0]}  t2i {a[1]}")
    print(f"  int8  i2t {b[0]}  t2i {b[1]}")
    return img32, txt32


# ---------------------------------------------------------------------------------------------------------------------
GALLERY_CATS = [
    ("dog", ["dog"]), ("kids", ["child", "children", "kid", "boy", "girl"]), ("bike", ["bike", "bicycle", "cyclist", "biker"]),
    ("water", ["ocean", "beach", "surf", "lake", "swim", "pool"]), ("city", ["street", "city", "sidewalk"]),
    ("food", ["food", "cook", "eating", "kitchen", "restaurant"]), ("ball", ["soccer", "football", "basketball", "baseball", "tennis"]),
    ("snow", ["snow", "ski"]), ("music", ["guitar", "drum", "band", "singing", "singer"]), ("horse", ["horse"]),
    ("climb", ["climb", "climbing", "rock"]), ("boat", ["boat"]), ("crowd", ["crowd"]), ("market", ["market", "vendor"]),
    ("work", ["worker", "construction"]), ("dance", ["dance", "dancing", "dancer"]),
]


def pick_gallery(ctx, avoid=()):
    rng = random.Random(0)
    used, picks = set(avoid), []
    for cat, words in GALLERY_CATS:
        pool = [n for n in range(len(ctx.val)) if n not in used
                and set(ctx.DL._tokenize_str(ctx.DL.captions[ctx.val[n]][0])) & set(words)]
        rng.shuffle(pool)
        for n in pool[:3]:
            used.add(n)
            picks.append((cat, n))
    return picks


def export_refs(ctx, txt32, featured):
    gdir = os.path.join(OUT, "gallery")
    os.makedirs(gdir, exist_ok=True)
    x = ctx.inputs()
    picks = pick_gallery(ctx, avoid={featured})
    gallery = []
    for k, (cat, n) in enumerate(picks):
        name = f"{k:02d}"
        crop(ctx.pil(n), 320).save(os.path.join(gdir, f"{name}.jpg"), quality=85)
        to_pil(denorm_u8(x[n])).save(os.path.join(gdir, f"{name}-128.png"))
        gallery.append({"file": name, "val": n, "category": cat, "captions": ctx.DL.captions[ctx.val[n]]})
    # contact sheet to eyeball
    sheet = Image.new("RGB", (8 * 160, 6 * 160), "white")
    for k, g in enumerate(gallery):
        sheet.paste(Image.open(os.path.join(gdir, f"{g['file']}.jpg")).resize((160, 160)), ((k % 8) * 160, (k // 8) * 160))
    sheet.save(os.path.join(SCRATCH, "gallery_sheet.jpg"))

    t = txt32
    s = t.abs().amax(1) / 127
    q = (t / s[:, None]).round().clamp(-127, 127).to(torch.int8)
    raw = q.numpy().tobytes()
    with open(os.path.join(OUT, "refs.bin"), "wb") as f:
        f.write(raw)
        f.write(s.numpy().astype("<f4").tobytes())
    with open(os.path.join(OUT, "refs.json"), "w") as f:
        json.dump({"n": len(ctx.caps), "dim": 256, "images": len(ctx.val), "int8Offset": 0, "scaleOffset": len(raw),
                   "captions": ctx.caps, "owner": ctx.owner, "gallery": gallery}, f, ensure_ascii=False)
    print(f"gallery {len(gallery)} images; refs {len(ctx.caps)} captions")
    return gallery


# ---------------------------------------------------------------------------------------------------------------------
def candidates(ctx, img32, txt32):
    sims = img32 @ txt32.t()
    top = sims.argmax(1)
    words = {"dog", "boy", "girl", "cyclist", "biker", "surfer", "skier", "child", "horse", "skateboarder", "snowboarder"}
    cands = [n for n in range(len(ctx.val)) if ctx.owner[top[n]] == n
             and set(ctx.DL._tokenize_str(ctx.DL.captions[ctx.val[n]][0])) & words]
    print(len(cands), "rank-1 candidates with a clear subject")
    cands = cands[:24]
    sheet = Image.new("RGB", (6 * 220, 4 * 250), "white")
    d = ImageDraw.Draw(sheet)
    for k, n in enumerate(cands):
        x, y = (k % 6) * 220, (k // 6) * 250
        sheet.paste(crop(ctx.pil(n), 210), (x + 5, y + 5))
        d.text((x + 6, y + 218), f"#{n}: {ctx.caps[top[n]][:34]}", fill="black")
    path = os.path.join(SCRATCH, "candidates.jpg")
    sheet.save(path)
    for n in cands:
        print(n, "|", ctx.caps[top[n]])
    print("->", path)


# ---------------------------------------------------------------------------------------------------------------------
def attn_probs(blk, x, mask=None):
    """Recompute the softmax attention of blk on input x (1, N, d); check it reproduces blk.attn."""
    h = blk.norm1(x)
    a = blk.attn
    B, N, d = h.shape
    qkv = a.qkv(h).view(B, N, 3, a.h, a.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(a.head_dim)
    if mask is not None:
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
    p = s.softmax(-1)                                            # (1, h, N, N)
    out = a.proj((p @ v).transpose(1, 2).reshape(B, N, d))
    ref = a(h, mask)
    assert (out - ref).abs().max().item() < 1e-4, (out - ref).abs().max().item()
    return p[0]


def sprite_rows(stages):
    """list of (N, 256) -> uint8 rows, per-stage absmax encoding."""
    absmax = [s.abs().max().item() for s in stages]
    rows = [torch.round(127.5 + 127.5 * s / m).clamp(0, 255) for s, m in zip(stages, absmax)]
    return torch.cat(rows).to(torch.uint8).numpy(), absmax


def export_trace(ctx, img32, txt32, featured, gallery):
    tdir = os.path.join(OUT, "trace")
    os.makedirs(tdir, exist_ok=True)
    ie, te, pr = ctx.ie, ctx.te, ctx.pr
    n = featured
    caps5 = ctx.DL.captions[ctx.val[n]]
    pil = ctx.pil(n)
    crop(pil, 512).save(os.path.join(tdir, "image.jpg"), quality=88)
    x = ctx.inputs()[n:n + 1]
    to_pil(denorm_u8(x[0])).save(os.path.join(tdir, "input.png"))

    # ---- image encoder
    conv, h = [], x
    for k, stage in enumerate([ie.convStem1, ie.convStem2, ie.convStem3, ie.convStem4]):
        h = stage(h)
        m = h[0]                                                 # (C, S, S)
        picks = m.flatten(1).var(1).topk(8).indices.tolist()
        S = m.shape[-1]
        strip = []
        for c in picks:
            fm = m[c]
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            strip.append(fm)
        Image.fromarray((torch.cat(strip, 1) * 255).round().to(torch.uint8).numpy(), "L").save(os.path.join(tdir, f"conv{k + 1}.png"))
        conv.append({"size": S, "channels": m.shape[0], "sprite": f"trace/conv{k + 1}.png", "picks": picks, "n": 8})
    tok = h.flatten(2).transpose(1, 2)                           # (1, 64, 256)
    stages = [tok[0].clone()]
    tok = tok + ie.pos_emb
    stages.append(tok[0].clone())
    attn = []
    for blk in ie.blocks:
        attn.append(attn_probs(blk, tok))
        tok = blk(tok)
        stages.append(tok[0].clone())
    tok = ie.ln(tok)
    stages.append(tok[0].clone())
    assert (tok - ie(x)).abs().max().item() < 1e-4
    rows, absmax = sprite_rows(stages)
    Image.fromarray(rows, "L").save(os.path.join(tdir, "img_tokens.png"))
    A = torch.stack(attn)                                        # (6, 4, 64, 64)
    A = A / A.amax(-1, keepdim=True)                             # each head's row scaled to its own max
    Image.fromarray((A.permute(0, 2, 1, 3).reshape(6 * 64, 4 * 64) * 255).round().to(torch.uint8).numpy(), "L").save(os.path.join(tdir, "img_attn.png"))

    # ---- predictor
    p = pr.linear1(tok)
    p = torch.cat([p, pr.q.expand(1, -1, -1)], 1)                # (1, 65, 256), q last
    q_attn = []
    for blk in pr.blocks:
        q_attn.append(attn_probs(blk, p)[:, 64, :])               # (4, 65)
        p = blk(p)
    pvec = F.normalize(pr.linear2(pr.ln(p[:, -1:, :])), dim=-1)[0, 0]
    assert (pvec - img32[n]).abs().max().item() < 1e-4

    # ---- text: the featured image's best-scoring caption
    own = [5 * n + j for j in range(5)]
    scores = (txt32[own] @ img32[n]).tolist()
    used = int(np.argmax(scores))
    caption = caps5[used]
    words = ctx.DL._tokenize_str(caption)[: ctx.DL.max_len]
    unk = ctx.DL.vocab["<unk>"]
    ids = [ctx.DL.vocab.get(w, unk) for w in words]
    L = len(ids)
    vocab_words = [None] * len(ctx.DL.vocab)
    for w, i in ctx.DL.vocab.items():
        vocab_words[i] = w
    neighbors = [[[j, vocab_words[j]] for j in range(max(0, i - 2), min(len(vocab_words), i + 3))] for i in ids]
    ids_t = torch.tensor([ids])
    mask = torch.ones(1, L, dtype=torch.bool)
    t = te.tok_emb(ids_t)
    tst = [t[0].clone()]
    t = t + te.pos_emb[:, :L]
    tst.append(t[0].clone())
    tattn = []
    for blk in te.blocks:
        tattn.append(attn_probs(blk, t, mask))
        t = blk(t, mask)
        tst.append(t[0].clone())
    t = te.ln(t)
    tst.append(t[0].clone())
    pooled = masked_mean(t, mask)[0]
    tvec = F.normalize(pooled, dim=-1)
    assert (tvec - txt32[5 * n + used]).abs().max().item() < 1e-4
    rows, tabsmax = sprite_rows(tst)
    Image.fromarray(rows, "L").save(os.path.join(tdir, "txt_tokens.png"))

    # ---- a batch of 8: featured + 7 from different gallery categories
    others, seen = [], set()
    for g in gallery:
        if g["category"] not in seen and g["val"] != n and len(others) < 7:
            seen.add(g["category"]); others.append(g["val"])
    batch = [n] + others
    items = []
    for k, b in enumerate(batch):
        crop(ctx.pil(b), 160).save(os.path.join(tdir, f"b{k}.jpg"), quality=85)
        items.append({"thumb": f"trace/b{k}.jpg", "caption": caption if k == 0 else ctx.DL.captions[ctx.val[b]][0], "val": b})
    bt = ctx.embed_texts([it["caption"] for it in items])
    S = img32[batch] @ bt.t()
    logits = ctx.scale * S
    labels = torch.arange(8)
    loss_i, loss_t = F.cross_entropy(logits, labels).item(), F.cross_entropy(logits.t(), labels).item()

    # ---- image -> text over the pool
    sims = txt32 @ img32[n]
    top = sims.topk(5)
    trace = {
        "image": {"val": n, "display": "trace/image.jpg", "input": "trace/input.png", "captions": caps5, "caption_used": used},
        "conv": conv,
        "tokens": {"sprite": "trace/img_tokens.png", "stages": ["stem", "pos", "b1", "b2", "b3", "b4", "b5", "b6", "ln"],
                   "absmax": r4(absmax), "norms": r4([s.norm(dim=1) for s in stages])},
        "attn": {"sprite": "trace/img_attn.png", "blocks": 6, "heads": 4, "n": 64,
                 "note": "rows block*64+query, cols head*64+key; each (block, head, query) row scaled to its own max"},
        "predictor": {"q_attn": r4(torch.stack(q_attn)), "vec": r4(pvec)},
        "text": {"caption": caption, "words": words, "ids": ids, "unknown": [w for w, i in zip(words, ids) if i == unk],
                 "neighbors": neighbors, "sprite": "trace/txt_tokens.png",
                 "stages": ["emb", "pos", "b1", "b2", "b3", "b4", "b5", "b6", "ln"], "absmax": r4(tabsmax),
                 "attn": r4(torch.stack(tattn)), "pooled": r4(pooled), "vec": r4(tvec)},
        "score": r4((pvec @ tvec).item()), "scale": r4(ctx.scale),
        "batch": {"items": items, "S": r4(S), "probs_rows": r4(logits.softmax(1)), "probs_cols": r4(logits.softmax(0)),
                  "loss_i": r4(loss_i), "loss_t": r4(loss_t), "loss": r4(0.5 * (loss_i + loss_t))},
        "top": [{"caption": ctx.caps[j], "score": r4(s), "own": ctx.owner[j] == n} for s, j in zip(top.values.tolist(), top.indices.tolist())],
    }
    with open(os.path.join(OUT, "trace.json"), "w") as f:
        json.dump(trace, f, ensure_ascii=False)
    print(f"trace: val {n}, caption {used}: {caption!r}, score {trace['score']}, batch loss {trace['batch']['loss']}")
    print("  top:", [(d["score"], d["own"], d["caption"][:40]) for d in trace["top"]])


# ---------------------------------------------------------------------------------------------------------------------
def export_fixtures(ctx, gallery):
    os.makedirs(FIX, exist_ok=True)
    x = ctx.inputs()
    imgs, texts = [], []
    for k, g in enumerate(gallery[::6][:8]):
        u8 = denorm_u8(x[g["val"]])                                          # what the browser gets
        rgba = torch.cat([u8, torch.full((1, 128, 128), 255, dtype=torch.uint8)]).permute(1, 2, 0).contiguous()
        with open(os.path.join(FIX, f"img{k}.bin"), "wb") as f:
            f.write(rgba.numpy().tobytes())
        xin = (u8.float() / 255 - MEAN) / STD
        imgs.append({"file": f"img{k}.bin", "vec": ctx.embed_images(xin[None])[0].tolist()})
        texts.append(g["captions"][0])
    texts += ["a dog runs on the grass", "two people sitting on a bench", "Xylophonist zzqx playing!!"]
    tv = ctx.embed_texts(texts)
    iv = torch.tensor([i["vec"] for i in imgs])
    with open(os.path.join(FIX, "expected.json"), "w") as f:
        json.dump({"images": imgs, "captions": [{"text": t, "vec": v} for t, v in zip(texts, tv.tolist())],
                   "scores": (iv @ tv.t()).tolist()}, f)
    print(f"fixtures: {len(imgs)} images, {len(texts)} captions")


# ---------------------------------------------------------------------------------------------------------------------
def export_metrics(ctx, img32, txt32):
    epochs = []
    temps = {}
    sec, steps = [], []
    with open(os.path.join(RUN, "metrics.csv")) as f:
        next(f)
        for line in f:
            e, step, g, split, loss, temp, lr, sps = line.strip().split(",")
            if split == "val":
                temps[int(e)] = float(temp)
            else:
                steps.append((int(g), float(loss)))
                if sps:
                    sec.append(float(sps))
    pat = re.compile(r"=== epoch (\d+) train_loss ([\d.]+) val_loss ([\d.]+) .*?i2t R@1 ([\d.]+) R@5 ([\d.]+)")
    stamps = []
    with open(os.path.join(RUN, "train.log")) as f:
        for line in f:
            stamps.append(line[:8])
            m = pat.search(line)
            if m:
                e = int(m.group(1))
                epochs.append({"epoch": e, "train_loss": float(m.group(2)), "val_loss": float(m.group(3)),
                               "r1": float(m.group(4)), "r5": float(m.group(5)), "temp": temps.get(e)})
    t0, t1 = (datetime.strptime(s, "%H:%M:%S") for s in (stamps[0], stamps[-1]))
    hours = ((t1 - t0).total_seconds() % 86400) / 3600
    k = max(1, len(steps) // 1500)
    best = min(epochs, key=lambda e: e["val_loss"])
    i2t, t2i = retrieval(ctx, img32, txt32)
    count = lambda m: sum(p.numel() for p in m.parameters())
    params = {"image_encoder": count(ctx.ie), "text_encoder": count(ctx.te), "predictor": count(ctx.pr)}
    params["total"] = sum(params.values())
    out = {"epochs": epochs, "steps": [[g, round(l, 4)] for g, l in steps[::k]], "best_epoch": best["epoch"],
           "best_val_loss": best["val_loss"], "chance": round(math.log(len(ctx.val)), 4),
           "final": {"i2t": i2t, "t2i": t2i, "images": len(ctx.val), "captions": len(ctx.caps),
                     "chance_r1": {"i2t": round(5 / len(ctx.caps), 5), "t2i": round(1 / len(ctx.val), 5)}},
           "params": params, "vocab": len(ctx.DL.vocab), "train_images": len(ctx.DL.train_idx),
           "batch_size": 256, "epochs_run": len(epochs), "steps_per_epoch": 113,
           "sec_per_step": round(statistics.median(sec), 3), "hours": round(hours, 2), "scale": round(ctx.scale, 4)}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(out, f)
    print(f"metrics: best epoch {best['epoch']} ({best['val_loss']}), {hours:.2f} h; final i2t {i2t} t2i {t2i}; params {params}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--featured", type=int)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ctx = Ctx()
    if args.candidates:
        x = ctx.inputs()
        candidates(ctx, ctx.embed_images(x), ctx.embed_texts(ctx.caps))
        return
    assert args.featured is not None, "pass --featured N (see --candidates)"
    img32, txt32 = export_model(ctx)
    gallery = export_refs(ctx, txt32, args.featured)
    export_trace(ctx, img32, txt32, args.featured, gallery)
    export_fixtures(ctx, gallery)
    export_metrics(ctx, img32, txt32)


if __name__ == "__main__":
    main()
