"""Everything the v2 Results tab needs, computed once on the Mac (V-JEPA 2 and EmbeddingGemma are far too big to run
in a browser, so the page plays back what was recorded here).

  writeup/data/v2results.json   training curves, final retrieval, the two baselines
  writeup/data/reveal.json      the Reveal game: for each photo, 64 reveal steps, masked before vs after the encoder
  writeup/data/arith.json/.bin  image arithmetic: every validation photo's predicted vector + a word list's embeddings
  writeup/data/v2/val/*.jpg     small thumbnails of the 1,014 validation photos

usage (from the repo root, nothing else using the GPU):
  python writeup/scripts/export_v2_results.py --candidates   # contact sheet of Reveal candidates -> $SCRATCH
  python writeup/scripts/export_v2_results.py                # everything
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "v2"
OUT = ROOT / "writeup" / "data"
SCRATCH = Path(os.environ.get("SCRATCH", "/tmp")) / "v2-export"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(ROOT))
import train_masked as tm                                              # noqa: E402  encode_visible, to_video, ...
from models.predictor import Predictor                                 # noqa: E402
from utilities import retrieval_metrics, embed_texts                   # noqa: E402

dev = tm.device
REVEAL = [140, 58, 44, 14, 71, 166, 180, 41, 105, 116]                 # validation positions, picked from --candidates
WORDS = """dog cat horse cow sheep bird fish puppy
man woman boy girl child baby people crowd family couple team police soldier clown
beach snow water ocean lake river pool rain grass field forest mountain desert sand
city street road building bridge park playground stadium stage kitchen restaurant market church
night sunset sunny winter summer
bike bicycle motorcycle car truck bus boat surfboard skateboard skis horseback train
ball frisbee guitar drum microphone camera umbrella kite book phone flag
red blue green yellow black white pink orange
running jumping swimming climbing dancing singing eating sleeping playing riding walking sitting surfing skiing fishing cooking
hat helmet dress suit uniform costume glasses
old young happy""".split()


def r4(v):
    return [round(float(x), 4) for x in v]


def q8(mat):
    """(n, d) float -> int8 rows + float32 per-row scales, as bytes."""
    mat = mat.float()
    scale = mat.abs().amax(1).clamp(min=1e-8) / 127
    q = (mat / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q.numpy().tobytes(), scale.numpy().astype(np.float32).tobytes()


@torch.no_grad()
def predict(predictor, tokens, keep=None):
    pad = None if keep is None else torch.zeros_like(keep, dtype=torch.bool)
    return predictor(tokens, keep, pad)


def main(candidates=False):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    text = torch.load(V2 / "cache" / "text.pt")
    val_px = np.load(V2 / "cache" / "val_pixels.npy")                  # (1014, 3, 256, 256) uint8
    val_caps = text["val_captions"].to(dev)                            # (5070, 768)
    owner = text["val_owner"].to(dev)
    cap_text = text["val_caption_text"]

    from transformers import AutoModel
    vjepa = AutoModel.from_pretrained(tm.VJEPA, dtype=torch.float16).to(dev).eval()
    ck = torch.load(V2 / "checkpoints" / "best_masked.pt", map_location=dev)
    predictor = Predictor().to(dev).eval()
    predictor.load_state_dict(ck["predictor"])
    print("checkpoint epoch", ck["epoch"])

    ### every validation photo, whole ###
    vt = SCRATCH / "val_cells.pt"
    if not vt.exists():
        torch.save(tm.encode_full(vjepa, val_px).cpu(), vt)
    val_cells = torch.load(vt).to(dev)                                 # (1014, 64, 1024)
    val_pred = torch.cat([predict(predictor, val_cells[s:s + 256]) for s in range(0, 1014, 256)])   # (1014, 768)
    i2t, t2i = retrieval_metrics(val_pred, val_caps, owner)
    sims = F.normalize(val_pred, dim=-1) @ F.normalize(val_caps, dim=-1).t()   # (1014, 5070)
    own_rank = []                                                       # best rank of any own caption, per photo
    for n in range(1014):
        order = sims[n].argsort(descending=True)
        own_rank.append(int((owner[order] == n).nonzero()[0]) + 1)
    print("i2t", i2t, "t2i", t2i)

    if candidates:
        ok = [n for n in range(1014) if own_rank[n] == 1]
        pick = ok[:60]
        sheet = Image.new("RGB", (10 * 128, 6 * 140), "white")
        from PIL import ImageDraw
        d = ImageDraw.Draw(sheet)
        for j, n in enumerate(pick):
            im = Image.fromarray(val_px[n].transpose(1, 2, 0)).resize((120, 120))
            sheet.paste(im, ((j % 10) * 128, (j // 10) * 140))
            d.text(((j % 10) * 128 + 2, (j // 10) * 140 + 122), str(n), fill="black")
        sheet.save(SCRATCH / "reveal_candidates.jpg")
        print("contact sheet ->", SCRATCH / "reveal_candidates.jpg", "rank-1 photos:", len(ok))
        return

    ### the training captions, all ~145k, as the pool the guesses come from ###
    tp = SCRATCH / "train_caps.pt"
    from dataloader import DataLoader
    DL = DataLoader()
    DL._setup_text()
    train_caps = [c for i in DL.train_idx for c in DL.captions[i]]
    if not tp.exists():
        from sentence_transformers import SentenceTransformer
        gemma = SentenceTransformer("google/embeddinggemma-300m", device=str(dev))
        torch.save({"emb": embed_texts(train_caps, gemma).half(), "words": embed_texts(WORDS, gemma)}, tp)
        del gemma
    pool = torch.load(tp)
    pool_emb = F.normalize(pool["emb"].to(dev).float(), dim=-1)        # (~145k, 768)
    word_emb = pool["words"]                                            # (W, 768), layer-normed like the targets

    def guesses(vecs, k=3):
        s = F.normalize(vecs, dim=-1) @ pool_emb.t()
        top = s.topk(k, dim=1)
        return top.indices.tolist(), top.values.tolist()

    def rank_own(vecs, n):
        s = F.normalize(vecs, dim=-1) @ F.normalize(val_caps, dim=-1).t()   # (t, 5070)
        mine = s[:, owner == n].max(1).values                                # best own caption
        return ((s > mine[:, None]).sum(1) + 1).tolist()

    ### Reveal: 64 steps per photo, masked before vs after the encoder ###
    used = {}                                                           # pool index -> position in the shipped list
    reveal = []
    for n in REVEAL:
        g = torch.Generator().manual_seed(1000 + n)
        order = torch.randperm(64, generator=g)
        video = tm.to_video(torch.from_numpy(val_px[n:n + 1]))
        full = val_cells[n:n + 1]                                       # (1, 64, 1024) from the whole photo
        before, after = [], []
        for t in range(1, 65):
            cells = order[:t].sort().values.to(dev)[None]               # (1, t)
            feats = tm.encode_visible(vjepa, video, tm.cells_to_patches(cells)).float()   # (1, 4t, 1024)
            grid = torch.zeros(1, 64, 1024, device=dev)
            grid[0, cells[0]] = feats.view(1, t, 4, -1).mean(2)[0]
            before.append(predict(predictor, grid, cells)[0])
            after.append(predict(predictor, full, cells)[0])            # the leak: tokens already saw everything
        rows = {}
        for name, vecs in (("before", torch.stack(before)), ("after", torch.stack(after))):
            idx, val = guesses(vecs)
            for row in idx:
                for i in row:
                    used.setdefault(i, len(used))
            rows[name] = {"guess": [[used[i] for i in row] for row in idx], "score": [r4(v) for v in val], "rank": rank_own(vecs, n)}
        Image.fromarray(val_px[n].transpose(1, 2, 0)).save(OUT / "v2" / f"reveal-{n}.jpg", quality=88)
        reveal.append({"val": n, "image": f"v2/reveal-{n}.jpg", "order": order.tolist(), "captions": [cap_text[i] for i in (owner == n).nonzero().squeeze(1).tolist()], **rows})
        print(n, "before rank", rows["before"]["rank"][::8], "after", rows["after"]["rank"][::8])
    pool_text = [None] * len(used)
    for i, j in used.items():
        pool_text[j] = train_caps[i]
    (OUT / "reveal.json").write_text(json.dumps({"photos": reveal, "captions": pool_text, "pool": len(train_caps), "epoch": int(ck["epoch"]), "val_captions": 5070}))

    ### arithmetic: every validation photo's vector, the words, thumbnails ###
    (OUT / "v2" / "val").mkdir(parents=True, exist_ok=True)
    for n in range(1014):
        p = OUT / "v2" / "val" / f"{n}.jpg"
        if not p.exists():
            Image.fromarray(val_px[n].transpose(1, 2, 0)).resize((144, 144), Image.LANCZOS).save(p, quality=78)
    vq, vs = q8(val_pred.cpu())
    wq, ws = q8(word_emb.cpu())
    blob = vq + vs + wq + ws
    (OUT / "arith.bin").write_bytes(blob)
    (OUT / "arith.json").write_text(json.dumps({
        "dim": 768, "photos": 1014, "words": WORDS,
        "offsets": {"photos": 0, "photoScales": len(vq), "words": len(vq) + len(vs), "wordScales": len(vq) + len(vs) + len(wq)},
        "firstCaption": [cap_text[(owner == n).nonzero()[0].item()] for n in range(1014)]}))
    # a few sums, to see whether it works at all
    def search(v, exclude):
        s = F.normalize(v, dim=-1) @ F.normalize(val_pred, dim=-1).t()
        s[exclude] = -2
        return s.topk(3).indices.tolist()
    W = {w: i for i, w in enumerate(WORDS)}
    for n, a, b in [(140, "beach", "snow"), (140, "dog", "horse"), (166, "dog", "child"), (58, "surfing", "skiing"), (116, "guitar", "drum"),
                    (44, "car", "horse"), (180, "football", "soccer") if "soccer" in W else (180, "ball", "frisbee"), (105, "water", "snow"), (14, "woman", "man"), (71, "climbing", "swimming")]:
        v = val_pred[n] - word_emb[W[a]].to(dev) + word_emb[W[b]].to(dev)
        print(f"#{n} - {a} + {b}:", [cap_text[(owner == j).nonzero()[0].item()][:50] for j in search(v, n)])

    ### baselines: always the average target; ridge from the averaged V-JEPA 2 tokens ###
    tm_path = SCRATCH / "train_mean_tokens.pt"
    if not tm_path.exists():
        train_px = np.load(V2 / "cache" / "train_pixels.npy", mmap_mode="r")
        means = []
        for s in range(0, len(train_px), 128):
            px = torch.from_numpy(np.array(train_px[s:s + 128]))
            means.append(tm.encode_visible(vjepa, tm.to_video(px)).float().mean(1).cpu())
            if s % 2560 == 0:
                print("train tokens", s)
        torch.save(torch.cat(means), tm_path)
    X = torch.load(tm_path).double()                                    # (29000, 1024)
    Y = text["train_targets"].double()
    Xv = val_cells.mean(1).float().cpu().double()
    base = {}
    mean_pred = Y.mean(0, keepdim=True).expand(1014, -1).float().to(dev)
    i2, t2 = retrieval_metrics(mean_pred, val_caps, owner)
    base["mean"] = {"i2t": i2, "t2i": t2}
    xm, ym = X.mean(0), Y.mean(0)
    XtX, XtY = (X - xm).t() @ (X - xm), (X - xm).t() @ (Y - ym)
    best = None
    for lam in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        Wr = torch.linalg.solve(XtX + lam * torch.eye(1024, dtype=torch.float64), XtY)
        i2, t2 = retrieval_metrics(((Xv - xm) @ Wr + ym).float().to(dev), val_caps, owner)
        if best is None or i2[5] + t2[5] > best[1]["i2t"][5] + best[1]["t2i"][5]:
            best = (lam, {"i2t": i2, "t2i": t2})
    base["ridge"] = {**best[1], "lambda": best[0]}
    print("baselines", base)

    ### the training run ###
    run = next((V2 / "logs").glob("*-masked-encode"))
    steps, epochs = [], []
    with open(run / "metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                steps.append([int(r["global_step"]), round(float(r["loss"]), 2), round(float(r["pred_std"]), 4), round(float(r["target_std"]), 4)])
            else:
                epochs.append({"epoch": int(r["epoch"]), "val_loss": round(float(r["loss"]), 3), "pred_std": round(float(r["pred_std"]), 4),
                               "i2t": [round(float(r[f"i2t_r{k}"]), 4) for k in (1, 5, 10)], "t2i": [round(float(r[f"t2i_r{k}"]), 4) for k in (1, 5, 10)]})
    log = (run / "train.log").read_text().splitlines()
    first, last = log[0][:8], [l for l in log if "===" in l][-1][:8]
    h = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
    (OUT / "v2results.json").write_text(json.dumps({
        "epoch": int(ck["epoch"]), "epochs": epochs, "steps": steps, "steps_per_epoch": 56, "batch": 512,
        "final": {"i2t": i2t, "t2i": t2i, "own_rank_1": sum(r == 1 for r in own_rank)},
        "baselines": base, "params": sum(p.numel() for p in predictor.parameters()),
        "hours": round((h(last) - h(first)) / 3600, 2)}))
    print("done")


if __name__ == "__main__":
    main(candidates="--candidates" in sys.argv)
