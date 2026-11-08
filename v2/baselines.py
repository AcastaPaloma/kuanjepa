from pathlib import Path

import numpy as np
import torch

from utilities import retrieval_metrics

CACHE = Path(__file__).resolve().parent / "cache"


def mean_tokens(tokens, chunk=1024):
    out = []
    for s in range(0, len(tokens), chunk):
        out.append(torch.from_numpy(np.array(tokens[s:s + chunk, 0])).double().mean(1))   # (b, 1024)
    return torch.cat(out)                                                                              # (N, 1024)


def report(name, pred, text):
    i2t, t2i = retrieval_metrics(pred.float(), text["val_captions"], text["val_owner"])
    val_loss = ((pred.float() - text["val_targets"]) ** 2).sum(-1).mean().item()
    print(f"{name:<22} val_loss {val_loss:8.2f}  "
          f"i2t R@1/5/10 {i2t[1]:.3f}/{i2t[5]:.3f}/{i2t[10]:.3f}  "
          f"t2i R@1/5/10 {t2i[1]:.3f}/{t2i[5]:.3f}/{t2i[10]:.3f}")


if __name__ == "__main__":
    text = torch.load(CACHE / "text.pt")
    train_tokens = np.load(CACHE / "train_tokens.npy", mmap_mode="r")    # (N_train, 2, 64, 1024)
    val_tokens = np.load(CACHE / "val_tokens.npy", mmap_mode="r")        # (N_val, 1, 64, 1024)

    Y = text["train_targets"].double()                                   # (N_train, 768)
    N_val = len(val_tokens)

    ### MEAN PREDICTOR ###
    mean_pred = Y.mean(0, keepdim=True).expand(N_val, -1)                # (N_val, 768)
    report("mean predictor", mean_pred, text)

    ### RIDGE REGRESSION ###
    X = mean_tokens(train_tokens)                                        # (N_train, 1024)
    X_val = mean_tokens(val_tokens)                                      # (N_val, 1024)
    x_mean, y_mean = X.mean(0), Y.mean(0)                                # (1024,), (768,)
    Xc, Yc = X - x_mean, Y - y_mean                                      # (N_train, 1024), (N_train, 768)
    XtX = Xc.t() @ Xc                                                    # (1024, 1024)
    XtY = Xc.t() @ Yc                                                    # (1024, 768)
    I = torch.eye(XtX.size(0), dtype=XtX.dtype)                          # (1024, 1024)

    for lam in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        W = torch.linalg.solve(XtX + lam * I, XtY)                       # (1024, 768)
        pred = (X_val - x_mean) @ W + y_mean                             # (N_val, 768)
        report(f"ridge lambda={lam:g}", pred, text)
