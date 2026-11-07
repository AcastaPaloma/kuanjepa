"""Live loss curves for a training run.

usage:
    python plot_live.py                 # newest run in logs/
    python plot_live.py logs/<run_dir>  # a specific run
    python plot_live.py --once out.png  # render a single snapshot and exit
"""
import csv
import glob
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

REFRESH_MS = 5000
EMA = 0.95          # smoothing for the train curve


def newest_run():
    runs = sorted(glob.glob(os.path.join("logs", "*", "metrics.csv")))
    if not runs:
        sys.exit("no logs/*/metrics.csv found")
    return os.path.dirname(runs[-1])


def read_metrics(path):
    train, val = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:  # the last line can be half-written while training is running
                gs, loss = int(row["global_step"]), float(row["loss"])
                temp, lr = float(row["temperature"]), float(row["lr"])
            except (TypeError, ValueError, KeyError):
                continue
            (train if row["split"] == "train" else val).append((gs, loss, temp, lr, row["epoch"]))
    return train, val


def ema(xs, a=EMA):
    out, m = [], None
    for x in xs:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def main():
    args = sys.argv[1:]
    snapshot = None
    if args and args[0] == "--once":
        snapshot = args[1] if len(args) > 1 else "loss_curve.png"
        args = args[2:]
    run_dir = args[0] if args else newest_run()
    path = os.path.join(run_dir, "metrics.csv")

    fig, (ax_tr, ax_val, ax_aux) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax_temp = ax_aux.twinx()
    batch = 256

    def draw(_=None):
        train, val = read_metrics(path)
        for ax in (ax_tr, ax_val, ax_aux, ax_temp):
            ax.clear()

        if train:
            gs = [r[0] for r in train]
            loss = [r[1] for r in train]
            ax_tr.plot(gs, loss, color="tab:blue", alpha=0.25, lw=0.8, label="train (per step)")
            ax_tr.plot(gs, ema(loss), color="tab:blue", lw=1.8, label=f"train (EMA {EMA})")
            ax_tr.axhline(math.log(batch), color="gray", ls="--", lw=1, label=f"chance ln({batch})")

            ax_aux.plot(gs, [r[3] for r in train], color="tab:green", lw=1.5, label="lr")
            ax_temp.plot(gs, [r[2] for r in train], color="tab:purple", lw=1.5, label="temperature")

        if val:
            ax_val.plot([r[0] for r in val], [r[1] for r in val], "o-", color="tab:orange", lw=1.8, ms=4,
                        label="val (full set)")
            ax_val.axhline(math.log(1014), color="gray", ls="--", lw=1, label="chance ln(N_val)")

        last_ep = train[-1][4] if train else "-"
        best = min((r[1] for r in val), default=float("nan"))
        ax_tr.set_title(f"{run_dir}   epoch {last_ep}   best val {best:.4f}")
        ax_tr.set_ylabel("InfoNCE loss"); ax_val.set_ylabel("val loss")
        ax_aux.set_ylabel("lr", color="tab:green"); ax_temp.set_ylabel("temperature", color="tab:purple")
        ax_aux.set_xlabel("global step")
        ax_temp.yaxis.set_label_position("right"); ax_temp.yaxis.tick_right()
        for ax in (ax_tr, ax_val):
            ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
        ax_aux.grid(alpha=0.3)
        h1, l1 = ax_aux.get_legend_handles_labels(); h2, l2 = ax_temp.get_legend_handles_labels()
        if h1 or h2:
            ax_aux.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        fig.tight_layout()

    if snapshot:
        draw()
        fig.savefig(snapshot, dpi=110)
        print("wrote", snapshot)
        return

    _anim = FuncAnimation(fig, draw, interval=REFRESH_MS, cache_frame_data=False)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
