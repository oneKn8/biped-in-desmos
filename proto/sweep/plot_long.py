"""Training curves for long-walk runs.

Left: one run, each seed's mean policy rolled out from rest for 20 s every generation (the offline
check), hip distance against cumulative training ticks, coloured by curriculum stage, with the
generation whose in-episode mean row passed the 20 s bar marked. Right: for several recipes, the share
of seeds whose mean row has passed the 20 s bar against the estimated Desmos wall time (training
ticks including the passing episode, divided by the measured Desmos rate for P+1 bodies). A recipe
given as RUN_A+RUN_B pools the seeds of several runs of the same config.

Usage: python3 plot_long.py MAIN_RUN OUT.png RUN[+RUN] [...] [--labels A B ...] [--title TEXT]
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import analyze_long

plt.switch_backend("Agg")

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STAGE = ["#9db8d9", "#2a78d6", "#0f3e7a", "#06224a"]
SERIES = ["#2a78d6", "#eb6834", "#2f9e6a", "#8a5cc2", "#c2a03a", "#52514e"]


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def load(run):
    out = []
    for f in sorted(glob.glob(os.path.join(run, "hist_s*.json"))):
        out.append(json.load(open(f)))
    return out, json.load(open(os.path.join(run, "summary.json"))), json.load(open(os.path.join(run, "config.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main"); ap.add_argument("out"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*"); ap.add_argument("--title", default="")
    a = ap.parse_args()
    hists, summ, cfg = load(a.main)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4), facecolor=SURF)
    style(ax1); style(ax2)
    for h in hists:
        x = np.array([r["ticks_before"] for r in h]) / 1000
        y = np.array([r["or20_dist"] for r in h])
        sg = np.array([r["stage"] for r in h])
        ax1.plot(x, y, color=INK2, lw=0.5, alpha=0.4)
        for s in np.unique(sg):
            m = sg == s
            ax1.plot(x[m], y[m], ".", ms=3, color=STAGE[min(s, len(STAGE) - 1)])
    for sd, fp in zip(summ["seeds"], summ["first_final"]):
        if fp is not None:
            h = hists[summ["seeds"].index(sd)]
            ax1.plot(h[fp]["ticks_before"] / 1000, h[fp]["or20_dist"], "o", ms=8, mfc=STAGE[1], mec=SURF, mew=2, zorder=3)
    ax1.axhline(16, color=INK2, lw=1.0, ls="--")
    ax1.text(0.3, 16.4, "20 s bar: 16 m (0.8 m/s)", color=INK2, fontsize=8)
    for s, sec in enumerate(cfg["stages"]):
        if cfg.get("chain_stages") and cfg["chain_stages"][s]:
            lab = f"{sec:g} s training window (continues the last walk, or from rest)"
        elif cfg.get("cut_stages") and cfg["cut_stages"][s]:
            lab = f"{sec:g} s exam from rest (ends when the mean row falls)"
        else:
            lab = f"{sec:g} s training episodes from rest"
        ax1.plot([], [], ".", ms=8, color=STAGE[min(s, len(STAGE) - 1)], label=lab)
    ax1.plot([], [], "o", ms=8, mfc=STAGE[1], mec=SURF, label="stop: the mean row passed the 20 s bar in an exam")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_xlabel("cumulative training ticks (thousands)", color=INK2, fontsize=9)
    ax1.set_ylabel("mean policy each generation, 20 s from rest: hip distance (m)", color=INK2, fontsize=9)
    ax1.set_title(a.title or f"{os.path.basename(a.main)}: {len(hists)} seeds", color=INK, fontsize=10, loc="left")
    labels = a.labels or [os.path.basename(r) for r in a.runs]
    xmax = 0
    for k, (spec, lab) in enumerate(zip(a.runs, labels)):
        t, n = [], 0
        for run in spec.split("+"):
            _, s, c = load(run)
            rate = analyze_long.desmos_rate(c["P"] + 1)
            t += [x / rate / 60.0 for x in s["desmos_ticks"] if x is not None]
            n += len(s["seeds"])
        t = np.sort(t)
        if len(t):
            ax2.step(np.concatenate([[0], t]), np.arange(len(t) + 1) / n, where="post",
                     color=SERIES[k % len(SERIES)], lw=2, label=f"{lab} ({len(t)}/{n})")
            xmax = max(xmax, t.max())
    ax2.set_xlim(0, xmax * 1.05 if xmax else 1)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("estimated Desmos minutes (training ticks / measured steps per second for P+1 bodies)", color=INK2, fontsize=9)
    ax2.set_ylabel("share of seeds past the 20 s bar", color=INK2, fontsize=9)
    ax2.set_title("time to the 20 s bar by recipe (each step = one seed passing)", color=INK, fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(a.out, dpi=90, facecolor=SURF)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
