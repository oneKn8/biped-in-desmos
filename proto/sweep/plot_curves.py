"""Training curves: mean-policy 5 s hip distance per generation for every seed of a run, and the
same against cumulative training ticks for a comparison run (the cost Desmos pays).

Usage: python3 plot_curves.py RUN_DIR BASELINE_RUN_DIR OUT.png [--label-run TEXT --label-base TEXT]
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RUN, BASE = "#2a78d6", "#eb6834"


def load(run_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "hist_s*.json"))):
        h = json.load(open(f))
        out.append(dict(g=np.array([r["g"] for r in h]), d=np.array([r["dist"] for r in h]),
                        ticks=np.array([r["ticks_before"] for r in h]), ok=np.array([r["passed"] for r in h])))
    return out


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--label-run", default="recommended"); ap.add_argument("--label-base", default="baseline")
    a = ap.parse_args()
    run, base = load(a.run), load(a.base)
    bar_d = 0.8 * 5.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), facecolor=SURF)
    style(ax1); style(ax2)
    for s in run:
        ax1.plot(s["g"], s["d"], color=RUN, lw=1.0, alpha=0.35)
        if s["ok"].any():
            k = int(np.argmax(s["ok"]))
            ax1.plot(s["g"][k], s["d"][k], "o", ms=8, mfc=RUN, mec=SURF, mew=2, zorder=3)
    L = min(len(s["g"]) for s in run)
    med = np.median(np.stack([s["d"][:L] for s in run]), 0)
    ax1.plot(np.arange(L), med, color=RUN, lw=2.2, label=f"{a.label_run}: median of {len(run)} seeds")
    ax1.axhline(bar_d, color=INK2, lw=1.0, ls="--")
    ax1.text(0.5, bar_d + 0.15, "bar: 4 m in 5 s (0.8 m/s)", color=INK2, fontsize=8)
    ax1.set_xlabel("generation", color=INK2, fontsize=9); ax1.set_ylabel("mean-policy hip distance in 5 s (m)", color=INK2, fontsize=9)
    ax1.set_title(f"{a.label_run}: each seed (thin), first strict-bar pass (dot)", color=INK, fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    for runs, col, lab in ((base, BASE, a.label_base), (run, RUN, a.label_run)):
        grid = np.linspace(0, max(s["ticks"].max() for s in runs), 300)
        curves = np.stack([np.interp(grid, s["ticks"], s["d"], right=np.nan) for s in runs])
        for c in curves:
            ax2.plot(grid / 1000, c, color=col, lw=0.8, alpha=0.25)
        live = np.isfinite(curves).sum(0)
        m = np.full(curves.shape[1], np.nan)
        ok = live >= len(runs) / 2                      # median of live seeds, only where at least half still run
        m[ok] = np.nanmedian(curves[:, ok], 0)
        ax2.plot(grid / 1000, m, color=col, lw=2.2, label=f"{lab} (median of {len(runs)})")
    ax2.axhline(bar_d, color=INK2, lw=1.0, ls="--")
    ax2.set_xlabel("cumulative simulated training ticks (thousands): Desmos wall time is proportional", color=INK2, fontsize=9)
    ax2.set_ylabel("mean-policy hip distance in 5 s (m)", color=INK2, fontsize=9)
    ax2.set_title("cost axis: distance vs training ticks", color=INK, fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(a.out, dpi=90, facecolor=SURF)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
