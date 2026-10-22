"""Scatter of gait quality against training cost across gait-sweep configs (analyze_gait.row per config).

Left: median stance asymmetry of a config's walkers (reference rollouts, verify_gait.py) against the median
training ticks to the second 20 s exam pass, with whiskers for the walkers' interquartile range. Right: the
same for mean |torso angle|. The shaded corner is the goal: asymmetry <= 0.15 (or |torso| <= 8 degrees) at no
more than 1.5x the adopted recipe's median cost. Each config is given as LABEL=FAMILY=RUN[+RUN...]; FAMILY is one
of adopted, shaping, symmetric (three colours); a LABEL starting with _ draws a small unlabeled point.

Usage: python3 plot_gait.py OUT.png LABEL=FAMILY=RUN[+RUN] [...] [--title TEXT] [--cost-limit 54000]
       [--offset LABEL DX DY ...]
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import analyze_gait

plt.switch_backend("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))

INK, INK2, GRID, SURF, GOAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#eef4fb"
FAMILY = {"adopted": "#2a78d6", "shaping": "#eb6834", "symmetric": "#1baf7a"}
NEVER = 112.0                                # x position of configs whose median seed did not pass within 100k ticks
OFFSETS = {}                                 # label -> (dx, dy) annotation offset in points, set from the command line


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def walkers(spec):
    rows = []
    for name in spec.split("+"):
        p = os.path.join(HERE, "runs", name, "verify_gait.json")
        rows += json.load(open(p)) if os.path.exists(p) else []
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("configs", nargs="+")
    ap.add_argument("--title", default=""); ap.add_argument("--cost-limit", type=float, default=54000.0)
    ap.add_argument("--footnote", default="")
    ap.add_argument("--offset", nargs=3, action="append", default=[], metavar=("LABEL", "DX", "DY"),
                    help="move one annotation (points)")
    a = ap.parse_args()
    for lab, dx, dy in a.offset:
        OFFSETS[lab] = (float(dx), float(dy))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURF)
    seen = set()
    for spec in a.configs:
        label, fam, runs = spec.split("=", 2)
        r = analyze_gait.row(runs, label)
        w = walkers(runs)
        if not w:
            continue
        x = r["med"] / 1000.0 if np.isfinite(r["med"]) else NEVER   # median seed not through within the budget
        for ax, key, goal in ((axes[0], "asym20", 0.15), (axes[1], "abs_deg20", 8.0)):
            vals = np.array([v[key] for v in w])
            y = np.median(vals); lo, hi = np.percentile(vals, [25, 75])
            q1 = r["q1"] / 1000.0 if np.isfinite(r["q1"]) else x
            q3 = r["q3"] / 1000.0 if np.isfinite(r["q3"]) else NEVER
            quiet = label.startswith("_")          # a variant drawn without whiskers or label
            if not quiet:
                ax.plot([q1, q3], [y, y], color=FAMILY[fam], lw=1.2, alpha=0.6, solid_capstyle="round")
                ax.plot([x, x], [lo, hi], color=FAMILY[fam], lw=1.2, alpha=0.6, solid_capstyle="round")
            ax.scatter([x], [y], s=26 if quiet else 64, color=FAMILY[fam], alpha=0.55 if quiet else 1.0,
                       edgecolor=SURF, linewidth=1 if quiet else 2, zorder=3,
                       label=fam if (fam not in seen and ax is axes[0] and not quiet) else None)
            if not quiet:
                dx, dy = OFFSETS.get(label, (6, 5))
                ax.annotate(f"{label} ({r['npass']}/{r['n']})", (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=7.5, color=INK2, ha="right" if dx < 0 else "left")
        seen.add(fam)
    for ax, goal, ylab in ((axes[0], 0.15, "stance asymmetry |L - R| / (L + R), 20 s from rest"),
                           (axes[1], 8.0, "mean |torso angle|, degrees, 20 s from rest")):
        style(ax)
        ax.fill_between([0, a.cost_limit / 1000.0], 0, goal, color=GOAL, zorder=0, lw=0)
        ax.axhline(goal, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.axvline(a.cost_limit / 1000.0, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlabel("median training ticks to the second 20 s exam pass (thousands); whiskers = IQR", color=INK2,
                      fontsize=9)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.axvline(100, color=GRID, lw=6, zorder=0)
        ax.set_xlim(0, NEVER + 6); ax.set_ylim(bottom=0)
        ax.set_xticks([0, 20, 40, 60, 80, 100, NEVER])
        ax.set_xticklabels(["0", "20", "40", "60", "80", "100", "not within\n100k"])
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    if a.title:
        fig.suptitle(a.title, color=INK, fontsize=11, x=0.01, ha="left")
    if a.footnote:
        fig.text(0.01, 0.005, a.footnote, color=INK2, fontsize=8, ha="left", va="bottom")
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(a.out, dpi=90, facecolor=SURF)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
