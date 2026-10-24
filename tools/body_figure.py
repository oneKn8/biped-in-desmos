"""Draw docs/body.png: the ragdoll at its rest pose (from proto/sim.py) and the linear policy's inputs and outputs.

Usage: python3 tools/body_figure.py docs/body.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  the backend must be chosen before pyplot is imported
from matplotlib.patches import FancyArrowPatch  # noqa: E402  same reason

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proto"))
import sim  # noqa: E402  sim lives in proto/, which the line above puts on the path

INK, BODY, MOTOR, MUTED = "#222222", "#e8a33c", "#2d70b3", "#888888"
NAMES = {sim.HEAD: "head", sim.HIP: "hip", sim.KL: "left knee", sim.FL: "left foot", sim.KR: "right knee", sim.FR: "right foot"}
INPUTS = ["torso angle", "torso spin", "left hip angle", "left knee angle", "right hip angle", "right knee angle",
          "hip height", "hip speed", "left foot down", "right foot down", "clock sin", "clock cos", "bias"]
OUTPUTS = ["left hip target", "left knee target", "right hip target", "right knee target"]


def main(out):
    pos, _ = sim.initial_state(1)
    p = pos[0]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.5]})
    for a, b, _ in sim.BONES:
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=BODY, lw=6, solid_capstyle="round", zorder=2)
    for m in range(sim.M):
        ax.plot(*p[m], "o", color=INK, ms=7, zorder=3)
        dx = -0.09 if p[m, 0] < 0 else 0.09     # label on the outside of the leg it belongs to
        dy = 0.06 if m in (sim.FL, sim.FR) else 0.0
        ax.text(p[m, 0] + dx, p[m, 1] + dy, NAMES[m], ha="right" if dx < 0 else "left", va="center", fontsize=10, color=INK)
    for far, j, c in sim.MOTORS:     # motors: an arc at the joint between the parent and child bones
        ax.add_patch(FancyArrowPatch(p[j] + (p[far] - p[j]) * 0.25, p[j] + (p[c] - p[j]) * 0.25,
                                     connectionstyle="arc3,rad=0.5", arrowstyle="<->", color=MOTOR, lw=1.4,
                                     mutation_scale=9, zorder=4))
    ax.axhline(0, color=MUTED, lw=1)
    ax.text(0, -0.12, "ground (friction 0.85)", ha="center", fontsize=9, color=MUTED)
    ax.text(0.5, 1.02, "6 point masses, 5 bones, 4 motors (blue)", transform=ax.transAxes, ha="center", fontsize=11)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.2, 1.65); ax.set_aspect("equal"); ax.axis("off")

    bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
    ys_in = [0.95 - i * 0.07 for i in range(len(INPUTS))]
    ys_out = [0.72 - i * 0.14 for i in range(len(OUTPUTS))]
    for y, t in zip(ys_in, INPUTS):
        bx.text(0.0, y, t, fontsize=9.5, va="center", color=INK)
    for y, t in zip(ys_out, OUTPUTS):
        bx.text(0.72, y, t, fontsize=9.5, va="center", color=MOTOR)
    for yi in ys_in:
        for yo in ys_out:
            bx.plot([0.31, 0.70], [yi, yo], color=MUTED, lw=0.35, alpha=0.6)
    bx.text(0.5, 1.02, "the brain: 13 inputs x 4 outputs = 52 weights, tanh, one layer",
            transform=bx.transAxes, ha="center", fontsize=11)
    bx.text(0.0, -0.02, "Every body in the population runs this with its own noisy copy of the weights.",
            fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/body.png")
