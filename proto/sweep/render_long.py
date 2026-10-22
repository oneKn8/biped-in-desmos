"""Filmstrip of a long rollout from rest (../sim.py reference physics), split into rows of equal time,
plus a foot-contact timeline, with the horizon-scaled strict-bar verdict (verify_long.long_bar).

Usage: python3 render_long.py THETA.npy OUT.png [--seconds 20] [--rows 4] [--title TEXT]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import sym_expand
import verify_long

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ars  # noqa: E402 - the reference trainer lives one folder up and is imported after the path is set
import sim  # noqa: E402 - same: the reference simulator one folder up

plt.switch_backend("Agg")

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
LEFT, RIGHT, AIR = "#2a78d6", "#eb6834", "#c3c2b7"


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("theta"); ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    W = sym_expand.expand(np.load(a.theta))
    T = int(round(a.seconds / sim.DT))
    res = ars.rollout(W[None], T=T, record=True)
    traj = np.array(res["traj"])
    ok, out = verify_long.long_bar(W[None], T)
    g = verify_long.gait(W, T)
    req = int(round(3 * a.seconds / 5))
    fig = plt.figure(figsize=(18, 2.9 * a.rows + 2.2), facecolor=SURF)
    gs = fig.add_gridspec(a.rows + 1, 1, height_ratios=[3] * a.rows + [1.3])
    per = T // a.rows
    step = max(1, int(round(0.1 / sim.DT)))
    span = max(traj[(r + 1) * per - 1, :, 0].max() - traj[r * per, :, 0].min() for r in range(a.rows)) + 0.4
    for r in range(a.rows):
        ax = fig.add_subplot(gs[r]); style(ax)
        k0, k1 = r * per, (r + 1) * per
        for k in range(k0, k1, step):
            p = traj[k]; c = plt.cm.viridis((k - k0) / per)
            for i, j, _ in sim.BONES:
                ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color=c, lw=1.2, alpha=0.85, solid_capstyle="round")
        x0 = traj[k0, :, 0].min() - 0.2
        ax.set_xlim(x0, x0 + span); ax.set_ylim(-0.05, 1.6); ax.set_aspect("equal")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel(f"{k0 * sim.DT:.0f}-{k1 * sim.DT:.0f} s", color=INK2, fontsize=9)
        if r == 0:
            head = a.title + "\n" if a.title else ""
            ax.set_title(f"{head}{a.seconds:g} s from rest: hip {out['dist'][0]:.2f} m ({out['dist'][0] / a.seconds:.2f} m/s), "
                         f"fell: {'yes' if out['fallen'][0] else 'no'}, alternating single-support phases "
                         f"L{out['n_left'][0]} R{out['n_right'][0]} (need {req} each), both feet airborne "
                         f"{100 * out['air_frac'][0]:.0f}% -> {a.seconds:g} s bar {'PASS' if ok[0] else 'fail'}\n"
                         f"single-support time left {g['ss_left_s']:.1f} s / right {g['ss_right_s']:.1f} s "
                         f"(asymmetry {g['limp']:.2f}); mean torso angle {g['lean_deg']:+.1f} deg (+ forward, - backward)",
                         color=INK, fontsize=10, loc="left")
    ax2 = fig.add_subplot(gs[a.rows]); style(ax2)
    t = np.arange(len(traj)) * sim.DT
    cl = traj[:, sim.FL, 1] < 1e-3; cr = traj[:, sim.FR, 1] < 1e-3
    ax2.fill_between(t, 0, cl * 0.9, step="mid", color=LEFT, alpha=0.85, lw=0)
    ax2.fill_between(t, 1.1, 1.1 + cr * 0.9, step="mid", color=RIGHT, alpha=0.85, lw=0)
    ax2.fill_between(t, 2.2, 2.2 + (~cl & ~cr) * 0.5, step="mid", color=AIR, lw=0)
    ax2.set_yticks([0.45, 1.55, 2.45]); ax2.set_yticklabels(["left down", "right down", "both up"], color=INK2, fontsize=8)
    ax2.set_xlabel("seconds", color=INK2, fontsize=9); ax2.set_xlim(0, a.seconds)
    plt.tight_layout(); plt.savefig(a.out, dpi=90, facecolor=SURF)
    print(f"wrote {a.out}: dist {out['dist'][0]:.2f} bar {'PASS' if ok[0] else 'fail'} limp {g['limp']:.3f} lean {g['lean_deg']:.1f}")


if __name__ == "__main__":
    main()
