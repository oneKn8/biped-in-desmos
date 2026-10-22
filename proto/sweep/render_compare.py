"""Side-by-side filmstrips of two walkers over the same horizon from rest (../sim.py reference physics via
../ars.py rollout), each with its foot-contact timeline, plus both torso-angle traces on one axis.

Usage: python3 render_compare.py THETA_A.npy THETA_B.npy OUT.png [--labels A B] [--seconds 20] [--rows 2]
       [--title TEXT]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import sym_expand
import verify_gait

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ars  # noqa: E402 - the reference trainer lives one folder up and is imported after the path is set
import sim  # noqa: E402 - same: the reference simulator one folder up

plt.switch_backend("Agg")

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LEFT, RIGHT, AIR = "#2a78d6", "#eb6834", "#c3c2b7"
TRACE = ["#8a8a85", "#2a78d6"]


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def run(path, T):
    W = sym_expand.expand(np.load(path))
    out = ars.rollout(W[None], T=T, record=True)
    traj = np.array(out["traj"])
    g = verify_gait.gait_stats(traj, bool(out["fallen"][0]), float(out["alive_t"][0]))
    return traj, out, bool(ars.passes(out, T)[0]), g


def strip(ax, traj, k0, k1, span):
    step = max(1, int(round(0.1 / sim.DT)))
    for k in range(k0, k1, step):
        p = traj[k]
        c = plt.cm.viridis((k - k0) / (k1 - k0))
        for i, j, _ in sim.BONES:
            ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color=c, lw=1.1, alpha=0.85, solid_capstyle="round")
    x0 = traj[k0, :, 0].min() - 0.2
    ax.set_xlim(x0, x0 + span); ax.set_ylim(-0.05, 1.6); ax.set_aspect("equal")
    ax.axhline(0, color=INK2, lw=0.8)


def contacts(ax, traj, seconds):
    t = np.arange(len(traj)) * sim.DT
    cl = traj[:, sim.FL, 1] < 1e-3
    cr = traj[:, sim.FR, 1] < 1e-3
    ax.fill_between(t, 0, cl * 0.9, step="mid", color=LEFT, alpha=0.85, lw=0)
    ax.fill_between(t, 1.1, 1.1 + cr * 0.9, step="mid", color=RIGHT, alpha=0.85, lw=0)
    ax.fill_between(t, 2.2, 2.2 + (~cl & ~cr) * 0.5, step="mid", color=AIR, lw=0)
    ax.set_yticks([0.45, 1.55, 2.45]); ax.set_yticklabels(["left down", "right down", "both up"], color=INK2, fontsize=8)
    ax.set_xlim(0, seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("out")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    T = int(round(a.seconds / sim.DT))
    runs = [run(p, T) for p in (a.a, a.b)]
    per = T // a.rows
    span = max(tr[(r + 1) * per - 1, :, 0].max() - tr[r * per, :, 0].min() for tr, *_ in runs for r in range(a.rows)) + 0.4
    n_rows = 2 * (a.rows + 1) + 1
    ratios = ([2.6] * a.rows + [1.0]) * 2 + [1.6]
    fig = plt.figure(figsize=(18, sum(ratios) * 0.95 + 0.8), facecolor=SURF)
    gs = fig.add_gridspec(n_rows, 1, height_ratios=ratios, hspace=0.45)
    req = int(round(3 * a.seconds / 5))
    for w, (traj, out, ok, g) in enumerate(runs):
        base = w * (a.rows + 1)
        for r in range(a.rows):
            ax = fig.add_subplot(gs[base + r]); style(ax)
            strip(ax, traj, r * per, (r + 1) * per, span)
            ax.set_ylabel(f"{r * per * sim.DT:.0f}-{(r + 1) * per * sim.DT:.0f} s", color=INK2, fontsize=9)
            if r == 0:
                head = (a.title + "\n") if (a.title and w == 0) else ""
                ax.set_title(f"{head}{a.labels[w]}: {out['dist'][0]:.1f} m in {a.seconds:g} s "
                             f"({out['dist'][0] / a.seconds:.2f} m/s), fell {'yes' if out['fallen'][0] else 'no'}, "
                             f"phases L{out['n_left'][0]} R{out['n_right'][0]} (need {req}), airborne "
                             f"{100 * out['air_frac'][0]:.0f}%, {a.seconds:g} s bar {'PASS' if ok else 'fail'} | "
                             f"single support left {g['ss_left_s']:.1f} s / right {g['ss_right_s']:.1f} s "
                             f"(asymmetry {g['asym']:.2f}) | torso mean {g['mean_deg']:+.1f} deg, "
                             f"mean |angle| {g['abs_deg']:.1f} deg, std {g['std_deg']:.1f} deg",
                             color=INK, fontsize=9.5, loc="left")
        axc = fig.add_subplot(gs[base + a.rows]); style(axc)
        contacts(axc, traj, a.seconds)
    ax = fig.add_subplot(gs[n_rows - 1]); style(ax)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for w, (traj, *_rest) in enumerate(runs):
        torso = traj[:, sim.HEAD] - traj[:, sim.HIP]
        ang = np.degrees(np.arctan2(torso[:, 0], torso[:, 1]))
        ax.plot(np.arange(len(traj)) * sim.DT, ang, color=TRACE[w], lw=1.0, label=a.labels[w])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(0, a.seconds); ax.set_ylabel("torso angle deg\n(+ forward)", color=INK2, fontsize=9)
    ax.set_xlabel("seconds", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    plt.savefig(a.out, dpi=85, facecolor=SURF, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
