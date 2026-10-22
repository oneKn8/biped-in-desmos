"""Filmstrip of a trained policy under its run's physics: stick figures across the episode plus a
foot-contact timeline (single-support phases, both-feet-airborne ticks) and the strict-bar verdict.

Usage: python3 render_strip.py RUN_DIR SEED OUT.png [--which final|firstpass] [--seconds 5]
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import fastsim as fs
import train

plt.switch_backend("Agg")

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
LEFT, RIGHT, AIR = "#2a78d6", "#eb6834", "#c3c2b7"


def record(theta, ph, sym, seconds):
    T = int(round(seconds / ph.DT))
    pos, prev = fs.initial_state(1)
    alive = np.ones(1); traj = []
    W = theta[None]
    for k in range(T):
        obs = fs.observe(pos, prev, k * ph.DT, ph)
        pos, prev = fs.step(pos, prev, fs.targets(train.policy_act(W, obs, sym)), alive, ph)
        alive = np.where(fs.fallen(pos), 0.0, alive)
        traj.append(pos[:, :, 0].copy())
    return np.array(traj)                      # (T, M, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("seed", type=int); ap.add_argument("out")
    ap.add_argument("--which", default="final"); ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    cfg = {**train.DEFAULTS, **json.load(open(os.path.join(a.run_dir, "config.json")))}
    ph = train.make_phys(cfg)
    theta = np.load(os.path.join(a.run_dir, f"theta_{a.which}_s{a.seed}.npy"))
    traj = record(theta, ph, cfg["sym"], a.seconds)
    ev = train.evaluate(theta[None], ph, cfg["sym"], a.seconds)
    verdict = "PASS" if ev["passed"][0] else "fail"
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 5.8), gridspec_kw={"height_ratios": [4, 1.1]}, facecolor=SURF)
    for axx in (ax, ax2):
        axx.set_facecolor(SURF)
        for sp in ("top", "right"):
            axx.spines[sp].set_visible(False)
        axx.tick_params(colors=INK2, labelsize=8)
    step = max(1, int(round(0.1 / ph.DT)))
    for k in range(0, len(traj), step):
        p = traj[k]; c = plt.cm.viridis(k / len(traj))
        for i, j, _ in fs.BONES:
            ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color=c, lw=1.3, alpha=0.85, solid_capstyle="round")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_aspect("equal"); ax.set_ylabel("height (m)", color=INK2, fontsize=9)
    head = a.title or cfg["name"]
    ax.set_title(f"{head}\n{a.seconds:g} s rollout: hip {ev['dist'][0]:.2f} m ({ev['dist'][0] / a.seconds:.2f} m/s), "
                 f"fell: {'yes' if ev['fallen'][0] else 'no'}, alternating single-support phases L{ev['n_left'][0]} R{ev['n_right'][0]}, "
                 f"both feet airborne {100 * ev['air'][0]:.0f}% of ticks  ->  strict bar {verdict}",
                 color=INK, fontsize=10, loc="left")
    t = np.arange(len(traj)) * ph.DT
    cl = traj[:, fs.FL, 1] < 1e-3; cr = traj[:, fs.FR, 1] < 1e-3
    ax2.fill_between(t, 0, cl * 0.9, step="mid", color=LEFT, alpha=0.85, lw=0, label="left foot down")
    ax2.fill_between(t, 1.1, 1.1 + cr * 0.9, step="mid", color=RIGHT, alpha=0.85, lw=0, label="right foot down")
    ax2.fill_between(t, 2.2, 2.2 + (~cl & ~cr) * 0.5, step="mid", color=AIR, lw=0, label="both feet airborne")
    ax2.set_yticks([0.45, 1.55, 2.45]); ax2.set_yticklabels(["left", "right", "air"], color=INK2, fontsize=8)
    ax2.set_xlabel("seconds", color=INK2, fontsize=9); ax2.set_xlim(0, a.seconds)
    plt.tight_layout(); plt.savefig(a.out, dpi=90, facecolor=SURF)
    print(f"wrote {a.out}: dist {ev['dist'][0]:.2f} bar {verdict}")


if __name__ == "__main__":
    main()
