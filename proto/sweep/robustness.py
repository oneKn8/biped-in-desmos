"""Longer-horizon check of a run's first-pass and final policies, plus physics-quality numbers.

For each seed: does the policy still meet the bar over 10 s and 20 s (same thresholds: 0.8 m/s,
no fall, >= 3 alternating single-support phases per foot, < 25% both-airborne), and over the
5 s rollout how much do bones stretch and do knees touch the ground.

Usage: python3 robustness.py RUN_DIR [RUN_DIR ...]
"""
import glob
import json
import os
import sys

import numpy as np

import fastsim as fs
import train


def quality(theta, ph, sym, seconds=5.0):
    T = int(round(seconds / ph.DT))
    pos, prev = fs.initial_state(theta.shape[0])
    alive = np.ones(theta.shape[0]); stretch = []; knee = np.zeros(theta.shape[0])
    for k in range(T):
        obs = fs.observe(pos, prev, k * ph.DT, ph)
        pos, prev = fs.step(pos, prev, fs.targets(train.policy_act(theta, obs, sym)), alive, ph)
        alive = np.where(fs.fallen(pos), 0.0, alive)
        d = pos[fs.BB] - pos[fs.BA]
        stretch.append(np.abs(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) - fs.BL[:, None]) / fs.BL[:, None])
        knee += (np.minimum(pos[fs.KL, 1], pos[fs.KR, 1]) < 1e-3)
    st = np.array(stretch)                                     # (T, 5, R)
    return st.mean(axis=(0, 1)), st.max(axis=(0, 1)), knee


def main():
    for run in sys.argv[1:]:
        cfg = {**train.DEFAULTS, **json.load(open(os.path.join(run, "config.json")))}
        ph = train.make_phys(cfg)
        for which in ("firstpass", "final"):
            files = sorted(glob.glob(os.path.join(run, f"theta_{which}_s*.npy")))
            if not files:
                continue
            th = np.stack([np.load(f) for f in files])
            res = {h: train.evaluate(th, ph, cfg["sym"], h) for h in (5.0, 10.0, 20.0)}
            smean, smax, knee = quality(th, ph, cfg["sym"])
            print(f"{os.path.basename(run)} {which:9s} n={len(files)}  bar@5s {res[5.0]['passed'].mean():.2f}"
                  f"  @10s {res[10.0]['passed'].mean():.2f}  @20s {res[20.0]['passed'].mean():.2f}"
                  f"  fell@20s {res[20.0]['fallen'].mean():.2f}  speed@10s med {np.median(res[10.0]['dist']) / 10:.2f}"
                  f"  stretch mean {smean.mean():.3f} max {smax.max():.3f}  knee-ground ticks med {np.median(knee):.0f}")


if __name__ == "__main__":
    main()
