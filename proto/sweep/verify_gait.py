"""Reference gait check of trained policies: ../ars.py rollout from rest, ../walkbar.py bar, gait statistics.

For each policy file, two reference rollouts from rest (ars.rollout, record=True): 1200 ticks (20 s) and
3600 ticks (60 s). Bar verdicts come from ars.passes (walkbar.strict_bar with the phase requirement scaled to
the horizon: 12 alternating single-support phases per foot over 20 s, 36 over 60 s). Gait statistics come
from the recorded trajectory of the same rollout, over the ticks while upright:
  asym      |left - right single-support time| / (left + right)   (single support: that foot down, the other up)
  abs_deg   mean |torso angle| in degrees (torso angle 0 = upright, + = leaning forward)
  mean_deg  mean torso angle (signed), std_deg its standard deviation (the pitch swing)
  air       share of all ticks with both feet off the ground
Symmetric 2x13 policies are expanded to their exact 4x13 form first (sym_expand.py).

Usage: python3 verify_gait.py FILE_OR_DIR [...] [--glob 'theta_pass_s*.npy'] [--out OUT.json]
A directory argument checks every file in it that matches --glob.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

import sym_expand

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
import ars  # noqa: E402 - the reference trainer lives one folder up and is imported after the path is set
import sim  # noqa: E402 - same: the reference simulator one folder up


def gait_stats(traj, fallen, alive_t):
    """Gait statistics of one recorded reference trajectory (T, M, 2), over the upright ticks."""
    T = len(traj)
    up = np.ones(T, bool)
    if fallen:
        up[int(round(alive_t / sim.DT)):] = False
    torso = traj[:, sim.HEAD] - traj[:, sim.HIP]
    ang = np.degrees(np.arctan2(torso[:, 0], torso[:, 1]))[up]
    cl = traj[:, sim.FL, 1] < 1e-3
    cr = traj[:, sim.FR, 1] < 1e-3
    sl = int((cl & ~cr & up).sum())
    sr = int((cr & ~cl & up).sum())
    return dict(asym=abs(sl - sr) / max(sl + sr, 1), ss_left_s=sl * sim.DT, ss_right_s=sr * sim.DT,
                abs_deg=float(np.abs(ang).mean()), mean_deg=float(ang.mean()), std_deg=float(ang.std()),
                air=float((~cl & ~cr).sum() / T))


def check(W):
    """W (4, 13). Reference rollouts from rest over 20 s and 60 s: bar verdicts and gait statistics."""
    row = {}
    for T, tag in ((1200, "20"), (3600, "60")):
        out = ars.rollout(W[None], T=T, record=True)
        g = gait_stats(np.array(out["traj"]), bool(out["fallen"][0]), float(out["alive_t"][0]))
        row[f"pass{tag}"] = bool(ars.passes(out, T)[0])
        row[f"dist{tag}"] = float(out["dist"][0])
        row[f"fell{tag}"] = bool(out["fallen"][0])
        row[f"alive{tag}"] = float(out["alive_t"][0])
        row[f"nL{tag}"] = int(out["n_left"][0])
        row[f"nR{tag}"] = int(out["n_right"][0])
        row[f"air_frac{tag}"] = float(out["air_frac"][0])
        for k, v in g.items():
            row[f"{k}{tag}"] = v
    return row


def seed_of(path):
    m = re.search(r"_s(\d+)\.npy$", path)
    return int(m.group(1)) if m else -1


def files_from(args, pattern):
    files = []
    for a in args:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, pattern)), key=seed_of)
        else:
            files.append(a)
    return files


def summary_line(rows):
    def med(k):
        return float(np.median([r[k] for r in rows]))
    n = len(rows)
    return (f"n={n}  20 s bar {sum(r['pass20'] for r in rows)}/{n}  60 s bar {sum(r['pass60'] for r in rows)}/{n}"
            f"  asym med {med('asym20'):.3f}  |torso| med {med('abs_deg20'):.1f} deg  torso mean med {med('mean_deg20'):+.1f}"
            f"  torso std med {med('std_deg20'):.1f} deg  speed med {med('dist20') / 20:.2f} m/s  air med {med('air_frac20'):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--glob", default="theta_pass_s*.npy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = files_from(a.paths, a.glob)
    if not files:
        raise SystemExit(f"no policy files found in {a.paths} matching {a.glob}")
    rows = []
    for f in files:
        r = dict(file=os.path.relpath(f, HERE), seed=seed_of(f), **check(sym_expand.expand(np.load(f))))
        rows.append(r)
        print(f"  {os.path.basename(f):24s} 20s {'PASS' if r['pass20'] else 'fail'} {r['dist20']:6.2f} m"
              f" L{r['nL20']} R{r['nR20']} air {r['air_frac20']:.2f} | 60s {'PASS' if r['pass60'] else 'fail'}"
              f" alive {r['alive60']:5.1f} s | asym {r['asym20']:.3f} |torso| {r['abs_deg20']:5.1f}"
              f" mean {r['mean_deg20']:+6.1f} std {r['std_deg20']:4.1f}", flush=True)
    print(summary_line(rows))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
