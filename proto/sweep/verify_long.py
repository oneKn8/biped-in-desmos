"""Reference re-evaluation of trained policies with ../ars.py's rollout and ../walkbar.py's strict bar.

For each theta file: the 20 s bar (1200 ticks, >= 12 alternating single-support phases per foot) and
the 60 s bar (3600 ticks, >= 36 per foot), both with walkbar.strict_bar's speed / fall / airborne
tests, plus gait metrics over 20 s from a recorded reference rollout: limp = |left - right
single-support time| / (left + right), lean = mean torso angle while upright (degrees, + = forward).
Symmetric 2x13 policies are expanded to their exact 4x13 form first (sym_expand.py).

Usage: python3 verify_long.py RUN_DIR [RUN_DIR ...] [--which pass]   (writes RUN_DIR/verify_<which>.json)
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
import walkbar  # noqa: E402 - same: the reference strict bar one folder up


def long_bar(W, T):
    out = ars.rollout(W, T=T)
    req = int(round(walkbar.BAR_PHASES * T / 300))
    ok = walkbar.strict_bar(out["dist"], T * sim.DT, out["fallen"], out["n_left"], out["n_right"], out["air_frac"])
    ok = ok & (out["n_left"] >= req) & (out["n_right"] >= req)
    return ok, out


def jitter_pass(W, T, K, sigma, seed=0):
    """Share of K rollouts per policy that pass the scaled bar when every initial coordinate is
    perturbed by N(0, sigma) metres (zero initial velocity). Same loop as ars.rollout (sim.py physics,
    walkbar phase tracker), only the start differs: a check that a pass does not hinge on one exact
    floating-point trajectory."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    Wr = np.repeat(W, K, 0)
    R = Wr.shape[0]
    pos, _ = sim.initial_state(R)
    pos = pos + sigma * rng.standard_normal(pos.shape)
    pos[..., 1] = np.maximum(pos[..., 1], 0.0)
    prev = pos.copy()
    x0 = pos[:, sim.HIP, 0].copy()
    alive = np.ones(R); dist = np.zeros(R); air_ticks = np.zeros(R); fallen = np.zeros(R, bool)
    trk = walkbar.PhaseTracker(R)
    for k in range(T):
        obs = sim.observe(pos, prev, k * sim.DT)
        act = np.tanh(np.einsum('pao,po->pa', Wr, obs))
        pos, prev = sim.step(pos, prev, sim.targets(act), alive)
        cl = pos[:, sim.FL, 1] < 1e-3; cr = pos[:, sim.FR, 1] < 1e-3
        down = sim.fallen(pos) & (alive > 0)
        dist = np.where(alive > 0, pos[:, sim.HIP, 0] - x0, dist)
        alive = np.where(down, 0.0, alive)
        fallen |= down
        air_ticks += ~cl & ~cr
        trk.update(cl, cr, alive > 0)
    trk.finish()
    req = int(round(walkbar.BAR_PHASES * T / 300))
    ok = walkbar.strict_bar(dist, T * sim.DT, fallen, trk.cnt[0], trk.cnt[1], air_ticks / T)
    ok = ok & (trk.cnt[0] >= req) & (trk.cnt[1] >= req)
    return ok.reshape(n, K).mean(1)


def gait(Wi, T):
    res = ars.rollout(Wi[None], T=T, record=True)
    traj = np.array(res["traj"])
    up = np.ones(len(traj), bool)
    if res["fallen"][0]:
        up[int(round(res["alive_t"][0] / sim.DT)):] = False
    torso = traj[:, sim.HEAD] - traj[:, sim.HIP]
    ang = np.degrees(np.arctan2(torso[:, 0], torso[:, 1]))
    cl = traj[:, sim.FL, 1] < 1e-3; cr = traj[:, sim.FR, 1] < 1e-3
    sl = (cl & ~cr & up).sum(); sr = (cr & ~cl & up).sum()
    return dict(lean_deg=float(ang[up].mean()), limp=float(abs(sl - sr) / max(sl + sr, 1)),
                ss_left_s=float(sl * sim.DT), ss_right_s=float(sr * sim.DT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--which", default="pass")
    ap.add_argument("--jitter", type=int, default=0, help="K jittered starts per policy for a 20 s robustness rate")
    ap.add_argument("--sigma", type=float, default=1e-3, help="initial-position jitter (m)")
    a = ap.parse_args()
    for run in a.runs:
        files = sorted(glob.glob(os.path.join(run, f"theta_{a.which}_s*.npy")),
                       key=lambda f: int(re.search(r"_s(\d+)\.npy$", f).group(1)))
        if not files:
            print(f"{run}: no theta_{a.which}_s*.npy"); continue
        W = np.stack([sym_expand.expand(np.load(f)) for f in files])
        ok20, o20 = long_bar(W, 1200)
        ok60, o60 = long_bar(W, 3600)
        jit = jitter_pass(W, 1200, a.jitter, a.sigma) if a.jitter else [None] * len(files)
        rows = []
        for i, f in enumerate(files):
            g = gait(W[i], 1200)
            rows.append(dict(file=os.path.basename(f), pass20=bool(ok20[i]), pass60=bool(ok60[i]),
                             dist20=float(o20["dist"][i]), fell20=bool(o20["fallen"][i]),
                             nL20=int(o20["n_left"][i]), nR20=int(o20["n_right"][i]), air20=float(o20["air_frac"][i]),
                             dist60=float(o60["dist"][i]), fell60=bool(o60["fallen"][i]),
                             alive60=float(o60["alive_t"][i]), nL60=int(o60["n_left"][i]), nR60=int(o60["n_right"][i]),
                             jitter20=None if jit[i] is None else float(jit[i]), **g))
        json.dump(rows, open(os.path.join(run, f"verify_{a.which}.json"), "w"), indent=1)
        print(f"{os.path.basename(os.path.normpath(run))} [{a.which}] n={len(rows)}  20s bar {sum(r['pass20'] for r in rows)}/{len(rows)}"
              f"  60s bar {sum(r['pass60'] for r in rows)}/{len(rows)}"
              f"  limp med {np.median([r['limp'] for r in rows]):.3f}  lean med {np.median([r['lean_deg'] for r in rows]):.1f} deg"
              + (f"  jittered-start 20 s pass rate mean {np.mean(jit):.2f}" if a.jitter else ""), flush=True)
        for r in rows:
            print(f"   {r['file']:22s} 20s {'PASS' if r['pass20'] else 'fail'} {r['dist20']:6.2f} m L{r['nL20']} R{r['nR20']}"
                  f" air {r['air20']:.2f} | 60s {'PASS' if r['pass60'] else 'fail'} {r['dist60']:6.2f} m alive {r['alive60']:5.1f} s"
                  f" | limp {r['limp']:.3f} lean {r['lean_deg']:5.1f}")


if __name__ == "__main__":
    main()
