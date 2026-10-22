"""Checks for train.py: fitness equals ../ars.py, phase counting equals an independent
implementation, the bar on known policies, and the early-termination tick count."""
import os
import sys

import numpy as np

import fastsim as fs
import train

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)


def load_reference():
    """Import ../ars.py (the reference trainer) without putting it ahead of this folder."""
    sys.path.append(PROTO)
    import ars
    return ars


def phases_reference(cl, cr, min_ticks=3):
    """Independent count: split into maximal runs, keep single-support runs >= min_ticks,
    then count a run only when its foot differs from the previously kept run's foot."""
    ss = [0 if (a and not b) else 1 if (b and not a) else -1 for a, b in zip(cl, cr)]
    runs, start = [], 0
    for k in range(1, len(ss) + 1):
        if k == len(ss) or ss[k] != ss[start]:
            runs.append((ss[start], k - start)); start = k
    cnt, last = [0, 0], -1
    for side, n in runs:
        if side >= 0 and n >= min_ticks and side != last:
            cnt[side] += 1; last = side
    return cnt


def main():
    ars = load_reference()
    thetas = np.stack([np.load(os.path.join(PROTO, d, f"theta_s{s}.npy")) for d in ("cap", "alive1") for s in range(4)])
    n = len(thetas)
    ref = ars.rollout(thetas, T=300)
    ref_fit, ref_dist, ref_alive = ref["fit"], ref["dist"], ref["alive_t"]
    ph = fs.Phys()
    # the reference fitness is the adopted recipe's (sim.ALIVE_W, W_AIR, AIR_FREE)
    cfg = {**train.DEFAULTS, "alive_w": ars.sim.ALIVE_W, "w_air": ars.sim.W_AIR, "air_free": ars.sim.AIR_FREE}
    out = train.rollout(np.concatenate([thetas, thetas]), ph, cfg, 300, 300, n, False)
    fit = train.fitness(out, cfg)[:n]
    err = np.abs(fit - ref_fit).max()
    print(f"fitness vs ars.py: max abs diff {err:.2e}")
    assert err < 1e-8, "fitness mismatch"

    # phase counter vs independent implementation, on recorded contact sequences
    rng = np.random.default_rng(3)
    for trial in range(300):
        T = 200
        cl = rng.random(T) < rng.uniform(0.2, 0.8); cr = rng.random(T) < rng.uniform(0.2, 0.8)
        # add structured runs so long phases occur
        if trial % 2:
            per = rng.integers(4, 30)
            cl = (np.arange(T) // per) % 2 == 0; cr = ~cl
            flip = rng.random(T) < 0.08; cl = cl ^ flip
        trk = train.PhaseTracker(1)
        for k in range(T):
            trk.update(np.array([cl[k]]), np.array([cr[k]]), np.array([True]))
        trk.finish()
        assert list(trk.cnt[:, 0]) == phases_reference(cl, cr), (trial, trk.cnt[:, 0], phases_reference(cl, cr))
    print("phase counter matches the independent implementation on 300 sequences")

    ev = train.evaluate(thetas, ph, False, 5.0)
    for i in range(n):
        print(f"policy {i}: dist {ev['dist'][i]:6.2f} fallen {ev['fallen'][i]!s:5s} phases L{ev['n_left'][i]} R{ev['n_right'][i]}"
              f" air {ev['air'][i]:.2f} -> bar {'PASS' if ev['passed'][i] else 'fail'}")
    assert np.allclose(ev["dist"], np.where(ref_alive >= 5.0 - 1e-9, ref_dist, ev["dist"])), "eval distance differs"

    # early termination: all individuals fall -> seed tick count equals the last fall tick
    W = np.zeros((5, 4, 13)); W[:, 1, 12] = 3.0; W[:, 3, 12] = 3.0   # both knees fully flexed: collapses
    cfg_et = {**cfg, "early_term": True}
    out = train.rollout(W, ph, cfg_et, 300, 300, 4, False)
    print(f"early-term ticks used {out['seed_ticks'][0]} (alive_t of pop {out['alive_t'][:4].round(3)})")
    assert out["seed_ticks"][0] == int(round(out["alive_t"][:4].max() / ph.DT)) + 1
    print("all checks passed")


if __name__ == "__main__":
    main()
