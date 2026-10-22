"""Augmented Random Search (Mania et al. 2018, V1-t) on a linear policy, population evaluated in parallel.

Each generation rolls out 2*N perturbed policies (antithetic pairs) plus the noise-free mean policy
as the last row, all at once. The perturbed rows give the update; the mean row is scored against the
strict walking bar (walkbar.py), and training stops at the first generation whose mean policy passes.

Usage: python3 ars.py [gens] [seed] [--dirs N] [--top b] [--nu nu] [--alpha alpha] [--keep-going]
"""
import argparse
import json

import numpy as np

import sim
import walkbar


def rollout(W, T=300, record=False, start=None, t0=None, cut=False):
    """W: (R, N_ACT, N_OBS). Every row runs T ticks; a fallen body freezes and stops scoring.

    start = (pos, prev), each (R, M, 2), replaces the rest pose; t0 (R,) offsets the clock input in
    seconds. cut: the last row is the mean policy, and once it has fallen every body freezes, which
    scores the same as ending the episode there (the long-walk exam).
    Returns a dict of per-row arrays: fit (training fitness), dist, alive_t, air_t (both feet up
    while upright, seconds), air_frac (both feet up, fraction of all T ticks), fallen (ever),
    n_left / n_right (alternating single-support phases), the final alive flags (0 for every body after
    an exam cut), the final pos / prev, traj (row 0 positions) if record, and the gait sums while upright:
    lean2 (dt * torso angle^2), lean_abs (dt * |torso angle|), s_l / s_r (seconds in single support).
    """
    R = W.shape[0]
    pos, prev = sim.initial_state(R) if start is None else (start[0].copy(), start[1].copy())
    x0 = pos[:, sim.HIP, 0].copy()
    alive = np.ones(R)
    dist = np.zeros(R)
    alive_t = np.zeros(R)
    air_t = np.zeros(R)
    air_ticks = np.zeros(R)
    lean2 = np.zeros(R); lean_abs = np.zeros(R); s_l = np.zeros(R); s_r = np.zeros(R)
    fallen = np.zeros(R, bool)
    trk = walkbar.PhaseTracker(R)
    traj = []
    for k in range(T):
        if cut:
            alive = alive * alive[-1]
        obs = sim.observe(pos, prev, k * sim.DT if t0 is None else t0 + k * sim.DT)
        act = np.tanh(np.einsum('pao,po->pa', W, obs))
        pos, prev = sim.step(pos, prev, sim.targets(act), alive)
        cl = pos[:, sim.FL, 1] < 1e-3
        cr = pos[:, sim.FR, 1] < 1e-3
        down = sim.fallen(pos) & (alive > 0)
        dist = np.where(alive > 0, pos[:, sim.HIP, 0] - x0, dist)
        alive = np.where(down, 0.0, alive)
        fallen |= down
        alive_t += alive * sim.DT
        both_up = ~cl & ~cr
        air_t += alive * sim.DT * both_up
        air_ticks += both_up
        th = np.arctan2(pos[:, sim.HEAD, 0] - pos[:, sim.HIP, 0], pos[:, sim.HEAD, 1] - pos[:, sim.HIP, 1])
        lean2 += alive * sim.DT * th * th
        lean_abs += alive * sim.DT * np.abs(th)
        s_l += alive * sim.DT * (cl & ~cr)
        s_r += alive * sim.DT * (cr & ~cl)
        trk.update(cl, cr, alive > 0)
        if record:
            traj.append(pos[0].copy())
    trk.finish()
    air_pen = np.maximum(0.0, air_t - sim.AIR_FREE * alive_t)
    fit = dist + sim.ALIVE_W * alive_t - sim.W_AIR * air_pen
    return dict(fit=fit, dist=dist, alive_t=alive_t, air_t=air_t, air_frac=air_ticks / T, fallen=fallen,
                n_left=trk.cnt[0], n_right=trk.cnt[1], alive=alive, pos=pos, prev=prev, traj=traj,
                lean2=lean2, lean_abs=lean_abs, s_l=s_l, s_r=s_r)


def passes(out, T):
    """The strict bar over T ticks, phases scaled with the horizon."""
    return walkbar.strict_bar(out["dist"], T * sim.DT, out["fallen"], out["n_left"], out["n_right"], out["air_frac"],
                              phases=walkbar.phases_needed(T * sim.DT))


MIRROR_OBS = [0, 1, 4, 5, 2, 3, 6, 7, 9, 8, 10, 11, 12]     # inputs of the other leg; clock shifted by half a period
MIRROR_SGN = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1]


def mirror(theta):
    """Mirror image of policies (..., N_ACT, N_OBS): the legs' rows swapped, every row's inputs mirrored."""
    t = np.asarray(theta)
    return np.concatenate([t[..., 2:, :], t[..., :2, :]], -2)[..., MIRROR_OBS] * np.array(MIRROR_SGN, float)


def carry(out, row=-1):
    """Start of a window that continues row's walk: its last pose with the hip shifted to x = 0."""
    p, q = out["pos"][row].copy(), out["prev"][row].copy()
    x = p[sim.HIP, 0]
    p[:, 0] -= x
    q[:, 0] -= x
    return p, q


def main(gens=200, N=64, b=32, nu=0.12, alpha=0.15, seed=0, T=300, stop_at_pass=True):
    rng = np.random.default_rng(seed)
    theta = np.zeros((sim.N_ACT, sim.N_OBS))
    hist = []
    for g in range(gens):
        delta = rng.standard_normal((N, sim.N_ACT, sim.N_OBS))
        W = np.concatenate([theta + nu * delta, theta - nu * delta, theta[None]])
        out = rollout(W, T)
        ok = bool(passes(out, T)[-1])
        hist.append(dict(g=g, ticks=g * T, dist=float(out["dist"][-1]), fallen=bool(out["fallen"][-1]),
                         n_left=int(out["n_left"][-1]), n_right=int(out["n_right"][-1]),
                         air=float(out["air_frac"][-1]), passed=ok, pop_best=float(out["dist"][:-1].max())))
        if g % 5 == 0 or ok:
            h = hist[-1]
            print(f"gen {g:3d}  mean policy {h['dist']:5.2f} m  fell {'yes' if h['fallen'] else 'no '}  "
                  f"phases L{h['n_left']} R{h['n_right']}  air {h['air']:.2f}  pop best {h['pop_best']:5.2f}"
                  + ("  PASS" if ok else ""), flush=True)
        if ok and stop_at_pass:
            break
        fit = out["fit"]
        rp, rm = fit[:N], fit[N:2 * N]
        top = np.argsort(-np.maximum(rp, rm))[:b]
        sigma = np.concatenate([rp[top], rm[top]]).std() + 1e-8
        theta += alpha / (b * sigma) * np.einsum('n,nao->ao', rp[top] - rm[top], delta[top])
    np.save(f'theta_s{seed}.npy', theta)
    json.dump(hist, open(f'hist_s{seed}.json', 'w'))
    first = next((h["g"] for h in hist if h["passed"]), None)
    print(f"seed {seed}: " + (f"mean policy passed the strict bar at generation {first} ({first * T} training ticks)"
                              if first is not None else f"no pass in {gens} generations") + f"; saved theta_s{seed}.npy")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("gens", nargs="?", type=int, default=200)
    ap.add_argument("seed", nargs="?", type=int, default=0)
    ap.add_argument("--dirs", type=int, default=64, help="antithetic directions N (population 2N + 1 mean row)")
    ap.add_argument("--top", type=int, default=32, help="top directions b used in the update")
    ap.add_argument("--nu", type=float, default=0.12)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--keep-going", action="store_true", help="train all gens instead of stopping at the first pass")
    a = ap.parse_args()
    main(gens=a.gens, N=a.dirs, b=a.top, nu=a.nu, alpha=a.alpha, seed=a.seed, stop_at_pass=not a.keep_going)
