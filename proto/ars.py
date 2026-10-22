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


def rollout(W, T=300, record=False):
    """W: (R, N_ACT, N_OBS). Every row runs T ticks; a fallen body freezes and stops scoring.

    Returns a dict of per-row arrays: fit (training fitness), dist, alive_t, air_t (both feet up
    while upright, seconds), air_frac (both feet up, fraction of all T ticks), fallen (ever),
    n_left / n_right (alternating single-support phases), and traj (row 0 positions) if record.
    """
    R = W.shape[0]
    pos, prev = sim.initial_state(R)
    x0 = pos[:, sim.HIP, 0].copy()
    alive = np.ones(R)
    dist = np.zeros(R)
    alive_t = np.zeros(R)
    air_t = np.zeros(R)
    air_ticks = np.zeros(R)
    fallen = np.zeros(R, bool)
    trk = walkbar.PhaseTracker(R)
    traj = []
    for k in range(T):
        obs = sim.observe(pos, prev, k * sim.DT)
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
        trk.update(cl, cr, alive > 0)
        if record:
            traj.append(pos[0].copy())
    trk.finish()
    air_pen = np.maximum(0.0, air_t - sim.AIR_FREE * alive_t)
    fit = dist + sim.ALIVE_W * alive_t - sim.W_AIR * air_pen
    return dict(fit=fit, dist=dist, alive_t=alive_t, air_t=air_t, air_frac=air_ticks / T, fallen=fallen,
                n_left=trk.cnt[0], n_right=trk.cnt[1], traj=traj)


def passes(out, T):
    return walkbar.strict_bar(out["dist"], T * sim.DT, out["fallen"], out["n_left"], out["n_right"], out["air_frac"])


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
