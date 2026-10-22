"""Augmented Random Search (Mania et al. 2018, V1-t) on a linear policy, population evaluated in parallel."""
import sys, json
import numpy as np
import sim

def rollout(W, T=300, record=False):
    """W: (P, N_ACT, N_OBS). Returns fitness, distance, alternations, optional trajectory of agent 0."""
    P = W.shape[0]
    pos, prev = sim.initial_state(P)
    x0 = pos[:, sim.HIP, 0].copy()
    alive = np.ones(P)
    dist = np.zeros(P)
    alive_t = np.zeros(P)
    last_contact = np.full(P, -1)
    switches = np.zeros(P)
    traj = []
    for k in range(T):
        obs = sim.observe(pos, prev, k * sim.DT)
        act = np.tanh(np.einsum('pao,po->pa', W, obs))
        pos, prev = sim.step(pos, prev, sim.targets(act), alive)
        cl = pos[:, sim.FL, 1] < 1e-3; cr = pos[:, sim.FR, 1] < 1e-3
        side = np.where(cl & ~cr, 0, np.where(cr & ~cl, 1, -1))
        newsw = (side >= 0) & (side != last_contact) & (last_contact >= 0) & (alive > 0)
        switches += newsw
        last_contact = np.where(side >= 0, side, last_contact)
        down = sim.fallen(pos) & (alive > 0)
        dist = np.where(alive > 0, pos[:, sim.HIP, 0] - x0, dist)
        alive = np.where(down, 0.0, alive)
        alive_t += alive * sim.DT
        if record: traj.append(pos[0].copy())
    fit = dist + 0.3 * alive_t
    return fit, dist, switches, alive_t, traj

def main(gens=300, N=16, b=8, nu=0.04, alpha=0.03, seed=0, T=300):
    rng = np.random.default_rng(seed)
    theta = np.zeros((sim.N_ACT, sim.N_OBS))
    hist = []
    for g in range(gens):
        delta = rng.standard_normal((N, sim.N_ACT, sim.N_OBS))
        W = np.concatenate([theta + nu * delta, theta - nu * delta])
        fit, dist, sw, at, _ = rollout(W, T)
        rp, rm = fit[:N], fit[N:]
        top = np.argsort(-np.maximum(rp, rm))[:b]
        sigma = np.concatenate([rp[top], rm[top]]).std() + 1e-8
        theta += alpha / (b * sigma) * np.einsum('n,nao->ao', rp[top] - rm[top], delta[top])
        f0, d0, s0, a0, _ = rollout(theta[None], T)
        hist.append((g, float(f0[0]), float(d0[0]), int(s0[0]), float(a0[0]), float(dist.max())))
        if g % 10 == 0 or g == gens - 1:
            print(f"gen {g:3d}  mean-policy dist {d0[0]:6.2f} m  alive {a0[0]:4.1f}s  switches {int(s0[0]):2d}  pop best dist {dist.max():6.2f}", flush=True)
    np.save(f'theta_s{seed}.npy', theta)
    json.dump(hist, open(f'hist_s{seed}.json', 'w'))

if __name__ == '__main__':
    main(gens=int(sys.argv[1]) if len(sys.argv) > 1 else 300, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
