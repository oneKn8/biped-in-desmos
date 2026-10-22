"""Check fastsim.py against the reference sim.py / ars.py at baseline physics.

1. Single step from random perturbed states: max abs difference in positions.
2. Full 300-tick rollouts of trained policies: hip distance, alive time, max position drift.
Exit code 1 if the single-step error exceeds 1e-10.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import sim  # noqa: E402 -- must come after the sys.path inserts above that make proto/ importable
import fastsim as fs  # noqa: E402 -- must come after the sys.path insert above that makes this folder importable

ph = fs.Phys()
rng = np.random.default_rng(0)
P = 256
pos, prev = sim.initial_state(P)
pos = pos + 0.05 * rng.standard_normal(pos.shape); pos[..., 1] = np.abs(pos[..., 1])
prev = pos - 0.01 * rng.standard_normal(pos.shape)
tgt = rng.uniform(-1.5, 1.0, (P, 4))
alive = (rng.random(P) > 0.2).astype(float)
a1, b1 = sim.step(pos.copy(), prev.copy(), tgt, alive)
a2, b2 = fs.step(fs.from_rm2(pos), fs.from_rm2(prev), tgt.T.copy(), alive, ph); a2 = fs.to_rm2(a2)
err1 = float(np.abs(a1 - a2).max())
o1 = sim.observe(pos, prev, 0.37); o2 = fs.observe(fs.from_rm2(pos), fs.from_rm2(prev), 0.37, ph).T
print(f"single step max |dpos| = {err1:.3e}   observe max diff = {np.abs(o1 - o2).max():.3e}")


def roll_ref(W, T):
    pos, prev = sim.initial_state(W.shape[0]); alive = np.ones(W.shape[0]); traj = []
    for k in range(T):
        act = np.tanh(np.einsum('pao,po->pa', W, sim.observe(pos, prev, k * sim.DT)))
        pos, prev = sim.step(pos, prev, sim.targets(act), alive)
        alive = np.where(sim.fallen(pos), 0.0, alive); traj.append(pos.copy())
    return np.array(traj)


def roll_fast(W, T):
    pos, prev = fs.initial_state(W.shape[0]); alive = np.ones(W.shape[0]); traj = []
    for k in range(T):
        act = np.tanh(np.einsum('pao,op->ap', W, fs.observe(pos, prev, k * ph.DT, ph)))
        pos, prev = fs.step(pos, prev, fs.targets(act), alive, ph)
        alive = np.where(fs.fallen(pos), 0.0, alive); traj.append(fs.to_rm2(pos).copy())
    return np.array(traj)


base = os.path.dirname(HERE)
thetas = [np.load(os.path.join(base, d, f"theta_s{s}.npy")) for d in ("cap", "alive1") for s in range(4)]
W = np.stack(thetas)
t0 = time.time(); tr = roll_ref(W, 300); t1 = time.time(); tf = roll_fast(W, 300); t2 = time.time()
drift = np.abs(tr - tf).max(axis=(2, 3))            # (T, n)
for i in range(len(thetas)):
    first_bad = int(np.argmax(drift[:, i] > 1e-6)) if (drift[:, i] > 1e-6).any() else -1
    print(f"policy {i}: ref hip dx {tr[-1, i, 1, 0] - tr[0, i, 1, 0]:7.3f}  fast {tf[-1, i, 1, 0] - tf[0, i, 1, 0]:7.3f}"
          f"  max drift {drift[:, i].max():.2e}  first tick drift>1e-6: {first_bad}")
print(f"timing for {len(thetas)} rows x 300 ticks: ref {t1 - t0:.2f}s fast {t2 - t1:.2f}s")
Wbig = np.repeat(W, 130, 0)
t0 = time.time(); roll_fast(Wbig, 300); print(f"fast, {Wbig.shape[0]} rows x 300 ticks: {time.time() - t0:.2f}s")
sys.exit(0 if err1 < 1e-10 else 1)
