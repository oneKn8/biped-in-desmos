"""Is a re-discretized body (other DT / Jacobi passes) still the same body?

For each (DT, ITERS) and scaling mode it reports:
  * peak joint speed (rad/s) under a bang-bang hip command (should match baseline if the cap is
    scaled right),
  * bone stretch: mean and max |len - L| / L over 5 s for the 8 policies trained at baseline,
  * those 8 baseline-trained policies' hip distance and survival under the new discretization
    (a consistent re-discretization should roughly preserve what a policy does).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastsim as fs  # noqa: E402 -- must come after the sys.path insert above that makes this folder importable

base = os.path.dirname(HERE)
thetas = np.stack([np.load(os.path.join(base, d, f"theta_s{s}.npy")) for d in ("cap", "alive1") for s in range(4)])


def run(ph, W=None, forced=None, seconds=5.0):
    T = int(round(seconds / ph.DT))
    R = W.shape[0] if W is not None else 1
    pos, prev = fs.initial_state(R)
    alive = np.ones(R); x0 = pos[fs.HIP, 0].copy(); dist = np.zeros(R); alive_t = np.zeros(R)
    stretch = []; ja_hist = []
    for k in range(T):
        t = k * ph.DT
        if forced is not None:
            act = forced(t, R)
        else:
            act = np.tanh(np.einsum('pao,op->ap', W, fs.observe(pos, prev, t, ph)))
        pos, prev = fs.step(pos, prev, fs.targets(act), alive, ph)
        d = pos[fs.BB] - pos[fs.BA]
        stretch.append(np.abs(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) - fs.BL[:, None]) / fs.BL[:, None])
        ja_hist.append(fs.joint_angles(pos))
        down = fs.fallen(pos) & (alive > 0)
        dist = np.where(alive > 0, pos[fs.HIP, 0] - x0, dist)
        alive = np.where(down, 0.0, alive); alive_t += alive * ph.DT
    return dist, alive_t, np.array(stretch), np.array(ja_hist)


def bang(t, R):
    s = np.sign(np.sin(2 * np.pi * 1.0 * t))
    a = np.zeros((4, R)); a[0] = s; a[2] = -s; a[1] = -1; a[3] = -1     # straight knees, scissor hips
    return a


rows = []
for mode in ("full", "cap-only", "none"):
    for dt in (1 / 60, 1 / 30):
        for it in (4, 3, 2):
            if mode == "full":
                ph = fs.Phys.scaled(dt, it)
            elif mode == "cap-only":
                ph = fs.Phys.scaled(dt, it, scale_k=False, scale_damp=False, scale_fric=False)
            else:
                ph = fs.Phys(DT=dt, ITERS=it)
            _, _, _, ja = run(ph, forced=bang, seconds=3.0)
            hip_rel = ja[:, 0, 0] - ja[:, 2, 0]                  # scissor angle
            w = np.abs(np.diff(hip_rel)) / ph.DT
            dist, at, st, _ = run(ph, W=thetas)
            rows.append((mode, f"1/{round(1 / dt)}", it, ph.MAX_ROT, ph.MOTOR_K, ph.DAMP, ph.FRICTION,
                         w.max(), np.percentile(w, 95), st.mean(), st.max(), dist, at))

print(f"{'mode':9s} {'DT':5s} it  MAX_ROT  K      DAMP    FRIC   | peak w  p95 w | stretch mean  max  | dist of 8 baseline-trained policies (alive s)")
for r in rows:
    mode, dt, it, mr, k, dmp, fr, wmax, w95, smean, smax, dist, at = r
    ds = " ".join(f"{d:5.2f}({a:.1f})" for d, a in zip(dist, at))
    print(f"{mode:9s} {dt:5s} {it}  {mr:.4f}  {k:.3f}  {dmp:.4f}  {fr:.4f} | {wmax:6.2f} {w95:6.2f} | {smean:.4f} {smax:.3f} | {ds}")
