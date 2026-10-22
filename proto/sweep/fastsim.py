"""Vectorized, parametrized copy of ../sim.py for recipe sweeps.

Same math as sim.py term for term (Verlet, K Jacobi passes of bone constraints plus joint-angle
motors, ground clamp inside each pass, friction after the passes). The only differences:
  * physics constants live in a Phys object instead of module globals, so one process can run
    any (DT, ITERS, MAX_ROT, ...) combination;
  * bones and motors are applied with incidence matrices (one small matrix multiply each, population axis last) instead of Python
    loops, which changes floating-point summation order (verified to ~1e-12 per step against
    sim.py, see verify_fastsim.py).
Everything stays elementwise per individual, so it still maps onto Desmos lists.
"""
from dataclasses import dataclass, asdict
import numpy as np

HEAD, HIP, KL, FL, KR, FR = range(6)
M = 6
TORSO, THIGH, SHIN = 0.6, 0.45, 0.45
BONES = [(HEAD, HIP, TORSO), (HIP, KL, THIGH), (KL, FL, SHIN), (HIP, KR, THIGH), (KR, FR, SHIN)]
MOTORS = [(HEAD, HIP, KL), (HIP, KL, FL), (HEAD, HIP, KR), (HIP, KR, FR)]
N_OBS, N_ACT = 13, 4
CLOCK_HZ = 1.1
G = -9.8

BA = np.array([b[0] for b in BONES]); BB = np.array([b[1] for b in BONES])
BL = np.array([b[2] for b in BONES])
MF = np.array([m[0] for m in MOTORS]); MJ = np.array([m[1] for m in MOTORS])
MC = np.array([m[2] for m in MOTORS])

# incidence matrices: corr[m] = sum_k I[m,k] * contribution[k]
I_BONE = np.zeros((M, len(BONES)))
for k, (a, b, _) in enumerate(BONES):
    I_BONE[a, k] += 1.0; I_BONE[b, k] -= 1.0
I_MC = np.zeros((M, len(MOTORS))); I_MF = np.zeros((M, len(MOTORS))); I_MJ = np.zeros((M, len(MOTORS)))
for k, (f, j, c) in enumerate(MOTORS):
    I_MC[c, k] += 1.0; I_MF[f, k] += 1.0; I_MJ[j, k] += 1.0
CNT = np.zeros(M)
for a, b, _ in BONES:
    CNT[a] += 1; CNT[b] += 1
for f, j, c in MOTORS:
    CNT[c] += 1; CNT[f] += 1; CNT[j] += 1

BASE_DT, BASE_ITERS, BASE_MAX_ROT, BASE_K, BASE_DAMP, BASE_FRIC = 1 / 60, 4, 0.05, 0.35, 0.995, 0.85


@dataclass
class Phys:
    DT: float = BASE_DT
    ITERS: int = BASE_ITERS
    MAX_ROT: float = BASE_MAX_ROT
    MOTOR_K: float = BASE_K
    DAMP: float = BASE_DAMP
    FRICTION: float = BASE_FRIC

    @staticmethod
    def scaled(dt, iters, scale_cap=True, scale_k=True, scale_damp=True, scale_fric=True):
        """Re-discretize the baseline (1/60 s, 4 passes) at a new DT and pass count.

        cap:   MAX_ROT * ITERS / DT held constant  -> same joint-speed cap in rad/s (12 rad/s).
        k:     per-pass error retention (1-K) raised to (seconds per pass / baseline seconds per pass)
               -> same motor time constant in seconds.
        damp:  per-tick velocity factor raised to (DT / baseline DT) -> same damping per second.
        fric:  per-tick slip retention (1-F) raised to (DT / baseline DT) -> same slip decay per second.
        """
        r_tick = dt / BASE_DT
        r_pass = (dt / iters) / (BASE_DT / BASE_ITERS)
        return Phys(
            DT=dt, ITERS=iters,
            MAX_ROT=BASE_MAX_ROT * (BASE_ITERS / iters) * r_tick if scale_cap else BASE_MAX_ROT,
            MOTOR_K=1 - (1 - BASE_K) ** r_pass if scale_k else BASE_K,
            DAMP=BASE_DAMP ** r_tick if scale_damp else BASE_DAMP,
            FRICTION=1 - (1 - BASE_FRIC) ** r_tick if scale_fric else BASE_FRIC,
        )

    def as_dict(self):
        return asdict(self)


def wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def initial_state(P):
    """Layout (M, 2, P): mass, coordinate (x, y), individual. Population last = contiguous lists."""
    pos = np.zeros((M, 2, P))
    hip_y = THIGH + SHIN - 0.02
    pos[HIP, 1] = hip_y
    pos[HEAD, 1] = hip_y + TORSO
    for knee, foot, s in ((KL, FL, 1.0), (KR, FR, -1.0)):
        a = 0.12 * s
        pos[knee, 0] = THIGH * np.sin(a)
        pos[knee, 1] = hip_y - THIGH * np.cos(a)
        pos[foot, 0] = pos[knee, 0] + SHIN * np.sin(a)
        pos[foot, 1] = pos[knee, 1] - SHIN * np.cos(a)
    return pos, pos.copy()


def joint_angles(pos):
    """(4, P) relative angle at each motor joint."""
    pdir = pos[MJ] - pos[MF]
    cdir = pos[MC] - pos[MJ]
    return wrap(np.arctan2(cdir[:, 1], cdir[:, 0]) - np.arctan2(pdir[:, 1], pdir[:, 0]))


def observe(pos, prev, t, ph):
    """Returns (N_OBS, P). t: scalar time in seconds; ph supplies DT for finite differences."""
    torso = pos[HEAD] - pos[HIP]
    torso_ang = np.arctan2(torso[0], torso[1])
    ptorso = prev[HEAD] - prev[HIP]
    torso_w = wrap(torso_ang - np.arctan2(ptorso[0], ptorso[1])) / ph.DT
    ja = joint_angles(pos)
    hip_vx = (pos[HIP, 0] - prev[HIP, 0]) / ph.DT
    P = pos.shape[2]
    phase = 2 * np.pi * CLOCK_HZ * t
    obs = np.empty((N_OBS, P))
    obs[0] = torso_ang; obs[1] = 0.1 * torso_w; obs[2:6] = ja
    obs[6] = pos[HIP, 1] - 0.85; obs[7] = 0.5 * hip_vx
    obs[8] = pos[FL, 1] < 1e-3; obs[9] = pos[FR, 1] < 1e-3
    obs[10] = np.sin(phase); obs[11] = np.cos(phase); obs[12] = 1.0
    return obs


def targets(act):
    """act (4, P) -> joint angle targets (4, P)."""
    a = np.clip(act, -1, 1)
    out = np.empty_like(a)
    out[0] = 0.9 * a[0]; out[2] = 0.9 * a[2]
    out[1] = -0.75 * (a[1] + 1); out[3] = -0.75 * (a[3] + 1)
    return out


def _rot_delta(r, th):
    """r (k, 2, P) rotated by th (k, P) minus r."""
    cs, sn = np.cos(th), np.sin(th)
    out = np.empty_like(r)
    out[:, 0] = cs * r[:, 0] - sn * r[:, 1] - r[:, 0]
    out[:, 1] = sn * r[:, 0] + cs * r[:, 1] - r[:, 1]
    return out


I_MOT = np.concatenate([I_MC - I_MJ, I_MF - I_MJ], 1)       # (6, 8): child deltas then far deltas
INV_CNT_L = (1.0 / np.maximum(CNT, 1))[:, None, None]


def step(pos, prev, tgt, alive, ph):
    """pos, prev (M, 2, P); tgt (4, P); alive (P,)."""
    P = pos.shape[2]
    vel = (pos - prev) * (ph.DAMP * alive)
    prev = pos
    pos = pos + vel
    pos[:, 1] += (G * ph.DT * ph.DT) * alive
    for _ in range(ph.ITERS):
        d = pos[BB] - pos[BA]                                     # (5, 2, P)
        n = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) + 1e-9           # (5, P)
        c = (0.5 * (n - BL[:, None]) / n)[:, None, :] * d
        corr = (I_BONE @ c.reshape(len(BONES), 2 * P)).reshape(M, 2, P)
        ja = joint_angles(pos)
        err = np.clip(wrap(tgt - ja) * ph.MOTOR_K, -ph.MAX_ROT, ph.MAX_ROT)   # (4, P)
        pj = pos[MJ]
        dm = np.concatenate([_rot_delta(pos[MC] - pj, 0.5 * err), _rot_delta(pos[MF] - pj, -0.5 * err)], 0)
        corr += (I_MOT @ dm.reshape(2 * len(MOTORS), 2 * P)).reshape(M, 2, P)
        pos = pos + corr * INV_CNT_L
        np.maximum(pos[:, 1], 0.0, out=pos[:, 1])
    ground = pos[:, 1] <= 1e-6
    slip = pos[:, 0] - prev[:, 0]
    pos[:, 0] = np.where(ground, prev[:, 0] + slip * (1 - ph.FRICTION), pos[:, 0])
    return pos, prev


def fallen(pos):
    return (pos[HIP, 1] < 0.45) | (pos[HEAD, 1] < pos[HIP, 1] + 0.25)


def to_rm2(pos):
    """(M, 2, P) -> (P, M, 2), the layout of ../sim.py."""
    return np.transpose(pos, (2, 0, 1))


def from_rm2(pos):
    return np.ascontiguousarray(np.transpose(pos, (1, 2, 0)))


def mirror_obs(obs):
    """Left/right mirror for the symmetric policy, obs (N_OBS, P): swap leg joints and contacts,
    shift the clock by half a period (sin, cos -> -sin, -cos)."""
    return obs[MIRROR_IDX] * MIRROR_SGN[:, None]


MIRROR_IDX = np.array([0, 1, 4, 5, 2, 3, 6, 7, 9, 8, 10, 11, 12])
MIRROR_SGN = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1], float)
