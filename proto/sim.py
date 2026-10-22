"""Planar ragdoll biped, written only with operations Desmos can do as lists.

Every array has a leading population axis P, so each line maps onto a Desmos list of length P.
Masses: 0 head, 1 hip, 2 left knee, 3 left foot, 4 right knee, 5 right foot.
Physics: Verlet integration, then K Jacobi passes of (bone length constraints + joint angle
motors), then ground projection and friction. Jacobi, not Gauss-Seidel, because Desmos
evaluates whole lists at once.
"""
import numpy as np

HEAD, HIP, KL, FL, KR, FR = range(6)
M = 6
TORSO, THIGH, SHIN = 0.6, 0.45, 0.45
BONES = [(HEAD, HIP, TORSO), (HIP, KL, THIGH), (KL, FL, SHIN), (HIP, KR, THIGH), (KR, FR, SHIN)]
# motor = (parent_far, joint, child): rotates child around joint toward target relative angle
MOTORS = [(HEAD, HIP, KL), (HIP, KL, FL), (HEAD, HIP, KR), (HIP, KR, FR)]

DT = 1.0 / 60.0
G = -9.8
DAMP = 0.995
ITERS = 4
MOTOR_K = 0.35
MAX_ROT = 0.05          # cap on motor rotation per Jacobi pass (rad): a strength / joint-speed limit
FRICTION = 0.85          # fraction of horizontal slip removed while a mass is on the ground
CLOCK_HZ = 1.1
N_OBS = 13
# fitness = distance before falling + ALIVE_W * seconds upright - W_AIR * max(0, air - AIR_FREE * upright),
# where air = seconds with both feet off the ground while upright. Chosen by the recipe sweep in
# sweep/: an alive bonus slows learning, and the airborne penalty above 15% stops running gaits.
ALIVE_W = 0.0
W_AIR = 3.0
AIR_FREE = 0.15
N_ACT = 4


def wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def initial_state(P):
    pos = np.zeros((P, M, 2))
    hip_y = THIGH + SHIN - 0.02
    pos[:, HIP] = (0.0, hip_y)
    pos[:, HEAD] = (0.0, hip_y + TORSO)
    for knee, foot, s in ((KL, FL, 1.0), (KR, FR, -1.0)):
        a = 0.12 * s
        pos[:, knee, 0] = THIGH * np.sin(a)
        pos[:, knee, 1] = hip_y - THIGH * np.cos(a)
        pos[:, foot, 0] = pos[:, knee, 0] + SHIN * np.sin(a)
        pos[:, foot, 1] = pos[:, knee, 1] - SHIN * np.cos(a)
    return pos, pos.copy()


def angle(v):
    return np.arctan2(v[..., 1], v[..., 0])


def joint_angles(pos):
    """Relative angle at each motor joint: child direction minus parent direction."""
    out = []
    for far, j, c in MOTORS:
        parent_dir = angle(pos[:, j] - pos[:, far])
        child_dir = angle(pos[:, c] - pos[:, j])
        out.append(wrap(child_dir - parent_dir))
    return np.stack(out, 1)


def observe(pos, prev, t):
    torso = pos[:, HEAD] - pos[:, HIP]
    torso_ang = np.arctan2(torso[:, 0], torso[:, 1])               # 0 = upright, + = leaning forward
    ptorso = prev[:, HEAD] - prev[:, HIP]
    torso_w = wrap(torso_ang - np.arctan2(ptorso[:, 0], ptorso[:, 1])) / DT
    ja = joint_angles(pos)
    hip_vx = (pos[:, HIP, 0] - prev[:, HIP, 0]) / DT
    contact_l = (pos[:, FL, 1] < 1e-3).astype(float)
    contact_r = (pos[:, FR, 1] < 1e-3).astype(float)
    ph = 2 * np.pi * CLOCK_HZ * t
    P = pos.shape[0]
    return np.stack([torso_ang, 0.1 * torso_w, ja[:, 0], ja[:, 1], ja[:, 2], ja[:, 3],
                     pos[:, HIP, 1] - 0.85, 0.5 * hip_vx, contact_l, contact_r,
                     np.broadcast_to(np.sin(ph), (P,)), np.broadcast_to(np.cos(ph), (P,)), np.ones(P)], 1)


def targets(act):
    """Map policy outputs in [-1, 1] to joint angle targets (radians)."""
    a = np.clip(act, -1, 1)
    hip = 0.9 * a[:, [0, 2]]                 # thigh swing relative to torso axis
    knee = -0.75 * (a[:, [1, 3]] + 1)        # knee flexes one way only: 0 to -1.5
    return np.stack([hip[:, 0], knee[:, 0], hip[:, 1], knee[:, 1]], 1)


def step(pos, prev, tgt, alive):
    vel = (pos - prev) * DAMP * alive[:, None, None]
    prev = pos
    pos = pos + vel
    pos[..., 1] += G * DT * DT * alive[:, None]
    for _ in range(ITERS):
        corr = np.zeros_like(pos)
        cnt = np.zeros(pos.shape[:2])
        for a, b, L in BONES:
            d = pos[:, b] - pos[:, a]
            n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            c = 0.5 * (n - L) / n * d
            corr[:, a] += c; corr[:, b] -= c
            cnt[:, a] += 1; cnt[:, b] += 1
        ja = joint_angles(pos)
        for m, (far, j, c) in enumerate(MOTORS):
            ref = tgt[:, m]          # 0 = child in line with parent (straight leg / straight knee)
            err = np.clip(wrap(ref - ja[:, m]) * MOTOR_K, -MAX_ROT, MAX_ROT)
            # rotate child by +err/2 and parent-far by -err/2 around the joint, then move the
            # joint by minus their sum so the motor adds no net translation (no self-propulsion)
            moved = 0.0
            for idx, s in ((c, 0.5), (far, -0.5)):
                r = pos[:, idx] - pos[:, j]
                th = s * err
                cs, sn = np.cos(th), np.sin(th)
                rot = np.stack([cs * r[:, 0] - sn * r[:, 1], sn * r[:, 0] + cs * r[:, 1]], 1)
                corr[:, idx] += rot - r
                cnt[:, idx] += 1
                moved = moved + (rot - r)
            corr[:, j] -= moved
            cnt[:, j] += 1
        pos = pos + corr / np.maximum(cnt, 1)[..., None]
        pos[..., 1] = np.maximum(pos[..., 1], 0.0)
    ground = pos[..., 1] <= 1e-6
    slip = pos[..., 0] - prev[..., 0]
    pos[..., 0] = np.where(ground, prev[..., 0] + slip * (1 - FRICTION), pos[..., 0])
    return pos, prev


def fallen(pos):
    return (pos[:, HIP, 1] < 0.45) | (pos[:, HEAD, 1] < pos[:, HIP, 1] + 0.25)
