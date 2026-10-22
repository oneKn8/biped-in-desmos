"""Expand a mirror-symmetric 2x13 leg controller into the equivalent 4x13 policy matrix.

The symmetric policy (train.policy_act with sym=True) drives the left leg with W @ obs and the right
leg with W @ mirror(obs), where mirror swaps the leg joints and contacts and shifts the clock by half a
period. mirror is a signed permutation, mirror(obs) = Mm @ obs, so the right rows are W @ Mm and the
full policy [left hip, left knee, right hip, right knee] is linear in obs like ../sim.py's.
"""
import numpy as np

import fastsim as fs

MM = np.zeros((fs.N_OBS, fs.N_OBS))
MM[np.arange(fs.N_OBS), fs.MIRROR_IDX] = fs.MIRROR_SGN


def expand(theta):
    """theta (..., 2, 13) symmetric -> (..., 4, 13); a (..., 4, 13) input is returned unchanged."""
    theta = np.asarray(theta)
    if theta.shape[-2] == fs.N_ACT:
        return theta
    return np.concatenate([theta, theta @ MM], -2)
