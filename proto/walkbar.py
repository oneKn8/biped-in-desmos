"""The strict walking bar, scored on a noise-free rollout of the mean policy.

A policy passes when, over the evaluation horizon, it averages at least BAR_SPEED, never falls,
makes at least BAR_PHASES alternating single-support phases with each foot, and has both feet off
the ground for less than BAR_AIR of the ticks. Same definition as proto/sweep/train.py, where the
training recipe was chosen against it.
"""
import numpy as np

BAR_SPEED = 0.8          # m/s average hip advance over the evaluation horizon
BAR_PHASES = 3           # alternating single-support phases per foot
BAR_MIN_TICKS = 3        # minimum length of a single-support phase, in ticks
BAR_AIR = 0.25           # max fraction of ticks with both feet airborne


class PhaseTracker:
    """Counts alternating single-support phases per foot, for R rollouts at once.

    A single-support phase of foot X is a maximal run of ticks with X down and the other foot up.
    It counts if it lasts >= min_ticks and the previous counted phase belonged to the other foot,
    so a chattering stance foot cannot score twice: phases must alternate L, R, L, R, ...
    """

    def __init__(self, R, min_ticks=BAR_MIN_TICKS):
        self.side = np.full(R, -1)
        self.len = np.zeros(R, int)
        self.lastq = np.full(R, -1)
        self.cnt = np.zeros((2, R), int)
        self.min = min_ticks

    def _close(self, mask):
        q = mask & (self.side >= 0) & (self.len >= self.min) & (self.side != self.lastq)
        idx = np.nonzero(q)[0]
        self.cnt[self.side[idx], idx] += 1
        self.lastq[idx] = self.side[idx]

    def update(self, cl, cr, active):
        """cl, cr: foot-down flags after this tick; active: rows still upright."""
        ss = np.where(cl & ~cr, 0, np.where(cr & ~cl, 1, -1))
        ss = np.where(active, ss, -1)
        same = ss == self.side
        self._close(~same)
        self.len = np.where(same, self.len + 1, 1)
        self.side = ss

    def finish(self):
        self._close(np.ones_like(self.side, bool))
        self.side[:] = -1


def phases_needed(horizon_s):
    """BAR_PHASES per foot per 5 s of horizon (3 over 5 s, 12 over 20 s, 36 over 60 s)."""
    return int(round(BAR_PHASES * horizon_s / 5.0))


def strict_bar(dist, horizon_s, ever_fallen, n_left, n_right, air_frac, phases=BAR_PHASES):
    """Elementwise over rollouts (or scalars). phases: per foot over the whole horizon."""
    return ((dist / horizon_s >= BAR_SPEED) & (~ever_fallen) & (n_left >= phases)
            & (n_right >= phases) & (air_frac < BAR_AIR))
