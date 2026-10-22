"""Filmstrip of one rollout: overlapping stick figures across the episode, plus foot-contact timeline."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sim
from ars import rollout

theta = np.load(sys.argv[1]); out = sys.argv[2]
fit, dist, sw, at, traj = rollout(theta[None], T=300, record=True)
traj = np.array(traj)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 5.5), gridspec_kw={'height_ratios': [4, 1]})
for k in range(0, len(traj), 6):
    p = traj[k]; c = plt.cm.viridis(k / len(traj))
    for a, b, _ in sim.BONES:
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=c, lw=1.2, alpha=0.8)
ax.axhline(0, color='k', lw=0.8); ax.set_aspect('equal'); ax.set_title(f'dist {dist[0]:.2f} m, alive {at[0]:.1f} s, contact switches {int(sw[0])}')
t = np.arange(len(traj)) * sim.DT
ax2.fill_between(t, 0, (traj[:, sim.FL, 1] < 1e-3) * 1.0, step='mid', alpha=0.6, label='left foot down')
ax2.fill_between(t, 1.1, 1.1 + (traj[:, sim.FR, 1] < 1e-3) * 1.0, step='mid', alpha=0.6, label='right foot down')
ax2.plot(t, traj[:, [sim.KL, sim.KR], 1].min(1) < 1e-3, 'r.', ms=2, label='knee on ground')
ax2.legend(loc='upper right', fontsize=7); ax2.set_xlabel('s')
plt.tight_layout(); plt.savefig(out, dpi=80)
print('knee-ground frames', int((traj[:, [sim.KL, sim.KR], 1].min(1) < 1e-3).sum()), 'hip y range', traj[:, sim.HIP, 1].min().round(2), traj[:, sim.HIP, 1].max().round(2), 'head-hip min', (traj[:, sim.HEAD, 1]-traj[:, sim.HIP, 1]).min().round(2))
