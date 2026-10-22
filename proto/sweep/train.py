"""ARS (V1-t) recipe trainer: several seeds of one config batched along the population axis.

Same algorithm as ../ars.py (antithetic directions, top-b by max(r+, r-), reward-std scaling),
with these recipe knobs (all elementwise per individual, so they map onto Desmos lists):
  P, b, nu, alpha              population (= 2 * directions), top directions, noise, step
  dt, iters                    physics re-discretized by fastsim.Phys.scaled (same rad/s, same
                               per-second damping, friction and motor time constant)
  ep_s                         training episode length in seconds
  alive_w                      fitness = dist + alive_w * seconds upright (as ars.py) + shaping:
  w_air   * max(0, air_t - air_free * alive_t)
                               air_t = seconds with both feet off the ground (penalty on the
                               part above a free fraction of the upright time; air_free=0: all)
  w_up    * up_t               up_t  = sum dt * (head.y - hip.y) / TORSO      (uprightness)
  w_clk   * |C|                C     = sum dt * (cl - cr) * exp(i * clock phase): amplitude of the
                               left-minus-right contact signal at the clock frequency
  w_alt   * min(alts, alt_cap) alts  = alternating single-support phases >= 3 ticks
  early_term                   end the episode when every individual of the population has fallen
  sym                          mirror-symmetric policy: one 2x13 leg controller; the right leg sees
                               the left/right-swapped observation with the clock shifted by pi

Every generation the current mean policy theta_g is rolled out (noise-free) for a fixed 5 s
evaluation horizon and scored with the strict walking bar (strict_bar below). Ticks are counted as
the ticks the training population was simulated for (what Desmos pays), not the evaluation ticks.

Usage: python3 train.py config.json      (writes into runs/<name>/)
"""
import json
import os
import pickle
import sys
import time

import numpy as np

import fastsim as fs

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    name="baseline", P=32, b=8, nu=0.04, alpha=0.03, dt=1 / 60, iters=4, phys="full",
    ep_s=5.0, alive_w=1.0, w_air=0.0, air_free=0.0, w_up=0.0, w_clk=0.0, w_alt=0.0, alt_cap=12,
    early_term=False, sym=False, seeds=[0, 1, 2, 3, 4, 5, 6, 7], budget_ticks=120_000,
    eval_s=5.0, post_pass_gens=40, ep_sched=None,
)

# strict walking bar (fixed before the sweep)
BAR_SPEED = 0.8          # m/s average hip advance over the evaluation horizon
BAR_PHASES = 3           # alternating single-support phases per foot
BAR_MIN_TICKS = 3        # minimum length of a single-support phase, in ticks
BAR_AIR = 0.25           # max fraction of ticks with both feet airborne


class PhaseTracker:
    """Counts alternating single-support phases per foot.

    A single-support phase of foot X = maximal run of ticks with X down and the other foot up.
    It counts if it lasts >= min_ticks AND the previous counted phase belonged to the other foot
    (so a chattering stance foot cannot score twice: phases must alternate L, R, L, R, ...).
    """

    def __init__(self, R, min_ticks=BAR_MIN_TICKS):
        self.side = np.full(R, -1); self.len = np.zeros(R, int)
        self.lastq = np.full(R, -1); self.cnt = np.zeros((2, R), int); self.min = min_ticks

    def _close(self, mask):
        q = mask & (self.side >= 0) & (self.len >= self.min) & (self.side != self.lastq)
        idx = np.nonzero(q)[0]
        self.cnt[self.side[idx], idx] += 1
        self.lastq[idx] = self.side[idx]

    def update(self, cl, cr, active):
        ss = np.where(cl & ~cr, 0, np.where(cr & ~cl, 1, -1))
        ss = np.where(active, ss, -1)
        same = ss == self.side
        self._close(~same)
        self.len = np.where(same, self.len + 1, 1)
        self.side = ss

    def finish(self):
        self._close(np.ones_like(self.side, bool))
        self.side[:] = -1


def strict_bar(dist, horizon_s, ever_fallen, n_left, n_right, air_frac):
    """The strict walking bar. All arguments are arrays over individuals (or scalars)."""
    return ((dist / horizon_s >= BAR_SPEED) & (~ever_fallen) & (n_left >= BAR_PHASES)
            & (n_right >= BAR_PHASES) & (air_frac < BAR_AIR))


def policy_act(W, obs, sym):
    """W (R, A, N_OBS); obs (N_OBS, R) -> act (4, R)."""
    if not sym:
        return np.tanh(np.einsum('rao,or->ar', W, obs))
    left = np.tanh(np.einsum('rao,or->ar', W, obs))
    right = np.tanh(np.einsum('rao,or->ar', W, fs.mirror_obs(obs)))
    return np.concatenate([left, right], 0)


def rollout(W, ph, cfg, T_train, T_eval, n_pop_rows, sym):
    """Roll out R = n_pop_rows + S rows (S = one eval row per seed, the last S rows).

    Population rows score fitness over T_train ticks, or until every individual of their seed has
    fallen when cfg["early_term"]; the tick count each seed actually used is seed_ticks. Eval rows
    are scored with the strict bar over exactly T_eval ticks (all bar quantities frozen there).
    """
    R = W.shape[0]
    S = R - n_pop_rows
    P = n_pop_rows // S
    pos, prev = fs.initial_state(R)
    x0 = pos[fs.HIP, 0].copy()
    alive = np.ones(R); dist = np.zeros(R); alive_t = np.zeros(R); air_t = np.zeros(R)
    up_t = np.zeros(R); cs = np.zeros(R); cc = np.zeros(R); ever_fallen = np.zeros(R, bool)
    air_ticks = np.zeros(S)
    trk = PhaseTracker(R)
    seed_done = np.zeros(S, bool); seed_ticks = np.full(S, T_train)
    rows = np.arange(R)                       # rows still simulated (indices into the full arrays)
    Wk = W
    DT = ph.DT
    pop_active = True
    for k in range(max(T_train, T_eval)):
        if pop_active and (k == T_train or seed_done.all()):
            pop_active = False                # population finished: simulate only the eval rows
            keep = rows >= n_pop_rows
            rows = rows[keep]; pos = pos[:, :, keep]; prev = prev[:, :, keep]; Wk = W[rows]
        if not pop_active and k >= T_eval:
            break
        a = alive[rows]
        obs = fs.observe(pos, prev, k * DT, ph)
        pos, prev = fs.step(pos, prev, fs.targets(policy_act(Wk, obs, sym)), a, ph)
        cl = pos[fs.FL, 1] < 1e-3; cr = pos[fs.FR, 1] < 1e-3
        down = fs.fallen(pos) & (a > 0)
        dist[rows] = np.where(a > 0, pos[fs.HIP, 0] - x0[rows], dist[rows])
        a = np.where(down, 0.0, a)
        alive[rows] = a
        ever_fallen[rows] |= down
        alive_t[rows] += a * DT
        both_up = ~cl & ~cr
        air_t[rows] += a * DT * both_up
        up_t[rows] += a * DT * (pos[fs.HEAD, 1] - pos[fs.HIP, 1]) / fs.TORSO
        phase = 2 * np.pi * fs.CLOCK_HZ * (k + 1) * DT
        diff = cl.astype(float) - cr.astype(float)
        cs[rows] += a * DT * diff * np.sin(phase); cc[rows] += a * DT * diff * np.cos(phase)
        cl_f = np.zeros(R, bool); cr_f = np.zeros(R, bool); act_f = np.zeros(R, bool)
        cl_f[rows] = cl; cr_f[rows] = cr; act_f[rows] = a > 0
        if k < T_eval:
            air_ticks += both_up[rows >= n_pop_rows]
        else:
            act_f[n_pop_rows:] = False        # eval rows stop counting phases at the horizon
        trk.update(cl_f, cr_f, act_f)
        if k == T_eval - 1:
            dist_eval = dist[n_pop_rows:].copy()
            fallen_eval = ever_fallen[n_pop_rows:].copy()
        if cfg["early_term"] and pop_active:
            dead = (alive[:n_pop_rows].reshape(S, P) == 0).all(1)
            seed_ticks[dead & ~seed_done] = k + 1
            seed_done |= dead
    trk.finish()
    return dict(dist=dist, alive_t=alive_t, air_t=air_t, up_t=up_t, clk=np.sqrt(cs ** 2 + cc ** 2),
                alts=trk.cnt.sum(0), n_left=trk.cnt[0], n_right=trk.cnt[1],
                fallen_eval=fallen_eval, air_frac_eval=air_ticks / T_eval, dist_eval=dist_eval,
                nL_eval=trk.cnt[0, n_pop_rows:], nR_eval=trk.cnt[1, n_pop_rows:], seed_ticks=seed_ticks)


def fitness(out, cfg):
    air_pen = np.maximum(0.0, out["air_t"] - cfg["air_free"] * out["alive_t"])
    return (out["dist"] + cfg["alive_w"] * out["alive_t"] - cfg["w_air"] * air_pen
            + cfg["w_up"] * out["up_t"] + cfg["w_clk"] * out["clk"]
            + cfg["w_alt"] * np.minimum(out["alts"], cfg["alt_cap"]))


def episode_seconds(cfg, g):
    if cfg.get("ep_sched"):
        s0, s1, gr = cfg["ep_sched"]
        return s0 + (s1 - s0) * min(1.0, g / gr)
    return cfg["ep_s"]


def evaluate(theta, ph, sym, seconds):
    """Noise-free rollout of a batch of policies over a horizon; returns the bar components."""
    T = int(round(seconds / ph.DT))
    S = theta.shape[0]
    out = rollout(np.concatenate([theta, theta]), ph, dict(early_term=False), 1, T, S, sym)
    passed = strict_bar(out["dist_eval"], seconds, out["fallen_eval"], out["nL_eval"], out["nR_eval"],
                        out["air_frac_eval"])
    return dict(dist=out["dist_eval"], fallen=out["fallen_eval"], n_left=out["nL_eval"],
                n_right=out["nR_eval"], air=out["air_frac_eval"], passed=passed)


def make_phys(cfg):
    if cfg["phys"] == "full":
        return fs.Phys.scaled(cfg["dt"], cfg["iters"])
    if cfg["phys"] == "cap":
        return fs.Phys.scaled(cfg["dt"], cfg["iters"], scale_k=False, scale_damp=False, scale_fric=False)
    raise ValueError(f"unknown phys mode {cfg['phys']!r}")


def train(cfg):
    cfg = {**DEFAULTS, **cfg}
    out_dir = os.path.join(HERE, "runs", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    ph = make_phys(cfg)
    seeds = cfg["seeds"]; S = len(seeds)
    P = cfg["P"]; N = P // 2; b = cfg["b"]; sym = cfg["sym"]
    if P % 2 or not 1 <= b <= N:
        raise ValueError(f"need even P and 1 <= b <= P/2, got P={P} b={b}")
    A = 2 if sym else fs.N_ACT
    rngs = [np.random.default_rng(1_000_003 * s + 17) for s in seeds]
    theta = np.zeros((S, A, fs.N_OBS))
    final_theta = np.zeros_like(theta)
    T_eval = int(round(cfg["eval_s"] / ph.DT))
    hist = [[] for _ in range(S)]
    cum = np.zeros(S)                         # training ticks spent to produce the current theta
    first_pass = [None] * S
    done = np.zeros(S, bool)                  # seed finished: passed + post_pass_gens, or budget spent
    t_start = time.time()
    g = 0
    ckpt_path = os.path.join(out_dir, "ckpt.pkl")
    if os.path.exists(ckpt_path):            # resume an interrupted run (file written by this script)
        with open(ckpt_path, "rb") as fh:
            ck = pickle.load(fh)
        if any(ck["cfg"].get(k, DEFAULTS.get(k)) != cfg.get(k, DEFAULTS.get(k)) for k in set(ck["cfg"]) | set(cfg)):
            raise SystemExit(f"{ckpt_path} was written for a different config; move it away to restart")
        g, theta, final_theta, hist, cum, first_pass, done = (ck[k] for k in ("g", "theta", "final_theta", "hist", "cum", "first_pass", "done"))
        for r, st in zip(rngs, ck["rng"]):
            r.bit_generator.state = st
    log = open(os.path.join(out_dir, "log.txt"), "a")
    if g:
        print(f"resumed at gen {g}", file=log, flush=True)
    while not done.all():
        T_train = int(round(episode_seconds(cfg, g) / ph.DT))
        delta = np.stack([r.standard_normal((N, A, fs.N_OBS)) for r in rngs])      # (S, N, A, O)
        Wp = np.concatenate([theta[:, None] + cfg["nu"] * delta, theta[:, None] - cfg["nu"] * delta], 1)
        W = np.concatenate([Wp.reshape(S * P, A, fs.N_OBS), theta], 0)
        out = rollout(W, ph, cfg, T_train, T_eval, S * P, sym)
        fit = fitness(out, cfg)[:S * P].reshape(S, P)
        passed = strict_bar(out["dist_eval"], cfg["eval_s"], out["fallen_eval"], out["nL_eval"],
                            out["nR_eval"], out["air_frac_eval"])
        pop_dist = out["dist"][:S * P].reshape(S, P)
        for i in np.nonzero(~done)[0]:
            hist[i].append(dict(g=g, ticks_before=float(cum[i]), ticks_gen=int(out["seed_ticks"][i]),
                                dist=float(out["dist_eval"][i]), fallen=bool(out["fallen_eval"][i]),
                                nL=int(out["nL_eval"][i]), nR=int(out["nR_eval"][i]),
                                air=float(out["air_frac_eval"][i]), passed=bool(passed[i]),
                                pop_best=float(pop_dist[i].max()), fit_mean=float(fit[i].mean())))
            if passed[i] and first_pass[i] is None:
                first_pass[i] = g
                np.save(os.path.join(out_dir, f"theta_firstpass_s{seeds[i]}.npy"), theta[i])
        for i in range(S):                    # ARS V1-t update
            rp, rm = fit[i, :N], fit[i, N:]
            top = np.argsort(-np.maximum(rp, rm))[:b]
            sigma = np.concatenate([rp[top], rm[top]]).std() + 1e-8
            theta[i] += cfg["alpha"] / (b * sigma) * np.einsum('n,nao->ao', rp[top] - rm[top], delta[i, top])
        cum += np.where(done, 0, out["seed_ticks"])
        if g % 10 == 0:
            print(f"gen {g:4d} ticks~{int(np.median(cum)):7d}  dist " + " ".join(f"{h[-1]['dist']:5.2f}" for h in hist)
                  + "  pass " + "".join("P" if h[-1]["passed"] else "." for h in hist)
                  + f"  [{time.time() - t_start:6.0f}s]", file=log, flush=True)
        g += 1
        for i in np.nonzero(~done)[0]:
            fp = first_pass[i]
            if (fp is not None and g - fp > cfg["post_pass_gens"]) or cum[i] >= cfg["budget_ticks"]:
                done[i] = True
                final_theta[i] = theta[i]     # theta after the last recorded generation's update
        if g % 10 == 0:
            tmp = ckpt_path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(dict(cfg=cfg, g=g, theta=theta, final_theta=final_theta, hist=hist, cum=cum,
                                 first_pass=first_pass, done=done, rng=[r.bit_generator.state for r in rngs]), fh)
            os.replace(tmp, ckpt_path)
    for i, s in enumerate(seeds):
        np.save(os.path.join(out_dir, f"theta_final_s{s}.npy"), final_theta[i])
        json.dump(hist[i], open(os.path.join(out_dir, f"hist_s{s}.json"), "w"))
    rob = evaluate(final_theta, ph, sym, 10.0)
    summary = dict(
        name=cfg["name"], seeds=seeds, first_pass=first_pass, gens=[len(h) for h in hist],
        ticks_to_bar=[hist[i][fp]["ticks_before"] if fp is not None else None for i, fp in enumerate(first_pass)],
        post_pass_rate=[float(np.mean([r["passed"] for r in hist[i][fp + 1:]])) if fp is not None and fp + 1 < len(hist[i]) else None
                        for i, fp in enumerate(first_pass)],
        final_dist=[float(np.mean([r["dist"] for r in hist[i][-10:]])) for i in range(S)],
        final_pass_last10=[float(np.mean([r["passed"] for r in hist[i][-10:]])) for i in range(S)],
        robust10_final=[bool(x) for x in rob["passed"]], robust10_final_dist=[float(x) for x in rob["dist"]],
        wall_s=time.time() - t_start, phys=ph.as_dict())
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"done {cfg['name']} in {time.time() - t_start:.0f}s first_pass={first_pass}", file=log, flush=True)
    return summary


if __name__ == "__main__":
    s = train(json.load(open(sys.argv[1])))
    print(json.dumps({k: s[k] for k in ("name", "first_pass", "ticks_to_bar", "post_pass_rate")}))
