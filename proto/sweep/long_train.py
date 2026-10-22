"""Long-walk trainer: ARS V1-t with the mean row inside the episode and an episode-length curriculum.

Reuses train.py's physics (fastsim), policy and phase tracker. What differs from train.py:
  * The noise-free mean row is simulated inside the training episode, as in the Desmos graph: it runs
    the current episode length, it is scored with the strict bar over that episode, and it counts for
    early termination (the episode ends when every body, mean row included, has fallen).
  * The bar's phase requirement scales with the horizon: 3 alternating single-support phases per foot
    per 5 s (12 over 20 s, 36 over 60 s). Speed, fall and airborne thresholds are unchanged.
  * Episode length follows `stages` (seconds). advance="bar": a seed moves to the next stage once its
    mean row passes the bar of the current stage `pass_streak` generations in a row. advance="gens":
    stage i lasts stage_gens[i] generations. demote=True: in a stage > 0 a failed mean row sends the
    seed back one stage (the longer episode then acts as an exam taken after each pass). A seed stops
    when its mean row passes the final stage's bar `final_streak` generations in a row (then it runs
    `post_pass_gens` more generations).
  * Per-stage schedules: nu_stages, alpha_stages, b_stages (lists indexed by stage; None = constant).
  * Extra shaping, all per-body sums accumulated tick by tick while upright:
      w_lean * sum dt * torso_angle^2            (torso angle 0 = upright, + = forward)
      w_asym * |sum dt * [left single support] - sum dt * [right single support]|
      w_late * (distance at the end - distance at the middle tick of the episode)
      w_alt  * min(alternating phases, alt_cap)
  * Chained starts (chain_stages: list of flags per stage): an episode of a flagged stage starts from
    the state the mean row ended the previous episode in (positions shifted to x = 0, the clock input
    continuing), when that mean row passed the bar; otherwise from rest. With stages [5, 20],
    chain_stages [true, false], pass_streak 4 and demote, the population trains on 5 s windows of an
    ongoing walk and takes a from-rest 20 s exam after 4 passing windows in a row.
  * cut_stages (list of flags per stage): an episode of a flagged stage ends as soon as the mean row
    falls (the population's fitness is then the truncated episode's), so a failed exam is cheap. With
    pace_cut = D metres the episode also ends once the mean row is more than D behind the bar's pace
    line 0.8 m/s * t: the noise-free mean row paces the episode, whose length then tracks how long
    the current mean policy keeps up with the bar.
  * nu2 / alpha2: step size from the update after the mean row first passes any bar, noise from the
    generation after it.
  * w_pace * max(0, v_pace * seconds upright - distance): a penalty on walking slower than v_pace while
    upright (aimed at the slow, safe gaits long episodes converge to).
  * avg_beta: the mean row (and the saved passing policy) runs an exponential moving average of the
    ARS iterates, avg <- beta * avg + (1 - beta) * theta after every update; the perturbations stay
    centred on the raw iterate.
  * v_cap: the distance term counts at most v_cap * episode seconds (no reward for averaging faster).
  * An offline oracle that Desmos does not have and that is not counted: each generation the mean
    policy is also rolled out from rest for each horizon in `oracle_s`, to see when it first could
    pass the long bar.

Seeds are batched along the population axis; each seed has its own stage and episode length.
Ticks are counted as in train.py: ticks the training population (with its mean row) is simulated for,
summed over generations. ticks_to_bar excludes the generation whose mean row passes the final bar
(train.py's convention); desmos_ticks includes it, since Desmos has to run that episode to see the pass.

Usage: python3 long_train.py config.json      (writes into runs/<name>/)
"""
import json
import os
import pickle
import sys
import time

import numpy as np

import fastsim as fs
import train

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    name="long_baseline", P=128, b=32, nu=0.12, alpha=0.15, alive_w=0.0, w_air=3.0, air_free=0.15,
    sym=False, seeds=list(range(8)), early_term=True,
    stages=[5.0, 10.0, 20.0], advance="bar", stage_gens=None, pass_streak=1, final_streak=1, demote=False,
    nu_stages=None, alpha_stages=None, b_stages=None,
    w_lean=0.0, w_asym=0.0, w_late=0.0, w_alt=0.0, alt_cap=40,
    oracle_s=[20.0], post_pass_gens=0, budget_ticks=150_000, ckpt_every=5, chain_stages=None,
    cut_stages=None, pace_cut=None, nu2=None, alpha2=None, w_pace=0.0, v_pace=0.9, avg_beta=None, v_cap=None,
)

PHASES_PER_5S = train.BAR_PHASES


def required_phases(seconds):
    """3 alternating single-support phases per foot per 5 s of horizon."""
    return int(round(PHASES_PER_5S * seconds / 5.0))


def bar(dist, T, ever_fallen, n_left, n_right, air_ticks, DT):
    """The strict bar over a horizon of T ticks, phases scaled with the horizon (elementwise)."""
    secs = T * DT
    req = np.round(PHASES_PER_5S * secs / 5.0)
    return ((dist / secs >= train.BAR_SPEED) & (~ever_fallen) & (n_left >= req) & (n_right >= req)
            & (air_ticks / T < train.BAR_AIR))


def rollout(W, ph, end, grp, sym, early_term, start=None, t0=None, cut=None, pace_cut=None):
    """Roll out R rows; row r runs end[r] ticks.

    grp[r] >= 0 groups rows for early termination (all rows of a group stop once every one of them
    has fallen; and, with cut (R,) bool, as soon as any cut row of the group falls, or, with pace_cut
    = D metres, falls more than D behind the bar's pace line BAR_SPEED * t); grp[r] = -1
    rows always run their full length. start = (pos, prev), each (M, 2, R),
    replaces the rest pose; t0 (R,) offsets the clock input in seconds. Returns per-row sums, `ticks`
    (the number of ticks each row was actually simulated for) and each row's last state (fin_pos,
    fin_prev, fin_t).
    """
    R = W.shape[0]
    end = np.asarray(end, int).copy()
    pos, prev = fs.initial_state(R) if start is None else (start[0].copy(), start[1].copy())
    fin_pos = np.zeros_like(pos); fin_prev = np.zeros_like(prev)
    x0 = pos[fs.HIP, 0].copy()
    DT = ph.DT
    alive = np.ones(R)
    dist, alive_t, air_t, air_ticks, lean1, lean2, s_l, s_r, dist_half = np.zeros((9, R))
    ever = np.zeros(R, bool)
    half = end // 2
    trk = train.PhaseTracker(R)
    gm = grp >= 0
    n_grp = int(grp.max()) + 1 if gm.any() else 0
    rows = np.arange(R)
    Wk = W
    for k in range(int(end.max())):
        if (end[rows] <= k).any():
            keep = end[rows] > k
            gone = rows[~keep]
            fin_pos[:, :, gone] = pos[:, :, ~keep]; fin_prev[:, :, gone] = prev[:, :, ~keep]
            rows = rows[keep]; pos = pos[:, :, keep]; prev = prev[:, :, keep]; Wk = W[rows]
            if rows.size == 0:
                break
        a = alive[rows]
        obs = fs.observe(pos, prev, k * DT if t0 is None else t0[rows] + k * DT, ph)
        pos, prev = fs.step(pos, prev, fs.targets(train.policy_act(Wk, obs, sym)), a, ph)
        cl = pos[fs.FL, 1] < 1e-3; cr = pos[fs.FR, 1] < 1e-3
        down = fs.fallen(pos) & (a > 0)
        dist[rows] = np.where(a > 0, pos[fs.HIP, 0] - x0[rows], dist[rows])
        a = np.where(down, 0.0, a)
        alive[rows] = a
        ever[rows] |= down
        alive_t[rows] += a * DT
        both_up = ~cl & ~cr
        air_t[rows] += a * DT * both_up
        air_ticks[rows] += both_up
        th = np.arctan2(pos[fs.HEAD, 0] - pos[fs.HIP, 0], pos[fs.HEAD, 1] - pos[fs.HIP, 1])
        lean1[rows] += a * DT * th
        lean2[rows] += a * DT * th * th
        s_l[rows] += a * DT * (cl & ~cr)
        s_r[rows] += a * DT * (cr & ~cl)
        at_half = half[rows] == k + 1
        if at_half.any():
            dist_half[rows[at_half]] = dist[rows[at_half]]
        cl_f = np.zeros(R, bool); cr_f = np.zeros(R, bool); act_f = np.zeros(R, bool)
        cl_f[rows] = cl; cr_f[rows] = cr; act_f[rows] = a > 0
        trk.update(cl_f, cr_f, act_f)
        if early_term and n_grp:
            alive_cnt = np.bincount(grp[gm], weights=alive[gm], minlength=n_grp)
            gend = np.zeros(n_grp, int)
            np.maximum.at(gend, grp[gm], end[gm])
            dead = alive_cnt == 0
            if cut is not None and cut.any():
                out_ = alive == 0
                if pace_cut is not None:
                    out_ = out_ | (dist < train.BAR_SPEED * (k + 1) * DT - pace_cut)
                cm = gm & cut & out_
                dead = dead | (np.bincount(grp[cm], minlength=n_grp) > 0)
            newly = dead & (gend > k + 1)
            if newly.any():
                hit = gm & np.isin(grp, np.nonzero(newly)[0])
                end[hit] = k + 1
    trk.finish()
    fin_pos[:, :, rows] = pos; fin_prev[:, :, rows] = prev
    short = end < half                        # cut by early termination before the middle: all fallen
    dist_half[short] = dist[short]
    fin_t = end * DT + (0.0 if t0 is None else t0)
    return dict(dist=dist, alive_t=alive_t, air_t=air_t, air_ticks=air_ticks, fallen=ever,
                n_left=trk.cnt[0], n_right=trk.cnt[1], alts=trk.cnt.sum(0), lean_mean=lean1 / np.maximum(alive_t, 1e-9),
                lean2=lean2, s_l=s_l, s_r=s_r, dist_half=dist_half, ticks=end,
                fin_pos=fin_pos, fin_prev=fin_prev, fin_t=fin_t)


def limp(out):
    """Stance asymmetry: |left - right single-support time| / (left + right)."""
    return np.abs(out["s_l"] - out["s_r"]) / np.maximum(out["s_l"] + out["s_r"], 1e-9)


def fitness(out, cfg):
    air_pen = np.maximum(0.0, out["air_t"] - cfg["air_free"] * out["alive_t"])
    return (out["dist"] + cfg["alive_w"] * out["alive_t"] - cfg["w_air"] * air_pen
            - cfg["w_lean"] * out["lean2"] - cfg["w_asym"] * np.abs(out["s_l"] - out["s_r"])
            + cfg["w_late"] * (out["dist"] - out["dist_half"])
            + cfg["w_alt"] * np.minimum(out["alts"], cfg["alt_cap"])
            - cfg["w_pace"] * np.maximum(0.0, cfg["v_pace"] * out["alive_t"] - out["dist"]))


def per_stage(cfg, key, stage):
    lst = cfg.get(f"{key}_stages")
    return lst[min(stage, len(lst) - 1)] if lst else cfg[key]


def evaluate(theta, ph, sym, seconds):
    """Noise-free rollouts from rest of a batch of policies; bar with phases scaled to the horizon."""
    T = int(round(seconds / ph.DT))
    S = theta.shape[0]
    out = rollout(theta, ph, np.full(S, T), np.full(S, -1), sym, False)
    out["passed"] = bar(out["dist"], T, out["fallen"], out["n_left"], out["n_right"], out["air_ticks"], ph.DT)
    out["limp"] = limp(out)
    return out


def train_long(cfg):
    cfg = {**DEFAULTS, **cfg}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    out_dir = os.path.join(HERE, "runs", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    ph = fs.Phys()
    seeds = cfg["seeds"]; S = len(seeds)
    P = cfg["P"]; N = P // 2; sym = cfg["sym"]
    stages = [float(s) for s in cfg["stages"]]; last = len(stages) - 1
    if P % 2 or any(not 1 <= b <= N for b in (cfg["b_stages"] or [cfg["b"]])):
        raise ValueError(f"need even P and 1 <= b <= P/2, got P={P}")
    if cfg["advance"] not in ("bar", "gens"):
        raise ValueError(f"advance must be 'bar' or 'gens', got {cfg['advance']!r}")
    if cfg["advance"] == "gens" and len(cfg["stage_gens"] or []) != last:
        raise ValueError("advance='gens' needs stage_gens with one entry per non-final stage")
    A = 2 if sym else fs.N_ACT
    rngs = [np.random.default_rng(1_000_003 * s + 17) for s in seeds]
    theta = np.zeros((S, A, fs.N_OBS))
    st = dict(g=0, theta=theta, hist=[[] for _ in range(S)], cum=np.zeros(S), stage=np.zeros(S, int),
              streak=np.zeros(S, int), first_final=[None] * S, ticks_to_bar=[None] * S, desmos_ticks=[None] * S,
              oracle_first={h: [None] * S for h in cfg["oracle_s"]}, done=np.zeros(S, bool),
              theta_pass=np.zeros_like(theta), final_theta=np.zeros_like(theta), gens_in_stage=np.zeros(S, int),
              carry=[None] * S, ever=np.zeros(S, bool), theta_avg=np.zeros_like(theta))
    ckpt_path = os.path.join(out_dir, "ckpt.pkl")
    if os.path.exists(ckpt_path):            # resume an interrupted run (file written by this script)
        with open(ckpt_path, "rb") as fh:
            ck = pickle.load(fh)
        if ck["cfg"] != cfg:
            raise SystemExit(f"{ckpt_path} was written for a different config; move it away to restart")
        st = ck["st"]
        for r, s in zip(rngs, ck["rng"]):
            r.bit_generator.state = s
    theta = st["theta"]
    log = open(os.path.join(out_dir, "log.txt"), "a")
    if st["g"]:
        print(f"resumed at gen {st['g']}", file=log, flush=True)
    t_start = time.time()
    T_or = [int(round(h / ph.DT)) for h in cfg["oracle_s"]]
    while not st["done"].all():
        g = st["g"]
        act = np.nonzero(~st["done"])[0]
        n = len(act)
        T_s = np.array([int(round(stages[st["stage"][i]] / ph.DT)) for i in act])
        deltas = np.stack([rngs[i].standard_normal((N, A, fs.N_OBS)) for i in act])     # (n, N, A, O)
        nus = np.array([cfg["nu2"] if st["ever"][i] and cfg["nu2"] is not None else per_stage(cfg, "nu", st["stage"][i])
                        for i in act])
        Wp = np.concatenate([theta[act, None] + nus[:, None, None, None] * deltas,
                             theta[act, None] - nus[:, None, None, None] * deltas], 1)   # (n, P, A, O)
        mean_pol = st["theta_avg"] if cfg["avg_beta"] is not None else theta    # the policy the mean row runs
        W = np.concatenate([Wp, mean_pol[act, None]], 1).reshape(n * (P + 1), A, fs.N_OBS)
        end = np.repeat(T_s, P + 1)
        grp = np.repeat(np.arange(n), P + 1)
        for T in T_or:                        # oracle rows: mean policy from rest, not counted
            W = np.concatenate([W, mean_pol[act]]); end = np.concatenate([end, np.full(n, T)])
            grp = np.concatenate([grp, np.full(n, -1)])
        chained = np.array([bool(cfg["chain_stages"] and cfg["chain_stages"][st["stage"][i]]
                                 and st["carry"][i] is not None) for i in act], bool)
        start = t0 = None
        if chained.any():                     # chained episodes start from the last mean-row state
            pos0, prev0 = fs.initial_state(W.shape[0]); t0 = np.zeros(W.shape[0])
            for jj in np.nonzero(chained)[0]:
                cp, cv, ct = st["carry"][act[jj]]
                sl = slice(jj * (P + 1), (jj + 1) * (P + 1))
                pos0[:, :, sl] = cp[:, :, None]; prev0[:, :, sl] = cv[:, :, None]; t0[sl] = ct
            start = (pos0, prev0)
        cut = None
        if cfg["cut_stages"]:                 # flagged stages end when the mean row falls
            cut = np.zeros(W.shape[0], bool)
            for jj, i in enumerate(act):
                cut[jj * (P + 1) + P] = bool(cfg["cut_stages"][st["stage"][i]])
        out = rollout(W, ph, end, grp, sym, cfg["early_term"], start, t0, cut, cfg["pace_cut"])
        nb = n * (P + 1)
        fit = fitness(out, cfg)[:nb]
        if cfg["v_cap"] is not None:          # distance counts only up to v_cap * episode seconds
            d = out["dist"][:nb]
            fit = fit - d + np.minimum(d, cfg["v_cap"] * np.repeat(T_s, P + 1) * ph.DT)
        fit = fit.reshape(n, P + 1)[:, :P]
        mrow = np.arange(n) * (P + 1) + P
        m_pass = bar(out["dist"][mrow], T_s, out["fallen"][mrow], out["n_left"][mrow], out["n_right"][mrow],
                     out["air_ticks"][mrow], ph.DT)
        ticks_gen = out["ticks"][mrow]
        lp = limp(out)
        or_pass = {}
        for j, (h, T) in enumerate(zip(cfg["oracle_s"], T_or)):
            sl = slice(nb + j * n, nb + (j + 1) * n)
            or_pass[h] = bar(out["dist"][sl], T, out["fallen"][sl], out["n_left"][sl], out["n_right"][sl],
                             out["air_ticks"][sl], ph.DT)
        for jj, i in enumerate(act):
            mi = mrow[jj]
            stage = int(st["stage"][i])
            rec = dict(g=g, stage=stage, T=int(T_s[jj]), chained=bool(chained[jj]), ticks_before=float(st["cum"][i]),
                       ticks_gen=int(ticks_gen[jj]),
                       nu=float(nus[jj]), dist=float(out["dist"][mi]), fallen=bool(out["fallen"][mi]),
                       alive_t=float(out["alive_t"][mi]), nL=int(out["n_left"][mi]), nR=int(out["n_right"][mi]),
                       air=float(out["air_ticks"][mi] / T_s[jj]), limp=float(lp[mi]), lean=float(out["lean_mean"][mi]),
                       passed=bool(m_pass[jj]), pop_best=float(out["dist"][jj * (P + 1):jj * (P + 1) + P].max()))
            for j, h in enumerate(cfg["oracle_s"]):
                oi = nb + j * n + jj
                rec[f"or{h:g}"] = bool(or_pass[h][jj])
                rec[f"or{h:g}_dist"] = float(out["dist"][oi])
                rec[f"or{h:g}_alive"] = float(out["alive_t"][oi])
                if or_pass[h][jj] and st["oracle_first"][h][i] is None:
                    st["oracle_first"][h][i] = dict(g=g, ticks_before=float(st["cum"][i]))
                    np.save(os.path.join(out_dir, f"theta_oracle{h:g}_s{seeds[i]}.npy"), mean_pol[i])
            st["hist"][i].append(rec)
            if not m_pass[jj]:                 # carry the mean row's last state (shifted to x = 0) only
                st["carry"][i] = None          # when it passed the bar, i.e. it ends mid-walk
            else:
                cp = out["fin_pos"][:, :, mi].copy(); cv = out["fin_prev"][:, :, mi].copy()
                x = cp[fs.HIP, 0]; cp[:, 0] -= x; cv[:, 0] -= x
                st["carry"][i] = (cp, cv, float(np.mod(out["fin_t"][mi], 1.0 / fs.CLOCK_HZ)))
            # curriculum logic (all of it is a few scalar variables per run in Desmos)
            st["ever"][i] |= bool(m_pass[jj])
            st["streak"][i] = st["streak"][i] + 1 if m_pass[jj] else 0
            st["gens_in_stage"][i] += 1
            if stage == last:
                if st["streak"][i] >= cfg["final_streak"] and st["first_final"][i] is None:
                    st["first_final"][i] = g
                    st["ticks_to_bar"][i] = float(st["cum"][i])
                    st["desmos_ticks"][i] = float(st["cum"][i] + ticks_gen[jj])
                    st["theta_pass"][i] = mean_pol[i]
                    np.save(os.path.join(out_dir, f"theta_pass_s{seeds[i]}.npy"), mean_pol[i])
                elif cfg["demote"] and not m_pass[jj] and st["first_final"][i] is None and last > 0:
                    st["stage"][i] -= 1; st["streak"][i] = 0; st["gens_in_stage"][i] = 0
            elif cfg["advance"] == "bar":
                if st["streak"][i] >= cfg["pass_streak"]:
                    st["stage"][i] += 1; st["streak"][i] = 0; st["gens_in_stage"][i] = 0
                elif cfg["demote"] and stage > 0 and not m_pass[jj]:
                    st["stage"][i] -= 1; st["streak"][i] = 0; st["gens_in_stage"][i] = 0
            elif st["gens_in_stage"][i] >= cfg["stage_gens"][stage]:
                st["stage"][i] += 1; st["streak"][i] = 0; st["gens_in_stage"][i] = 0
        for jj, i in enumerate(act):          # ARS V1-t update
            b = int(per_stage(cfg, "b", int(st["hist"][i][-1]["stage"])))
            alpha = per_stage(cfg, "alpha", int(st["hist"][i][-1]["stage"]))
            if st["ever"][i] and cfg["alpha2"] is not None:
                alpha = cfg["alpha2"]
            rp, rm = fit[jj, :N], fit[jj, N:]
            top = np.argsort(-np.maximum(rp, rm))[:b]
            sigma = np.concatenate([rp[top], rm[top]]).std() + 1e-8
            theta[i] += alpha / (b * sigma) * np.einsum('n,nao->ao', rp[top] - rm[top], deltas[jj, top])
            if cfg["avg_beta"] is not None:
                st["theta_avg"][i] = cfg["avg_beta"] * st["theta_avg"][i] + (1 - cfg["avg_beta"]) * theta[i]
            st["cum"][i] += ticks_gen[jj]
        if g % 5 == 0:
            print(f"gen {g:4d} ticks~{int(np.median(st['cum'])):7d} stage " + "".join(str(s) for s in st["stage"])
                  + "  dist " + " ".join(f"{h[-1]['dist']:5.2f}" for h in st["hist"])
                  + "  pass " + "".join("P" if h[-1]["passed"] else "." for h in st["hist"])
                  + "  final " + "".join("F" if f is not None else "." for f in st["first_final"])
                  + f"  [{time.time() - t_start:6.0f}s]", file=log, flush=True)
        st["g"] = g + 1
        for i in act:
            fp = st["first_final"][i]
            if (fp is not None and st["g"] - fp > cfg["post_pass_gens"]) or st["cum"][i] >= cfg["budget_ticks"]:
                st["done"][i] = True
                st["final_theta"][i] = theta[i]
        if st["g"] % cfg["ckpt_every"] == 0:
            tmp = ckpt_path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(dict(cfg=cfg, st=st, rng=[r.bit_generator.state for r in rngs]), fh)
            os.replace(tmp, ckpt_path)
    for i, s in enumerate(seeds):
        np.save(os.path.join(out_dir, f"theta_final_s{s}.npy"), st["final_theta"][i])
        json.dump(st["hist"][i], open(os.path.join(out_dir, f"hist_s{s}.json"), "w"))
    passed_idx = [i for i in range(S) if st["first_final"][i] is not None]
    ev = {}
    if passed_idx:
        for h in (20.0, 60.0):
            e = evaluate(st["theta_pass"][passed_idx], ph, sym, h)
            ev[h] = dict(passed=[bool(x) for x in e["passed"]], dist=[float(x) for x in e["dist"]],
                         alive_t=[float(x) for x in e["alive_t"]], limp=[float(x) for x in e["limp"]],
                         lean=[float(x) for x in e["lean_mean"]])
    summary = dict(name=cfg["name"], seeds=seeds, first_final=st["first_final"], ticks_to_bar=st["ticks_to_bar"],
                   desmos_ticks=st["desmos_ticks"], gens=[len(h) for h in st["hist"]],
                   oracle_first={f"{h:g}": v for h, v in st["oracle_first"].items()},
                   passed_seeds=[seeds[i] for i in passed_idx], eval_pass={f"{h:g}": v for h, v in ev.items()},
                   wall_s=time.time() - t_start)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"done {cfg['name']} in {time.time() - t_start:.0f}s first_final={st['first_final']}", file=log, flush=True)
    return summary


if __name__ == "__main__":
    s = train_long(json.load(open(sys.argv[1])))
    print(json.dumps({k: s[k] for k in ("name", "first_final", "ticks_to_bar", "desmos_ticks")}))
