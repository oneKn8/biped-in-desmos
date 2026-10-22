"""Emit a Desmos graph state in which a population of ragdoll bipeds learns to walk by ARS.

Mirrors proto/sim.py, proto/ars.py and proto/walkbar.py term for term, so they can be checked against
each other. Layout: positions of all individuals live in one point list S of length 6*n_p, mass-major
(all heads, then all hips, ...). Within each mass block the last individual is the noise-free mean
policy; the graph scores it against the strict walking bar every episode and stops learning at its
first pass, then replays that walk. Learning state is the mean parameter list w_0 plus a generation
counter g; per-individual parameters are regenerated from seeded noise every generation.

Usage: python3 gen_desmos.py out_state.json [--theta proto/theta_s0.npy --nu 0] [--dirs 64 --top 32]
                             [--steps 6] [--random-seed HEX32] [--paused]
"""
import argparse
import json
import re
import sys

import numpy as np

sys.path.insert(0, 'proto')
import sim  # noqa: E402  sim lives in proto/, which the line above puts on the path
import walkbar  # noqa: E402  imported after the sys.path insert above, which adds proto/

NPAR = sim.N_ACT * sim.N_OBS
N_LO = 7                    # weights for observations 0..6 live in W_{q1}, 7..12 in W_{q2} (list cap, see build)
LIST_CAP = 10000            # Desmos list length limit
# Desmos seeds every random() from the state's randomSeed and invents a new one on each load when it
# is missing, so it is pinned here. With this seed and the default recipe the graph learns to walk
# at generation 16 (seen 2026-10-22).
DEFAULT_RANDOM_SEED = "437b79efcc42e4f3ec1675a0e0615340"


def fx(v):
    """Shortest round-trip decimal, never exponent notation (Desmos must parse it to the same double)."""
    return np.format_float_positional(float(v), unique=True, trim='-')


def build(theta=None, nu=0.12, alpha=0.15, n_dir=64, top=32, episode=300, playing=True, steps_per_tick=6,
          min_step_ms=50, random_seed=DEFAULT_RANDOM_SEED):
    """Physics, policy, learning rule and walking bar, laid out for Desmos speed.

    Measured cost model (NOTES.md, 2026-10-21/22): every statement downstream of the action-written
    state is re-analysed by the worker on every tick and also costs main-thread progress messages;
    the action's write-back of the state literals is re-parsed every tick; after that, cost is
    roughly per list operation, so it grows with population. So: one vectorised formula per family
    (all bones at once, all motors at once, all four actions at once) gathered through static index
    lists, the four Jacobi passes nested as function calls (arguments are bound once, not
    substituted), several physics steps per ticker action (steps_per_tick) so the write-back is paid
    once per tick, learning statistics moved off the per-tick path (a snapshot tick feeds them), the
    mean policy's phase counter nested inside the tick action instead of separate statements, every
    folder collapsed, and constants written at full double precision with the same operation order
    as proto/sim.py. Collapse the whole expression list while it runs: a visible list re-typesets.
    """
    n_p = 2 * n_dir + 1                    # antithetic pairs, then the noise-free mean policy as the last row
    sizes = {"D": NPAR * n_dir, "W_q1": sim.N_ACT * N_LO * n_p, "W_q2": sim.N_ACT * (sim.N_OBS - N_LO) * n_p,
             "S": sim.M * n_p, "K": (2 * len(sim.BONES) + 3 * len(sim.MOTORS)) * n_p + 1}
    over = {k: v for k, v in sizes.items() if v > LIST_CAP}
    if over:
        raise ValueError(f"n_dir={n_dir} puts lists over the {LIST_CAP} cap: {over}")
    if not re.fullmatch(r"[0-9a-f]{32}", random_seed):
        raise ValueError(f"random_seed must be 32 lowercase hex digits, got {random_seed!r}")
    if not 1 <= top <= n_dir:
        raise ValueError(f"need 1 <= top <= n_dir, got top={top} n_dir={n_dir}")
    items, fid = [], [0]
    draw_id = [None]

    def folder(title):
        fid[0] += 1
        items.append({"type": "folder", "id": f"f{fid[0]}", "title": title, "collapsed": True})
        return f"f{fid[0]}"

    def ex(latex, fold=None, **style):
        e = {"type": "expression", "id": f"e{len(items)}", "latex": latex, "color": style.pop("color", "#2d70b3")}
        if fold:
            e["folderId"] = fold
        # Desmos plots any one-variable function definition, so hide everything outside Drawing
        style.setdefault("hidden", fold != draw_id[0])
        e.update({k: (v if isinstance(v, (bool, dict)) else str(v)) for k, v in style.items()})
        items.append(e)

    def blocks(ms):
        """Index list selecting mass blocks ms (in order) of a mass-major list."""
        return r"\operatorname{join}\left(" + ",".join(rf"\left[1...n_{{p}}\right]+{m}n_{{p}}" for m in ms) + r"\right)"

    def consts(vals):
        return r"\operatorname{join}\left(" + ",".join(rf"0\cdot\left[1...n_{{p}}\right]+{fx(v)}" for v in vals) + r"\right)"

    DT = fx(sim.DT)
    horizon = fx(episode * sim.DT)
    items.append({"type": "text", "id": "t0", "text": "A population of ragdoll bipeds learns to walk. Physics, neural policy, the "
                  "learning rule and the walking test are all Desmos expressions. Press play on the ticker. Grey: the "
                  "population, each with its own noise. Orange: the current mean policy. Learning stops by itself the first "
                  "time the mean policy passes the walking bar, then the graph replays that walk. g = generation, k = step in the episode. "
                  "Tip: collapse this list (the arrows at its top right) before pressing play; it runs faster."})

    # ---------------- learning (changes once per generation) ----------------
    F = folder("Learning (ARS: mean parameters, seeded noise, top-b update; stops when the mean policy walks)")
    ex(rf"n_{{d}}={n_dir}", F); ex(r"n_{p}=2n_{d}+1", F); ex(rf"b_{{t}}={top}", F)
    ex(rf"T_{{e}}={episode}", F); ex(rf"\nu={fx(nu)}", F); ex(rf"\alpha={fx(alpha)}", F)
    ex(r"g=0", F); ex(r"k=0", F)
    ex(r"P_{s}=0", F)                       # 1 when the last episode's mean policy passed the bar
    ex(r"P_{w}=0", F)                       # 1 once it has passed: learning is over
    w0 = np.zeros(NPAR) if theta is None else np.asarray(theta, float).reshape(-1)
    ex(r"w_{0}=\left[" + ",".join(fx(v) for v in w0) + r"\right]", F)
    ex(r"U_{1}=\operatorname{random}\left(" + str(NPAR) + r"n_{d},2g+1\right)", F)
    ex(r"U_{2}=\operatorname{random}\left(" + str(NPAR) + r"n_{d},2g+2\right)", F)
    ex(r"D=\sqrt{-2\ln\left(1-U_{1}\right)}\cos\left(2\pi U_{2}\right)", F)
    # per individual: noise sign (+1 for the first n_d, -1 for their mirrors, 0 for the mean row) and noise index
    ex(r"s_{v}=\operatorname{join}\left(1+0\cdot\left[1...n_{d}\right],-1+0\cdot\left[1...n_{d}\right],\left[0\right]\right)", F)
    ex(r"i_{m}=\operatorname{join}\left(\left[1...n_{d}\right],\left[1...n_{d}\right],\left[1\right]\right)", F)
    # parameter p (0-based) of every individual; once the mean policy has walked, every row is the mean
    ex(r"v\left(p\right)=w_{0}\left[p+1\right]+\nu\left(1-P_{w}\right)s_{v}D\left[n_{d}p+i_{m}\right]", F)
    # policy weights regrouped once per generation so one tick evaluates all 4 actions as 4n_p-long lists;
    # 52 blocks of n_p would pass the list cap at n_p = 193, so the blocks are split over two lists
    for name, obs_range in (("W_{q1}", range(N_LO)), ("W_{q2}", range(N_LO, sim.N_OBS))):
        ex(name + r"=\operatorname{join}\left(" + ",".join(
            rf"v\left({a * sim.N_OBS + o}\right)" for o in obs_range for a in range(sim.N_ACT)) + r"\right)", F)
    # F_s is a fitness snapshot written once per episode, so none of this re-runs per tick
    ex(r"F_{s}=0\cdot\left[1...n_{p}\right]", F)
    ex(r"r_{p}=F_{s}\left[1...n_{d}\right]", F)
    ex(r"r_{m}=F_{s}\left[n_{d}+1...2n_{d}\right]", F)
    ex(r"O_{t}=\operatorname{sort}\left(\left[1...n_{d}\right],-\max\left(r_{p},r_{m}\right)\right)\left[1...b_{t}\right]", F)
    ex(r"\sigma_{r}=\operatorname{stdevp}\left(\operatorname{join}\left(r_{p}\left[O_{t}\right],r_{m}\left[O_{t}\right]\right)\right)+10^{-8}", F)
    ex(r"w_{n}=\left[w_{0}\left[p\right]+\frac{\alpha}{b_{t}\sigma_{r}}\operatorname{total}\left(\left(r_{p}\left[O_{t}\right]-r_{m}\left[O_{t}\right]\right)"
       r"D\left[n_{d}\left(p-1\right)+O_{t}\right]\right)\operatorname{for}p=\left[1..." + str(NPAR) + r"\right]\right]", F)
    ex(r"F_{h}=\left[\right]", F)            # mean-policy distance, one entry per generation

    # ---------------- state ----------------
    S0 = folder("Bodies (state)")
    pos, _ = sim.initial_state(1)
    init = [pos[0, m] for m in range(sim.M)]
    ex(r"h\left(L,a\right)=L\left[an_{p}+1...an_{p}+n_{p}\right]", S0)
    ex(r"S_{0}=\operatorname{join}\left(" + ",".join(
        rf"\left(0\cdot\left[1...n_{{p}}\right]+{fx(p[0])},0\cdot\left[1...n_{{p}}\right]+{fx(p[1])}\right)" for p in init) + r"\right)", S0)
    lit = r"\left[" + ",".join(rf"\left({fx(p[0])},{fx(p[1])}\right)" for p in init for _ in range(n_p)) + r"\right]"
    ex(r"S=" + lit, S0)
    ex(r"Q=" + lit, S0)
    ex(r"A=\left[" + ",".join(["1"] * n_p) + r"\right]", S0)
    zeros = r"\left[" + ",".join(["0"] * n_p) + r"\right]"
    ex(r"F_{d}=" + zeros, S0)                # hip distance while upright
    ex(r"F_{a}=" + zeros, S0)                # seconds upright
    ex(r"F_{air}=" + zeros, S0)              # seconds upright with both feet off the ground
    # mean policy's phase counter (walkbar.PhaseTracker): [side, run length, last counted side, n_left, n_right]
    z0 = r"\left[-1,0,-1,0,0\right]"
    ex(r"Z=" + z0, S0)

    # ---------------- static topology: gather lists ----------------
    G = folder("Body topology (static index lists)")
    ex(r"B_{a}=" + blocks([a for a, b, _ in sim.BONES]), G)
    ex(r"B_{b}=" + blocks([b for a, b, _ in sim.BONES]), G)
    ex(r"L_{b}=" + consts([L for _, _, L in sim.BONES]), G)
    ex(r"M_{f}=" + blocks([fa for fa, j, c in sim.MOTORS]), G)
    ex(r"M_{j}=" + blocks([j for fa, j, c in sim.MOTORS]), G)
    ex(r"M_{c}=" + blocks([c for fa, j, c in sim.MOTORS]), G)
    # contribution list K = [c (5 bones), -c (5), dc (4 motors), df (4), -(dc+df) (4), zero]; for each mass the
    # contributions are listed in exactly the order proto/sim.py accumulates them (bones, then motors: child,
    # far, joint), padded with the zero element, so the floating-point sums happen in the same order
    nb, nm_ = len(sim.BONES), len(sim.MOTORS)
    contrib = [[] for _ in range(sim.M)]
    for bi, (a, b, _) in enumerate(sim.BONES):
        contrib[a].append(bi); contrib[b].append(nb + bi)
    for mi, (fa, j, c) in enumerate(sim.MOTORS):
        contrib[c].append(2 * nb + mi); contrib[fa].append(2 * nb + nm_ + mi); contrib[j].append(2 * nb + 2 * nm_ + mi)
    zero = 2 * nb + 3 * nm_
    depth = max(len(cl) for cl in contrib)
    for r in range(depth):
        parts = []
        for m in range(sim.M):
            parts.append(rf"\left[1...n_{{p}}\right]+{contrib[m][r]}n_{{p}}" if r < len(contrib[m])
                         else rf"0\cdot\left[1...n_{{p}}\right]+{zero}n_{{p}}+1")
        ex(rf"G_{{{r + 1}}}=\operatorname{{join}}\left(" + ",".join(parts) + r"\right)", G)
    ex(r"N_{c}=" + consts([max(len(cl), 1) for cl in contrib]), G)

    # ---------------- functions (no state dependence, analysed once) ----------------
    Fn = folder("Physics, policy and walking-bar functions")
    ex(r"w_{r}\left(a\right)=\operatorname{mod}\left(a+\pi,2\pi\right)-\pi", Fn)
    ex(r"l_{a}\left(V\right)=\arctan\left(V.y,V.x\right)", Fn)
    ex(r"j_{a}\left(P,J,C\right)=w_{r}\left(l_{a}\left(C-J\right)-l_{a}\left(J-P\right)\right)", Fn)
    ex(r"t_{a}\left(P\right)=\arctan\left(h\left(P,0\right).x-h\left(P,1\right).x,h\left(P,0\right).y-h\left(P,1\right).y\right)", Fn)
    ex(r"n_{w}\left(V\right)=\sqrt{V.xV.x+V.yV.y}+0.000000001", Fn)
    ex(r"c_{z}\left(V,n,L\right)=\frac{0.5\left(n-L\right)}{n}V", Fn)
    ex(r"c_{w}\left(V,L\right)=c_{z}\left(V,n_{w}\left(V\right),L\right)", Fn)
    ex(rf"e_{{r}}\left(P,J,C,q\right)=\min\left(\max\left(w_{{r}}\left(q-j_{{a}}\left(P,J,C\right)\right)\cdot{fx(sim.MOTOR_K)},-{fx(sim.MAX_ROT)}\right),{fx(sim.MAX_ROT)}\right)", Fn)
    ex(r"r_{t}\left(V,a\right)=\left(\cos\left(a\right)V.x-\sin\left(a\right)V.y,\sin\left(a\right)V.x+\cos\left(a\right)V.y\right)-V", Fn)
    # both motor rotations (child by +u/2, parent-far by -u/2) as one rotation over a joined list; R = [dc, df]
    ex(r"k_{v}\left(C,R\right)=\operatorname{join}\left(C,-C,R,-\left(R\left[1...4n_{p}\right]+R\left[4n_{p}+1...8n_{p}\right]\right),\left[\left(0,0\right)\right]\right)", Fn)
    ex(r"k_{m}\left(C,P,J,X,u\right)=k_{v}\left(C,r_{t}\left(\operatorname{join}\left(X-J,P-J\right),\operatorname{join}\left(0.5u,-0.5u\right)\right)\right)", Fn)
    ex(r"k_{n}\left(C,P,J,X,q\right)=k_{m}\left(C,P,J,X,e_{r}\left(P,J,X,q\right)\right)", Fn)
    ex(r"g_{r}\left(V\right)=\left(V.x,\max\left(V.y,0\right)\right)", Fn)
    gsum = "+".join(rf"K\left[G_{{{r + 1}}}\right]" for r in range(depth))
    ex(rf"p_{{n}}\left(P,K\right)=g_{{r}}\left(P+\frac{{{gsum}}}{{N_{{c}}}}\right)", Fn)
    ex(r"p_{s}\left(P,q\right)=p_{n}\left(P,k_{n}\left(c_{w}\left(P\left[B_{b}\right]-P\left[B_{a}\right],L_{b}\right),P\left[M_{f}\right],P\left[M_{j}\right],P\left[M_{c}\right],q\right)\right)", Fn)
    ex(r"d_{f}\left(P\right)=\left\{h\left(P,1\right).y<0.45:1,h\left(P,0\right).y<h\left(P,1\right).y+0.25:1,0\right\}", Fn)
    ex(r"u_{4}\left(L\right)=\operatorname{join}\left(L,L,L,L\right)", Fn)
    ex(r"w_{q}\left(o\right)=W_{q1}\left[4on_{p}+1...4on_{p}+4n_{p}\right]", Fn)
    ex(r"w_{s}\left(o\right)=W_{q2}\left[4on_{p}+1...4on_{p}+4n_{p}\right]", Fn)
    # action a -> target: hips 0.9*a, knees -0.75*(a+1), written as m_1*(a+m_0) (exact: x+0 = x)
    ex(r"m_{1}=" + consts([0.9, -0.75, 0.9, -0.75]), Fn)
    ex(r"m_{0}=" + consts([0, 1, 0, 1]), Fn)
    # fitness pieces: 1 while both feet are off the ground; the airborne penalty above the free fraction
    ex(r"b_{u}\left(P\right)=\left\{h\left(P,3\right).y<0.001:0,h\left(P,5\right).y<0.001:0,1\right\}", Fn)
    ex(rf"p_{{a}}\left(u,l\right)=\max\left(0,u-{fx(sim.AIR_FREE)}l\right)", Fn)
    # walking bar on the mean row (walkbar.py): single-support side of one body (-1 none, 0 left, 1 right) ...
    ex(r"c_{s}\left(l,u,a\right)=\left\{a=0:-1,l<0.001:\left\{u<0.001:-1,0\right\},u<0.001:1,-1\right\}", Fn)
    # ... and one PhaseTracker.update on L = [side, run length, last counted side, n_left, n_right]: the run that
    # just ended counts (i_c = 1) if it was single support, long enough, and not the same foot as the last one
    ex(rf"i_{{c}}\left(L,x\right)=\left\{{x=L\left[1\right]:0,L\left[1\right]<0:0,L\left[2\right]<{walkbar.BAR_MIN_TICKS}:0,L\left[1\right]=L\left[3\right]:0,1\right\}}", Fn)
    ex(r"z_{v}\left(L,x,u\right)=\left[x,\left\{x=L\left[1\right]:L\left[2\right]+1,1\right\},\left\{u=1:L\left[1\right],L\left[3\right]\right\},"
       r"L\left[4\right]+u\cdot\left\{L\left[1\right]=0:1,0\right\},L\left[5\right]+u\cdot\left\{L\left[1\right]=1:1,0\right\}\right]", Fn)
    ex(r"z_{u}\left(L,x\right)=z_{v}\left(L,x,i_{c}\left(L,x\right)\right)", Fn)
    # pass = fast enough, never fell, enough phases per foot, both feet up for under BAR_AIR of the steps
    air_steps = fx(walkbar.BAR_AIR * episode - 0.5)
    ex(rf"p_{{c}}\left(q,a,L,s\right)=\left\{{\frac{{q}}{{{horizon}}}\ge{fx(walkbar.BAR_SPEED)}:\left\{{a=1:\left\{{L\left[4\right]\ge{walkbar.BAR_PHASES}:"
       rf"\left\{{L\left[5\right]\ge{walkbar.BAR_PHASES}:\left\{{\frac{{s}}{{{DT}}}<{air_steps}:1,0\right\}},0\right\}},0\right\}},0\right\}},0\right\}}", Fn)

    # ---------------- per tick: policy and physics, unrolled n_s steps ----------------
    # one ticker action advances n_s physics steps; the state is written back once per tick, and that
    # write-back (re-parsing the state literals) is the largest per-tick cost left after fusion
    if steps_per_tick < 1 or episode % steps_per_tick:
        raise ValueError(f"steps_per_tick={steps_per_tick} must be >= 1 and divide episode={episode}")
    Y = folder(f"One tick = {steps_per_tick} physics step(s): senses, brain, Verlet, 4 Jacobi passes, friction")

    def nm(base, s):                       # s = 0 keeps the plain names; later steps get a digit suffix
        return base if s == 0 else base[:-1] + str(s) + "}"
    P_, R_, al_ = "S", "Q", "A"            # positions, previous positions, alive flags entering step s
    alive_after = []
    for s in range(steps_per_tick):
        Ob, q, A6, Sv, Sk, Sf = (nm(b, s) for b in ("O_{b}", "q_{c}", "A_{6}", "S_{v}", "S_{k}", "S_{f}"))
        kk = r"k" if s == 0 else rf"\left(k+{s}\right)"
        obs = [rf"t_{{a}}\left({P_}\right)",
               rf"0.1\frac{{w_{{r}}\left(t_{{a}}\left({P_}\right)-t_{{a}}\left({R_}\right)\right)}}{{{DT}}}"]
        obs += [rf"j_{{a}}\left({P_}\left[M_{{f}}\right],{P_}\left[M_{{j}}\right],{P_}\left[M_{{c}}\right]\right)"]   # 4 blocks at once
        obs += [rf"h\left({P_},1\right).y-0.85",
                rf"0.5\frac{{h\left({P_},1\right).x-h\left({R_},1\right).x}}{{{DT}}}",
                rf"\left\{{h\left({P_},3\right).y<0.001:1,0\right\}}", rf"\left\{{h\left({P_},5\right).y<0.001:1,0\right\}}",
                rf"\sin\left(2\pi\cdot{fx(sim.CLOCK_HZ)}\cdot\left({kk}\cdot{DT}\right)\right)+0\cdot\left[1...n_{{p}}\right]",
                rf"\cos\left(2\pi\cdot{fx(sim.CLOCK_HZ)}\cdot\left({kk}\cdot{DT}\right)\right)+0\cdot\left[1...n_{{p}}\right]"]
        assert len(obs) == sim.N_OBS - 1 - (sim.N_ACT - 1)   # 13th input is the bias; joint angles packed
        ex(Ob + r"=\operatorname{join}\left(" + ",".join(obs) + r"\right)", Y)

        def wo(o):
            return rf"w_{{q}}\left({o}\right)" if o < N_LO else rf"w_{{s}}\left({o - N_LO}\right)"
        pre = "+".join(rf"{wo(o)}u_{{4}}\left(h\left({Ob},{o}\right)\right)" for o in range(sim.N_OBS - 1))
        ex(q + rf"=m_{{1}}\left(\tanh\left({pre}+{wo(sim.N_OBS - 1)}\right)+m_{{0}}\right)", Y)
        ex(A6 + rf"=\operatorname{{join}}\left({al_},{al_},{al_},{al_},{al_},{al_}\right)", Y)
        ex(Sv + rf"=\left({P_}.x+\left({P_}.x-{R_}.x\right)\cdot{fx(sim.DAMP)}\cdot {A6},{P_}.y+\left({P_}.y-{R_}.y\right)\cdot{fx(sim.DAMP)}\cdot {A6}+{fx(sim.G * sim.DT * sim.DT)}\cdot {A6}\right)", Y)
        nested = Sv
        for _ in range(sim.ITERS):
            nested = rf"p_{{s}}\left({nested},{q}\right)"
        ex(Sk + "=" + nested, Y)
        ex(Sf + rf"=\left(\left\{{{Sk}.y\le0.000001:{P_}.x+\left({Sk}.x-{P_}.x\right)\cdot{fx(1 - sim.FRICTION)},{Sk}.x\right\}},{Sk}.y\right)", Y)
        name = f"A_{{a{s + 1}}}"
        ex(name + rf"={al_}\left(1-d_{{f}}\left({Sf}\right){al_}\right)", Y)
        alive_after.append((al_, Sf, name))
        P_, R_, al_ = Sf, P_, name

    # ---------------- actions ----------------
    T = folder("Clock (tick = physics steps; then a fitness snapshot tick and a learning tick)")
    fd, fa, fair, z = r"F_{d}", r"F_{a}", r"F_{air}", "Z"
    for al_in, Sf, al_out in alive_after:   # same update order as proto/ars.py, one step at a time
        fd = rf"\left\{{{al_in}>0:h\left({Sf},1\right).x,{fd}\right\}}"
        fa = rf"{fa}+{DT}\cdot {al_out}"
        fair = rf"{fair}+{DT}\cdot {al_out}\cdot b_{{u}}\left({Sf}\right)"
        z = (rf"z_{{u}}\left({z},c_{{s}}\left({Sf}\left[4n_{{p}}\right].y,{Sf}\left[6n_{{p}}\right].y,"
             rf"{al_out}\left[n_{{p}}\right]\right)\right)")
    ex(rf"A_{{t}}=S\to {P_},Q\to {R_},F_{{d}}\to {fd},A\to {al_},F_{{a}}\to {fa},F_{{air}}\to {fair},Z\to {z},k\to k+{steps_per_tick}", T)
    fit = r"F_{d}"
    if sim.ALIVE_W:
        fit += rf"+{fx(sim.ALIVE_W)}F_{{a}}"
    if sim.W_AIR:
        fit += rf"-{fx(sim.W_AIR)}p_{{a}}\left(F_{{air}},F_{{a}}\right)"
    # snapshot: fitness for the update, and the mean row's bar test (its phase counter closed as in finish())
    ex(rf"A_{{s}}=F_{{s}}\to {fit},P_{{s}}\to p_{{c}}\left(F_{{d}}\left[n_{{p}}\right],A\left[n_{{p}}\right],"
       rf"z_{{u}}\left(Z,-1\right),F_{{air}}\left[n_{{p}}\right]\right),k\to T_{{e}}+1", T)
    zl = r"0\cdot\left[1...n_{p}\right]"
    reset = (rf"k\to0,S\to S_{{0}},Q\to S_{{0}},A\to1+{zl},F_{{d}}\to{zl},F_{{a}}\to{zl},F_{{air}}\to{zl},Z\to" + z0)
    log = r"F_{h}\to\operatorname{join}\left(F_{h},\left[F_{d}\left[n_{p}\right]\right]\right)"
    ex(r"A_{g}=w_{0}\to w_{n},g\to g+1," + log + "," + reset, T)      # learn
    ex(r"A_{f}=P_{w}\to1," + log + "," + reset, T)                     # first pass: keep w_0, stop learning
    ex(r"A_{r}=" + reset, T)                                           # after that: replay the walk
    # an episode ends at T_e steps, or early once every body has fallen (k < T_e * 0 is false)
    ex(r"A_{c}=\left\{k<T_{e}\operatorname{sign}\left(\operatorname{total}\left(A\right)\right):A_{t},k\le T_{e}:A_{s},"
       r"P_{w}=1:A_{r},P_{s}=1:A_{f},A_{g}\right\}", T)

    # ---------------- drawing ----------------
    V = folder("Drawing")
    draw_id[0] = V
    ex(r"S\left[B_{a}\right]+t\left(S\left[B_{b}\right]-S\left[B_{a}\right]\right)", V, color="#c8c2b3", lineWidth=2, lineOpacity=0.35)
    ia = ",".join(str(a) for a, b, _ in sim.BONES)
    ib = ",".join(str(b) for a, b, _ in sim.BONES)
    pa = rf"S\left[\left[{ia}\right]n_{{p}}+n_{{p}}\right]"
    pb = rf"S\left[\left[{ib}\right]n_{{p}}+n_{{p}}\right]"
    ex(rf"{pa}+t\left({pb}-{pa}\right)", V, color="#e8a33c", lineWidth=5)
    ex(r"y=0", V, color="#555555", lineWidth=1.5)
    # learning curve, top left: mean-policy distance per generation against the bar distance
    cx0, cdx, cy0, cs = -0.8, 0.08, 2.0, 0.2
    bar_y = cy0 + cs * walkbar.BAR_SPEED * episode * sim.DT
    ex(rf"\left({fx(cx0)}+{fx(cdx)}\cdot\left[1...\operatorname{{length}}\left(F_{{h}}\right)\right],{fx(cy0)}+{fx(cs)}F_{{h}}\right)", V,
       color="#6a9fb5", pointSize=5, lines=True)
    ex(rf"y={fx(bar_y)}\left\{{{fx(cx0)}\le x\le{fx(cx0 + 4)}\right\}}", V, color="#6a9fb5", lineWidth=1, lineStyle="DASHED")
    ex(rf"\left({fx(cx0)},{fx(cy0 + cs * 7.25)}\right)", V, color="#333333", showLabel=True, pointSize=0,
       label=f"mean policy: metres walked in {fx(episode * sim.DT)} s, per generation (dashed: the bar)")
    ex(r"\left(3.6,3.45\right)", V, color="#333333", showLabel=True, pointSize=0, label=r"generation ${g}   step ${k}")
    ex(r"\left(3.6,\left\{P_{w}=1:3.1\right\}\right)", V, color="#e8a33c", showLabel=True, pointSize=0,
       label=r"walks: passed the bar at generation ${g}, learning stopped")

    return {
        "version": 11,
        "randomSeed": random_seed,
        "graph": {"viewport": {"xmin": -1, "xmax": 7, "ymin": -0.6, "ymax": 3.6},
                  "showGrid": False, "showXAxis": False, "showYAxis": False},
        # minStep stops the ticker from dispatching (and re-rendering the UI) on every frame while the
        # worker is still busy; 50 ms is far below the measured action time, so it never limits throughput
        "expressions": {"list": items, "ticker": {"handlerLatex": "A_{c}", "minStepLatex": str(min_step_ms), "open": True, "playing": playing}},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--theta", help="start from these mean parameters (an .npy from proto/ars.py)")
    ap.add_argument("--nu", type=float, default=0.12, help="noise; 0 replays --theta on every body")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--dirs", type=int, default=64, help="antithetic directions n_d (population 2 n_d + 1)")
    ap.add_argument("--top", type=int, default=32, help="top directions b_t used in the update")
    ap.add_argument("--paused", action="store_true")
    ap.add_argument("--steps", type=int, default=6, help="physics steps per ticker action")
    ap.add_argument("--random-seed", default=DEFAULT_RANDOM_SEED, help="Desmos randomSeed (32 hex digits); sets the whole noise sequence")
    a = ap.parse_args()
    th = np.load(a.theta) if a.theta else None
    st = build(theta=th, nu=a.nu, alpha=a.alpha, n_dir=a.dirs, top=a.top, playing=not a.paused, steps_per_tick=a.steps,
               random_seed=a.random_seed)
    json.dump(st, open(a.out, "w"), separators=(",", ":"))
    n = len(st["expressions"]["list"])
    print(f"wrote {a.out}: {n} items")
