"""Emit a Desmos graph state in which a population of ragdoll bipeds learns to walk by ARS.

Mirrors proto/sim.py and proto/ars.py term for term, so the two can be checked against each
other. Layout: positions of all individuals live in one point list S of length 6*n_p, mass-major
(all heads, then all hips, ...). Learning state is the mean parameter list w_0 plus a generation
counter g; per-individual parameters are regenerated from seeded noise every generation.

Usage: python3 gen_desmos.py out_state.json [--theta proto/theta_s0.npy --nu 0] [--steps 6] [--legacy]
The default fast layout runs 6 physics steps per ticker action; --legacy emits the original layout.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, 'proto')
import sim  # noqa: E402  sim lives in proto/, which the line above puts on the path

NPAR = sim.N_ACT * sim.N_OBS


def f(v):
    s = f"{float(v):.10g}"
    return s if 'e' not in s else f"{float(v):.12f}".rstrip('0').rstrip('.')


def build(theta=None, nu=0.04, alpha=0.03, n_dir=16, top=8, episode=300, playing=True, fused=True):
    n_p = 2 * n_dir
    items, fid = [], [0]
    draw_id = [None]

    def folder(title, collapsed=True):
        fid[0] += 1
        items.append({"type": "folder", "id": f"f{fid[0]}", "title": title, "collapsed": collapsed})
        return f"f{fid[0]}"

    def ex(latex, fold=None, **style):
        e = {"type": "expression", "id": f"e{len(items)}", "latex": latex, "color": style.pop("color", "#2d70b3")}
        if fold:
            e["folderId"] = fold
        # Desmos plots any one-variable function definition, so hide everything outside Drawing
        style.setdefault("hidden", fold != draw_id[0])
        e.update({k: (v if isinstance(v, (bool, dict)) else str(v)) for k, v in style.items()})
        items.append(e)

    def note(text):
        items.append({"type": "text", "id": f"t{len(items)}", "text": text})

    def h(lst, m):                        # slice mass m (0..5) of a mass-major list
        return rf"h\left({lst},{m}\right)"

    note("A population of ragdoll bipeds learns to walk. Physics, neural policy and the learning rule "
         "are all Desmos expressions. Press play on the ticker. g = generation, k = tick in the episode.")

    # ---------------- constants and learning state ----------------
    F = folder("Learning (ARS: mean parameters, seeded noise, top-b update)", collapsed=False)
    ex(rf"n_{{d}}={n_dir}", F); ex(r"n_{p}=2n_{d}", F); ex(rf"b_{{t}}={top}", F)
    ex(rf"T_{{e}}={episode}", F); ex(rf"\nu={f(nu)}", F); ex(rf"\alpha={f(alpha)}", F)
    ex(r"g=0", F); ex(r"k=0", F)
    w0 = np.zeros(NPAR) if theta is None else np.asarray(theta, float).reshape(-1)
    ex(r"w_{0}=\left[" + ",".join(f(v) for v in w0) + r"\right]", F)
    # Box-Muller standard normals, seeded by generation: n_d directions x 52 params, param-major
    ex(r"U_{1}=\operatorname{random}\left(" + str(NPAR) + r"n_{d},2g+1\right)", F)
    ex(r"U_{2}=\operatorname{random}\left(" + str(NPAR) + r"n_{d},2g+2\right)", F)
    ex(r"D=\sqrt{-2\ln\left(1-U_{1}\right)}\cos\left(2\pi U_{2}\right)", F)
    ex(r"p_{f}\left(j\right)=\operatorname{floor}\left(\frac{j-1}{n_{p}}\right)", F)
    ex(r"i_{f}\left(j\right)=j-n_{p}p_{f}\left(j\right)", F)
    ex(r"s_{g}\left(i\right)=\left\{i\le n_{d}:1,-1\right\}", F)
    # per-individual parameters, param-major: element (p, i) at index p*n_p + i
    ex(r"W=\left[w_{0}\left[p_{f}\left(j\right)+1\right]+\nu s_{g}\left(i_{f}\left(j\right)\right)"
       r"D\left[n_{d}p_{f}\left(j\right)+\operatorname{mod}\left(i_{f}\left(j\right)-1,n_{d}\right)+1\right]"
       r"\operatorname{for}j=\left[1...52n_{p}\right]\right]".replace("52", str(NPAR)), F)
    ex(r"v\left(p\right)=W\left[pn_{p}+1...pn_{p}+n_{p}\right]", F)
    # fitness and the ARS update (V1-t)
    ex(r"R_{f}=F_{d}+" + f(sim.ALIVE_W) + r"F_{a}", F)
    ex(r"r_{p}=R_{f}\left[1...n_{d}\right]", F)
    ex(r"r_{m}=R_{f}\left[n_{d}+1...n_{p}\right]", F)
    ex(r"O_{t}=\operatorname{sort}\left(\left[1...n_{d}\right],-\max\left(r_{p},r_{m}\right)\right)\left[1...b_{t}\right]", F)
    ex(r"\sigma_{r}=\operatorname{stdevp}\left(\operatorname{join}\left(r_{p}\left[O_{t}\right],r_{m}\left[O_{t}\right]\right)\right)+10^{-8}", F)
    ex(r"w_{n}=\left[w_{0}\left[p\right]+\frac{\alpha}{b_{t}\sigma_{r}}\operatorname{total}\left(\left(r_{p}\left[O_{t}\right]-r_{m}\left[O_{t}\right]\right)"
       r"D\left[n_{d}\left(p-1\right)+O_{t}\right]\right)\operatorname{for}p=\left[1..." + str(NPAR) + r"\right]\right]", F)
    ex(r"F_{h}=\left[0\right]", F)       # history of mean fitness per generation

    # ---------------- state ----------------
    S0 = folder("Bodies (state)")
    pos, _ = sim.initial_state(1)
    init = [pos[0, m] for m in range(sim.M)]
    ex(r"h\left(L,a\right)=L\left[an_{p}+1...an_{p}+n_{p}\right]", S0)
    ex(r"S_{0}=\operatorname{join}(" + ",".join(
        rf"\left(0\cdot\left[1...n_{{p}}\right]+{f(p[0])},{f(p[1])}\right)" for p in init) + ")", S0, hidden=True)
    lit = r"\left[" + ",".join(rf"\left({f(p[0])},{f(p[1])}\right)" for p in init for _ in range(n_p)) + r"\right]"
    ex(r"S=" + lit, S0, hidden=True)
    ex(r"Q=" + lit, S0, hidden=True)
    ex(r"A=\left[" + ",".join(["1"] * n_p) + r"\right]", S0)
    ex(r"F_{d}=\left[" + ",".join(["0"] * n_p) + r"\right]", S0)
    ex(r"F_{a}=\left[" + ",".join(["0"] * n_p) + r"\right]", S0)

    # ---------------- observation and policy ----------------
    P = folder("Senses and brain (linear policy with a clock)")
    for m in range(sim.M):
        ex(rf"X_{{{m}}}={h('S', m)}", P, hidden=True)
        ex(rf"Z_{{{m}}}={h('Q', m)}", P, hidden=True)
    ex(r"w_{r}\left(a\right)=\operatorname{mod}\left(a+\pi,2\pi\right)-\pi", P)
    ex(r"l_{a}\left(V\right)=\arctan\left(V.y,V.x\right)", P)
    ex(r"o_{1}=\arctan\left(X_{0}.x-X_{1}.x,X_{0}.y-X_{1}.y\right)", P)
    ex(r"o_{2}=0.1\frac{w_{r}\left(o_{1}-\arctan\left(Z_{0}.x-Z_{1}.x,Z_{0}.y-Z_{1}.y\right)\right)}{" + f(sim.DT) + "}", P)
    for i, (far, j, c) in enumerate(sim.MOTORS):
        ex(rf"o_{{{i + 3}}}=w_{{r}}\left(l_{{a}}\left(X_{{{c}}}-X_{{{j}}}\right)-l_{{a}}\left(X_{{{j}}}-X_{{{far}}}\right)\right)", P)
    ex(r"o_{7}=X_{1}.y-0.85", P)
    ex(r"o_{8}=0.5\frac{X_{1}.x-Z_{1}.x}{" + f(sim.DT) + "}", P)
    ex(r"o_{9}=\left\{X_{3}.y<0.001:1,0\right\}", P)
    ex(r"o_{10}=\left\{X_{5}.y<0.001:1,0\right\}", P)
    ex(r"o_{11}=\sin\left(2\pi\cdot" + f(sim.CLOCK_HZ * sim.DT) + r"k\right)+0\cdot\left[1...n_{p}\right]", P)
    ex(r"o_{12}=\cos\left(2\pi\cdot" + f(sim.CLOCK_HZ * sim.DT) + r"k\right)+0\cdot\left[1...n_{p}\right]", P)
    ex(r"o_{13}=1+0\cdot\left[1...n_{p}\right]", P)
    for a in range(sim.N_ACT):
        terms = "+".join(rf"v\left({a * sim.N_OBS + o}\right)o_{{{o + 1}}}" for o in range(sim.N_OBS))
        ex(rf"a_{{{a + 1}}}=\tanh\left({terms}\right)", P)
    ex(r"q_{1}=0.9a_{1}", P); ex(r"q_{2}=-0.75\left(a_{2}+1\right)", P)
    ex(r"q_{3}=0.9a_{3}", P); ex(r"q_{4}=-0.75\left(a_{4}+1\right)", P)

    # ---------------- physics ----------------
    Y = folder("Physics (Verlet, Jacobi bones and joint motors, ground, friction)")
    ex(r"R\left(V,J,a\right)=J+\left(\cos\left(a\right)\left(V.x-J.x\right)-\sin\left(a\right)\left(V.y-J.y\right),"
       r"\sin\left(a\right)\left(V.x-J.x\right)+\cos\left(a\right)\left(V.y-J.y\right)\right)", Y)
    ex(r"A_{x}=\left[A\left[i_{f}\left(j\right)\right]\operatorname{for}j=\left[1...6n_{p}\right]\right]", Y)
    ex(rf"S_{{v0}}=\left(S.x+{f(sim.DAMP)}A_{{x}}\left(S.x-Q.x\right),S.y+{f(sim.DAMP)}A_{{x}}\left(S.y-Q.y\right)+{f(sim.G * sim.DT * sim.DT)}A_{{x}}\right)", Y, hidden=True)
    cnt = np.zeros(sim.M)
    for a, b, _ in sim.BONES:
        cnt[a] += 1; cnt[b] += 1
    for far, j, c in sim.MOTORS:
        cnt[c] += 1; cnt[far] += 1; cnt[j] += 1
    if fused:
        # one expression per Jacobi pass: per-expression overhead (~1.3 ms) dominates Desmos's
        # cost, per-element math is nearly free, so duplicated work inside functions is cheaper
        ex(r"c_{b}\left(U,V,L\right)=0.5\left(1-\frac{L}{\operatorname{distance}\left(U,V\right)}\right)\left(V-U\right)", Y)
        ex(rf"z_{{e}}\left(F,J,C,q\right)=\max\left(-{f(sim.MAX_ROT)},\min\left({f(sim.MAX_ROT)},{f(sim.MOTOR_K)}w_{{r}}\left(q-w_{{r}}\left(l_{{a}}\left(C-J\right)-l_{{a}}\left(J-F\right)\right)\right)\right)\right)", Y)
        ex(r"m_{c}\left(F,J,C,q\right)=R\left(C,J,0.5z_{e}\left(F,J,C,q\right)\right)-C", Y)
        ex(r"m_{f}\left(F,J,C,q\right)=R\left(F,J,-0.5z_{e}\left(F,J,C,q\right)\right)-F", Y)
        ex(r"k_{g}\left(V\right)=\left(V.x,\max\left(V.y,0\right)\right)", Y)
        ex(r"r_{c}\left(C,J,u\right)=R\left(C,J,0.5u\right)-C", Y)
        ex(r"r_{f}\left(F,J,u\right)=R\left(F,J,-0.5u\right)-F", Y)
        for it in range(sim.ITERS):
            src = f"S_{{v{it}}}"
            B = [h(src, m) for m in range(sim.M)]
            corr = [[] for _ in range(sim.M)]
            # packed per pass: bone corrections (5 blocks), motor errors (4), motor deltas (8)
            Cn, En, Mn = f"C_{{b{it}}}", f"E_{{m{it}}}", f"M_{{d{it}}}"
            bones_l = ",".join(rf"c_{{b}}\left({B[a]},{B[b]},{f(L)}\right)" for a, b, L in sim.BONES)
            errs_l = ",".join(rf"z_{{e}}\left({B[far]},{B[j]},{B[c]},q_{{{mi + 1}}}\right)" for mi, (far, j, c) in enumerate(sim.MOTORS))
            ex(Cn + r"=\operatorname{join}\left(" + bones_l + r"\right)", Y)
            ex(En + r"=\operatorname{join}\left(" + errs_l + r"\right)", Y)
            deltas = []
            for mi, (far, j, c) in enumerate(sim.MOTORS):
                e_mi = h(En, mi)
                deltas.append(rf"r_{{c}}\left({B[c]},{B[j]},{e_mi}\right)")
                deltas.append(rf"r_{{f}}\left({B[far]},{B[j]},{e_mi}\right)")
            ex(Mn + r"=\operatorname{join}\left(" + ",".join(deltas) + r"\right)", Y)
            for bi, (a, b, L) in enumerate(sim.BONES):
                corr[a].append("+" + h(Cn, bi)); corr[b].append("-" + h(Cn, bi))
            for mi, (far, j, c) in enumerate(sim.MOTORS):
                dc, df = h(Mn, 2 * mi), h(Mn, 2 * mi + 1)
                corr[c].append("+" + dc); corr[far].append("+" + df)
                corr[j].append(f"-{dc}-{df}")
            parts = [rf"k_{{g}}\left({B[m]}+\frac{{1}}{{{int(cnt[m])}}}\left({''.join(corr[m]).lstrip('+')}\right)\right)" for m in range(sim.M)]
            nxt = f"S_{{v{it + 1}}}" if it < sim.ITERS - 1 else "S_{k}"
            ex(rf"{nxt}=\operatorname{{join}}\left({','.join(parts)}\right)", Y, hidden=True)
    for it in ([] if fused else range(sim.ITERS)):
        src = f"S_{{v{it}}}"
        B = [rf"{h(src, m)}" for m in range(sim.M)]
        corr = [[] for _ in range(sim.M)]
        for bi, (a, b, L) in enumerate(sim.BONES):
            name = f"c_{{{it}{bi}}}"
            ex(rf"{name}=0.5\left(1-\frac{{{f(L)}}}{{\operatorname{{distance}}\left({B[a]},{B[b]}\right)}}\right)\left({B[b]}-{B[a]}\right)", Y, hidden=True)
            corr[a].append(f"+{name}"); corr[b].append(f"-{name}")
        for mi, (far, j, c) in enumerate(sim.MOTORS):
            err = f"u_{{{it}{mi}}}"
            ex(rf"{err}=\max\left(-{f(sim.MAX_ROT)},\min\left({f(sim.MAX_ROT)},{f(sim.MOTOR_K)}w_{{r}}\left(q_{{{mi + 1}}}-w_{{r}}\left(l_{{a}}\left({B[c]}-{B[j]}\right)-l_{{a}}\left({B[j]}-{B[far]}\right)\right)\right)\right)\right)", Y)
            dc, df = f"m_{{c{it}{mi}}}", f"m_{{f{it}{mi}}}"
            ex(rf"{dc}=R\left({B[c]},{B[j]},0.5{err}\right)-{B[c]}", Y, hidden=True)
            ex(rf"{df}=R\left({B[far]},{B[j]},-0.5{err}\right)-{B[far]}", Y, hidden=True)
            corr[c].append(f"+{dc}"); corr[far].append(f"+{df}"); corr[j].append(f"-{dc}-{df}")
        parts = []
        for m in range(sim.M):
            num = ''.join(corr[m]).lstrip('+')
            parts.append(rf"{B[m]}+\frac{{1}}{{{int(cnt[m])}}}\left({num}\right)")
        nxt = f"S_{{v{it + 1}}}" if it < sim.ITERS - 1 else "S_{k}"
        ex(rf"G_{{{it}}}=\operatorname{{join}}\left({','.join(parts)}\right)", Y, hidden=True)
        ex(rf"{nxt}=\left(G_{{{it}}}.x,\max\left(G_{{{it}}}.y,0\right)\right)", Y, hidden=True)
    ex(rf"S_{{f}}=\left(\left\{{S_{{k}}.y\le10^{{-6}}:S.x+{f(1 - sim.FRICTION)}\left(S_{{k}}.x-S.x\right),S_{{k}}.x\right\}},S_{{k}}.y\right)", Y, hidden=True)
    # fall test and bookkeeping on the new state
    ex(r"H_{n}=h\left(S_{f},1\right)", Y, hidden=True)
    ex(r"N_{0}=h\left(S_{f},0\right)", Y, hidden=True)
    ex(r"d_{n}=\left\{H_{n}.y<0.45:1,N_{0}.y<H_{n}.y+0.25:1,0\right\}A", Y)

    # ---------------- actions ----------------
    T = folder("Clock (one tick = one physics step; episode end = learning step)")
    ex(r"A_{t}=S\to S_{f},Q\to S,F_{d}\to\left\{A>0:H_{n}.x,F_{d}\right\},A\to A\left(1-d_{n}\right),"
       r"F_{a}\to F_{a}+" + f(sim.DT) + r"A\left(1-d_{n}\right),k\to k+1", T)
    ex(r"A_{g}=w_{0}\to w_{n},g\to g+1,k\to0,S\to S_{0},Q\to S_{0},A\to1+0\cdot\left[1...n_{p}\right],"
       r"F_{d}\to0\cdot\left[1...n_{p}\right],F_{a}\to0\cdot\left[1...n_{p}\right],"
       r"F_{h}\to\operatorname{join}\left(F_{h},\left[\operatorname{mean}\left(R_{f}\right)\right]\right)", T)
    ex(r"A_{c}=\left\{k<T_{e}:A_{t},A_{g}\right\}", T)

    # ---------------- drawing ----------------
    V = folder("Drawing", collapsed=False)
    draw_id[0] = V
    ex(r"b_{e}=\operatorname{sort}\left(\left[1...n_{p}\right],-F_{d}\right)\left[1\right]", V)
    for a, b, _ in sim.BONES:
        ex(rf"{h('S', a)}+t\left({h('S', b)}-{h('S', a)}\right)", V, color="#c8c2b3", lineWidth=2, lineOpacity=0.35)
    for a, b, _ in sim.BONES:
        ex(rf"{h('S', a)}\left[b_{{e}}\right]+t\left({h('S', b)}\left[b_{{e}}\right]-{h('S', a)}\left[b_{{e}}\right]\right)", V, color="#e8a33c", lineWidth=5)
    ex(r"y=0", V, color="#555555", lineWidth=1.5)
    ex(r"\left(\left[1...\operatorname{length}\left(F_{h}\right)\right]\cdot0.04-0.5,0.25F_{h}+2.2\right)", V, color="#6a9fb5", pointSize=4)
    ex(r"\left(0,3.2\right)", V, color="#000000", hidden=False, showLabel=True, label=r"generation ${g}   tick ${k}", pointSize=0)

    state = {
        "version": 11,
        "graph": {"viewport": {"xmin": -1, "xmax": 7, "ymin": -0.6, "ymax": 3.6},
                  "showGrid": False, "showXAxis": False, "showYAxis": False},
        "expressions": {"list": items, "ticker": {"handlerLatex": "A_{c}", "minStepLatex": "0", "open": True, "playing": playing}},
    }
    return state


def fx(v):
    """Shortest round-trip decimal, never exponent notation (Desmos must parse it to the same double)."""
    return np.format_float_positional(float(v), unique=True, trim='-')


def build_fast(theta=None, nu=0.04, alpha=0.03, n_dir=16, top=8, episode=300, playing=True, steps_per_tick=6,
               min_step_ms=50):
    """Same physics, policy and learning rule as build(), restructured for Desmos speed.

    Measured cost model (NOTES.md, 2026-10-21/22): every statement downstream of the action-written
    state is re-analysed by the worker on every tick and also costs main-thread progress messages;
    the action's write-back of the S and Q literals is re-parsed every tick; after that, cost is
    roughly per list operation. So: one vectorised formula per family (all bones at once, all motors
    at once, all four actions at once) gathered through static index lists, the four Jacobi passes
    nested as function calls (arguments are bound once, not substituted), several physics steps per
    ticker action (steps_per_tick) so the write-back is paid once per tick, learning statistics moved
    off the per-tick path (a snapshot tick feeds them), two drawing statements, every folder
    collapsed so the expression list never re-typesets a changing value, and constants written at
    full double precision with the same operation order as proto/sim.py.
    """
    n_p = 2 * n_dir
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
        style.setdefault("hidden", fold != draw_id[0])
        e.update({k: (v if isinstance(v, (bool, dict)) else str(v)) for k, v in style.items()})
        items.append(e)

    def blocks(ms):
        """Index list selecting mass blocks ms (in order) of a mass-major list."""
        return r"\operatorname{join}\left(" + ",".join(rf"\left[1...n_{{p}}\right]+{m}n_{{p}}" for m in ms) + r"\right)"

    def consts(vals):
        return r"\operatorname{join}\left(" + ",".join(rf"0\cdot\left[1...n_{{p}}\right]+{fx(v)}" for v in vals) + r"\right)"

    DT = fx(sim.DT)
    items.append({"type": "text", "id": "t0", "text": "A population of ragdoll bipeds learns to walk. Physics, neural policy and the "
                  "learning rule are all Desmos expressions. Press play on the ticker. g = generation, k = tick in the episode."})

    # ---------------- learning (changes once per generation) ----------------
    F = folder("Learning (ARS: mean parameters, seeded noise, top-b update)")
    ex(rf"n_{{d}}={n_dir}", F); ex(r"n_{p}=2n_{d}", F); ex(rf"b_{{t}}={top}", F)
    ex(rf"T_{{e}}={episode}", F); ex(rf"\nu={fx(nu)}", F); ex(rf"\alpha={fx(alpha)}", F)
    ex(r"g=0", F); ex(r"k=0", F)
    w0 = np.zeros(NPAR) if theta is None else np.asarray(theta, float).reshape(-1)
    ex(r"w_{0}=\left[" + ",".join(fx(v) for v in w0) + r"\right]", F)
    ex(r"U_{1}=\operatorname{random}\left(" + str(NPAR) + r"n_{d},2g+1\right)", F)
    ex(r"U_{2}=\operatorname{random}\left(" + str(NPAR) + r"n_{d},2g+2\right)", F)
    ex(r"D=\sqrt{-2\ln\left(1-U_{1}\right)}\cos\left(2\pi U_{2}\right)", F)
    ex(r"p_{f}\left(j\right)=\operatorname{floor}\left(\frac{j-1}{n_{p}}\right)", F)
    ex(r"i_{f}\left(j\right)=j-n_{p}p_{f}\left(j\right)", F)
    ex(r"s_{g}\left(i\right)=\left\{i\le n_{d}:1,-1\right\}", F)
    ex(r"W=\left[w_{0}\left[p_{f}\left(j\right)+1\right]+\nu s_{g}\left(i_{f}\left(j\right)\right)"
       r"D\left[n_{d}p_{f}\left(j\right)+\operatorname{mod}\left(i_{f}\left(j\right)-1,n_{d}\right)+1\right]"
       r"\operatorname{for}j=\left[1..." + str(NPAR) + r"n_{p}\right]\right]", F)
    ex(r"v\left(p\right)=W\left[pn_{p}+1...pn_{p}+n_{p}\right]", F)
    # policy weights regrouped once per generation so one tick evaluates all 4 actions as 4n_p-long lists
    ex(r"W_{q}=\operatorname{join}\left(" + ",".join(
        rf"v\left({a * sim.N_OBS + o}\right)" for o in range(sim.N_OBS) for a in range(sim.N_ACT)) + r"\right)", F)
    # F_s is a fitness snapshot written once per episode, so none of this re-runs per tick
    ex(r"F_{s}=0\cdot\left[1...n_{p}\right]", F)
    ex(r"r_{p}=F_{s}\left[1...n_{d}\right]", F)
    ex(r"r_{m}=F_{s}\left[n_{d}+1...n_{p}\right]", F)
    ex(r"O_{t}=\operatorname{sort}\left(\left[1...n_{d}\right],-\max\left(r_{p},r_{m}\right)\right)\left[1...b_{t}\right]", F)
    ex(r"\sigma_{r}=\operatorname{stdevp}\left(\operatorname{join}\left(r_{p}\left[O_{t}\right],r_{m}\left[O_{t}\right]\right)\right)+10^{-8}", F)
    ex(r"w_{n}=\left[w_{0}\left[p\right]+\frac{\alpha}{b_{t}\sigma_{r}}\operatorname{total}\left(\left(r_{p}\left[O_{t}\right]-r_{m}\left[O_{t}\right]\right)"
       r"D\left[n_{d}\left(p-1\right)+O_{t}\right]\right)\operatorname{for}p=\left[1..." + str(NPAR) + r"\right]\right]", F)
    ex(r"F_{h}=\left[0\right]", F)

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
    ex(r"F_{d}=\left[" + ",".join(["0"] * n_p) + r"\right]", S0)
    ex(r"F_{a}=\left[" + ",".join(["0"] * n_p) + r"\right]", S0)

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
    nb, nm = len(sim.BONES), len(sim.MOTORS)
    contrib = [[] for _ in range(sim.M)]
    for bi, (a, b, _) in enumerate(sim.BONES):
        contrib[a].append(bi); contrib[b].append(nb + bi)
    for mi, (fa, j, c) in enumerate(sim.MOTORS):
        contrib[c].append(2 * nb + mi); contrib[fa].append(2 * nb + nm + mi); contrib[j].append(2 * nb + 2 * nm + mi)
    zero = 2 * nb + 3 * nm
    depth = max(len(cl) for cl in contrib)
    for r in range(depth):
        parts = []
        for m in range(sim.M):
            parts.append(rf"\left[1...n_{{p}}\right]+{contrib[m][r]}n_{{p}}" if r < len(contrib[m])
                         else rf"0\cdot\left[1...n_{{p}}\right]+{zero}n_{{p}}+1")
        ex(rf"G_{{{r + 1}}}=\operatorname{{join}}\left(" + ",".join(parts) + r"\right)", G)
    ex(r"N_{c}=" + consts([max(len(cl), 1) for cl in contrib]), G)

    # ---------------- functions (no state dependence, analysed once) ----------------
    Fn = folder("Physics and policy functions")
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
    ex(r"w_{q}\left(o\right)=W_{q}\left[4on_{p}+1...4on_{p}+4n_{p}\right]", Fn)
    # action a -> target: hips 0.9*a, knees -0.75*(a+1), written as m_1*(a+m_0) (exact: x+0 = x)
    ex(r"m_{1}=" + consts([0.9, -0.75, 0.9, -0.75]), Fn)
    ex(r"m_{0}=" + consts([0, 1, 0, 1]), Fn)

    # ---------------- per tick: policy and physics, unrolled n_s steps ----------------
    # one ticker action advances n_s physics steps; the state is written back once per tick, and that
    # write-back (re-parsing the S and Q literals) is the largest per-tick cost left after fusion
    if steps_per_tick < 1 or episode % steps_per_tick:
        raise ValueError(f"steps_per_tick={steps_per_tick} must be >= 1 and divide episode={episode}")
    Y = folder(f"One tick = {steps_per_tick} physics step(s): senses, brain, Verlet, 4 Jacobi passes, friction")

    def nm(base, s):                       # s = 0 keeps the plain names; later steps get a digit suffix
        return base if s == 0 else base[:-1] + str(s) + "}"
    P_, R_, al_ = "S", "Q", "A"            # positions, previous positions, alive flags entering step s
    alive_after = []
    for s in range(steps_per_tick):
        O, q, A6, Sv, Sk, Sf = (nm(b, s) for b in ("O_{b}", "q_{c}", "A_{6}", "S_{v}", "S_{k}", "S_{f}"))
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
        ex(O + r"=\operatorname{join}\left(" + ",".join(obs) + r"\right)", Y)

        pre = "+".join(rf"w_{{q}}\left({o}\right)u_{{4}}\left(h\left({O},{o}\right)\right)" for o in range(sim.N_OBS - 1))
        ex(q + rf"=m_{{1}}\left(\tanh\left({pre}+w_{{q}}\left({sim.N_OBS - 1}\right)\right)+m_{{0}}\right)", Y)
        ex(A6 + rf"=\operatorname{{join}}\left({al_},{al_},{al_},{al_},{al_},{al_}\right)", Y)
        ex(Sv + rf"=\left({P_}.x+\left({P_}.x-{R_}.x\right)\cdot{fx(sim.DAMP)}\cdot {A6},{P_}.y+\left({P_}.y-{R_}.y\right)\cdot{fx(sim.DAMP)}\cdot {A6}+{fx(sim.G * sim.DT * sim.DT)}\cdot {A6}\right)", Y)
        nested = Sv
        for _ in range(sim.ITERS):
            nested = rf"p_{{s}}\left({nested},{q}\right)"
        ex(Sk + "=" + nested, Y)
        ex(Sf + rf"=\left(\left\{{{Sk}.y\le0.000001:{P_}.x+\left({Sk}.x-{P_}.x\right)\cdot{fx(1 - sim.FRICTION)},{Sk}.x\right\}},{Sk}.y\right)", Y)
        nxt_alive = rf"{al_}\left(1-d_{{f}}\left({Sf}\right){al_}\right)"
        if s < steps_per_tick - 1:
            name = f"A_{{a{s + 1}}}"
            ex(name + "=" + nxt_alive, Y)
            alive_after.append((al_, Sf, name))
            P_, R_, al_ = Sf, P_, name
        else:
            alive_after.append((al_, Sf, nxt_alive))

    # ---------------- actions ----------------
    T = folder("Clock (tick = physics steps; then a fitness snapshot tick and a learning tick)")
    fd = r"F_{d}"
    fa_sum = r"F_{a}"
    for al_in, Sf, al_out in alive_after:   # same update order as proto/ars.py, one step at a time
        fd = rf"\left\{{{al_in}>0:h\left({Sf},1\right).x,{fd}\right\}}"
        fa_sum = rf"{fa_sum}+{DT}\cdot {al_out}" if al_out.startswith("A_") else rf"{fa_sum}+{DT}{al_out}"
    last_al_in, last_Sf, last_al_out = alive_after[-1]
    prev_pos = alive_after[-2][1] if steps_per_tick > 1 else "S"
    ex(rf"A_{{t}}=S\to {last_Sf},Q\to {prev_pos},F_{{d}}\to {fd},A\to {last_al_out},F_{{a}}\to {fa_sum},k\to k+{steps_per_tick}", T)
    ex(rf"A_{{s}}=F_{{s}}\to F_{{d}}+{fx(sim.ALIVE_W)}F_{{a}},k\to k+1", T)
    ex(r"A_{g}=w_{0}\to w_{n},g\to g+1,k\to0,S\to S_{0},Q\to S_{0},A\to1+0\cdot\left[1...n_{p}\right],"
       r"F_{d}\to0\cdot\left[1...n_{p}\right],F_{a}\to0\cdot\left[1...n_{p}\right],"
       r"F_{h}\to\operatorname{join}\left(F_{h},\left[\operatorname{mean}\left(F_{s}\right)\right]\right)", T)
    ex(r"A_{c}=\left\{k<T_{e}:A_{t},k=T_{e}:A_{s},A_{g}\right\}", T)

    # ---------------- drawing ----------------
    V = folder("Drawing")
    draw_id[0] = V
    ex(r"S\left[B_{a}\right]+t\left(S\left[B_{b}\right]-S\left[B_{a}\right]\right)", V, color="#c8c2b3", lineWidth=2, lineOpacity=0.35)
    best = r"\operatorname{sort}\left(\left[1...n_{p}\right],-F_{d}\right)\left[1\right]"
    ia = "[" + ",".join(str(a) for a, b, _ in sim.BONES) + "]"
    ib = "[" + ",".join(str(b) for a, b, _ in sim.BONES) + "]"
    ex(rf"b_{{e}}={best}", V, hidden=True)
    pa = rf"S\left[\left[{ia[1:-1]}\right]n_{{p}}+b_{{e}}\right]"
    pb = rf"S\left[\left[{ib[1:-1]}\right]n_{{p}}+b_{{e}}\right]"
    ex(rf"{pa}+t\left({pb}-{pa}\right)", V, color="#e8a33c", lineWidth=5)
    ex(r"y=0", V, color="#555555", lineWidth=1.5)
    ex(r"\left(\left[1...\operatorname{length}\left(F_{h}\right)\right]\cdot0.04-0.5,0.25F_{h}+2.2\right)", V, color="#6a9fb5", pointSize=4)
    ex(r"\left(0,3.2\right)", V, color="#000000", hidden=False, showLabel=True, label=r"generation ${g}   tick ${k}", pointSize=0)

    return {
        "version": 11,
        "graph": {"viewport": {"xmin": -1, "xmax": 7, "ymin": -0.6, "ymax": 3.6},
                  "showGrid": False, "showXAxis": False, "showYAxis": False},
        # minStep stops the ticker from dispatching (and re-rendering the UI) on every frame while the
        # worker is still busy; 50 ms is far below the measured action time, so it never limits throughput
        "expressions": {"list": items, "ticker": {"handlerLatex": "A_{c}", "minStepLatex": str(min_step_ms), "open": True, "playing": playing}},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--theta")
    ap.add_argument("--nu", type=float, default=0.04)
    ap.add_argument("--paused", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="the original one-expression-per-term layout (slow)")
    ap.add_argument("--steps", type=int, default=6, help="physics steps per ticker action (fast layout only)")
    a = ap.parse_args()
    th = np.load(a.theta) if a.theta else None
    if a.legacy:
        st = build(theta=th, nu=a.nu, playing=not a.paused)
    else:
        st = build_fast(theta=th, nu=a.nu, playing=not a.paused, steps_per_tick=a.steps)
    json.dump(st, open(a.out, "w"), separators=(",", ":"))
    n = len(st["expressions"]["list"])
    print(f"wrote {a.out}: {n} items")
