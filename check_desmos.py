"""Check the Desmos learning graph against the numpy reference (proto/ars.py, proto/walkbar.py).

Data flows numpy -> page easily (file upload) but page -> here only in small pieces, so:

  prepare  Writes a test graph whose noise list D is numpy's own noise (a literal instead of Desmos's
           random()), with the ticker stopped after one episode, plus the numpy results for that same
           episode: fitness, distance, airborne time, alive flags, the mean row's bar result and phase
           counts, and the updated mean parameters. The page then compares and reports max differences.
  history  Re-scores mean parameters read from a full Desmos run (52 numbers for each chosen generation)
           with the numpy rollout and the strict bar, and compares with the graph's own distances.

  window / exam  The same for the long-walk graph (gen_desmos --long): "window" is a 5 s window that
           continues a walk (start pose and clock offset carried from THETA's own 5 s walk from rest; also
           checks the next carry the graph computes), "exam" is a 20 s exam from rest that ends when the
           mean row falls. gwindow / gexam: the same for the gait graph (gen_desmos --gait), past the first pass.

Usage: python3 check_desmos.py prepare THETA.npy|zero SEED graph.json expected.json [--dirs 64 --top 32]
       python3 check_desmos.py window|exam THETA.npy SEED graph.json expected.json [--dirs 48 --top 24]
       python3 check_desmos.py history run.json
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, 'proto')
import ars  # noqa: E402  ars lives in proto/, which the line above puts on the path
import sim  # noqa: E402  imported after the sys.path insert above, which adds proto/
import gen_desmos  # noqa: E402  it imports sim from proto/, so it must follow the sys.path insert above

T = 300


def update(theta, fit, delta, N, b, alpha):
    """The ARS V1-t step exactly as in proto/ars.py main()."""
    rp, rm = fit[:N], fit[N:2 * N]
    top = np.argsort(-np.maximum(rp, rm))[:b]
    sigma = np.concatenate([rp[top], rm[top]]).std() + 1e-8
    return theta + alpha / (b * sigma) * np.einsum('n,nao->ao', rp[top] - rm[top], delta[top])


def prepare(theta_arg, seed, graph_out, expected_out, N, b, nu=0.12, alpha=0.15):
    theta = np.zeros((sim.N_ACT, sim.N_OBS)) if theta_arg == "zero" else np.load(theta_arg).reshape(sim.N_ACT, sim.N_OBS)
    delta = np.random.default_rng(seed).standard_normal((N, sim.N_ACT, sim.N_OBS))
    W = np.concatenate([theta + nu * delta, theta - nu * delta, theta[None]])
    out = ars.rollout(W, T)
    st = gen_desmos.build(theta=theta, nu=nu, alpha=alpha, n_dir=N, top=b, playing=False)
    D = delta.reshape(N, -1).T.reshape(-1)           # parameter-major, as the graph indexes it
    hits = [e for e in st["expressions"]["list"] if e.get("latex", "").startswith("D=")]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one D expression, found {len(hits)}")
    hits[0]["latex"] = "D=\\left[" + ",".join(gen_desmos.fx(v) for v in D) + "\\right]"
    st["expressions"]["ticker"]["handlerLatex"] = "\\left\\{k\\le T_{e}:A_{c}\\right\\}"   # stop before the update
    json.dump(st, open(graph_out, "w"), separators=(",", ":"))
    exp = dict(F_s=out["fit"].tolist(), F_d=out["dist"].tolist(), F_air=out["air_t"].tolist(),
               A=(~out["fallen"]).astype(float).tolist(), P_s=int(ars.passes(out, T)[-1]),
               n_left=int(out["n_left"][-1]), n_right=int(out["n_right"][-1]),
               w_n=update(theta, out["fit"], delta, N, b, alpha).reshape(-1).tolist())
    json.dump(exp, open(expected_out, "w"))
    print(f"mean row: dist {out['dist'][-1]:.4f}  phases L{exp['n_left']} R{exp['n_right']}  "
          f"air {out['air_frac'][-1]:.3f}  pass {exp['P_s']}  fallen in population {int(out['fallen'][:-1].sum())}/{2 * N}")


def literal_points(pos):
    """(R, M, 2) array -> the graph's mass-major point-list literal."""
    R = pos.shape[0]
    return "\\left[" + ",".join(rf"\left({gen_desmos.fx(pos[i, m, 0])},{gen_desmos.fx(pos[i, m, 1])}\right)"
                                 for m in range(sim.M) for i in range(R)) + "\\right]"


def set_latex(st, prefix, latex):
    hits = [e for e in st["expressions"]["list"] if e.get("latex", "").startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one expression starting with {prefix!r}, found {len(hits)}")
    hits[0]["latex"] = latex


def prepare_long(kind, theta_path, seed, graph_out, expected_out, N, b, nu1=0.12, nu2=0.08, alpha1=0.15, alpha2=0.08,
                 gait=False, w_air2=8.0, w_lean2=3.0, lam=0.05, bar_asym=0.15, bar_lean=8.0):
    """One long-walk episode in numpy with the graph's layout; the graph gets numpy's noise and start.
    gait: the episode is past the first pass, so the noise is symmetrised (the graph is given the raw noise and
    symmetrises it itself), fitness uses w_air2 / w_lean2, the update is pulled toward the mirror image, and an
    exam also needs the gait limits."""
    theta = np.load(theta_path).reshape(sim.N_ACT, sim.N_OBS)
    raw = np.random.default_rng(seed).standard_normal((N, sim.N_ACT, sim.N_OBS))
    delta = 0.5 * raw + 0.5 * ars.mirror(raw) if gait else raw
    R = 2 * N + 1
    if kind == "window":                     # continue the policy's own 5 s walk from rest; past the first pass
        T, exam, P_e, nu = 300, 0, 1, nu2
        first = ars.rollout(theta[None], 300)
        p0, q0 = ars.carry(first, 0)
        t0 = float(np.mod(300 * sim.DT + 0.0, 1.0 / sim.CLOCK_HZ))
        start = (np.repeat(p0[None], R, 0), np.repeat(q0[None], R, 0))
    else:                                    # exam from rest, before any pass
        T, exam, P_e, nu = 1200, 1, 0, nu1
        t0, start = None, None
    if gait:                                 # windows and exams in the gait phase both come after the first pass
        P_e, nu = 1, nu2
    W = np.concatenate([theta + nu * delta, theta - nu * delta, theta[None]])
    out = ars.rollout(W, T, start=start, t0=None if t0 is None else np.full(R, t0), cut=bool(exam))
    passed = int(ars.passes(out, T)[-1])
    if gait and exam:                        # long_train.gait_ok on the mean row
        limp = abs(out["s_l"][-1] - out["s_r"][-1]) / max(out["s_l"][-1] + out["s_r"][-1], 1e-9)
        lean = np.degrees(out["lean_abs"][-1] / max(out["alive_t"][-1], 1e-9))
        passed = int(passed and limp <= bar_asym and lean <= bar_lean)
    st = gen_desmos.build(theta=theta, n_dir=N, top=b, playing=False, long_walk=True, nu=nu1, alpha=alpha1, nu2=nu2, alpha2=alpha2,
                          gait=gait, w_air2=w_air2, w_lean2=w_lean2, sym_decay2=lam, bar_asym=bar_asym, bar_lean=bar_lean)
    D = (raw if gait else delta).reshape(N, -1).T.reshape(-1)
    set_latex(st, "D=", "D=\\left[" + ",".join(gen_desmos.fx(v) for v in D) + "\\right]")
    set_latex(st, "X_{e}=", f"X_{{e}}={exam}")
    set_latex(st, "P_{e}=", f"P_{{e}}={P_e}")
    if start is not None:
        set_latex(st, "S=", "S=" + literal_points(start[0]))
        set_latex(st, "Q=", "Q=" + literal_points(start[1]))
        set_latex(st, "t_{0}=", "t_{0}=" + gen_desmos.fx(t0))
    st["expressions"]["ticker"]["handlerLatex"] = "\\left\\{k\\le T_{e}:A_{c}\\right\\}"
    json.dump(st, open(graph_out, "w"), separators=(",", ":"))
    fit = out["fit"]
    if gait:
        pen = np.maximum(0.0, out["air_t"] - sim.AIR_FREE * out["alive_t"])
        fit = out["dist"] - w_air2 * pen - w_lean2 * out["lean2"]
    w_n = update(theta, fit, delta, N, b, alpha2 if (P_e or passed) else alpha1)
    exp = dict(F_s=fit.tolist(), F_d=out["dist"].tolist(), F_air=out["air_t"].tolist(),
               A=out["alive"].tolist(), P_s=passed, w_n=w_n.reshape(-1).tolist())
    if gait:                                 # the pull toward the mirror image, and the gait sums
        exp["w_u"] = (w_n - lam * 0.5 * (w_n - ars.mirror(w_n))).reshape(-1).tolist()
        exp["F_l"] = out["lean2"].tolist()
        exp["G"] = [float(out["s_l"][-1]), float(out["s_r"][-1]), float(out["lean_abs"][-1])]
    if kind == "window":                     # what the graph should carry into the next window if this one passed
        p1, q1 = ars.carry(out)
        exp["next_S"] = np.repeat(p1[None], R, 0).transpose(1, 0, 2).reshape(-1, 2).tolist()
        exp["next_Q"] = np.repeat(q1[None], R, 0).transpose(1, 0, 2).reshape(-1, 2).tolist()
        exp["next_t0"] = float(np.mod(T * sim.DT + t0, 1.0 / sim.CLOCK_HZ))
    json.dump(exp, open(expected_out, "w"))
    print(f"{kind}: mean row dist {out['dist'][-1]:.4f} m, fell {bool(out['fallen'][-1])}, phases L{int(out['n_left'][-1])} "
          f"R{int(out['n_right'][-1])}, pass {passed}; fallen in population {int(out['fallen'][:-1].sum())}/{2 * N}")


def history(path):
    """d = {"w0": {"<g>": [52 numbers], ...} for chosen generations, "F_h": [graph distance per generation], "stopped_at": g}."""
    d = json.load(open(path))
    rows = []
    for g in sorted(int(k) for k in d["w0"]):
        w = np.asarray(d["w0"][str(g)], float)
        out = ars.rollout(w.reshape(1, sim.N_ACT, sim.N_OBS), T)
        fh = d["F_h"][g]
        rows.append(dict(g=g, desmos=fh, numpy=float(out["dist"][0]), err=abs(fh - float(out["dist"][0])),
                         passed=bool(ars.passes(out, T)[0]), n_left=int(out["n_left"][0]), n_right=int(out["n_right"][0])))
        r = rows[-1]
        print(f"g {g:3d}  desmos {fh:8.4f}  numpy {r['numpy']:8.4f}  err {r['err']:.1e}  numpy pass {r['passed']} (L{r['n_left']} R{r['n_right']})")
    print(json.dumps(dict(checked=len(rows), max_dist_err=max((r["err"] for r in rows), default=None),
                          desmos_stopped_at=d.get("stopped_at"),
                          numpy_pass_at_stop=next((r["passed"] for r in rows if r["g"] == d.get("stopped_at")), None))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "window", "exam", "gwindow", "gexam", "history"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--dirs", type=int, default=64)
    ap.add_argument("--top", type=int, default=32)
    a = ap.parse_args()
    if a.mode == "prepare":
        th, seed, g_out, e_out = a.args
        prepare(th, int(seed), g_out, e_out, a.dirs, a.top)
    elif a.mode in ("window", "exam", "gwindow", "gexam"):
        th, seed, g_out, e_out = a.args
        prepare_long(a.mode.lstrip("g"), th, int(seed), g_out, e_out, a.dirs, a.top, gait=a.mode.startswith("g"))
    else:
        history(a.args[0])
