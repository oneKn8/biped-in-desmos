"""Check the Desmos learning graph against the numpy reference (proto/ars.py, proto/walkbar.py).

Data flows numpy -> page easily (file upload) but page -> here only in small pieces, so:

  prepare  Writes a test graph whose noise list D is numpy's own noise (a literal instead of Desmos's
           random()), with the ticker stopped after one episode, plus the numpy results for that same
           episode: fitness, distance, airborne time, alive flags, the mean row's bar result and phase
           counts, and the updated mean parameters. The page then compares and reports max differences.
  history  Re-scores mean parameters read from a full Desmos run (52 numbers for each chosen generation)
           with the numpy rollout and the strict bar, and compares with the graph's own distances.

Usage: python3 check_desmos.py prepare THETA.npy|zero SEED graph.json expected.json [--dirs 64 --top 32]
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
    ap.add_argument("mode", choices=["prepare", "history"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--dirs", type=int, default=64)
    ap.add_argument("--top", type=int, default=32)
    a = ap.parse_args()
    if a.mode == "prepare":
        th, seed, g_out, e_out = a.args
        prepare(th, int(seed), g_out, e_out, a.dirs, a.top)
    else:
        history(a.args[0])
