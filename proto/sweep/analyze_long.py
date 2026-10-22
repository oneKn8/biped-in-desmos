"""Aggregate long-walk runs (long_train.py) into a table: stdout and results_long.txt.

Columns: 20 s pass = seeds whose mean row passed the 20 s bar inside a 20 s training episode within
budget; ticks = training ticks to that pass (train.py's convention, excluding the passing episode), a
seed that never passed counts as +inf; Desmos min = ticks including the passing episode divided by
the measured Desmos rate for P+1 bodies; 60 s = how many of the passing walkers also pass the 60 s
bar; limp / lean = medians over the passing walkers (20 s rollout). Reference numbers (ars.py +
walkbar.py, verify_long.py) are used when RUN_DIR/verify_pass.json exists, marked "ref".
oracle = median ticks until the mean policy, rolled out from rest offline, first passes the 20 s bar
(Desmos cannot see this; it bounds what a better stopping rule could save).

Usage: python3 analyze_long.py [name-prefix ...]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# measured Desmos physics steps per second (fresh page, laptop) by number of bodies
RATE_BODIES = np.array([65, 97, 129, 193]); RATE_SPS = np.array([41.0, 30.0, 26.0, 18.0])
_MS = np.polyfit(RATE_BODIES, 1000.0 / RATE_SPS, 1)          # ms per step = a * bodies + b


def desmos_rate(bodies):
    """Steps per second: measured value when tabulated, else the linear ms-per-step fit."""
    hit = np.nonzero(RATE_BODIES == bodies)[0]
    return float(RATE_SPS[hit[0]]) if hit.size else float(1000.0 / np.polyval(_MS, bodies))


def describe(cfg):
    bits = [f"P{cfg['P']}/b{cfg['b']}", f"nu{cfg['nu']:g}", f"a{cfg['alpha']:g}",
            "stages " + "-".join(f"{s:g}" for s in cfg["stages"]) + "s"]
    if cfg["advance"] == "gens":
        bits.append("gens " + "/".join(str(x) for x in cfg["stage_gens"]))
    if cfg.get("pass_streak", 1) > 1:
        bits.append(f"streak{cfg['pass_streak']}")
    if cfg.get("final_streak", 1) > 1:
        bits.append(f"final{cfg['final_streak']}")
    if cfg.get("demote"):
        bits.append("demote")
    if cfg.get("chain_stages"):
        bits.append("chain[" + ",".join("1" if x else "0" for x in cfg["chain_stages"]) + "]")
    if cfg.get("cut_stages"):
        bits.append("cut[" + ",".join("1" if x else "0" for x in cfg["cut_stages"]) + "]")
    if cfg.get("pace_cut") is not None:
        bits.append(f"pacecut{cfg['pace_cut']:g}m")
    if cfg.get("nu2") is not None or cfg.get("alpha2") is not None:
        bits.append(f"after-pass nu{cfg.get('nu2')} a{cfg.get('alpha2')}")
    if cfg.get("w_pace"):
        bits.append(f"pace{cfg['w_pace']:g}@{cfg['v_pace']:g}")
    if cfg.get("v_cap") is not None:
        bits.append(f"cap{cfg['v_cap']:g}m/s")
    if cfg.get("avg_beta") is not None:
        bits.append(f"avg{cfg['avg_beta']:g}")
    for k in ("nu", "alpha", "b"):
        if cfg.get(f"{k}_stages"):
            bits.append(f"{k}[" + ",".join(f"{v:g}" for v in cfg[f"{k}_stages"]) + "]")
    for k, lab in (("w_lean", "lean"), ("w_asym", "asym"), ("w_late", "late"), ("w_alt", "alt")):
        if cfg.get(k):
            bits.append(f"{lab}{cfg[k]:g}")
    if cfg.get("alive_w"):
        bits.append(f"aw{cfg['alive_w']:g}")
    if not cfg.get("early_term", True):
        bits.append("noET")
    if cfg.get("sym"):
        bits.append("SYM")
    return " ".join(bits)


def fk(v):
    return "inf" if not np.isfinite(v) else f"{v / 1000:.1f}k"


def load_run(run_dir):
    """Per-seed numbers of one run; gait and long-bar numbers from verify_pass.json (reference) if present."""
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    n = len(s["seeds"])
    rate = desmos_rate(cfg["P"] + 1)
    out = dict(name=s["name"], desc=describe(cfg), n=n,
               ticks=[x if x is not None else np.inf for x in s["ticks_to_bar"]],
               minutes=[x / rate / 60.0 if x is not None else np.inf for x in s["desmos_ticks"]],
               oracle=[x["ticks_before"] if x is not None else np.inf for x in s["oracle_first"].get("20", [None] * n)])
    vpath = os.path.join(run_dir, "verify_pass.json")
    if os.path.exists(vpath):
        v = json.load(open(vpath))
        out.update(ref="ref", pass20=[r["pass20"] for r in v], pass60=[r["pass60"] for r in v],
                   limp=[r["limp"] for r in v], lean=[r["lean_deg"] for r in v],
                   jitter=[r["jitter20"] for r in v if r.get("jitter20") is not None])
    else:
        e = s.get("eval_pass", {})
        out.update(ref="fast", pass20=e.get("20", {}).get("passed", []), pass60=e.get("60", {}).get("passed", []),
                   limp=e.get("20", {}).get("limp", []), lean=list(np.degrees(e.get("20", {}).get("lean", []))), jitter=[])
    return out


def summarize(name, runs):
    """One table row from one or more runs of the same recipe (seeds pooled)."""
    def cat(k):
        return [x for r in runs for x in r[k]]
    t = np.array(cat("ticks"), float); m = np.array(cat("minutes"), float); o = np.array(cat("oracle"), float)
    npass = int(np.isfinite(t).sum())
    p20, p60, jit = cat("pass20"), cat("pass60"), cat("jitter")
    return dict(name=name, desc=runs[0]["desc"], n=len(t), npass=npass, med=np.median(t),
                q1=np.percentile(t, 25), q3=np.percentile(t, 75), p90=np.percentile(t, 90), worst=t.max(),
                dmed=np.median(m), dp90=np.percentile(m, 90), dworst=m.max(),
                n20=sum(p20), n60=sum(p60), nv=len(p20), ref="+".join(sorted({r["ref"] for r in runs})),
                limp=np.median(cat("limp")) if cat("limp") else np.nan, lean=np.median(cat("lean")) if cat("lean") else np.nan,
                jit=np.mean(jit) if jit else np.nan, oracle=np.median(o))


def fm(v):
    return f"{v:.0f}" if np.isfinite(v) else "-"


def table(rows):
    lines = ["| config | recipe | 20 s pass | median ticks to 20 s pass (IQR) | p90 / worst | Desmos min median / p90 / worst | "
             "passers re-checked: 20 s / 60 s bar | jittered 20 s pass rate | limp | lean deg | oracle median |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (-(r["npass"] / r["n"] >= 0.5), r["med"], -r["npass"])):
        jit = f"{r['jit']:.2f}" if np.isfinite(r["jit"]) else "-"
        lines.append(f"| {r['name']} | {r['desc']} | {r['npass']}/{r['n']} | {fk(r['med'])} ({fk(r['q1'])}-{fk(r['q3'])}) | "
                     f"{fk(r['p90'])} / {fk(r['worst'])} | {fm(r['dmed'])} / {fm(r['dp90'])} / {fm(r['dworst'])} | "
                     f"{r['n20']}/{r['nv']} , {r['n60']}/{r['nv']} ({r['ref']}) | {jit} | "
                     f"{r['limp']:.3f} | {r['lean']:.1f} | {fk(r['oracle'])} |")
    return "\n".join(lines)


def main():
    prefixes = sys.argv[1:] or ["L_"]
    dirs = sorted(d for d in glob.glob(os.path.join(HERE, "runs", "*"))
                  if os.path.exists(os.path.join(d, "summary.json"))
                  and any(os.path.basename(d).startswith(p) for p in prefixes)
                  and os.path.exists(os.path.join(d, "request.json")))
    runs = [load_run(d) for d in dirs]
    text = "Per run:\n\n" + table([summarize(r["name"], [r]) for r in runs])
    groups = {}
    for r in runs:
        groups.setdefault(r["desc"], []).append(r)
    pooled = [summarize(" + ".join(r["name"] for r in g), g) for g in groups.values() if len(g) > 1]
    if pooled:
        text += "\n\nSame recipe, seeds pooled:\n\n" + table(pooled)
    stopped = stopped_runs(prefixes)
    if stopped:
        text += "\n\nStopped before finishing (state at the last checkpoint):\n\n" + stopped
    print(text)
    open(os.path.join(HERE, "results_long.txt"), "w").write(text + "\n")


def stopped_runs(prefixes):
    """Runs stopped early (moved to runs_stopped/): passes so far, ticks spent, last mean-row walk."""
    import pickle
    lines = ["| config | recipe | passed so far | median ticks spent per seed | ticks to pass of the passed seeds | "
             "last mean-row episode per seed (upright s / m) |", "|---|---|---|---|---|---|"]
    rows = 0
    for d in sorted(glob.glob(os.path.join(HERE, "runs_stopped", "*"))):
        ck_path = os.path.join(d, "ckpt.pkl")
        if not (os.path.exists(ck_path) and any(os.path.basename(d).startswith(p) for p in prefixes)):
            continue
        with open(ck_path, "rb") as fh:                # checkpoint written by long_train.py
            ck = pickle.load(fh)
        st, cfg = ck["st"], ck["cfg"]
        done = [t for t in st["ticks_to_bar"] if t is not None]
        last = " ".join(f"{h[-1]['alive_t']:.0f}/{h[-1]['dist']:.0f}" for h in st["hist"])
        lines.append(f"| {cfg['name']} | {describe(cfg)} | {len(done)}/{len(st['hist'])} | {fk(float(np.median(st['cum'])))} | "
                     f"{', '.join(fk(t) for t in sorted(done)) or '-'} | {last} |")
        rows += 1
    return "\n".join(lines) if rows else ""


if __name__ == "__main__":
    main()
