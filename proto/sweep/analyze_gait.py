"""Gait-sweep table: training cost of each config (long_train.py runs) next to the reference gait of its walkers.

Training columns come from runs/<name>/summary.json: pass = seeds whose mean row passed the 20 s exam twice in
a row within the budget; ticks = training ticks before the second passing exam (train.py's convention, a seed
that never passed counts as +inf); Desmos minutes = ticks including that exam / 22 steps per second / 60 (the
rate measured for 97 bodies in a real run; a different population would run at a different rate).
Reference columns come from runs/<name>/verify_gait.json (verify_gait.py: ../ars.py rollouts from rest, 1200 and
3600 ticks): 20 s / 60 s bar passes of the walkers, and medians of stance asymmetry, mean |torso angle|, signed
mean torso angle and torso-angle std over 20 s.

A config given as A+B pools the seeds of several runs of the same recipe.

Usage: python3 analyze_gait.py [LABEL=]CONFIG [...] [--stopped NAME ...] [--out results_gait.txt]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DESMOS_SPS = 22.0


def load(name):
    d = os.path.join(HERE, "runs", name)
    s = json.load(open(os.path.join(d, "summary.json")))
    cfg = json.load(open(os.path.join(d, "config.json")))
    vp = os.path.join(d, "verify_gait.json")
    v = json.load(open(vp)) if os.path.exists(vp) else []
    return s, cfg, v


def describe(cfg):
    """The knobs that differ from the adopted recipe; a 2 suffix means "from the first pass on"."""
    defaults = {"w_lean": 0.0, "w_abs": 0.0, "w_asym": 0.0, "anti_scale": 1.0, "sym_decay": 0.0, "w_air": 3.0,
                "air_free": 0.15, "v_cap": None, "bar_asym": None, "bar_lean": None, "nu": 0.12, "alpha": 0.15,
                "nu2": 0.08, "alpha2": 0.08, "w_lean2": None, "w_asym2": None, "w_abs2": None, "w_air2": None,
                "air_free2": None, "anti_scale2": None, "sym_decay2": None, "decay_on_pass": False,
                "decay_limp": None, "post_on": "pass"}
    labels = {"w_lean": "lean", "w_abs": "abs", "w_asym": "asym", "anti_scale": "anti", "sym_decay": "decay",
              "w_air": "air", "air_free": "airfree", "v_cap": "cap", "nu": "nu", "alpha": "a", "nu2": "nu2",
              "alpha2": "a2", "w_lean2": "lean2", "w_asym2": "asym2", "w_abs2": "abs2", "w_air2": "air2",
              "air_free2": "airfree2", "anti_scale2": "anti2", "sym_decay2": "decay2", "decay_limp": "decay-while-asym>"}
    bits = []
    for k, d in defaults.items():
        v = cfg.get(k, d)
        if v == d:
            continue
        if k == "bar_asym":
            bits.append(f"exam asym<={v:g}")
        elif k == "bar_lean":
            bits.append(f"exam torso<={v:g}deg")
        elif k == "decay_on_pass":
            bits.append("decay-on-pass")
        elif k == "post_on":
            bits.append(f"2-knobs from first {v}")
        else:
            bits.append(f"{labels[k]}={v:g}")
    if cfg.get("gait_stages") is not None and cfg.get("gait_stages") != [False, True]:
        bits.append("gait[" + ",".join("1" if x else "0" for x in cfg["gait_stages"]) + "]")
    if cfg.get("sym"):
        bits.append("SYM")
    if cfg.get("P") != 96 or cfg.get("b") != 24:
        bits.append(f"P{cfg['P']}/b{cfg['b']}")
    return " ".join(bits) or "adopted recipe"


def row(spec, label=None):
    names = spec.split("+")
    loaded = [load(n) for n in names]
    ticks, dticks, verify, seeds = [], [], [], []
    for s, cfg, v in loaded:
        ticks += [x if x is not None else np.inf for x in s["ticks_to_bar"]]
        dticks += [x if x is not None else np.inf for x in s["desmos_ticks"]]
        verify += v
        seeds += s["seeds"]
    t = np.array(ticks); dt = np.array(dticks)
    npass = int(np.isfinite(t).sum())

    def pct(x, q):                            # percentile with never-passed seeds (+inf) ranked last
        v = np.percentile(np.where(np.isfinite(x), x, 1e15), q, method="lower")
        return np.inf if v >= 1e15 else float(v)
    def med(k):
        return float(np.median([r[k] for r in verify])) if verify else np.nan
    return dict(label=label or spec, desc=describe(loaded[0][1]), n=len(t), npass=npass,
                med=np.median(t), q1=pct(t, 25), q3=pct(t, 75), worst=t.max(),
                dmin=np.median(dt) / DESMOS_SPS / 60.0, nv=len(verify),
                p20=sum(r["pass20"] for r in verify), p60=sum(r["pass60"] for r in verify),
                asym=med("asym20"), absd=med("abs_deg20"), meand=med("mean_deg20"), std=med("std_deg20"),
                asym_max=max((r["asym20"] for r in verify), default=np.nan),
                abs_max=max((r["abs_deg20"] for r in verify), default=np.nan),
                speed=med("dist20") / 20.0, air=med("air_frac20"),
                seed_range=f"{min(seeds)}-{max(seeds)}")


def fk(v):
    return "inf" if not np.isfinite(v) else f"{v / 1000:.1f}k"


def fmin(v):
    return "-" if not np.isfinite(v) else f"{v:.0f}"


def table(rows):
    head = ("| config | knobs | seeds | 20 s two-in-a-row pass | median ticks (IQR) | worst | Desmos min (median) | "
            "ref 20 s / 60 s bar | asym median (max) | mean abs torso deg median (max) | signed torso deg | "
            "torso std deg | speed m/s | airborne |")
    lines = [head, "|" + "---|" * 14]
    for r in rows:
        lines.append(f"| {r['label']} | {r['desc']} | {r['seed_range']} | {r['npass']}/{r['n']} | "
                     f"{fk(r['med'])} ({fk(r['q1'])}-{fk(r['q3'])}) | {fk(r['worst'])} | {fmin(r['dmin'])} | "
                     f"{r['p20']}/{r['nv']} , {r['p60']}/{r['nv']} | {r['asym']:.3f} ({r['asym_max']:.2f}) | "
                     f"{r['absd']:.1f} ({r['abs_max']:.1f}) | {r['meand']:+.1f} | {r['std']:.1f} | {r['speed']:.2f} | "
                     f"{r['air']:.2f} |")
    return "\n".join(lines)


def stopped(names):
    """Configs stopped before finishing (runs_stopped/<name>/ckpt.pkl, written by long_train.py): seeds through,
    median ticks spent, and the median over seeds of the mean row's gait over its last 20 training episodes."""
    import pickle
    lines = ["| config | knobs | seeds through / seeds | median ticks spent per seed | mean-row asym (last 20 episodes) | "
             "mean-row mean abs torso deg | mean-row airborne | mean-row passes of the last 20 |", "|" + "---|" * 8]
    for name in names:
        with open(os.path.join(HERE, "runs_stopped", name, "ckpt.pkl"), "rb") as fh:   # written by long_train.py
            ck = pickle.load(fh)
        st, cfg = ck["st"], ck["cfg"]
        def med(f):
            return float(np.median([np.mean([f(r) for r in h[-20:]]) for h in st["hist"]]))
        done = sum(t is not None for t in st["ticks_to_bar"])
        lines.append(f"| {name} | {describe(cfg)} | {done}/{len(st['hist'])} | {fk(float(np.median(st['cum'])))} | "
                     f"{med(lambda r: r['limp']):.2f} | "
                     f"{np.degrees(med(lambda r: r.get('lean_abs', np.nan))):.1f} | {med(lambda r: r['air']):.2f} | "
                     f"{med(lambda r: r['passed'] * 20.0):.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--stopped", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [row(*reversed(c.split("=", 1))) if "=" in c else row(c) for c in a.configs]
    text = table(rows)
    if a.stopped:
        text += "\n\nStopped before finishing (state at the last checkpoint):\n\n" + stopped(a.stopped)
    print(text)
    if a.out:
        open(os.path.join(HERE, a.out) if not os.path.isabs(a.out) else a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
