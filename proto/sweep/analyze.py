"""Aggregate runs/*/summary.json into a table (stdout and results_<tag>.txt).

Median ticks-to-bar counts a seed that never passed as +inf, so a config where more than half
the seeds fail shows "> budget". Estimated Desmos time uses tick cost = 50 ms + 25 ms per Jacobi
pass (fits the measured ~150 ms for the fused 4-pass graph; the per-pass figure is the measured
unfused cost and is an assumption for the fused graph).

Usage: python3 analyze.py [name-prefix ...]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def tick_ms(iters):
    return 50.0 + 25.0 * iters


def describe(cfg):
    bits = [f"P{cfg['P']}/b{cfg['b']}", f"nu{cfg['nu']:g}", f"a{cfg['alpha']:g}",
            f"dt1/{round(1 / cfg['dt'])}", f"it{cfg['iters']}", f"ep{cfg['ep_s']:g}s", f"aw{cfg['alive_w']:g}"]
    for k, lab in (("w_air", "air"), ("w_up", "up"), ("w_clk", "clk"), ("w_alt", "alt")):
        if cfg.get(k):
            bits.append(f"{lab}{cfg[k]:g}")
            if k == "w_air" and cfg.get("air_free"):
                bits[-1] += f">{100 * cfg['air_free']:g}%"
    if cfg.get("early_term"):
        bits.append("ET")
    if cfg.get("sym"):
        bits.append("SYM")
    if cfg.get("ep_sched"):
        s = cfg["ep_sched"]; bits.append(f"sched{s[0]:g}-{s[1]:g}/{s[2]}g")
    return " ".join(bits)


def row(run_dir):
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    t = np.array([x if x is not None else np.inf for x in s["ticks_to_bar"]], float)
    n = len(t); npass = int(np.isfinite(t).sum())
    med = float(np.median(t))
    q1, q3 = np.percentile(t, 25), np.percentile(t, 75)
    post = [x for x in s["post_pass_rate"] if x is not None]
    eval_s = cfg.get("eval_s", 5.0)
    return dict(name=s["name"], desc=describe(cfg), n=n, npass=npass, med=med, q1=q1, q3=q3,
                budget=cfg["budget_ticks"], post=float(np.median(post)) if post else float("nan"),
                speed=float(np.median(s["final_dist"])) / eval_s,
                rob=float(np.mean(s["robust10_final"])),
                minutes=med * tick_ms(cfg["iters"]) / 60000.0 if np.isfinite(med) else float("inf"),
                gens_med=float(np.median([h for h in s["gens"]])), iters=cfg["iters"])


def fmt_ticks(v, budget):
    return f"> {budget // 1000}k" if not np.isfinite(v) else f"{v / 1000:.1f}k"


def main():
    prefixes = sys.argv[1:] or [""]
    dirs = sorted(d for d in glob.glob(os.path.join(HERE, "runs", "*"))
                  if os.path.exists(os.path.join(d, "summary.json"))
                  and any(os.path.basename(d).startswith(p) for p in prefixes))
    rows = [row(d) for d in dirs]
    rows.sort(key=lambda r: (-(r["npass"] / r["n"] >= 0.5), r["med"], -r["npass"]))
    hdr = ("| config | recipe | pass | median ticks-to-bar (IQR) | est. Desmos min | post-pass rate | final speed m/s | 10 s robust |\n"
           "|---|---|---|---|---|---|---|---|")
    lines = [hdr]
    for r in rows:
        iqr = f"{fmt_ticks(r['q1'], r['budget'])}-{fmt_ticks(r['q3'], r['budget'])}"
        mins = f"{r['minutes']:.0f}" if np.isfinite(r["minutes"]) else "-"
        lines.append(f"| {r['name']} | {r['desc']} | {r['npass']}/{r['n']} | {fmt_ticks(r['med'], r['budget'])} ({iqr}) | "
                     f"{mins} | {r['post']:.2f} | {r['speed']:.2f} | {r['rob']:.2f} |")
    text = "\n".join(lines)
    print(text)
    tag = "_".join(p for p in prefixes if p) or "all"
    open(os.path.join(HERE, f"results_{tag}.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
