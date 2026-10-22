"""Run sweep stages in parallel: one process per config (its seeds are batched inside train.py).

Processes run at nice 19 with single-threaded BLAS, so interactive work on the machine keeps
priority. A config whose runs/<name>/summary.json exists is skipped, an interrupted one resumes from its\ncheckpoint, and at most --jobs sweep trainers run machine-wide, counting other dispatchers' jobs.

Usage: python3 run_sweep.py STAGE [STAGE ...] [--jobs 8]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
S8 = list(range(8))
S16 = list(range(8, 24))


def base(**kw):
    c = dict(P=64, b=16, nu=0.04, alpha=0.03, dt=1 / 60, iters=4, ep_s=5.0, alive_w=1.0, seeds=S8)
    c.update(kw)
    return c


def stage1():
    """Cost axes without shaping: population, Jacobi passes, DT, ARS step sizes."""
    cs = [dict(name="s1_base_P32", **base(P=32, b=8))]
    for P in (64, 128, 192):
        cs.append(dict(name=f"s1_P{P}", **base(P=P, b=P // 4)))
    for it in (3, 2):
        cs.append(dict(name=f"s1_P64_it{it}", **base(iters=it)))
    for it in (4, 3, 2):
        cs.append(dict(name=f"s1_P64_dt30_it{it}", **base(dt=1 / 30, iters=it)))
    cs.append(dict(name="s1_P128_a06", **base(P=128, b=32, alpha=0.06)))
    cs.append(dict(name="s1_P128_a10", **base(P=128, b=32, alpha=0.10)))
    cs.append(dict(name="s1_P64_nu02", **base(nu=0.02)))
    cs.append(dict(name="s1_P64_nu08", **base(nu=0.08, alpha=0.06)))
    return cs


def tuned(**kw):
    """Tuned base from the first results: wider exploration and a larger step, P=128."""
    c = base(P=128, b=32, nu=0.08, alpha=0.06)
    c.update(kw)
    return c


def stage2():
    """Fitness, episode, physics and policy knobs on the tuned base."""
    cs = [dict(name="t_base", **tuned()),
          dict(name="t_P192", **tuned(P=192, b=48)),
          dict(name="t_nu12", **tuned(nu=0.12, alpha=0.09)),
          dict(name="t_a10", **tuned(alpha=0.10))]
    for aw in (0.0, 0.5, 2.0):
        cs.append(dict(name=f"t_alive{aw:g}", **tuned(alive_w=aw)))
    cs += [dict(name="t_air3", **tuned(w_air=3.0)),
           dict(name="t_clk1", **tuned(w_clk=1.0)),
           dict(name="t_up1", **tuned(w_up=1.0, alive_w=0.0)),
           dict(name="t_alt0.2", **tuned(w_alt=0.2)),
           dict(name="t_early", **tuned(early_term=True)),
           dict(name="t_ep3", **tuned(ep_s=3.0)),
           dict(name="t_ep4", **tuned(ep_s=4.0)),
           dict(name="t_sched2to5", **tuned(ep_sched=[2.0, 5.0, 60])),
           dict(name="t_it3", **tuned(iters=3)),
           dict(name="t_it2", **tuned(iters=2)),
           dict(name="t_dt30_it4", **tuned(dt=1 / 30, iters=4)),
           dict(name="t_dt30_it2", **tuned(dt=1 / 30, iters=2)),
           dict(name="t_sym", **tuned(sym=True)),
           dict(name="t_sym_P384", **tuned(sym=True, P=384, b=96))]
    return cs


def a0(**kw):
    """alive_w = 0 (pure distance) on the tuned base, plus an airborne penalty by default."""
    c = tuned(alive_w=0.0, w_air=3.0)
    c.update(kw)
    return c


def stage3():
    """Pure-distance fitness with an airborne penalty, crossed with the cost knobs."""
    return [dict(name="a0_air3", **a0()),
            dict(name="a0_air1", **a0(w_air=1.0)),
            dict(name="a0_air3_free15", **a0(air_free=0.15)),
            dict(name="a0aw025_air3", **a0(alive_w=0.25)),
            dict(name="a0_air3_P32", **a0(P=32, b=8)),
            dict(name="a0_air3_P64", **a0(P=64, b=16)),
            dict(name="a0_air3_P192", **a0(P=192, b=48)),
            dict(name="a0_air3_nu04", **a0(nu=0.04)),
            dict(name="a0_air3_early", **a0(early_term=True)),
            dict(name="a0_air3_ep3", **a0(ep_s=3.0)),
            dict(name="a0_air3_it3", **a0(iters=3)),
            dict(name="a0_air3_it2", **a0(iters=2)),
            dict(name="a0_air3_dt30_it4", **a0(dt=1 / 30, iters=4)),
            dict(name="a0_air3_dt30_it2", **a0(dt=1 / 30, iters=2)),
            dict(name="a0_air3_sym", **a0(sym=True)),
            dict(name="a0_air3_clk1", **a0(w_clk=1.0))]


def core1():
    """Baseline-step-size cost axes that the comparison table needs."""
    keep = {"s1_base_P32", "s1_P64", "s1_P128", "s1_P192", "s1_P64_it3", "s1_P64_dt30_it4", "s1_P64_dt30_it2"}
    return [c for c in stage1() if c["name"] in keep]


def rerun2():
    """Tuned-base arms that were running under the previous dispatcher (resume if they died)."""
    keep = {"t_a10", "t_air3", "t_alive2", "t_base", "t_clk1", "t_nu12", "t_P192", "t_sym_P384"}
    return [c for c in stage2() if c["name"] in keep]


def f15(**kw):
    """Best base so far: pure distance + airborne penalty above 15% of upright time."""
    c = tuned(alive_w=0.0, w_air=3.0, air_free=0.15)
    c.update(kw)
    return c


def stage4():
    """Cost and shaping knobs on the thresholded-airborne base."""
    return [dict(name="f_free10", **f15(air_free=0.10)),
            dict(name="f_free20", **f15(air_free=0.20)),
            dict(name="f_w1.5", **f15(w_air=1.5)),
            dict(name="f_w6", **f15(w_air=6.0)),
            dict(name="f_P32", **f15(P=32, b=8)),
            dict(name="f_P64", **f15(P=64, b=16)),
            dict(name="f_P192", **f15(P=192, b=48)),
            dict(name="f_it3", **f15(iters=3)),
            dict(name="f_it2", **f15(iters=2)),
            dict(name="f_dt30_it4", **f15(dt=1 / 30, iters=4)),
            dict(name="f_dt30_it2", **f15(dt=1 / 30, iters=2)),
            dict(name="f_ep3", **f15(ep_s=3.0)),
            dict(name="f_ep4", **f15(ep_s=4.0)),
            dict(name="f_early", **f15(early_term=True)),
            dict(name="f_sym", **f15(sym=True)),
            dict(name="f_nu04", **f15(nu=0.04)),
            dict(name="f_a10", **f15(alpha=0.10)),
            dict(name="f_aw025", **f15(alive_w=0.25))]


def rerun3():
    keep = {"a0_air3", "a0_air1"}
    return [c for c in stage3() if c["name"] in keep] + rerun2()


def stage5():
    """Sustained gait: longer training episodes on the thresholded base (bar still scored at 5 s)."""
    return [dict(name="f_ep8", **f15(ep_s=8.0)),
            dict(name="f_ep10", **f15(ep_s=10.0)),
            dict(name="f_sched5to10", **f15(ep_sched=[5.0, 10.0, 60]))]


def confirm1():
    """Front-runners on 16 fresh seeds (8..23)."""
    return [dict(name="c_f15", **f15(seeds=S16)),
            dict(name="c_free20", **f15(air_free=0.20, seeds=S16))]


def rec(**kw):
    """Candidate recipe: thresholded-airborne base with alpha 0.10 and early termination."""
    c = f15(alpha=0.10, early_term=True)
    c.update(kw)
    return c


def confirm2():
    return [dict(name="c_rec", **rec(seeds=S16))]


def stage6():
    """Quick variants around the candidate recipe."""
    return [dict(name="r_a15", **rec(alpha=0.15)),
            dict(name="r_nu12_a15", **rec(nu=0.12, alpha=0.15)),
            dict(name="r_P192", **rec(P=192, b=48)),
            dict(name="r_P192_a15", **rec(P=192, b=48, alpha=0.15)),
            dict(name="r_b16", **rec(b=16)),
            dict(name="r_ep4", **rec(ep_s=4.0))]


def stage5r():
    """Sustained gait on the candidate recipe."""
    return [dict(name="r_ep8", **rec(ep_s=8.0)),
            dict(name="r_ep10", **rec(ep_s=10.0)),
            dict(name="r_sched5to10", **rec(ep_sched=[5.0, 10.0, 60]))]


def stage7():
    """Push population to the list cap and grow the step with it."""
    return [dict(name="c_P192_a15", **rec(P=192, b=48, alpha=0.15, seeds=S16)),
            dict(name="q_P192_a20", **rec(P=192, b=48, alpha=0.20)),
            dict(name="q_P192_a30", **rec(P=192, b=48, alpha=0.30)),
            dict(name="q_P192_a15_nu12", **rec(P=192, b=48, alpha=0.15, nu=0.12)),
            dict(name="q_P192_a15_nu05", **rec(P=192, b=48, alpha=0.15, nu=0.05)),
            dict(name="q_sym_P384_a15", **rec(sym=True, P=384, b=96, alpha=0.15)),
            dict(name="q_sym_P384_a25", **rec(sym=True, P=384, b=96, alpha=0.25))]


def stage8():
    """Sustained-gait schedule on the fastest recipe, and its 16-seed confirmation."""
    fast = dict(P=192, b=48, alpha=0.15)
    return [dict(name="q_P192_a15_sched", **rec(ep_sched=[5.0, 10.0, 60], **fast)),
            dict(name="c_P192_a15_sched", **rec(ep_sched=[5.0, 10.0, 60], seeds=S16, **fast))]


def confirm3():
    return [dict(name="c_sym_P384_a15", **rec(sym=True, P=384, b=96, alpha=0.15, seeds=S16))]


STAGES = {"stage1": stage1, "stage2": stage2, "stage3": stage3, "core1": core1, "rerun2": rerun2, "stage4": stage4, "rerun3": rerun3, "stage5": stage5, "confirm1": confirm1, "confirm2": confirm2, "stage6": stage6, "stage5r": stage5r, "stage7": stage7, "stage8": stage8, "confirm3": confirm3}


def _cmdlines():
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open(f"/proc/{p}/cmdline", "rb") as fh:
                    out.append(fh.read().replace(b"\0", b" ").decode(errors="replace"))
            except OSError:
                continue
    return out


def running_trainers():
    return sum(1 for c in _cmdlines() if os.path.join(HERE, "train.py") in c)


def is_running(name):
    tag = os.path.join(HERE, "runs", name, "config_in.json")
    return any(tag in c for c in _cmdlines())


GATE = threading.Lock()


def run_one(cfg, jobs_env, jobs):
    """Launch one config once fewer than `jobs` sweep trainers run machine-wide (any dispatcher)."""
    out_dir = os.path.join(HERE, "runs", cfg["name"])
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        return cfg["name"], "skipped", 0.0
    with GATE:
        while running_trainers() >= jobs:
            time.sleep(5)
        if is_running(cfg["name"]):
            return cfg["name"], "running-elsewhere", 0.0
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "config_in.json")
        json.dump(cfg, open(path, "w"), indent=1)
        t = time.time()
        fo = open(os.path.join(out_dir, "stdout.txt"), "a")
        proc = subprocess.Popen(["nice", "-n", "19", sys.executable, os.path.join(HERE, "train.py"), path],
                                stdout=fo, stderr=subprocess.STDOUT, env=jobs_env, cwd=HERE)
        time.sleep(2)                       # let the child exec so the next count sees it
    rc = proc.wait()
    fo.close()
    return cfg["name"], f"rc={rc}", time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    # stages run in the order given; inside a stage the biggest jobs start first
    def cost(c):
        return -c.get("P", 64) * c.get("ep_s", 5.0) / c.get("dt", 1 / 60)
    cfgs = [c for s in a.stages for c in sorted(STAGES[s](), key=cost)]
    names = [c["name"] for c in cfgs]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate config names")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with ThreadPoolExecutor(a.jobs) as ex:
        for name, status, wall in ex.map(lambda c: run_one(c, env, a.jobs), cfgs):
            print(f"{name:28s} {status:8s} {wall:7.0f}s", flush=True)


if __name__ == "__main__":
    main()
