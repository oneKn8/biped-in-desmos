"""Run long-walk configs (long_train.py) in parallel, one process per config, its seeds batched inside.

Trainers run at niceness 15 (absolute) with single-threaded BLAS. A config whose runs/<name>/summary.json exists is
skipped; an interrupted one resumes from its checkpoint.

Usage: python3 run_long.py GROUP [GROUP ...] [--jobs 4]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
S8 = list(range(8))
S16 = list(range(8, 24))


def c(name, **kw):
    return dict(name=name, seeds=kw.pop("seeds", S8), **kw)


CHAIN = dict(stages=[5, 20], chain_stages=[True, False], pass_streak=4, demote=True)

GROUPS = {
    # curriculum shape, base recipe (P128 b32 nu0.12 a0.15, air penalty, early termination)
    "g1": [c("L_s5_10_20", stages=[5, 10, 20]),
           c("L_direct20", stages=[20]),
           c("L_chain4", **CHAIN),
           c("L_sym_s5_10_20", stages=[5, 10, 20], sym=True)],
    # chained windows with a cheap exam (episode ends when the mean row falls)
    "g3": [c("L_chain4_cut", **CHAIN, cut_stages=[False, True]),
           c("L_chain2_cut", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True])],
    "g4": [c("L_chain1_cut", **{**CHAIN, "pass_streak": 1}, cut_stages=[False, True]),
           c("L_s5_20_pace_cut", stages=[5, 20], w_pace=0.5, v_pace=0.9, cut_stages=[False, True]),
           c("L_chain2_cut_dec", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08, alpha2=0.08),
           c("L_chain1_cut_dec", **{**CHAIN, "pass_streak": 1}, cut_stages=[False, True], nu2=0.08, alpha2=0.08)],
    "g5": [c("L_chain2_cut_c16", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], seeds=S16),
           c("L_chain2_cut_avg7", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], avg_beta=0.7),
           c("L_chain2_cut_asym", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], w_asym=0.5),
           c("L_sym_chain2_cut", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], sym=True)],
    # the mean row paces the episode: up to 20 s, ending when the mean row falls or drops 2 m behind 0.8 m/s
    "g6": [c("L_pace20_D2", stages=[20], cut_stages=[True], pace_cut=2.0),
           c("L_s5_20_cutD2", stages=[5, 20], cut_stages=[False, True], pace_cut=2.0),
           c("L_chain2_cut_avg7", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], avg_beta=0.7),
           c("L_s5_20_cut", stages=[5, 20], cut_stages=[False, True]),
           c("L_chain2_cut_asym", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], w_asym=0.5),
           c("L_sym_chain2_cut", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], sym=True)],
    "g7": [c("L_chain2_cut_dec_c16", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08, alpha2=0.08,
             seeds=S16),
           c("L_chain2_cut_dec_avg8", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08, alpha2=0.08,
             avg_beta=0.8),
           c("L_chain2_cut_b64", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], b=64),
           c("L_chain2_cut_asym", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], w_asym=0.5),
           c("L_sym_chain2_cut", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], sym=True)],
    "g8": [c("L_chain2_cut_dec_P64", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08, alpha2=0.08,
             P=64, b=16),
           c("L_chain2_cut_dec_P96", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08, alpha2=0.08,
             P=96, b=24),
           c("L_chain2_cut_asym", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], w_asym=0.5),
           c("L_sym_chain2_cut", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], sym=True)],
    "g9": [c("L_chain2_cut_dec_P96", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08, alpha2=0.08,
             P=96, b=24),
           c("L_chain2_cut_avg7_c16", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], avg_beta=0.7, seeds=S16),
           c("L_chain2_cut_asym", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], w_asym=0.5)],
    "g10": [c("L_chain2_cut_dec_P96_c16", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08, alpha2=0.08,
              P=96, b=24, seeds=S16)],
    "g11": [c("L_chain2_cut_dec_P96_f2", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08, alpha2=0.08,
              P=96, b=24, final_streak=2)],
    "g12": [c("L_chain2_cut_dec_P96_f2_c16", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08,
              alpha2=0.08, P=96, b=24, final_streak=2, seeds=S16, budget_ticks=100_000)],
    "g13": [c("L_sym_chain2_cut_dec_P96", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08,
              alpha2=0.08, P=96, b=24, sym=True)],
    "g14": [c("L_chain2_cut_dec_P96_cap10", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08,
              alpha2=0.08, P=96, b=24, v_cap=1.0),
            c("L_chain2_cut_dec_P96_cap12", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08,
              alpha2=0.08, P=96, b=24, v_cap=1.2),
            c("L_chain2_cut_dec_P96_f2_asym", **{**CHAIN, "pass_streak": 2}, cut_stages=[False, True], nu2=0.08,
              alpha2=0.08, P=96, b=24, final_streak=2, w_asym=0.5)],
    "g2": [c("L_s5_20", stages=[5, 20]),
           c("L_s5_10_20_demote", stages=[5, 10, 20], demote=True),
           c("L_gens20_15", stages=[5, 10, 20], advance="gens", stage_gens=[20, 15]),
           c("L_s10_20", stages=[10, 20])],
}


LAUNCH = threading.Condition()
NEXT = [0]                                    # index of the next config allowed to launch (keeps the order)


NICE = 15


def set_nice():
    """Run the trainer at niceness NICE exactly, whatever the launcher's own niceness is."""
    os.nice(max(0, NICE - os.nice(0)))


def running_trainers():
    """long_train.py processes running machine-wide (any dispatcher)."""
    r = subprocess.run(["pgrep", "-f", "long_train.py"], capture_output=True, text=True, timeout=10)
    return len(r.stdout.split())


def run_one(idx, cfg, env, jobs):
    out_dir = os.path.join(HERE, "runs", cfg["name"])
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        with LAUNCH:
            LAUNCH.wait_for(lambda: NEXT[0] == idx)
            NEXT[0] += 1; LAUNCH.notify_all()
        return cfg["name"], "skipped", 0.0
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "request.json")
    json.dump(cfg, open(path, "w"), indent=1)
    with LAUNCH:                              # launch in the given order, only below the machine-wide cap
        LAUNCH.wait_for(lambda: NEXT[0] == idx)
        while running_trainers() >= jobs:
            time.sleep(5)
        t0 = time.time()
        fo = open(os.path.join(out_dir, "stdout.txt"), "a")
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, "long_train.py"), path],
                                stdout=fo, stderr=subprocess.STDOUT, env=env, cwd=HERE, preexec_fn=set_nice)
        time.sleep(2)
        NEXT[0] += 1; LAUNCH.notify_all()
    rc = proc.wait()
    fo.close()
    return cfg["name"], "ok" if rc == 0 else f"exit {rc}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+"); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    unknown = [g for g in a.groups if g not in GROUPS]
    if unknown:
        raise SystemExit(f"unknown groups {unknown}; known: {sorted(GROUPS)}")
    cfgs = [cfg for g in a.groups for cfg in GROUPS[g]]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with ThreadPoolExecutor(len(cfgs)) as ex:
        for name, status, wall in ex.map(lambda ic: run_one(ic[0], ic[1], env, min(a.jobs, 4)), enumerate(cfgs)):
            print(f"{name}: {status} ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
