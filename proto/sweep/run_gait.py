"""Gait-quality sweep on top of the adopted long-walk recipe.

Each config trains with long_train.py and is then re-checked with the reference (verify_gait.py writes
runs/<name>/verify_gait.json) in the same job slot. Jobs run at niceness 15 (absolute) with single-threaded
BLAS; at most --jobs (<= 4) jobs run machine-wide, counting every long_train.py and verify_gait.py process.
A config whose summary.json and verify_gait.json exist is skipped; an interrupted one resumes from its checkpoint.

BASE is the adopted recipe (long_walkers/P96_two-exam-passes): 97 bodies (48 directions + mean row), top 24,
nu 0.12 -> 0.08 and alpha 0.15 -> 0.08 after the first pass, 5 s chained windows, a 20 s cut exam from rest after
2 passing windows in a row, stop after 2 exam passes in a row. Each config adds gait knobs (long_train.py).

Usage: python3 run_gait.py GROUP [GROUP ...] [--jobs 4]
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
NICE = 15
S8 = list(range(8))
S16 = list(range(8, 24))
BASE = dict(P=96, b=24, stages=[5, 20], chain_stages=[True, False], pass_streak=2, final_streak=2, demote=True,
            cut_stages=[False, True], nu2=0.08, alpha2=0.08, budget_ticks=100_000)


def c(name, **kw):
    return dict(name=name, seeds=kw.pop("seeds", S8), **{**BASE, **kw})


GROUPS = {
    # one knob per mechanism: torso penalty (square), stance-asymmetry penalty, symmetric-biased noise.
    # All four were stopped (state in runs_stopped/): G_lean10, G_asym1.5 and G_anti25 at ~36k ticks, as their
    # walkers went airborne; G_lean3 at ~56k ticks with 2/8 seeds through.
    "k1_stopped": [c("G_lean3", w_lean=3.0),
                   c("G_lean10", w_lean=10.0),
                   c("G_asym1.5", w_asym=1.5),
                   c("G_anti25", anti_scale=0.25)],
    # every gait penalty above pushed the walkers into flight (air 0.3-0.45 > the bar's 0.25): pair the penalties
    # with a strong airborne penalty (w_air 8 above the free 15%) or a speed cap
    "k2": [c("G_lean5_air8", w_lean=5.0, w_air=8.0),
           c("G_anti0_air8", anti_scale=0.0, w_air=8.0)],
    # stopped (runs_stopped/): G_anti0_cap10 0/8 through at 48k ticks, G_asym1.5_lean5_air8 0/8 at 65k
    "k2_stopped": [c("G_asym1.5_lean5_air8", w_asym=1.5, w_lean=5.0, w_air=8.0),
                   c("G_anti0_cap10", anti_scale=0.0, v_cap=1.0)],
    # penalties on from the start keep the walkers hopping (air 0.24-0.52 at 12-17k ticks, no window passes, while
    # the adopted recipe already passes windows): learn the walk unshaped, shape it after the first pass, and let
    # the 20 s exam pass only a walk with asymmetry <= 0.15 and mean |torso angle| <= 8 degrees
    # stopped (runs_stopped/) at ~40-58k ticks with no seed through: the exam gate rejected limping walks
    # (asymmetry 0.28-0.38 from rest even with w_asym2) while the torso was fixed (5-8 degrees with w_lean2 3)
    "k3_stopped": [c("G_post_l3a1_bar", w_lean2=3.0, w_asym2=1.0, w_air2=8.0, bar_asym=0.15, bar_lean=8.0,
                     gait_stages=[False, True]),
                   c("G_post_l5a2_bar", w_lean2=5.0, w_asym2=2.0, w_air2=8.0, bar_asym=0.15, bar_lean=8.0,
                     gait_stages=[False, True])],
    # continuation: plain ARS finds a (limping) walk fast; after the first pass the noise moves into the
    # mirror-symmetric subspace and the antisymmetric part of theta decays, while w_lean2 shapes the torso
    "k6": [c("G_homo05_l3_bar", anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0, bar_asym=0.15,
             bar_lean=8.0, gait_stages=[False, True]),
           c("G_homo05s_l3_bar", anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0, bar_asym=0.15,
             bar_lean=8.0, gait_stages=[False, True], nu2=0.113, alpha2=0.113),
           c("G_homo10_l3_bar", anti_scale2=0.0, sym_decay2=0.10, w_lean2=3.0, w_air2=8.0, bar_asym=0.15,
             bar_lean=8.0, gait_stages=[False, True]),
           ],
    # symmetric search kept falling while hopping (air 0.25-0.33): an airborne penalty from 0% (air_free 0) makes the
    # grounded walk the only cheap gait from the first generation
    # stopped (runs_stopped/): symmetric search from zero stayed unreliable (1/8 through at ~33-37k ticks, seeds stuck
    # leaning 20-45 degrees or on one leg); G_anti0s_air8_post_l3_bar 3/8 at ~77k
    "k8_stopped": [c("G_anti0_air8_free0", anti_scale=0.0, w_air=8.0, air_free=0.0),
                   c("G_anti0s_air8_free0", anti_scale=0.0, w_air=8.0, air_free=0.0, nu=0.17, alpha=0.21, nu2=0.113,
                     alpha2=0.113),
                   c("G_anti0_air8_free0_post_l3_bar", anti_scale=0.0, w_air=8.0, air_free=0.0, w_lean2=3.0,
                     bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True]),
                   c("G_anti0s_air8_post_l3_bar", anti_scale=0.0, w_air=8.0, nu=0.17, alpha=0.21, nu2=0.113,
                     alpha2=0.113, w_lean2=3.0, bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True])],
    "not_run_2": [c("G_homo05_free0_l3_bar", anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0, air_free2=0.0,
             bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True])],
    # the continuation reaches the gait goal (6/8, asym 0.01, torso 7 degrees, 60 s 6/6) at ~65k ticks; its windows
    # fail by falls while theta is being symmetrized. Cheaper variants:
    "k9": [c("G_homo05p_l3_bar", anti_scale2=0.0, sym_decay2=0.05, decay_on_pass=True, w_lean2=3.0, w_air2=8.0,
             bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True]),
           c("G_homo05_l3", anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0),
           c("G_homo03_l3_bar", anti_scale2=0.0, sym_decay2=0.03, w_lean2=3.0, w_air2=8.0, bar_asym=0.15,
             bar_lean=8.0, gait_stages=[False, True]),
           c("G_homo10p_l3_bar", anti_scale2=0.0, sym_decay2=0.10, decay_on_pass=True, w_lean2=3.0, w_air2=8.0,
             bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True])],
    # post-pass windows of the continuation fell 60-90% of the time: the switch at the first window pass (2.6-8.7k
    # ticks) pushes a barely-walking policy. Switch at the first exam that passes the plain bar instead
    # stopped (runs_stopped/) at ~37-58k ticks with no seed through: symmetrizing a walk that already passes 20 s
    # still breaks it (exams keep falling), and starting that late leaves no time to repair it
    "k10_stopped": [c("G_homoX05_l3_bar", post_on="exam", anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0,
              bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True]),
            c("G_homoX05p_l3_bar", post_on="exam", anti_scale2=0.0, sym_decay2=0.05, decay_on_pass=True, w_lean2=3.0,
              w_air2=8.0, bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True]),
            c("G_homoX10p_l3_bar", post_on="exam", anti_scale2=0.0, sym_decay2=0.10, decay_on_pass=True, w_lean2=3.0,
              w_air2=8.0, bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True]),
            c("G_homoX20p_l3_bar", post_on="exam", anti_scale2=0.0, sym_decay2=0.20, decay_on_pass=True, w_lean2=3.0,
              w_air2=8.0, bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True])],
    # confirmations on seeds 8-23 of the best continuation, gated and ungated, and the cost of the symmetry alone
    "c1": [c("G_homo05_l3_bar_c16", anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0, bar_asym=0.15,
             bar_lean=8.0, gait_stages=[False, True], seeds=S16),
           c("G_homo05_l3_c16", anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0, seeds=S16)],
    "k11": [c("G_homo05_asymbar", anti_scale2=0.0, sym_decay2=0.05, w_air2=8.0, bar_asym=0.15,
              gait_stages=[False, True])],
    # post-pass windows fall 75% of the time while theta is symmetrized (49% in the adopted recipe): start the first
    # pass with less asymmetry to remove, by a small decay or half-strength antisymmetric noise from the start
    "k12": [c("G_pre01_homo05_l3_bar", sym_decay=0.01, anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0,
              bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True]),
            c("G_anti50_homo05_l3_bar", anti_scale=0.5, anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, w_air2=8.0,
              bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True])],
    # does the post-pass switch to w_air 8 or the torso weight cause the extra falls? (the walkers end 0-1% airborne)
    # stopped (runs_stopped/): G_homo05_l3_bar_air3 1/8 through at ~70k ticks, G_homo05_l2_bar 1/8 at ~50k
    "k13_stopped": [c("G_homo05_l3_bar_air3", anti_scale2=0.0, sym_decay2=0.05, w_lean2=3.0, bar_asym=0.15, bar_lean=8.0,
              gait_stages=[False, True]),
            c("G_homo05_l2_bar", anti_scale2=0.0, sym_decay2=0.05, w_lean2=2.0, w_air2=8.0, bar_asym=0.15,
              bar_lean=8.0, gait_stages=[False, True]),
            ],
    "k13": [c("G_homo05_l5_bar", anti_scale2=0.0, sym_decay2=0.05, w_lean2=5.0, w_air2=8.0, bar_asym=0.15,
              bar_lean=8.0, gait_stages=[False, True])],
    # pooled over 24 seeds the continuation reaches 14/24 at a median of ~74-78k ticks, gate or not: the cost is the
    # push to exact symmetry. Decay only while the mean row's stance asymmetry in the episode exceeds 0.10
    "k14": [c("G_homoL10_l3_bar", anti_scale2=0.0, sym_decay2=0.05, decay_limp=0.10, w_lean2=3.0, w_air2=8.0,
              bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True]),
            c("G_homoL10a50_l3_bar", anti_scale2=0.5, sym_decay2=0.05, decay_limp=0.10, w_lean2=3.0, w_air2=8.0,
              bar_asym=0.15, bar_lean=8.0, gait_stages=[False, True])],
    # never run (superseded once the continuation above worked)
    "not_run": [           c("G_anti0s_air8", anti_scale=0.0, w_air=8.0, nu=0.17, alpha=0.21, nu2=0.113, alpha2=0.113),
           c("G_anti25_air8", anti_scale=0.25, w_air=8.0)],
    # stopped (runs_stopped/): 1/8 through at ~50k ticks; walkers fell or leaned back 30-40 degrees
    "k7_stopped": [c("G_anti0_airfree0", anti_scale=0.0, air_free=0.0)],
    # parked, not run: superseded by the configs above
    "parked": [c("G_post_l3a1", w_lean2=3.0, w_asym2=1.0, w_air2=8.0),
               c("G_anti0_air8_post_l3_bar", anti_scale=0.0, w_air=8.0, w_lean2=3.0, bar_asym=0.15, bar_lean=8.0,
                 gait_stages=[False, True]),
               c("G_anti0_air8_lean2", anti_scale=0.0, w_air=8.0, w_lean=2.0),
               c("G_asym1.5_lean3_air15", w_asym=1.5, w_lean=3.0, w_air=15.0)],
}

LAUNCH = threading.Lock()


def set_nice():
    """Run the job at niceness NICE or above, whatever the launcher's own niceness is."""
    os.nice(max(0, NICE - os.nice(0)))


def running_jobs():
    """Trainer and reference-check processes running machine-wide."""
    r = subprocess.run(["pgrep", "-f", r"python3? .*(long_train|verify_gait)\.py"], capture_output=True, text=True,
                       timeout=10)
    return len(r.stdout.split())


def trainer_running(name):
    pat = f"(long_train.py .*runs/{name}/request.json|verify_gait.py .*runs/{name}( |$))"
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=10)
    return bool(r.stdout.split())


def start(argv, out_dir, env, jobs):
    """Start argv once fewer than jobs jobs run machine-wide; the lock keeps this dispatcher's threads in line."""
    while True:
        with LAUNCH:
            if running_jobs() < jobs:
                fo = open(os.path.join(out_dir, "stdout.txt"), "a")
                proc = subprocess.Popen(argv, stdout=fo, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                                        preexec_fn=set_nice)
                time.sleep(1)
                return proc, fo
        time.sleep(5)


def run_one(cfg, env, jobs):
    out_dir = os.path.join(HERE, "runs", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    while trainer_running(cfg["name"]):        # a job started elsewhere for this config: wait for it
        time.sleep(10)
    if not os.path.exists(os.path.join(out_dir, "summary.json")):
        path = os.path.join(out_dir, "request.json")
        json.dump(cfg, open(path, "w"), indent=1)
        proc, fo = start([sys.executable, os.path.join(HERE, "long_train.py"), path], out_dir, env, jobs)
        rc = proc.wait(); fo.close()
        if rc:
            return cfg["name"], f"train exit {rc}", time.time() - t0
    if not os.path.exists(os.path.join(out_dir, "verify_gait.json")):
        proc, fo = start([sys.executable, os.path.join(HERE, "verify_gait.py"), out_dir,
                          "--out", os.path.join(out_dir, "verify_gait.json")], out_dir, env, jobs)
        rc = proc.wait(); fo.close()
        if rc:
            return cfg["name"], f"verify exit {rc}", time.time() - t0
    return cfg["name"], "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+"); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    unknown = [g for g in a.groups if g not in GROUPS]
    if unknown:
        raise SystemExit(f"unknown groups {unknown}; known: {sorted(GROUPS)}")
    cfgs = [cfg for g in a.groups for cfg in GROUPS[g]]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    jobs = min(a.jobs, 4)
    with ThreadPoolExecutor(len(cfgs)) as ex:
        futs = []
        for cfg in cfgs:                      # submit in order, a moment apart, so earlier configs start first
            futs.append(ex.submit(run_one, cfg, env, jobs)); time.sleep(0.2)
        for f in futs:
            name, status, wall = f.result()
            print(f"{name}: {status} ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
