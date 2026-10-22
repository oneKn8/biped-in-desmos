"""Jittered-start robustness of walkers: share of K reference rollouts per walker that pass the horizon-scaled bar
when every initial coordinate is perturbed by N(0, sigma) metres (verify_long.jitter_pass: ../sim.py physics and
../walkbar.py's phase tracker, only the start differs).

The gaits are chaotic (a 1e-15 m difference grows to metres within 15 s), so a single from-rest pass can hinge on
one floating-point trajectory; this rate measures whether a walker passes for a family of nearby starts.

Usage: python3 robust_gait.py OUT.json RUN_OR_DIR [...] [--glob 'theta_pass_s*.npy'] [--k 20] [--sigma 0.001]
"""
import argparse
import json
import os

import numpy as np

import sym_expand
import verify_gait
import verify_long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("paths", nargs="+")
    ap.add_argument("--glob", default="theta_pass_s*.npy")
    ap.add_argument("--k", type=int, default=20); ap.add_argument("--sigma", type=float, default=1e-3)
    a = ap.parse_args()
    files = verify_gait.files_from(a.paths, a.glob)
    if not files:
        raise SystemExit(f"no policy files in {a.paths} matching {a.glob}")
    W = np.stack([sym_expand.expand(np.load(f)) for f in files])
    res = {}
    for T, tag in ((1200, "20"), (3600, "60")):
        res[tag] = [float(x) for x in verify_long.jitter_pass(W, T, a.k, a.sigma)]
    rows = [dict(file=os.path.relpath(f, verify_gait.HERE), jitter20=res["20"][i], jitter60=res["60"][i])
            for i, f in enumerate(files)]
    json.dump(dict(k=a.k, sigma=a.sigma, rows=rows), open(a.out, "w"), indent=1)
    print(f"n={len(files)} K={a.k} sigma={a.sigma * 1000:g} mm  jittered 20 s pass rate mean {np.mean(res['20']):.2f}"
          f"  60 s mean {np.mean(res['60']):.2f}")


if __name__ == "__main__":
    main()
