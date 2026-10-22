"""An interrupted-then-resumed run must reproduce an uninterrupted run exactly."""
import json
import os
import shutil

import numpy as np

import train

CFG = dict(P=8, b=2, nu=0.08, alpha=0.06, ep_s=1.0, eval_s=1.0, seeds=[0, 1], budget_ticks=1800,
           post_pass_gens=5)


class Boom(Exception):
    pass


def run(name, crash_after=None):
    d = os.path.join(train.HERE, "runs", name)
    shutil.rmtree(d, ignore_errors=True)
    if crash_after is not None:
        real = train.rollout
        calls = [0]

        def flaky(*a, **k):
            calls[0] += 1
            if calls[0] > crash_after:
                raise Boom()
            return real(*a, **k)
        train.rollout = flaky
        try:
            train.train(dict(CFG, name=name))
        except Boom:
            pass
        finally:
            train.rollout = real
        assert os.path.exists(os.path.join(d, "ckpt.pkl")) and not os.path.exists(os.path.join(d, "summary.json"))
    train.train(dict(CFG, name=name))
    hist = [json.load(open(os.path.join(d, f"hist_s{s}.json"))) for s in CFG["seeds"]]
    th = [np.load(os.path.join(d, f"theta_final_s{s}.npy")) for s in CFG["seeds"]]
    shutil.rmtree(d)
    return hist, th


def main():
    h1, t1 = run("test_resume_a")
    h2, t2 = run("test_resume_b", crash_after=17)
    assert h1 == h2, "history differs after resume"
    assert all(np.array_equal(a, b) for a, b in zip(t1, t2)), "final theta differs after resume"
    print(f"resume reproduces the uninterrupted run exactly ({len(h1[0])} gens, seed 0)")


if __name__ == "__main__":
    main()
