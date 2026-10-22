"""Checks for long_train.py: rollout equals ../ars.py (fitness, bar quantities) on known policies,
per-row episode lengths equal separate reference rollouts, the symmetric policy equals its expanded
4x13 form, early termination stops a group at its last fall, and the scaled phase requirement."""
import os
import sys

import numpy as np

import fastsim as fs
import long_train as lt
import sym_expand

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)


def load_reference():
    """Import ../ars.py (the reference trainer) without putting it ahead of this folder."""
    sys.path.append(PROTO)
    import ars
    return ars


def main():
    ars = load_reference()
    thetas = np.stack([np.load(os.path.join(PROTO, d, f"theta_s{s}.npy")) for d in ("cap", "alive1") for s in range(4)])
    n = len(thetas)
    ph = fs.Phys()
    cfg = {**lt.DEFAULTS}
    assert cfg["alive_w"] == ars.sim.ALIVE_W and cfg["w_air"] == ars.sim.W_AIR and cfg["air_free"] == ars.sim.AIR_FREE

    # 1. same horizon for every row: fitness and bar quantities equal ars.rollout
    ref = ars.rollout(thetas, T=300)
    out = lt.rollout(thetas, ph, np.full(n, 300), np.full(n, -1), False, False)
    err = np.abs(lt.fitness(out, cfg) - ref["fit"]).max()
    print(f"fitness vs ars.py: max abs diff {err:.2e}")
    assert err < 1e-8
    for k_lt, k_ref in (("n_left", "n_left"), ("n_right", "n_right"), ("fallen", "fallen")):
        assert (out[k_lt] == ref[k_ref]).all(), k_lt
    assert np.allclose(out["air_ticks"] / 300, ref["air_frac"])
    ok_ref = ars.passes(ref, 300)
    ok = lt.bar(out["dist"], 300, out["fallen"], out["n_left"], out["n_right"], out["air_ticks"], ph.DT)
    assert (ok == ok_ref).all(), (ok, ok_ref)
    print(f"bar verdicts equal ars.passes: {ok.astype(int)}")

    # 2. mixed per-row horizons equal separate reference rollouts
    ends = np.array([60, 300, 150, 420, 300, 37, 200, 300])
    out = lt.rollout(thetas, ph, ends, np.full(n, -1), False, False)
    for i in range(n):
        r = ars.rollout(thetas[i:i + 1], T=int(ends[i]))
        assert abs(out["dist"][i] - r["dist"][0]) < 1e-9 and out["n_left"][i] == r["n_left"][0] \
            and out["n_right"][i] == r["n_right"][0] and out["fallen"][i] == r["fallen"][0], i
        assert out["ticks"][i] == ends[i]
    print("per-row horizons match separate reference rollouts")

    # 3. symmetric policy equals its expanded 4x13 matrix under the reference simulator
    rng = np.random.default_rng(5)
    th_sym = 0.3 * rng.standard_normal((3, 2, fs.N_OBS))
    out = lt.rollout(th_sym, ph, np.full(3, 300), np.full(3, -1), True, False)
    r = ars.rollout(sym_expand.expand(th_sym), T=300)
    err = np.abs(out["dist"] - r["dist"]).max()
    print(f"symmetric policy vs expanded 4x13 under ars.py: dist max abs diff {err:.2e}")
    assert err < 1e-8 and (out["n_left"] == r["n_left"]).all()

    # 4. early termination: a group that collapses stops at its last fall tick; an ungrouped row runs on
    W = np.zeros((5, 4, 13)); W[:, 1, 12] = 3.0; W[:, 3, 12] = 3.0      # knees fully flexed: collapses
    out = lt.rollout(W, ph, np.full(5, 300), np.array([0, 0, 0, 0, -1]), False, True)
    last_fall = int(round(out["alive_t"][:4].max() / ph.DT)) + 1
    print(f"early-term ticks {out['ticks'][:4]} (last fall tick {last_fall}), ungrouped row {out['ticks'][4]}")
    assert (out["ticks"][:4] == last_fall).all() and out["ticks"][4] == 300

    # 5. a chained start (last state shifted to x = 0, clock continued) continues a rollout exactly
    th2 = thetas[2:3]
    full = lt.rollout(th2, ph, np.array([600]), np.array([-1]), False, False)
    first = lt.rollout(th2, ph, np.array([300]), np.array([-1]), False, False)
    cp, cv = first["fin_pos"].copy(), first["fin_prev"].copy()
    x = cp[fs.HIP, 0].copy(); cp[:, 0] -= x; cv[:, 0] -= x
    t0 = np.mod(first["fin_t"], 1.0 / fs.CLOCK_HZ)
    second = lt.rollout(th2, ph, np.array([300]), np.array([-1]), False, False, (cp, cv), t0)
    err = abs(first["dist"][0] + second["dist"][0] - full["dist"][0])
    print(f"chained 300 + 300 ticks vs 600 continuous: distance diff {err:.2e}")
    assert err < 1e-6 and second["fallen"][0] == full["fallen"][0]

    # 6. cut: a group ends when its flagged row falls, even if other rows still stand
    W2 = np.concatenate([np.zeros((3, 4, 13)), W[:1]])                 # 3 standing rows, 1 collapsing row
    out = lt.rollout(W2, ph, np.full(4, 300), np.zeros(4, int), False, True, cut=np.array([False, False, False, True]))
    print(f"cut on the collapsing row: ticks {out['ticks']}")
    assert (out["ticks"] == last_fall).all()

    # 7. pace cut: a walker never more than D behind the 0.8 m/s line runs on; a line set 1 m ahead of
    #    the start (D = -1) cuts at the first tick; a falling flagged row still cuts at its fall
    Wp = np.concatenate([thetas[2:3]] * 2)
    out = lt.rollout(Wp, ph, np.full(2, 300), np.zeros(2, int), False, True, cut=np.array([False, True]), pace_cut=2.0)
    ref_ok = lt.rollout(thetas[2:3], ph, np.array([300]), np.array([-1]), False, False)
    behind = ref_ok["dist"][0] < 0.8 * 5.0 - 2.0
    print(f"pace cut 2 m on a 5 s walker ({ref_ok['dist'][0]:.2f} m): ticks {out['ticks']}")
    assert behind or (out["ticks"] == 300).all()
    out = lt.rollout(Wp, ph, np.full(2, 300), np.zeros(2, int), False, True, cut=np.array([False, True]), pace_cut=-1.0)
    assert (out["ticks"] == 1).all(), out["ticks"]
    out = lt.rollout(W2, ph, np.full(4, 300), np.zeros(4, int), False, True, cut=np.array([False, False, False, True]),
                     pace_cut=2.0)
    assert (out["ticks"] == last_fall).all()
    print("pace cut checks passed")

    # 8. scaled phase requirement
    assert [lt.required_phases(s) for s in (5, 10, 20, 60)] == [3, 6, 12, 36]

    # 9. gait knobs: the mirror image of an expanded symmetric policy is itself, the mirror is an involution,
    #    a mirror-image policy on mirrored inputs gives the leg-swapped actions, anti_scale 0 noise is symmetric,
    #    and the trainer's stance-asymmetry and |torso angle| accumulators equal verify_gait's statistics of an
    #    ars.py trajectory
    import verify_gait
    E = sym_expand.expand(th_sym)
    assert np.abs(lt.mirror_theta(E) - E).max() == 0.0
    X = rng.standard_normal((6, 4, fs.N_OBS))
    assert np.abs(lt.mirror_theta(lt.mirror_theta(X)) - X).max() == 0.0
    obs = rng.standard_normal((fs.N_OBS, 3))
    assert np.abs(lt.mirror_theta(X[0]) @ fs.mirror_obs(obs) - (X[0] @ obs)[[2, 3, 0, 1]]).max() < 1e-12
    D = 0.5 * X + 0.5 * lt.mirror_theta(X)
    assert np.abs(lt.mirror_theta(D) - D).max() < 1e-15
    out = lt.rollout(thetas, ph, np.full(n, 300), np.full(n, -1), False, False)
    for i in range(n):
        r = ars.rollout(thetas[i:i + 1], T=300, record=True)
        g = verify_gait.gait_stats(np.array(r["traj"]), bool(r["fallen"][0]), float(r["alive_t"][0]))
        up = max(out["alive_t"][i], 1e-9)
        assert abs(lt.limp(out)[i] - g["asym"]) < 1e-9, (i, lt.limp(out)[i], g["asym"])
        assert abs(np.degrees(out["lean_abs"][i] / up) - g["abs_deg"]) < 1e-9, i
    cfg_g = {**cfg, "bar_asym": 0.15, "bar_lean": 8.0}
    ok_g = lt.gait_ok(out, np.arange(n), cfg_g)
    want = (lt.limp(out) <= 0.15) & (np.degrees(out["lean_abs"] / np.maximum(out["alive_t"], 1e-9)) <= 8.0)
    assert (ok_g == want).all()
    print(f"gait knobs: mirror algebra exact, gait accumulators equal the reference, gait bar verdicts {ok_g.astype(int)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
