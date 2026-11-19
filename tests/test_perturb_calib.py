"""M1a #21 perturbation effects; #8 swap cost; #10/#11 calibrations; #25 throughput (report)."""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from animalbench.core import compile_schedule, crosswind, gust, injury, recovery_time
from animalbench.scenarios.geese import GeeseConfig, make_rollout, rule_blind

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib", "out")


def paired(events, T=200.0, seed=0, policy=rule_blind):
    cfg = GeeseConfig(n_max=9, T=T, init_mode="v", log_stride=5, log_D=False)
    f = jax.jit(make_rollout(cfg, policy(cfg)))
    k = jax.random.PRNGKey(seed)
    _, p = f(k, jnp.int32(9), compile_schedule(events, cfg.e_max))
    _, c = f(k, jnp.int32(9), compile_schedule([], cfg.e_max))
    return cfg, jax.tree_util.tree_map(np.asarray, p), jax.tree_util.tree_map(np.asarray, c)


def e_series(cfg, rec):
    ef = rec["low"]["e_form"].reshape(-1, cfg.n_max) / cfg.b
    return np.sqrt((ef ** 2).mean(1)), rec["low"]["t"].reshape(-1)


def delta_E(cfg, p, c):
    return float((p["P_mean"] * p["alive"]).sum() - (c["P_mean"] * c["alive"]).sum()) * cfg.delta_h   # P_solo*s


def test_21_crosswind_finite_recovery_and_positive_cost():
    ev = [crosswind(60.0, 40.0, speed=3.0, ou_sigma=0.0)]
    cfg, p, c = paired(ev)
    dev, t = e_series(cfg, p)
    t_rec, cens = recovery_time(dev, t, 60.0, base_window=20.0, tol_rel=1.2, tol_abs=0.05, hold=5.0)
    assert not cens and 0 < t_rec < 60.0, t_rec
    dev_c, _ = e_series(cfg, c)
    assert dev[(t > 50) & (t < 60)].max() < 5e-3 and dev[(t > 62) & (t < 70)].max() > 0.05      # a real transient happened
    dE = delta_E(cfg, p, c)
    E_tot = float((c["P_mean"] * c["alive"]).sum() * cfg.delta_h)
    assert dE > 0 and dE / E_tot > 1e-4, (dE, E_tot)
    # with OU turbulence the recovery is still finite (recovered once the wind has gone)
    ev2 = [crosswind(60.0, 40.0, speed=3.0, ou_sigma=0.5)]
    cfg, p2, c2 = paired(ev2)
    dev2, t = e_series(cfg, p2)
    t_rec2, cens2 = recovery_time(dev2, t, 60.0)
    assert not cens2 and np.isfinite(t_rec2), t_rec2
    assert delta_E(cfg, p2, c2) > 0


def test_21_zero_wind_costs_nothing():
    ev = [crosswind(60.0, 40.0, speed=0.0, ou_sigma=0.0)]
    cfg, p, c = paired(ev, T=120.0)
    dE = delta_E(cfg, p, c)
    E_tot = float((c["P_mean"] * c["alive"]).sum() * cfg.delta_h)
    assert abs(dE) / E_tot < 0.005


def test_21_injury_and_gust_effects():
    cfg, p, c = paired([injury(40.0, "leader")], T=120.0)
    assert delta_E(cfg, p, c) > 0
    assert p["E_used"][-1].max() > c["E_used"][-1].max()
    cfg, p, c = paired([gust(40.0, 3.0, amp=6.0, dir_deg=90.0, spatial=1.0)], T=120.0)
    dev, t = e_series(cfg, p)
    assert dev[(t > 40) & (t < 60)].max() > 1e-4                                  # the front-sweeping gust disturbs the V
    assert np.isfinite(p["P_mean"]).all()


# --------------------------------------------------------------------------- #8
@pytest.fixture(scope="module")
def swap_cost():
    from calib import cal_swap
    res = cal_swap.run(100)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "swap_cost.json"), "w"), indent=1)
    return json.load(open(os.path.join(OUT, "swap_cost.json")))


def test_8_swap_cost(swap_cost):
    for name, r in swap_cost["pairs"].items():
        assert 2.0 <= r["t_swap_min"] and r["t_swap_max"] <= 6.0, (name, r)
        assert r["min_sep_b"] > 0.3, (name, r)
        assert r["others_e_form_max_b"] < 0.3, (name, r)
        assert 0.1 <= r["c_swap_min"] and r["c_swap_max"] <= 2.0, (name, r)
    assert 0.1 <= swap_cost["c_swap_apex"] <= 2.0


# --------------------------------------------------------------------------- #10 / #11
@pytest.fixture(scope="module")
def cprog():
    from calib import cal_cprog
    res = cal_cprog.run()
    json.dump(res, open(os.path.join(OUT, "cprog.json"), "w"), indent=1)
    return res


def test_10_cprog_single_bird_argmax(cprog):
    assert abs(cprog["c_prog_closed_form"] - 0.7076) < 1e-3
    assert cprog["single_ok"] and 0.99 <= cprog["single_argmax_closed_form"] <= 1.01


@pytest.mark.xfail(strict=True, reason="spec #10 second clause fails with the closed-form c_prog: wake strength ~1/v "
                   "makes a collective 10% slow-down pay (+0.24% P_solo per step). See M1a_stage1_decisions.md")
def test_10_cprog_team_slowdown_strictly_lower_closed_form(cprog):
    assert cprog["team_slowdown_lower_ok"]


def test_10_cprog_window_that_satisfies_both_exists(cprog):
    lo, hi = cprog["c_prog_both_ok_range"]
    assert lo > cprog["c_prog_closed_form"] and hi - lo > 0.005


@pytest.fixture(scope="module")
def ciso():
    from calib import cal_ciso
    res = cal_ciso.run()
    json.dump(res, open(os.path.join(OUT, "ciso.json"), "w"), indent=1)
    return res


def test_11_ciso_default_passes_and_negative_control_fails(ciso):
    d, n = ciso["default"], ciso["negative_control"]
    assert d["order_ok"] and d["gap_ok"]
    assert d["lead_minus_iso"] >= 0.03 and d["lead_minus_iso"] <= 0.5 * d["follow_minus_lead"]
    assert not n["order_ok"]


# --------------------------------------------------------------------------- #25
def test_25_throughput_report():
    from calib import bench_throughput
    r = bench_throughput.run(B=256, n_dec=4)
    json.dump(r, open(os.path.join(OUT, "throughput.json"), "w"), indent=1)
    assert r["agent_steps_per_s"] > 0 and np.isfinite(r["seconds"])
