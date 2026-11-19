"""Survival-migration small game (analysis/survival_game.py): hand-checkable degenerate cases + classification logic."""
import numpy as np
import pytest

from animalbench.analysis import survival_game as sv

PR = sv.load_params()
DH = sv.DH


def _noise(n, D):
    return sv.make_gusts(n, D, 0.0, 0)            # pg=0 -> no gusts


def _used(res, E0, h=0.1):
    cap = np.array([E0 * (1 + h), E0, E0])
    return (1 - res["efrac"]) * cap                # energy spent per bird (nobody died)


def test_params_come_from_calib_files():
    assert PR["p_lead"] == pytest.approx(1.0) and PR["p_follow"] == pytest.approx(0.875)
    assert PR["c_old"] + PR["c_new"] + PR["c_by"] == pytest.approx(0.8688, abs=1e-3)


def test_no_gust_huge_E0_fixed_is_plain_energy_ledger():
    D, E0 = 60, 1e6
    hit, size = _noise(4, D)
    r = sv.simulate("fixed", E0, D, hit, size, PR)
    used = _used(r, E0)
    assert r["arrived"].all() and r["n_swaps"].sum() == 0
    assert used[:, 0] == pytest.approx(np.full(4, D * DH * 1.0), rel=1e-9)
    assert used[:, 1:] == pytest.approx(np.full((4, 2), D * DH * 0.875), rel=1e-9)


def test_no_gust_huge_E0_rotation_matches_hand_count():
    D, tau, E0 = 90, 15, 1e7                      # leaders: 0 1 2 0 1 2 -> 5 swaps, each bird leads 30 decisions
    hit, size = _noise(3, D)
    r = sv.simulate("rot", E0, D, hit, size, PR, tau=tau)
    co, cn, cb = PR["c_old"], PR["c_new"], PR["c_by"]
    steady = 30 * DH * 1.0 + 60 * DH * 0.875
    want = steady + np.array([2 * co + 2 * cb + cn, 2 * cn + 2 * co + cb, 2 * cb + 2 * cn + co])
    assert (r["n_swaps"] == 5).all()
    assert _used(r, E0)[0] == pytest.approx(want, rel=1e-6)


def test_leader_death_promotes_successor_and_follower_saving_stays():
    # bird 0 cap = 0.5*E0 = 30 hits E<=0 when its cumulative lead cost 2/decision reaches 30 (after 15 decisions, t=14); no gusts, no swap on death
    D, E0 = 20, 60.0
    hit, size = _noise(2, D)
    r = sv.simulate("fixed", E0, D, hit, size, PR, h=-0.5)
    assert (r["death_t"][:, 0] == 14).all() and not r["arrived"][:, 0].any()
    assert r["efrac"][0, 1] == pytest.approx((60 - 15 * 1.75 - 5 * 2.0) / 60)       # promoted leader pays full cost
    assert r["efrac"][0, 2] == pytest.approx((60 - 20 * 1.75) / 60)                  # remaining follower still saves 12.5 %
    assert r["n_swaps"].sum() == 0


def test_lone_bird_pays_solo_and_cascade_deaths():
    D, E0 = 12, 11.0                              # h=0: all start 11; leader 2/step dies t=5; bird1 (promoted, 2/step) and bird2 (0.5 left, 1.75/step) both die t=6
    hit, size = _noise(1, D)
    r = sv.simulate("fixed", E0, D, hit, size, PR, h=0.0)
    assert r["death_t"][0].tolist() == [5, 6, 6] and r["efrac"].sum() == 0


def test_utility_mix_and_team_term():
    D = 30
    hit, size = _noise(2, D)
    r = sv.simulate("fixed", 1e6, D, hit, size, PR)
    u0, u1 = sv.utilities(r, 0.0, 0.5), sv.utilities(r, 1.0, 0.5)
    ind = 1.0 + 0.5 * r["efrac"]
    assert u0 == pytest.approx(ind) and u1 == pytest.approx(np.repeat(ind.mean(1, keepdims=True), 3, 1))
    assert sv.utilities(r, 0.25, 0.5) == pytest.approx(0.75 * u0 + 0.25 * u1)


def test_degenerate_rotation_not_equilibrium_without_gusts_alpha0():
    # nobody can die -> refusing a tenure just saves energy: the plain energy ledger (like ledger_game, lam=0)
    D, E0 = 90, 1e7
    hit, size = _noise(50, D)
    out = sv.eval_cell("rot", 15, E0, D, hit, size, PR, 0.0, 0.1, 1.0, [0.0], [0.5])
    e = out["per"]["0.0|0.5"]
    assert e["max_gain"] > 0 and e["lo_max"] > 0                                      # deviating strictly pays (tiny: ~1e-7 of E0)
    assert e["label"] == "EQ_EPS"                                                      # ... but far below any practical tolerance
    assert out["summary"]["team_arrival"] == 1.0


def test_classify_three_way():
    assert sv.classify([0.1, -0.2], [0.01, 0.01])[0] == "NOT_EQ"
    assert sv.classify([-0.1, -0.2], [0.01, 0.01])[0] == "EQ_STRICT"
    assert sv.classify([0.0, -0.2], [0.1, 0.01])[0] == "UNDECIDED"          # gain CI contains 0 and 0.3
    assert sv.classify([0.001, -0.2], [0.0005, 0.001])[0] == "EQ_EPS"       # real but immaterial gain
    assert sv.classify([], [])[0] == "EQ_STRICT"
    assert sv.deviations("fixed", 15, 300, rule="mutual") == []


# ---------------------------------------------------------------- v2: multipliers and the energy-aware emax strategy
def test_v2_mult_none_and_ones_are_identical():
    D, E0 = 60, 1e4
    hit, size = sv.make_gusts(50, D, 0.05, 1)
    a = sv.simulate("rot", E0, D, hit, size, PR, tau=15, m_g=1.0)
    b = sv.simulate("rot", E0, D, hit, size, PR, tau=15, m_g=1.0, mult=np.ones((50, 3)))
    assert (a["efrac"] == b["efrac"]).all() and (a["arrived"] == b["arrived"]).all()


def test_v2_mult_scales_energy_use_hand_count():
    D, E0 = 60, 1e7
    hit, size = _noise(2, D)
    mult = np.array([[1.05, 1.0, 0.95], [0.95, 1.0, 1.05]])
    r = sv.simulate("fixed", E0, D, hit, size, PR, mult=mult)
    used = _used(r, E0)
    exp = np.array([D * DH * 1.0, D * DH * 0.875, D * DH * 0.875])[None, :] * mult
    assert used == pytest.approx(exp, rel=1e-6)


def test_v2_emax_leads_with_highest_energy_and_h_gives_bird0_more():
    D, E0 = 90, 1e6
    hit, size = _noise(1, D)
    r = sv.simulate("emax", E0, D, hit, size, PR, tau=15, h=0.05)
    used = _used(r, E0, 0.05)[0]
    assert r["arrived"].all()
    assert used[0] > used[1] and used[0] > used[2]          # bird 0 (more energy) carries more of the lead


def test_v2_pathT_params_reproduce_old_cost_at_3s():
    import os
    if not os.path.exists(os.path.join(sv.OUT, "swap_cost_pathT.json")):
        pytest.skip("run python -m calib.cal_swap_pathT first")
    p3 = sv.load_params_pathT(3.0)
    assert p3["c_old"] + p3["c_new"] + p3["c_by"] == pytest.approx(0.8688, abs=0.05)
    assert sum(sv.load_params_pathT(1.0)[k] for k in ("c_old", "c_new", "c_by")) < sum(p3[k] for k in ("c_old", "c_new", "c_by"))


def test_v2_gate_keeps_leading_until_successor_has_more_energy():
    D, E0 = 90, 1e6
    hit, size = _noise(1, D)
    r = sv.simulate("gate", E0, D, hit, size, PR, tau=4, h=0.0)       # equal caps: leader burns faster -> first handover right at tau
    assert r["n_swaps"][0] >= 1 and r["arrived"].all()
    r2 = sv.simulate("gate", E0, D, hit, size, PR, tau=4, h=0.0, mult=np.array([[1.0, 0.95, 0.95]]) * np.array([[1.0, 1.0, 1.0]]))
    assert r2["lead_cnt"][0].sum() > 0
