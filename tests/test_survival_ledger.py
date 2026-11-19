"""Ledger-reciprocity survival game (analysis/survival_ledger.py): hand-computable degenerate cases."""
import numpy as np
import pytest

from animalbench.analysis import survival_game as sv
from animalbench.analysis import survival_ledger as sl

PR = sv.load_params()
DH = sv.DH
CO, CN, CB = PR["c_old"], PR["c_new"], PR["c_by"]
E0 = 1e7


def _trip(D, **kw):
    hit, size = sv.make_gusts(2, D, 0.0, 0)
    n = 2
    return sl.simulate_trip(E0, D, hit, size, PR, np.zeros((n, 3), np.int64), np.ones((n, 3), bool), tau=15, h=0.0, **kw)


def _used(r):
    return (1 - r["efrac"]) * E0


def test_baseline_ledger_is_plain_ring_and_matches_energy_count():
    # D=45: leaders 0,1,2 for 15 decisions each, two swaps (0->1: old0,new1,by2 ; 1->2: old1,new2,by0)
    r = _trip(45)
    assert r["lead"][0].tolist() == [15, 15, 15] and (r["n_swaps"] == 2).all()
    want = 15 * DH * (1.0 + 2 * 0.875) + np.array([CO + CB, CN + CO, CB + CN])
    assert _used(r)[0] == pytest.approx(want, rel=1e-9)
    assert r["u_end"][0].tolist() == [0, 0, 0]


def test_refuse_without_fine_just_postpones_and_ledger_records_it():
    # bird 1 refuses its tenure at t=15 (lam=0): bird 0 covers t=15..29, then bird 2 (lowest u, ring successor of 1) leads
    r = _trip(45, lam=0, dev=dict(mode="refuse", i=1, t_d=15))
    assert r["lead"][0].tolist() == [30, 0, 15] and (r["n_swaps"] == 1).all()
    want = np.array([30 * DH * 1.0 + 15 * DH * 0.875 + CO, 45 * DH * 0.875 + CB, 15 * DH * 1.0 + 30 * DH * 0.875 + CN])
    assert _used(r)[0] == pytest.approx(want, rel=1e-9)
    assert r["u_end"][0].tolist() == [30, 0, 15]


def test_refuse_with_fine_makes_the_refuser_lead_next_environment_only_punishment():
    # lam=15: u_1 = -15 -> at t=30 the others do not take over from bird 1 (it is still the lowest) -> bird 1 leads 15 decisions
    r = _trip(45, lam=15, dev=dict(mode="refuse", i=1, t_d=15))
    assert r["lead"][0].tolist() == [30, 15, 0] and (r["n_swaps"] == 1).all()
    want = np.array([30 * DH * 1.0 + 15 * DH * 0.875 + CO, 30 * DH * 0.875 + 15 * DH * 1.0 + CN, 45 * DH * 0.875 + CB])
    assert _used(r)[0] == pytest.approx(want, rel=1e-9)
    assert r["u_end"][0].tolist() == [30, 0, 0]
    # the punishers pay NOTHING extra: they only follow (compare with the no-fine run: bird 0 identical)


def test_ledger_carries_over_trips_and_continues_the_ring():
    D, K = 30, 2
    gusts = [sv.make_gusts(2, D, 0.0, k) for k in range(K)]
    st = (np.zeros((2, 3), np.int64), np.ones((2, 3), bool))
    res, starts = sl.run_trips(K, E0, D, gusts, PR, st, lam=0, tau=15, h=0.0)
    assert res[0]["lead"][0].tolist() == [15, 15, 0] and res[0]["u_end"][0].tolist() == [15, 15, 0]
    assert res[1]["lead"][0].tolist() == [15, 0, 15]            # trip 2 starts with the bird that has not led (bird 2), then bird 0
    assert starts[1][0][0].tolist() == [15, 15, 0]


def test_permanent_death_carries_into_next_trip():
    D = 12
    gusts = [sv.make_gusts(1, D, 0.0, k) for k in range(2)]
    st = (np.zeros((1, 3), np.int64), np.ones((1, 3), bool))
    # E0=11,h=0: the first leader (bird 0) dies at t=5 (2/step), as in test_survival_game.test_lone_bird...
    res, starts = sl.run_trips(2, 11.0, D, gusts, PR, st, lam=0, tau=100, h=0.0, perm_death=True)
    assert not res[0]["arrived"][0, 0] and not starts[1][1][0, 0]
    res2, starts2 = sl.run_trips(2, 11.0, D, gusts, PR, st, lam=0, tau=100, h=0.0, perm_death=False)
    assert starts2[1][1].all()


def test_value_discounting_and_bonferroni_z():
    D = 30
    hit, size = sv.make_gusts(2, D, 0.0, 0)
    r = _trip(30)
    v = sl.value([r, r], 0.0, 0.9, 0.5)
    assert v == pytest.approx(sv.utilities(r, 0.0, 0.5) * 1.9)
    assert sl.bonf_z(36) == pytest.approx(3.2, abs=0.05)


def test_degenerate_no_risk_lam0_deviation_pays_but_is_below_tolerance():
    # nobody can die, K=1: refusing saves 0.125*dh*15 energy and the ledger repayment falls outside the horizon late in the trip
    D = 90
    gusts = [sv.make_gusts(40, D, 0.0, 0)]
    out = sl.eval_ledger_cell(1, E0, D, gusts, PR, 0.0, 0, [0.0], [0.99])
    e = out["per"]["0.0|0.99"]["all"]
    assert e["max_gain"] > 0 and e["lo_max"] > 0
    assert e["label"] == "EQ_EPS" and out["summary"]["team_arrival_mean"] == 1.0
