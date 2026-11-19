"""M1a #15: analytically tractable small game (analysis/small_game.py).  Inputs p[slot], cost[edge][slot] are measured in
the real environment (calib/out/small_game_params.json, written by `python -m animalbench.analysis.slot_params`)."""
import json
import os

import numpy as np
import pytest

from animalbench.analysis import small_game as sg
from animalbench.analysis.slot_params import main as measure_params

S, B, F = 0, 1, 2


@pytest.fixture(scope="module", autouse=True)
def params():
    if not os.path.exists(sg.PARAMS_PATH):
        measure_params()
    return json.load(open(sg.PARAMS_PATH))


@pytest.fixture(scope="module")
def games():
    out = {}
    for N in (3, 4):
        p, cost = sg.load_params(N)
        for rule in ("unilateral", "mutual"):
            out[(N, rule)] = sg.Game(N, p, cost, rule)
    return out


# ---------------------------------------------------------------------------------------------- (a) 2-slot, hand solution
def test_15a_two_slot_known_solution():
    p1, c0, c1 = 0.8, 0.28, 0.19                     # synthetic parameters (not the measured ones)
    cost = {1: np.array([c0, c1])}
    dh, lock = 2.0, 4
    # hand calculation: after a swap both birds are locked for lock-1 = 3 decisions, so under 'apex asks BACK' the pair
    # swaps every 4 decisions; each bird spends 4 decisions in each slot and pays c0 (as retreater) and c1 (as advancer)
    # once per 8 decisions:   g_rot = -dh (1+p1)/2 - (c0+c1)/8 ;   fixed leader: apex -dh, follower -p1 dh.
    g_rot = -dh * (1 + p1) / 2 - (c0 + c1) / 8
    uni = sg.Game(2, [1.0, p1], cost, "unilateral")
    assert uni.average_reward((B, S)) == pytest.approx(g_rot, abs=1e-3)
    # unilateral: retreating beats staying for the apex iff dh (1-p1)/2 > (c0+c1)/8  (true here), and the follower's
    # FWD request is redundant (same edge) -> equilibria are exactly (B,S) and (B,F)
    assert dh * (1 - p1) / 2 > (c0 + c1) / 8
    eq = {tuple(int(x) for x in e[0]) for e in uni.equilibria()}
    assert eq == {(B, S), (B, F)}
    # mutual: (B,F) is not an equilibrium (the follower prefers to stay deep: -p1 dh > g_rot) -> fixed leader
    assert -p1 * dh > g_rot
    mut = sg.Game(2, [1.0, p1], cost, "mutual")
    eq_m = {tuple(int(x) for x in e[0]) for e in mut.equilibria()}
    assert (S, S) in eq_m and (B, F) not in eq_m


# ---------------------------------------------------------------------------------------------- (b) convergence
@pytest.mark.parametrize("key", [(3, "unilateral"), (3, "mutual"), (4, "mutual")])
def test_15b_convergence_residuals(games, key):
    g = games[key]
    r = g.check_pure(np.array([B] + [S] * (g.N - 1)))
    assert r["residual"] < 1e-8                                   # Bellman residual of the exact policy iteration
    pi, hist = g.br_iteration(iters=100)
    assert hist[-1] < 1e-8                                        # damped (0.5) best-response iteration converged
    assert len(hist) < 100


def test_15b_unilateral_n4_has_no_pure_symmetric_equilibrium(games):
    """Reported, not hidden: for N=4 the unilateral game has no pure symmetric equilibrium and the damped BR iteration
    cycles in the mixing of slots 1/2 (residual stays ~1/3); the apex retreat is nevertheless stable (see (c))."""
    g = games[(4, "unilateral")]
    assert g.equilibria() == []
    pi, hist = g.br_iteration(iters=30)
    assert hist[-1] > 0.1
    assert pi[0, B] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------- (c) null predictions
def test_15c_unilateral_n3_equilibrium_is_hot_potato(games):
    g = games[(3, "unilateral")]
    eq = [tuple(int(x) for x in e[0]) for e in g.equilibria()]
    assert eq == [(B, S, S)]
    pi, hist = g.br_iteration()
    assert pi[0, B] > 0.9                                          # P(back | apex, unlocked)
    # the treadmill is a tax: per-bird long-run cost vs. the fixed-leader average over slots
    tax = -g.average_reward((B, S, S)) / (g.p.sum() / 3 * g.dh) - 1
    assert 0.02 < tax < 0.08                                       # about 4 % more energy than a fixed leader


def test_15c_unilateral_n4_apex_retreat_is_best_response_in_most_profiles(games):
    n, nB, nStrict = games[(4, "unilateral")].apex_retreat_dominance()
    assert nB / n > 0.9


def test_15c_unilateral_n5_pure_equilibrium_has_apex_retreat():
    p, cost = sg.load_params(5)
    g = sg.Game(5, p, cost, "unilateral")
    r = g.check_pure(np.array([B, B, B, F, F]))
    assert r["gap"] <= 1e-9 and r["residual"] < 1e-8
    assert g.check_pure(np.array([S] * 5))["gap"] > 0.1           # fixed leader is far from equilibrium


@pytest.mark.parametrize("N", [3, 4, 5])
def test_15c_mutual_fixed_leader_is_equilibrium(N):
    p, cost = sg.load_params(N)
    g = sg.Game(N, p, cost, "mutual")
    r = g.check_pure(np.zeros(N, int))
    assert r["gap"] <= 1e-9                                        # deviation gain <= 0
    assert r["residual"] < 1e-8
    # nobody volunteers: also the profile where every slot asks to retreat has no deviation gain, i.e. no profitable
    # FWD volunteer exists at any N
    assert g.check_pure(np.array([B] * (N - 2) + [S, S]))["gap"] <= 1e-9 if N >= 4 else True


# ---------------------------------------------------------------------------------------------- (d) vs single-point test
def test_15d_n3_apex_retreat_advantage_matches_single_point_deviation(games):
    from animalbench.analysis import deviation as dv
    g = games[(3, "unilateral")]
    abstract = sg.finite_horizon_advantage(g, [S, S, S], 0, B, 30)          # H = 60 s, others stay
    real = dv.advantage("stay", "apex", seeds=range(5))["mean"]            # N=9 real environment (#12)
    assert abstract > 0 and real > 0
    assert abstract == pytest.approx(real, rel=0.10)
