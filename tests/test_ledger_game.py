"""M2-pre (B): ledger-state small game (analysis/ledger_game.py), N=3 mutual, kappa=0."""
import numpy as np
import pytest

from animalbench.analysis import small_game as sg
from animalbench.analysis.ledger_game import LedgerGame, fixed_leader_cost


@pytest.fixture(scope="module")
def p3():
    return sg.load_params(3)


def test_state_space_invariants(p3):
    lg = LedgerGame(3, *p3, lock_dec=4, theta=8, lam=2)
    n = lg.enumerate_states()
    assert n > 1000
    sl, lk, ct, uu = lg.state
    assert np.all(uu.min(1) == 0) and np.all(uu <= lg.Lc)                          # normalised ledger
    assert np.all(np.sort(sl, 1) == np.arange(3))                                  # slots are a permutation
    lg.sigma_chain()
    assert np.allclose(np.asarray(lg.P.sum(1)).ravel(), 1.0)                       # stochastic matrix


def test_ledger_rotation_is_spe_with_fine(p3):
    lg = LedgerGame(3, *p3, lock_dec=4, theta=8, lam=2)
    r = lg.analyse(0.995, tol=1e-6)
    assert r["residual"] < 1e-9
    assert r["spe_on_path"] and r["eps_on_path"] <= 1e-6                           # no profitable one-shot deviation on path
    assert r["spe_unsat"]                                                           # nor in any subgame with an unclipped ledger
    c = lg.avg_cost(lg.on_path())
    fx = fixed_leader_cost(p3[0])
    assert np.ptp(c) < 0.01                                                         # equal burden for all birds
    assert 0.0 < c.mean() / fx["mean"] - 1 < 0.05                                   # rotation costs a little more (kappa=0)


def test_pure_time_ledger_without_fine_is_not_equilibrium(p3):
    lg = LedgerGame(3, *p3, lock_dec=4, theta=8, lam=0)
    r = lg.analyse(0.99, tol=1e-6)
    assert not r["spe_on_path"] and r["eps_on_path"] > 0.05                         # the follower profits by refusing to take over
    assert r["worst_path"]["action"] == 0
