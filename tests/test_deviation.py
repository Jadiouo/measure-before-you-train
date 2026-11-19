"""M1a #12-#14: one-shot deviation tests (analysis/deviation.py).  Seeds are paired (same wind / start)."""
import json
import os

import numpy as np
import pytest

from animalbench.analysis import deviation as dv
from animalbench.analysis.batch import run_batch
from animalbench.scenarios.geese import GeeseConfig, stay_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = 60.0


@pytest.fixture(scope="module")
def swap_json():
    return json.load(open(os.path.join(ROOT, "calib", "out", "swap_cost.json")))


@pytest.fixture(scope="module")
def sp_json():
    p = os.path.join(ROOT, "calib", "out", "small_game_params.json")
    if not os.path.exists(p):
        from animalbench.analysis.slot_params import main
        main()
    return json.load(open(p))


@pytest.fixture(scope="module")
def apex():
    """100 paired seeds with OU turbulence (so the CI is not degenerate) and the calm deterministic version."""
    return {(o, noisy): dv.advantage(o, "apex", seeds=range(100), noisy=noisy) for o in ("stay", "rule_blind")
            for noisy in (True, False)}


# ------------------------------------------------------------------------------------------------ #12
@pytest.mark.parametrize("others", ["stay", "rule_blind"])
def test_12_apex_retreat_advantage_positive(apex, others):
    for noisy in (True, False):
        r = apex[(others, noisy)]
        assert r["mean"] > 0
        assert r["ci"][0] > 0                                      # bootstrap 95 % CI lower bound > 0


def test_12_slot1_advance_is_negative():
    r = dv.advantage("stay", "slot1", seeds=range(100))
    assert r["mean"] < 0 and r["ci"][1] < 0
    # with rotating others: the slot-1 bird advances early (apex still within its hold time)
    r2 = dv.advantage("rule_blind", "slot1", seeds=range(20), d0=6)
    assert r2["mean"] < 0


def test_12_matches_formula_with_individual_swap_cost(apex, sp_json):
    """Advantage = (p_apex - p_slot1) * H - c_retreater, with the retreater's own transient cost measured in the real
    environment (intercept definition, analysis/slot_params.py).  Holds to ~1e-3."""
    p = sp_json["9"]["p"]
    c_ret = sp_json["9"]["edges"]["1"]["cost"][0]
    pred = (p[0] - p[1]) * H - c_ret
    assert apex[("stay", False)]["mean"] == pytest.approx(pred, rel=0.01)
    # rotating others: the bystander cost of the follow-up swap (apex -> other arm) is paid by the test bird as well
    c_by = sp_json["9"]["edges"]["2"]["cost"][1]
    assert apex[("rule_blind", False)]["mean"] == pytest.approx(pred - c_by, rel=0.02)


def test_12_v03_group_cost_formula_underpredicts(apex, swap_json):
    """Why v0.4 changed the formula: the v0.3 form 0.125*(H-t_swap)-c_swap with the GROUP c_swap of #8 (all birds' extra
    energy) under-predicts the retreater's own advantage by ~19 % (the advancer and the bystanders pay most of it)."""
    pred = 0.125 * (H - swap_json["t_swap_apex"]) - swap_json["c_swap_apex"]
    meas = apex[("stay", False)]["mean"]
    assert 0.15 < (meas - pred) / meas < 0.25


# ------------------------------------------------------------------------------------------------ #13
def test_13_group_mean_equal_but_individual_advantage_positive(apex):
    fixed = dv.mean_individual_return("stay", seeds=range(10))
    rot = dv.mean_individual_return("rule_blind", seeds=range(10))
    assert rot["n_swaps"].mean() > 10 and fixed["n_swaps"].mean() == 0
    assert abs(rot["mean"] - fixed["mean"]) / abs(fixed["mean"]) < 0.01          # group mean equal (< 1 %)
    assert rot["mean"] < fixed["mean"]                                           # ...up to the swap costs
    assert apex[("stay", False)]["mean"] > 0                                     # both sentences hold at once


def test_13_kappa_advantage_monotone():
    adv = [dv.advantage("stay", "apex", seeds=range(5), kappa=k)["mean"] for k in (0.0, 0.5, 1.0, 2.0)]
    assert all(b >= a for a, b in zip(adv, adv[1:]))
    assert adv[-1] > adv[0] * 1.5


# ------------------------------------------------------------------------------------------------ #14
def test_14_kappa_none_equals_12_and_pressure_increases(apex):
    base = dv.advantage("stay", "apex", seeds=range(5), kappa=0.0)["mean"]
    assert base == pytest.approx(apex[("stay", False)]["mean"], abs=1e-9)
    assert dv.advantage("stay", "apex", seeds=range(5), kappa=1.0)["mean"] > base


def test_14_nonbinding_E0_equals_no_E0():
    a = dv.e0_advantage(None, seeds=range(3))
    b = dv.e0_advantage(2.0, seeds=range(3))
    assert np.allclose(a["adv"], b["adv"], atol=1e-9)


def test_14_E0_death_penalty_is_remaining_time_times_solo_cost():
    """v0.4: after an E0 death every remaining low step pays e0_death_cost (= solo cost 1.0 P_solo): death is no longer
    better than staying alive.  Checked on the reward stream of a bird that dies of the cap (others stay)."""
    cfg = GeeseConfig(n_max=9, T=600.0, log_low=False, log_D=False, init_mode="v", E0_coef=0.85)
    r = run_batch(cfg, stay_policy(cfg), [0], 9, keep=("reward", "alive"))
    rew, alive = r["reward"][0], r["alive"][0]                  # (300, 9)
    apex = 0
    dead = ~alive[:, apex]
    assert dead.any() and dead[-1] and not dead[0]
    first = int(np.argmax(dead))
    assert 0 < first < 299
    assert np.allclose(rew[dead, apex], -1.0, atol=1e-12)        # -1 per decision (mean of 20 low steps of -1)
    assert rew[first - 1, apex] == pytest.approx(-1.0, abs=0.02)   # the apex flew at ~solo power just before
    # legacy semantics (flag off): reward 0 after death -> death beat living (the v0.3 bug)
    cfg0 = GeeseConfig(n_max=9, T=600.0, log_low=False, log_D=False, init_mode="v", E0_coef=0.85, e0_death_penalty=False)
    r0 = run_batch(cfg0, stay_policy(cfg0), [0], 9, keep=("reward", "alive"))
    assert np.all(r0["reward"][0][~r0["alive"][0][:, apex], apex] == 0.0)


def test_14_E0_binding_retreat_advantage_positive_and_survival_gain():
    none = dv.e0_advantage(None, seeds=range(6))
    for c in (0.95, 0.90, 0.85):
        r = dv.e0_advantage(c, seeds=range(6))
        assert r["alive_gain"].mean() > 5.0                       # retreating keeps the bird alive longer
        assert r["adv"].mean() > 10.0                             # and now pays (legacy flag: ~0 for c<=0.90)
        assert r["adv"].mean() <= none["adv"].mean() + 1e-9       # honest: the cap does NOT add reward advantage
    old = dv.e0_advantage(0.90, seeds=range(6), e0_death_penalty=False)
    assert old["adv"].mean() < 1.0                                 # the v0.3 bug, kept as a regression record


@pytest.mark.xfail(strict=True, reason="spec v0.4 #14: 'E0 pressure does not lower the #12 advantage (monotone non-decreasing)' "
                   "does NOT hold even with the death penalty = remaining time x solo cost: advantage 24.7 (no cap) -> 21.8 "
                   "(0.95) -> 18.2 (0.90) -> 14.1 (0.85).  Dying costs the same as leading (-1/step), so the advantage "
                   "(1-p)*alive_time shrinks with alive_time.  Larger death costs (2, 3) give 37/36/28 and 52/55/42, still "
                   "non-monotone.  E0 is therefore NOT a pressure knob; it stays a survival-analysis ablation.")
def test_14_E0_pressure_increases_reward_advantage():
    adv = [dv.e0_advantage(c, seeds=range(6))["adv"].mean() for c in (None, 0.95, 0.90, 0.85)]
    assert all(b >= a - 1e-9 for a, b in zip(adv, adv[1:]))
