"""M1a #19 known-answer test: scripted agents + null agents in the REAL environment -> metric pipeline -> confusion
matrix.  Thresholds are the pre-registered null 99th percentiles (analysis/known_answer.py); the matrix is written to
runs/known_answer/ and notes/impl/M1a_stage2_confusion.md."""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from animalbench.analysis import known_answer as ka
from animalbench.analysis.batch import run_batch
from animalbench.scenarios.geese import GeeseConfig, SCRIPTED, build_obs, layout, run_episode
from animalbench.scenarios.geese.env import reset
from animalbench.scenarios.geese.swap import action_mask
from animalbench.core import compile_schedule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "known_answer")
N_SEEDS = 100


@pytest.fixture(scope="module")
def res():
    r = ka.evaluate(n_seeds=N_SEEDS, K=100)
    os.makedirs(RUNS, exist_ok=True)
    ka.write_report(r, ka.CLASSES7, os.path.join(RUNS, "confusion.json"),
                    os.path.join(ROOT, "notes", "impl", "M1a_stage2_confusion.md"))
    json.dump(r["feats"], open(os.path.join(RUNS, "features.json"), "w"), default=float)
    return r


def diag(res, rule, tag="test"):
    return res[rule][tag]["diag"]


def test_19_scripted_agents_run_in_real_env_and_have_their_signature(res):
    f = res["feats"]
    med = lambda c, k: float(np.nanmedian([x[k] for x in f[c]]))
    assert med("S1_treadmill", "rate") >= 1 / 20 and med("S1_treadmill", "support") <= 3          # treadmill: front 3 only
    assert med("S3_reciprocity", "support") == 2 and med("S3_reciprocity", "return_rate") > 0.9     # ledger ping-pong
    assert med("S2_state_rotation", "support") >= 4 and med("S2_state_rotation", "rate") < 1 / 20   # slow, wide rotation
    assert med("S2f_state_rotation_fair", "rls") > 0.9                                              # fair rotation
    assert med("N_cyc", "rls") == 1.0 and med("N_fix", "rate") == 0.0
    # null agents' handover rates are matched to the reciprocity agent (spec: same rate)
    assert med("N_cyc", "rate") == pytest.approx(med("S3_reciprocity", "rate"), rel=0.1)
    assert med("N_rnd", "rate") == pytest.approx(med("S3_reciprocity", "rate"), rel=0.1)


@pytest.mark.parametrize("name", ["S1_treadmill", "S2_state_rotation", "S3_reciprocity", "N_cyc", "N_rnd"])
def test_19_run_episode_matches_batch(name):
    cfg, pol = ka.make_cfg(name)
    st, traj = run_episode(cfg, pol, seed=3, N=9, events=ka.EVENTS)
    b = run_batch(cfg, pol, [3], 9, events=ka.EVENTS, chunk=1)
    assert np.array_equal(np.asarray(traj.global_fields["leader"]), b["leader"][0])
    assert traj.global_fields["D"].shape == (300, 9, 9)


@pytest.mark.parametrize("name", ["S1_treadmill", "S2_state_rotation"])
def test_19_anonymous_agents_use_L0_obs_only(name):
    """S1/S2 are anonymous: NaN in every non-L0 obs column (command, identity) must not change the output."""
    cfg, pol = ka.make_cfg(name, init_mode="v")
    st = reset(cfg, jax.random.PRNGKey(0), 9, compile_schedule([], cfg.e_max))
    st = st._replace(lock=jnp.zeros_like(st.lock))
    obs = build_obs(cfg, st)
    mask = action_mask(cfg, st)
    lay = layout(cfg)
    ps = pol.init(st.key_policy, cfg)
    a0, _ = pol.act(ps, obs, st, mask, st.key_policy)
    a1, _ = pol.act(ps, obs.at[:, lay["L0_end"][0]:].set(jnp.nan), st, mask, st.key_policy)
    assert np.array_equal(np.asarray(a0), np.asarray(a1))


def test_19_s3_ledger_pingpong_flips():
    """S3: apex hands over to bird b, then b is relieved by the debtor (the previous apex) -> strict 2-bird alternation."""
    cfg, pol = ka.make_cfg("S3_reciprocity")
    r = run_batch(cfg, pol, [0, 1, 2], 9, events=ka.EVENTS, chunk=3)
    for lead in r["leader"]:
        ids = lead[np.r_[True, lead[1:] != lead[:-1]]]
        assert len(ids) >= 8 and len(set(ids.tolist())) == 2
        assert np.all(ids[2:] == ids[:-2])


def test_19_confusion_matrix_written(res):
    assert os.path.exists(os.path.join(RUNS, "confusion.json"))
    assert os.path.exists(os.path.join(ROOT, "notes", "impl", "M1a_stage2_confusion.md"))
    M = np.array(res["spec"]["test"]["M"])
    assert M.sum(1).tolist() == [N_SEEDS // 2] * len(ka.CLASSES7)


# ------------------------------------------------------------------ what the pre-registered rule does separate
def test_19_treadmill_detected(res):
    assert diag(res, "spec")["S1_treadmill"] >= 0.90


def test_19_fair_state_rotation_detected(res):
    assert diag(res, "spec")["S2f_state_rotation_fair"] >= 0.90


def test_19_ncyc_is_fair_rotation_class_not_a_null(res):
    """v0.4: N-cyc's single role is the formal-level fair-rotation reference (true label state_rotation, same as S2f)."""
    assert ka.TRUE_LABEL["N_cyc"] == "state_rotation"
    assert diag(res, "spec")["N_cyc"] >= 0.90
    assert diag(res, "spec")["S2f_state_rotation_fair"] >= 0.90


@pytest.mark.parametrize("c", ["N_rnd", "N_fix"])
def test_19_random_and_fixed_nulls_are_none(res, c):
    assert diag(res, "spec")[c] >= 0.90


# ------------------------------------------------------------------ the acceptance criterion itself
@pytest.mark.xfail(strict=True, reason="spec v0.4 #19 (observational part) still not met (reported, not tuned): anonymous "
                   "mutual state rotation S2 is read as 'none' (49/50; RLS at the N-rnd level) and the 2-bird ledger "
                   "ping-pong S3 is mostly 'none' (45/50; PSI at the N-perm 99th pct, rho_D never above null).  N-cyc is now "
                   "correctly the fair-rotation class (diag 1.0).  Observational RLS/PSI/Var(B)/rho_D stay 'unvalidated' "
                   "(L3 = identifiability negative result); identity dependence is decided by interventions I1/I2.")
def test_19_acceptance_all_diagonals_ge_90(res):
    d = diag(res, "spec")
    assert all(d[c] >= 0.90 for c in ka.CLASSES6)


def test_19_threshold_fit_is_out_of_sample(res):
    """thresholds use only the calibration (even) seeds of the null agents, the headline matrix uses odd seeds."""
    assert res["rho_thr"] == pytest.approx(ka.fit_rho_threshold({c: res["feats"][c][0::2] for c in ka.CLASSES7}))
