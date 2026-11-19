"""Interventional classifier (metrics/intervention.py) + the code-keyed scripted agents (scenarios/geese/scripted_id.py) +
the learned-policy adapter.  Fast version (CPU, short episodes).  The full 50-episode known-answer run is
scripts/m2gate_rollout.py + scripts/m2gate_classify.py (notes/impl/M2gate_intervention.md)."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from animalbench.analysis import known_answer as ka
from animalbench.core import compile_schedule, immune_on, relabel
from animalbench.metrics import intervention as iv
from animalbench.scenarios.geese import GeeseConfig, SCRIPTED, make_rollout
from animalbench.scenarios.geese.scripted_id import REGISTRY_ID

N = 9
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rollouts(name, event_lists, seeds=(0, 1), T=200.0, registry=None):
    """leader/immune records for several event lists in ONE compile: -> dict cond_idx -> (leader (S,n_dec), recs)."""
    mk, rule = (registry or {**SCRIPTED, **REGISTRY_ID})[name]
    cfg = GeeseConfig(n_max=N, T=T, swap_rule=rule, log_low=False, log_D=False, init_mode="scatter")
    f = make_rollout(cfg, mk(cfg))
    g = jax.jit(jax.vmap(lambda k, s: f(k, jnp.int32(N), s)))
    rows = [(i, s) for i in range(len(event_lists)) for s in seeds]
    sch = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *[compile_schedule(list(ka.EVENTS) + event_lists[i], cfg.e_max) for i, _ in rows])
    _, r = g(jnp.stack([jax.random.PRNGKey(s) for _, s in rows]), sch)
    L = np.asarray(r["leader"]).reshape(len(event_lists), len(seeds), -1)
    return L, {k: np.asarray(v).reshape((len(event_lists), len(seeds)) + v.shape[1:]) for k, v in r.items() if k in ("slot", "immune", "alive")}


# ------------------------------------------------------------------ pure statistics
def pingpong(n_ho, gap=15, a=0, b=1):
    out = []
    for k in range(n_ho + 1):
        out += [a if k % 2 == 0 else b] * gap
    return np.array(out)


def test_q_return_pingpong_is_one_and_cycle_is_zero():
    L = pingpong(10)
    W = iv.window_W(L, 2.0, t_event=2.0 * len(L))
    assert W >= int(iv.W_FLOOR / 2.0)
    assert iv.q_return(L, 0, len(L), W) == 1.0 or iv.q_return(L, 0, len(L), W) == pytest.approx(9 / 10)   # first handover has no credit
    cyc = np.repeat(np.arange(9), 15)
    assert iv.q_return(cyc, 0, len(cyc), 40) == 0.0
    assert np.isnan(iv.q_return(np.zeros(50, int), 0, 50, 20))


def test_q_return_window_and_W_respect_credit_horizon():
    L = np.r_[np.zeros(10, int), np.ones(100, int), np.zeros(10, int)]      # 0 -> 1 at 10, 1 -> 0 at 110
    assert iv.q_return(L, 100, 120, W=99) == 0.0       # repayment later than W => no credit
    assert iv.q_return(L, 100, 120, W=100) == 1.0


def test_null_q_perm_below_pingpong():
    L = pingpong(8)
    rng = np.random.default_rng(0)
    nd = iv.null_q(L, 0, len(L), 40, 9, rng, K=200)
    assert np.nanquantile(nd["perm"], 0.99) < 1.0
    assert np.nanmedian(nd["rnd"]) < 0.6


def test_i1_label_rules_are_the_spec_ones():
    base = dict(q_pre=1.0, null99=0.4, null95=0.3, null_med=0.1, q_post_R=0.9, q_post_S=0.95, delta_R=0.1, delta_S=0.05,
                delta_M=0.1, delta_RM=0.1)
    assert iv.i1_label(base) == "state-driven"
    assert iv.i1_label(dict(base, q_post_R=0.2, delta_R=0.9)) == "identity"
    assert iv.i1_label(dict(base, q_post_R=0.2, delta_R=0.9, delta_S=0.5)) == "mixed"          # sham moved too
    assert iv.i1_label(dict(base, q_pre=0.2)) == "no-signal"                                    # premise fails
    assert iv.i1_label(dict(base, delta_R=0.35, q_post_R=0.8)) == "mixed"
    assert iv.i1_memory_sublabel(dict(delta_R=0.9, delta_M=0.85)) == "memory"
    assert iv.i1_memory_sublabel(dict(delta_R=0.9, delta_M=0.05)) == "lookup"


def test_thresholds_are_the_preregistered_values():
    assert (iv.DELTA_ID, iv.DELTA_SHAM, iv.MEM_CLOSE, iv.I2_DROP, iv.I2_TOL, iv.P_PREMISE, iv.P_RETURN_IN, iv.K_NULL) == \
        (0.5, 0.2, 0.2, 0.5, 0.2, 0.99, 0.95, 1000)
    assert (iv.PRE, iv.POST, iv.T_EVENT, iv.W_FLOOR, iv.W_MULT) == ((60.0, 300.0), (360.0, 600.0), 300.0, 40.0, 4.0)


def test_i2_label_rules():
    f = dict(y_pre=2.0, ls_pre=3, dy_F=-0.8, dy_S=0.0, dls_F=-0.3, bstar_ok=True)
    assert iv.i2_label(f) == "punish"
    assert iv.i2_label(dict(f, bstar_ok=False, dls_F=0.0, dy_F=-0.8)) == "mixed"
    assert iv.i2_label(dict(f, dy_F=0.05, dls_F=0.0)) == "tolerate"
    assert iv.i2_label(dict(f, y_pre=0.0)) == "no-signal"
    assert iv.i2_label(dict(f, ls_pre=1)) == "no-signal"
    assert iv.i2_label(dict(f, dy_F=-0.3, dls_F=0.0)) == "mixed"


def test_bstar_check_sign():
    D = np.zeros((100, 3, 3)); D[:, :, 2] = 0.5            # everybody gives bird 2 savings, bird 2 gives nothing
    ok, m = iv.bstar_check(D, 2, 0, 100, np.random.default_rng(0))
    assert ok and m < 0
    ok0, _ = iv.bstar_check(D, 0, 0, 100, np.random.default_rng(0))
    assert not ok0                                          # bird 0 only gives -> B* > 0


# ------------------------------------------------------------------ code-keyed agents in the real env
@pytest.fixture(scope="module")
def s3c_runs():
    ev = [[], [relabel(100.0, perm=list(range(N)))], [relabel(100.0)], [relabel(100.0, perm=list(range(N)), reset_mem=True)]]
    return rollouts("S3c_reciprocity_code", ev, seeds=(0, 1))


def test_s3c_equals_s3_when_codes_never_change():
    L, _ = rollouts("S3c_reciprocity_code", [[]], seeds=(0,))
    Lo, _ = rollouts("S3_reciprocity", [[]], seeds=(0,))
    assert np.array_equal(L, Lo)


def test_s3c_sham_is_bitwise_baseline_but_relabel_and_memreset_change_the_rotation(s3c_runs):
    L, _ = s3c_runs
    assert np.array_equal(L[1], L[0])                                   # sham
    assert not np.array_equal(L[2], L[0])                               # relabel: ledger points to the wrong birds
    assert np.array_equal(L[2][:, :50], L[0][:, :50])                   # nothing before the event
    assert not np.array_equal(L[3], L[0])                               # memory wipe alone also changes it


def test_anonymous_agents_are_unchanged_by_relabel():
    L, _ = rollouts("S1_treadmill", [[relabel(100.0, perm=list(range(N)))], [relabel(100.0)]], seeds=(0,))
    assert np.array_equal(L[0], L[1])                                   # identity=none-like control (X): bitwise


def test_queue_agent_ia_equals_literal_when_nobody_is_immune():
    for nm, ia in (("N_cyc", "N_cyc_ia"), ("S2f_state_rotation_fair", "S2f_ia")):
        a, _ = rollouts(nm, [[]], seeds=(0,))
        b, _ = rollouts(ia, [[]], seeds=(0,))
        assert np.array_equal(a, b), nm


def test_literal_queue_agent_deadlocks_on_immune_bird_and_ia_does_not():
    ev = [[immune_on(100.0, "last")]]
    lit, r = rollouts("S2f_state_rotation_fair", ev, seeds=(0,))
    ia, _ = rollouts("S2f_ia", ev, seeds=(0,))
    def n_ho(x, a=60, b=100):
        return int((x[0, 0, a + 1:b] != x[0, 0, a:b - 1]).sum())
    assert n_ho(ia) >= 2
    assert n_ho(lit, 80, 100) <= n_ho(ia, 80, 100)


# ------------------------------------------------------------------ learned-policy adapter (M1.5)
@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "runs", "m15", "unilateral_shared_s0", "final.ckpt")), reason="no m15 ckpt")
def test_m15_policy_adapter_is_blind_to_relabel():
    from animalbench.marl.policy_adapter import load_marl_policy
    cfg, pol = load_marl_policy(os.path.join(ROOT, "runs", "m15", "unilateral_shared_s0", "final.ckpt"))
    assert cfg.identity == "none" and cfg.n_max == 3
    f = make_rollout(cfg, pol, n_dec=60)
    g = jax.jit(f)
    outs = []
    for ev in ([], [relabel(40.0)], [relabel(40.0, reset_mem=True)]):
        _, r = g(jax.random.PRNGKey(0), jnp.int32(3), compile_schedule(ev, cfg.e_max))
        outs.append(np.asarray(r["leader"]))
    assert np.array_equal(outs[0], outs[1])            # identity=none policy: I1-X control is a bitwise no-op
    assert (np.diff(outs[0]) != 0).sum() > 5           # and it does rotate


# ------------------------------------------------------------------ false-positive / lookup agents and the frozen V3 / block rules
@pytest.fixture(scope="module")
def pair_runs():
    ev = [[], [relabel(60.0, perm=list(range(N)))], [relabel(60.0)], [relabel(60.0, perm=list(range(N)), reset_mem=True)]]
    return {nm: rollouts(nm, ev, seeds=(0, 1), T=200.0)[0] for nm in ("P_body", "P_body_codenoise", "P_code_lookup")}


def test_pair_agents_rotate_between_two_birds_and_sham_is_baseline(pair_runs):
    for nm, L in pair_runs.items():
        assert np.array_equal(L[1], L[0]), nm                              # sham is bitwise baseline
        for s in range(L.shape[1]):
            assert len(set(L[0, s].tolist())) == 2, nm                     # fixed pair
            assert (np.diff(L[0, s]) != 0).sum() >= 4, nm                  # and it does ping-pong


def test_p_body_ignores_relabel_and_memory_reset(pair_runs):
    L = pair_runs["P_body"]
    assert np.array_equal(L[2], L[0]) and np.array_equal(L[3], L[0])      # never reads the code: bitwise


def test_p_body_codenoise_reads_code_only_for_timing(pair_runs):
    L = pair_runs["P_body_codenoise"]
    assert np.array_equal(L[2][:, :30], L[0][:, :30])                      # nothing before the event
    assert not np.array_equal(L[2], L[0])                                  # timing moves (it does read the code)
    assert np.array_equal(L[3], L[0])                                      # no memory: reset is a no-op
    for s in range(L.shape[1]):                                            # but WHO leads is unchanged
        assert set(L[2, s].tolist()) == set(L[0, s].tolist())


def test_p_code_lookup_is_stateless(pair_runs):
    L = pair_runs["P_code_lookup"]
    assert np.array_equal(L[3], L[0])                                      # memory reset (with identity relabel) changes nothing
    assert np.array_equal(L[2][:, :30], L[0][:, :30])


def test_set_changed_and_v3_label():
    S = np.repeat([0, 1] * 60, 5)
    same_set_shifted = np.roll(S, 3)
    other_pair = np.repeat([2, 3] * 60, 5)
    assert not iv.set_changed(same_set_shifted, S, 2.0, 9)                 # timing only
    assert iv.set_changed(other_pair, S, 2.0, 9)                           # who leads changed
    f = dict(q_pre=1.0, null99=0.5, delta_S=0.0)
    assert iv.i1_label_v3(f, False) == "state-driven"
    assert iv.i1_label_v3(f, True) == "identity"
    assert iv.i1_label_v3(dict(f, delta_S=0.6), True) == "mixed"
    assert iv.i1_label_v3(dict(f, q_pre=0.2), True) == "no-signal"


def test_i2_block_label_averages_out_count_noise():
    base = dict(y_pre=2.0, ls_pre=3, dy_S=-0.2, dls_S=0.0, bstar_ok=False)
    noisy = [dict(base, dy_F=-0.2 + e, dls_F=0.0 + e) for e in (0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3)]
    assert all(iv.i2_label_v2(f) == "mixed" for f in noisy)                # episode level: noise breaks the tolerate band
    assert iv.i2_label_block(noisy) == "tolerate"                          # block of 10: averaged out
    drop = [dict(base, dy_F=-0.9, dls_F=-0.3, bstar_ok=True) for _ in range(10)]
    assert iv.i2_label_block(drop) == "punish"
    assert iv.i2_label_block([dict(base, y_pre=0.0, dy_F=0.0, dls_F=0.0)] * 10) == "no-signal"
    assert iv.BLOCK_SIZE == 10
