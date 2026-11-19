"""v0.4 intervention interface: identity relabel event (I1) and mid-episode immune/free-rider event (I2).
The environment only relabels / sets immunity and logs it; policy memory is the policy's business (flag `mem_reset`)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from animalbench.analysis.batch import run_batch
from animalbench.core import compile_schedule, immune_on, relabel, swap_pair_perm
from animalbench.scenarios.geese import GeeseConfig, make_rollout, rule_blind, run_episode, stay_policy
from animalbench.scenarios.geese.env import reset
from animalbench.scenarios.geese.obs import build_obs, layout

N = 9


def cfg_(**kw):
    kw.setdefault("identity", "nbr+self")
    return GeeseConfig(n_max=N, T=200.0, log_low=False, log_D=False, init_mode="v", **kw)


def run(cfg, events, seed=0, policy=None):
    pol = policy or rule_blind(cfg)
    return run_episode(cfg, pol, seed, N, events=events)


# ------------------------------------------------------------------ I1: relabel
def test_relabel_default_is_derangement_and_logged():
    cfg = cfg_()
    for seed in range(5):
        _, tr = run(cfg, [relabel(100.0)], seed)
        own = tr.agent_fields["id_owner"]
        assert np.array_equal(own[49], np.arange(N))                       # before (t < 100 s -> decision 50)
        assert sorted(own[-1].tolist()) == list(range(N))                  # still a permutation
        assert np.all(own[-1] != np.arange(N))                             # no fixed point
        assert tr.global_fields["relabel_n"][49] == 0 and tr.global_fields["relabel_n"][-1] == 1
        assert np.all(own[50:] == own[-1])                                 # a single event
        assert tr.events["perm"].shape[1] >= N


def test_relabel_codes_follow_permutation_and_obs_changes():
    cfg = cfg_()
    ev = [relabel(100.0)]
    sched = compile_schedule(ev, cfg.e_max)
    f = jax.jit(make_rollout(cfg, stay_policy(cfg), n_dec=60))
    st, _ = f(jax.random.PRNGKey(3), jnp.int32(N), sched)
    st0 = reset(cfg, jax.random.PRNGKey(3), N, sched)
    own = np.asarray(st.id_owner)
    assert np.array_equal(np.asarray(st.id_code), np.asarray(st0.id_code)[own])   # bird i carries the code of bird own[i]
    lay = layout(cfg)
    a, b = lay["identity"]
    o0 = np.asarray(build_obs(cfg, st0))[:, a:b]
    o1 = np.asarray(build_obs(cfg, st._replace(slot=st0.slot, alive=st0.alive)))[:, a:b]
    assert np.abs(o0[:, :3] - o1[:, :3]).max() > 1e-3                      # own-code block moved


def test_relabel_does_not_touch_physics_for_anonymous_policy():
    """rule_blind reads L0 only, so relabel / sham change nothing but the logged codes (bitwise)."""
    cfg = cfg_()
    _, base = run(cfg, [])
    for ev in (relabel(100.0), relabel(100.0, perm=list(range(N))), relabel(100.0, reset_mem=True)):
        _, tr = run(cfg, [ev])
        for k in ("slot", "P_mean", "reward", "swapped"):
            assert np.array_equal(tr.agent_fields[k], base.agent_fields[k]), k


def test_sham_is_identity_but_still_logged():
    cfg = cfg_()
    _, tr = run(cfg, [relabel(100.0, perm=list(range(N)))])
    assert np.array_equal(tr.agent_fields["id_owner"][-1], np.arange(N))
    assert tr.global_fields["relabel_n"][-1] == 1


def test_pair_swap_and_composition():
    cfg = cfg_()
    ev = [relabel(60.0, perm=swap_pair_perm(N, 2, 5)), relabel(120.0, perm=swap_pair_perm(N, 5, 7))]
    _, tr = run(cfg, ev)
    own = tr.agent_fields["id_owner"]
    exp1 = np.arange(N); exp1[[2, 5]] = [5, 2]
    assert np.array_equal(own[31], exp1) and np.array_equal(own[29], np.arange(N))
    assert np.array_equal(own[-1], exp1[swap_pair_perm(N, 5, 7)])
    with pytest.raises(ValueError):
        compile_schedule([relabel(1.0, perm=[0, 0, 1])], 8)


def test_mem_reset_flag_is_one_shot_and_only_when_requested():
    cfg = cfg_()
    _, tr = run(cfg, [relabel(100.0, reset_mem=True)])
    f = tr.global_fields["mem_reset"]
    assert f.sum() == 1 and f[50]                                          # event inside decision 50 -> visible to the policy at 51
    _, tr2 = run(cfg, [relabel(100.0, reset_mem=False)])
    assert tr2.global_fields["mem_reset"].sum() == 0


def test_policy_sees_mem_reset_before_acting():
    """A probe policy that copies st.mem_reset into its action: the flag is visible at decision 51 only."""
    from animalbench.scenarios.geese import Policy
    cfg = cfg_()
    pol = Policy("probe", lambda key, c=None: dict(),
                 lambda ps, obs, st, mask, key=None: (jnp.full((N,), st.mem_reset.astype(jnp.int32)), ps))
    sched = compile_schedule([relabel(100.0, reset_mem=True)], cfg.e_max)
    _, recs = jax.jit(make_rollout(cfg, pol, n_dec=80))(jax.random.PRNGKey(0), jnp.int32(N), sched)
    seen = np.asarray(recs["action_req"])[:, 0]
    assert seen.sum() == 1 and seen[51] == 1 and not bool(np.asarray(recs["mem_reset"])[49])


def test_relabel_fixed_shape_single_trace_and_vmap():
    cfg = cfg_()
    f = make_rollout(cfg, rule_blind(cfg), n_dec=60)
    g = jax.jit(f)
    for evs, n in (([], 9), ([relabel(40.0)], 9), ([relabel(40.0), immune_on(20.0, "last")], 6),
                   ([relabel(40.0, perm=swap_pair_perm(6, 0, 1)), relabel(60.0)], 6)):
        g(jax.random.PRNGKey(0), jnp.int32(n), compile_schedule(evs, cfg.e_max))
    assert f.trace_count[0] == 1
    # vmap over seeds == single seed
    r = run_batch(cfg, rule_blind(cfg), [0, 1], N, n_dec=60, events=[relabel(40.0)], chunk=2,
                  keep=("id_owner", "slot"))
    _, tr = run_episode(cfg, rule_blind(cfg), 1, N, events=[relabel(40.0)], n_dec=60)
    assert np.array_equal(r["id_owner"][1], tr.agent_fields["id_owner"])
    # different N: dead/dormant slots map to themselves, derangement among the 6 alive
    _, t6 = run_episode(cfg, rule_blind(cfg), 0, 6, events=[relabel(40.0)], n_dec=60)
    own = t6.agent_fields["id_owner"][-1]
    assert sorted(own[:6].tolist()) == list(range(6)) and np.all(own[:6] != np.arange(6))
    assert np.array_equal(own[6:], np.arange(6, N))


# ------------------------------------------------------------------ I2: free-rider mid-episode
def test_immune_event_mid_episode_stops_swaps_of_target():
    cfg = cfg_()
    _, base = run(cfg, [])
    t_on = 100.0
    _, tr = run(cfg, [immune_on(t_on, "leader")])
    d = int(t_on / cfg.delta_h)
    slot = tr.agent_fields["slot"]
    lead_id = int(np.argmin(slot[d + 1]))
    assert np.array_equal(slot[: d + 1], base.agent_fields["slot"][: d + 1])      # identical before the event
    imm = tr.agent_fields["immune"]
    assert not imm[: d].any() and imm[d:, lead_id].all() and imm[d:].sum() == imm.shape[0] - d
    # the immune bird never moves again; in the base run the rotation would have moved it
    assert np.all(slot[d + 1:, lead_id] == slot[d + 1, lead_id])
    assert (base.agent_fields["slot"][d + 1:, lead_id] != base.agent_fields["slot"][d + 1, lead_id]).any()
    assert not tr.agent_fields["swapped"][d + 1:, lead_id].any()


def test_immune_last_target_and_off_switch():
    cfg = cfg_()
    _, tr = run(cfg, [immune_on(40.0, "last"), immune_on(120.0, 8, on=False)])
    imm = tr.agent_fields["immune"]
    assert imm[25:55, 8].all() and imm[:, :8].sum() == 0 and not imm[70:, 8].any()
    # free-rider (last slot at event time) never leads while immune
    slot = tr.agent_fields["slot"]
    assert (slot[20:, 8] > 0).all()


def test_immune_event_vs_initial_immune_argument_agree_after_t0():
    """Event at the first decision boundary + 1 decision: same effect as the rollout-start `immune=` mask for
    a bird that never swaps (stay policy keeps everything fixed; the rotation policy shows it)."""
    cfg = cfg_()
    mask = np.zeros(N, bool); mask[0] = True
    _, a = run_episode(cfg, rule_blind(cfg), 0, N, immune=mask)
    _, b = run_episode(cfg, rule_blind(cfg), 0, N, events=[immune_on(0.0, 0)])
    assert np.array_equal(a.agent_fields["slot"][1:], b.agent_fields["slot"][1:])
