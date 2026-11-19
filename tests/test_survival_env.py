"""Survival-migration mode of the geese env (cfg.survival, default OFF).  Hand-computed rewards, death, arrival, alpha
endpoints, dense-shaping telescoping, and bit-identity of the old env when the switch is off."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import animalbench  # noqa
from animalbench.core import compile_schedule
from animalbench.scenarios.geese import GeeseConfig, make_rollout, reset, rule_blind, layout, stay_policy, decision_step
from animalbench.scenarios.geese import survival as surv
from animalbench.scenarios.geese.state import slot_offset

SCHED = compile_schedule([], 8)


def _h(*a):
    return hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(x)).tobytes() for x in a)).hexdigest()[:16]


# --------------------------------------------------------------------------- switch off == old env, bit for bit
# golden hashes were produced with the pre-change code (git HEAD ec2f9f0 worktree), see notes/impl/survival_ippo_prereg.md
GOLD = {"plain": ("af328bcd2cd9d0a6", "26105e5ff5101736", "27c9f9931481ecd9", "8949f3e8d7df50fd"),
        "e0": ("ff9ee9387e0a1ca4", "40edc65d1f1d7c7e", "8e0f5a5abb38eec8", "4cf89f678e24a935")}


@pytest.mark.parametrize("name,kw", [("plain", dict(n_max=5)), ("e0", dict(n_max=5, E0_coef=0.55, swap_rule="mutual"))])
def test_switch_off_bitwise_old_env(name, kw):
    # survival params are set to non-default values but the switch is off => must change nothing
    cfg = GeeseConfig(T=60.0, dtype="float64", log_low=False, log_D=False, use_events=False,
                      surv_alpha=1.0, surv_burst_m=9.0, surv_h=0.3, surv_shaping="energy", **kw)
    assert not cfg.survival and layout(cfg)["L0_end"][0] == 21
    st, rec = jax.jit(make_rollout(cfg, rule_blind(cfg)))(jax.random.PRNGKey(3), jnp.int32(5), SCHED)
    got = (_h(rec["reward"]), _h(st.E_rem, st.x, st.y), _h(rec["slot"]), _h(rec["n_alive"]))
    assert got == GOLD[name]
    assert "arrived" not in rec


def _cfg(**kw):
    base = dict(n_max=3, dtype="float64", log_low=False, log_D=False, use_events=False, E0_coef=1.0, E0_sigma=0.0,
                survival=True, obs_clock=True, T=600.0)
    base.update(kw)
    return GeeseConfig(**base)


def _crafted(cfg, efrac, alive, died, late=()):
    """State at the end of the crossing decision with hand-set energy fractions."""
    st = reset(cfg, jax.random.PRNGKey(0), jnp.int32(3), SCHED)
    D = cfg.D_route
    R = jnp.array([D + 1.0, 0.0], cfg.jdtype)
    x = R[0] + slot_offset(cfg, st.slot)[:, 0]
    x = x.at[jnp.array(list(late), int)].add(-100.0) if late else x
    return st._replace(R=R, x=x, E_rem=jnp.asarray(efrac, cfg.jdtype) * st.E0, alive=jnp.asarray(alive, bool),
                       died=jnp.asarray(died, bool))


def _r(cfg, st, R0_before=None):
    R0 = cfg.D_route - 1.0 if R0_before is None else R0_before
    return surv.reward(cfg, jnp.asarray(R0, cfg.jdtype), jnp.clip(st.E_rem, 0.0, None), st)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_reward_hand_computed(alpha):
    cfg = _cfg(surv_alpha=alpha, surv_beta=0.5)
    # birds 0,1 arrive with 50% / 20% energy; bird 2 died.  indiv = [1.25, 1.10, 0]; team = 2.35/3
    st = _crafted(cfg, [0.5, 0.2, -0.01], [True, True, False], [False, False, True])
    o = _r(cfg, st)
    indiv = np.array([1.25, 1.10, 0.0])
    team = 2.35 / 3
    np.testing.assert_allclose(np.asarray(o["reward"]), (1 - alpha) * indiv + alpha * team, atol=1e-12)
    assert list(np.asarray(o["arrived"])) == [True, True, False] and bool(o["cross"])


def test_alpha_endpoints_individual_vs_team():
    st_args = ([0.5, 0.2, -0.01], [True, True, False], [False, False, True])
    r0 = np.asarray(_r(_cfg(surv_alpha=0.0), _crafted(_cfg(), *st_args))["reward"])
    r1 = np.asarray(_r(_cfg(surv_alpha=1.0), _crafted(_cfg(), *st_args))["reward"])
    assert r0[2] == 0.0 and r0[0] > r0[1] > 0                     # alpha=0: own outcome only, the dead bird gets 0
    assert np.allclose(r1, r1[0]) and abs(r1[0] - 2.35 / 3) < 1e-12   # alpha=1: everybody the same team value


def test_late_bird_and_no_cross_and_after_done():
    cfg = _cfg()
    st = _crafted(cfg, [0.5, 0.5, 0.5], [True] * 3, [False] * 3, late=(1,))
    o = _r(cfg, st)
    assert list(np.asarray(o["arrived"])) == [True, False, True]            # timeout: alive but too far behind => not arrived
    np.testing.assert_allclose(np.asarray(o["reward"]), [1.25, 0.0, 1.25], atol=1e-12)
    o2 = _r(cfg, st, R0_before=cfg.D_route + 5.0)                           # already finished earlier: reward 0 (done)
    assert float(np.abs(np.asarray(o2["reward"])).sum()) == 0.0 and not bool(o2["cross"])
    o3 = surv.reward(cfg, jnp.asarray(0.0), jnp.ones(3), st._replace(R=jnp.array([10.0, 0.0])))   # not yet there
    assert float(np.abs(np.asarray(o3["reward"])).sum()) == 0.0


def test_death_closes_v_and_leader_succession():
    cfg = _cfg(surv_burst_p=0.0)
    st = reset(cfg, jax.random.PRNGKey(1), jnp.int32(3), SCHED)
    assert list(np.asarray(st.slot)) == [0, 1, 2]
    st = st._replace(E_rem=st.E_rem.at[0].set(-1.0))                         # leader out of energy
    st = surv.kill_dying(cfg, st)
    assert not bool(st.alive[0]) and bool(st.died[0]) and int(st.slot[0]) == -1
    assert int(st.slot[1]) == 0 and int(st.slot[2]) == 2                    # slot-1 bird takes the apex; right-arm bird keeps slot 2 (existing kill_bird compaction)
    st2, rec = decision_step(cfg, st, jnp.zeros(3, jnp.int32))               # dead bird: reward 0, others alive
    assert float(rec["reward"][0]) == 0.0 and not bool(rec["arrived"].any())


def test_death_inside_step_when_energy_runs_out():
    cfg = _cfg(surv_burst_p=0.0)
    st = reset(cfg, jax.random.PRNGKey(1), jnp.int32(3), SCHED)
    st = st._replace(E_rem=st.E_rem.at[2].set(1e-3))                        # one decision costs far more
    st2, rec = decision_step(cfg, st, jnp.zeros(3, jnp.int32))
    assert not bool(st2.alive[2]) and bool(st2.died[2]) and bool(st2.alive[0]) and bool(st2.alive[1])


def test_burst_costs_energy_with_probability():
    cfg = _cfg(surv_burst_p=1.0, surv_burst_m=1.0)
    st = reset(cfg, jax.random.PRNGKey(1), jnp.int32(3), SCHED)
    st2 = surv.apply_burst(cfg, st)
    d = np.asarray(st.E_rem - st2.E_rem)
    assert (d > 0).all()                                                    # p=1 => everybody hit; extra = m*dh*P_solo*Exp(1)
    cfg0 = _cfg(surv_burst_p=0.0, surv_burst_m=1.0)
    assert float(np.abs(np.asarray(st.E_rem - surv.apply_burst(cfg0, st).E_rem)).max()) == 0.0
    cfg1 = _cfg(surv_burst_p=0.05, surv_burst_m=1.0)                         # mean extra over many draws ~ p*m*dh*P_solo
    tot = []
    for i in range(300):
        s = reset(cfg1, jax.random.PRNGKey(i), jnp.int32(3), SCHED)
        tot.append(float((s.E_rem - surv.apply_burst(cfg1, s).E_rem).sum()) / 3)
    exp = 0.05 * 1.0 * cfg1.delta_h * cfg1.P_solo
    assert abs(np.mean(tot) - exp) < 0.35 * exp


def test_full_episode_arrival_and_shaping_telescopes():
    # short route; bird 0 (E cap lowest after h<0 not used) dies in the eps=0.5 run, everybody arrives in eps=2 run
    for eps, mg in ((2.0, 0.0), (0.5, 1.0)):
        outs = {}
        for shp in ("none", "energy"):
            for al in (0.0, 0.5):
                cfg = _cfg(T=120.0, E0_coef=eps, surv_burst_m=mg, surv_h=0.05, surv_shaping=shp, surv_alpha=al)
                st, rec = jax.jit(make_rollout(cfg, stay_policy(cfg)))(jax.random.PRNGKey(5), jnp.int32(3), SCHED)
                outs[(shp, al)] = (np.asarray(rec["reward"]).sum(0), np.asarray(rec["arrived"]).sum(0), np.asarray(st.died))
        for al in (0.0, 0.5):
            a, b = outs[("none", al)], outs[("energy", al)]
            np.testing.assert_allclose(a[0], b[0], atol=1e-9)               # dense == sparse return exactly (gamma=1)
            assert (a[1] == b[1]).all()
        if eps == 2.0:
            assert outs[("none", 0.0)][1].tolist() == [1, 1, 1] and not outs[("none", 0.0)][2].any()
            assert all(r > 1.0 for r in outs[("none", 0.0)][0])
        else:
            assert outs[("none", 0.0)][2].any()                             # somebody died: energy too short
            assert (outs[("none", 0.0)][0][outs[("none", 0.0)][2]] == 0.0).all()   # dead => individual return 0


def test_reserved_interfaces_not_implemented():
    with pytest.raises(NotImplementedError):
        _cfg(surv_K=3)
    with pytest.raises(NotImplementedError):
        _cfg(surv_unknown_end=True)
    with pytest.raises(ValueError):
        GeeseConfig(survival=True)                                          # needs an energy cap


# --------------------------------------------------------------------------- v2 switches (default OFF => bit-identical)
GOLD_V2_OFF = ("1717b0fee5593b92", "b383ebc05414198b", "f6ff25f2e9ef45b2", "17069b420058c128")   # pre-change code, survival=True


def _v2_cfg(**kw):
    base = dict(n_max=3, T=120.0, dtype="float64", log_low=False, log_D=False, use_events=False, E0_coef=1.02,
                E0_sigma=0.0, survival=True, obs_clock=True, surv_h=0.05, surv_burst_m=1.0, swap_rule="mutual",
                random_slots=True)
    base.update(kw)
    return GeeseConfig(**base)


def _v2_run(cfg):
    st, rec = jax.jit(make_rollout(cfg, rule_blind(cfg), record_obs=True))(jax.random.PRNGKey(3), jnp.int32(3), SCHED)
    return st, rec, (_h(rec["reward"]), _h(st.E_rem, st.x, st.y), _h(rec["slot"]), _h(rec["obs"]))


def test_v2_switches_off_bitwise_survival_env():
    _, _, got = _v2_run(_v2_cfg())
    assert got == GOLD_V2_OFF
    _, _, got2 = _v2_run(_v2_cfg(surv_obs_abs=False, surv_pw_spread=0.0, surv_pw_fixed=True))   # fixed flag alone does nothing
    assert got2 == GOLD_V2_OFF


def test_v2_flags_ignored_when_survival_off():
    kw = dict(n_max=5, T=60.0, dtype="float64", log_low=False, log_D=False, use_events=False,
              surv_obs_abs=True, surv_pw_spread=0.05)
    cfg = GeeseConfig(**kw)
    st, rec = jax.jit(make_rollout(cfg, rule_blind(cfg)))(jax.random.PRNGKey(3), jnp.int32(5), SCHED)
    assert (_h(rec["reward"]), _h(st.E_rem, st.x, st.y), _h(rec["slot"]), _h(rec["n_alive"])) == GOLD["plain"]


def test_v2_obs_abs_energy_makes_h_visible():
    cfg = _v2_cfg(surv_obs_abs=True)
    lay = layout(cfg)
    st = reset(cfg, jax.random.PRNGKey(0), jnp.int32(3), SCHED)
    from animalbench.scenarios.geese.obs import build_obs
    o = np.asarray(build_obs(cfg, st))
    a, b = lay["e_self"]
    ref = cfg.E0_coef * cfg.P_solo * cfg.T
    np.testing.assert_allclose(o[:, a], np.asarray(st.E_rem) / ref, rtol=1e-12)
    assert np.ptp(o[:, a]) > 0.04                           # strong bird starts at 1.05 vs 1.0 (old obs: all 1.0)
    assert layout(cfg) == layout(_v2_cfg())                 # observation dimension unchanged
    o0 = np.asarray(build_obs(_v2_cfg(), st))
    np.testing.assert_allclose(o0[:, a], 1.0)


def test_v2_power_multiplier_range_fixed_vs_redrawn():
    cfg = _v2_cfg(surv_pw_spread=0.05)
    s1 = reset(cfg, jax.random.PRNGKey(1), jnp.int32(3), SCHED)
    s2 = reset(cfg, jax.random.PRNGKey(2), jnp.int32(3), SCHED)
    m1, m2 = np.asarray(s1.power_mult), np.asarray(s2.power_mult)
    assert ((m1 >= 0.95) & (m1 <= 1.05)).all() and not np.allclose(m1, 1.0)
    assert not np.allclose(m1, m2)                           # redrawn per episode
    cf = _v2_cfg(surv_pw_spread=0.05, surv_pw_fixed=True)
    f1 = np.asarray(reset(cf, jax.random.PRNGKey(1), jnp.int32(3), SCHED).power_mult)
    f2 = np.asarray(reset(cf, jax.random.PRNGKey(2), jnp.int32(3), SCHED).power_mult)
    np.testing.assert_array_equal(f1, f2)
    assert ((f1 >= 0.95) & (f1 <= 1.05)).all()


def test_v2_power_multiplier_scales_energy_use():
    cfg = _v2_cfg(surv_pw_spread=0.05, surv_burst_m=0.0)
    base = _v2_cfg(surv_burst_m=0.0)
    st, _, _ = _v2_run(cfg)
    st0, _, _ = _v2_run(base)
    pm = np.asarray(reset(cfg, jax.random.PRNGKey(3), jnp.int32(3), SCHED).power_mult)
    ratio = np.asarray(st.E_used) / np.asarray(st0.E_used)
    np.testing.assert_allclose(ratio, pm, rtol=2e-2)         # trajectories nearly identical (same slots/wake), power scales by the multiplier
