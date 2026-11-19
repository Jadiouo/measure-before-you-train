"""M1a #9 timing, #18 D_ij (env side), #22 determinism, #23 fixed shapes, #24 interface/obs, trajectory I/O."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from animalbench.core import Trajectory, compile_schedule, crosswind, injury, join, leave
from animalbench.scenarios.geese import (GeeseConfig, build_obs, layout, make_rollout, obs_dim, rule_blind, stay_policy,
                                         table_policy, run_episode, resolve_swaps, decision_step, llc, step_continuous)
from animalbench.scenarios.geese import wake
from animalbench.scenarios.geese.env import reset, to_trajectory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cfg, pol, seed, N, events=(), n_dec=None, **kw):
    f = jax.jit(make_rollout(cfg, pol, n_dec, **kw))
    sched = compile_schedule(list(events), cfg.e_max)
    return f(jax.random.PRNGKey(seed), jnp.int32(N), sched)


def tonp(t):
    return jax.tree_util.tree_map(np.asarray, t)


# --------------------------------------------------------------------------- #9
def test_9_timing_and_interval_reward():
    cfg = GeeseConfig()
    assert cfg.n_dec == 300 and cfg.n_sub == 20 and cfg.lock_dec == 4
    cfg = GeeseConfig(n_max=9, log_stride=1, log_D=False, c_prog=0.3, c_iso=0.05, kappa=1.0)  # non-trivial reward
    st, rec = run(cfg, rule_blind(cfg), 0, 9, n_dec=4)
    r = tonp(rec)
    low_r = r["low"]["r"]                       # (dec, 20, n)
    assert low_r.shape[:2] == (4, 20)
    assert np.abs(low_r.mean(1) - r["reward"]).max() < 1e-9
    # obs P of the next decision == interval mean of P/P_solo
    P_low = r["low"]["P"].mean(1) / cfg.P_solo
    assert np.abs(P_low[:-1] - r["P_prev"][1:]).max() < 1e-9
    assert np.abs(P_low - r["P_mean"]).max() < 1e-9
    st1 = reset(cfg, jax.random.PRNGKey(0), 9, compile_schedule([], cfg.e_max))
    assert np.allclose(np.asarray(build_obs(cfg, st1))[:, layout(cfg)["P_prev"][0]], 1.0)


def test_9_lock_window_is_four_decisions():
    cfg = GeeseConfig(n_max=9, log_low=False, log_D=False)
    tab = np.zeros((12, 9), np.int32)
    tab[2, 0] = 1                              # apex retreats at decision 2
    tab[3:, 1] = 1                             # new apex (bird 1) keeps asking to retreat
    st, rec = run(cfg, table_policy(cfg, tab, respect_mask=False), 0, 9, n_dec=12)
    r = tonp(rec)
    sw = r["swapped"]
    assert sw[2, 0] and sw[2, 1]
    assert not sw[3:6].any()                   # locked decisions 3,4,5
    assert r["blocked"][3:6, 1].all()          # forced stay is reported as blocked
    assert sw[6, 1]                            # free again exactly t_lock=8 s after the swap
    assert r["lock"][3, 0] == 3 and r["lock"][4, 0] == 2 and r["lock"][5, 0] == 1 and r["lock"][6, 0] == 0


# --------------------------------------------------------------------------- #22
def test_22_determinism_and_pairing():
    cfg = GeeseConfig(n_max=9, T=60.0, init_mode="scatter")
    ev = [crosswind(10.0, 30.0)]
    a = tonp(run(cfg, rule_blind(cfg), 5, 9, ev)[1])
    b = tonp(run(cfg, rule_blind(cfg), 5, 9, ev)[1])
    for k in a:
        if k == "low":
            for kk in a["low"]:
                assert np.array_equal(a["low"][kk], b["low"][kk]), kk
        else:
            assert np.array_equal(a[k], b[k]), k
    c = tonp(run(cfg, stay_policy(cfg), 5, 9, ev)[1])
    # different agent, same seed: identical exogenous wind and initial state
    assert np.array_equal(a["wind_ou"], c["wind_ou"])
    assert np.array_equal(a["low"]["wind"], c["low"]["wind"])
    for k in ("x", "y", "psi", "v", "slot"):
        assert np.array_equal(a[k][0], c[k][0]), k
    d = tonp(run(cfg, rule_blind(cfg), 6, 9, ev)[1])
    assert not np.array_equal(a["x"][0], d["x"][0])
    s0 = reset(cfg, jax.random.PRNGKey(5), 9, compile_schedule(ev, cfg.e_max))
    s1 = reset(cfg, jax.random.PRNGKey(5), 9, compile_schedule(ev, cfg.e_max))
    assert all(np.array_equal(np.asarray(u), np.asarray(w)) for u, w in zip(jax.tree_util.tree_leaves(s0), jax.tree_util.tree_leaves(s1)))


# --------------------------------------------------------------------------- #23
def test_23a_single_trace_for_any_events_and_N():
    cfg = GeeseConfig(n_max=20, e_max=8, log_low=False, log_D=False)
    fn = make_rollout(cfg, rule_blind(cfg), 2)
    jf = jax.jit(fn)
    evs = [crosswind(1.0, 2.0), injury(1.0, "leader"), join(2.0, 2), leave(1.0, "random"), crosswind(2.0, 1.0),
           injury(1.0, 3), join(3.0, 1), leave(2.0, 4)]
    for ne in (0, 1, 3, 5, 8):
        for N in (3, 9, 20):
            sched = compile_schedule(evs[:ne], 8)
            out = jf(jax.random.PRNGKey(ne * 31 + N), jnp.int32(N), sched)
            jax.block_until_ready(out[1]["reward"])
    assert fn.trace_count[0] == 1
    with pytest.raises(ValueError):
        compile_schedule(evs + [crosswind(0.0)], 8)


def _bit(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_23b_dead_slot_state_does_not_leak():
    cfg = GeeseConfig(n_max=20, T=40.0, init_mode="scatter")
    pol = rule_blind(cfg)
    ev = [crosswind(5.0, 20.0), injury(10.0, 2)]
    sched = compile_schedule(ev, cfg.e_max)
    dead = jnp.arange(20) >= 9
    jx = jnp.zeros(20).at[9:].set(jnp.array([1e30, -5e9, jnp.nan, jnp.inf, 7.0, 7.0, 0.0, -1e-30, 3.0, 2.0, 1.0]))
    jy = jnp.zeros(20).at[9:].set(jnp.array([jnp.nan, 4.0, 1e12, -jnp.inf, 7.0, 7.0, 0.0, 1.0, 3.0, 2.0, 1.0]))
    junk = dict(mask=dead, x=jx, y=jy, psi=jnp.full(20, 123.0), v=jnp.full(20, jnp.nan))
    f = jax.jit(make_rollout(cfg, pol, 20, record_obs=True))
    _, a = f(jax.random.PRNGKey(2), jnp.int32(9), sched, None)
    _, b = f(jax.random.PRNGKey(2), jnp.int32(9), sched, junk)
    a, b = tonp(a), tonp(b)
    for k in ("x", "y", "psi", "v", "P_mean", "reward", "save_mean", "E_used", "e_form_end", "obs", "action_req"):
        assert _bit(a[k][:, :9], b[k][:, :9]), k
    assert _bit(a["D"][:, :9, :9], b["D"][:, :9, :9])
    assert _bit(a["low"]["P"][..., :9], b["low"]["P"][..., :9])
    for k in a["obs"].shape[:1]:
        pass
    assert not np.isnan(b["obs"][:, :9]).any() and not np.isnan(b["reward"]).any()


def test_23c_join_equals_native_bird():
    cfg = GeeseConfig(n_max=20, T=30.0, init_mode="v", init_noise=0.0, log_stride=1)
    pol = stay_policy(cfg)
    sched_j = compile_schedule([join(0.0, 1)], cfg.e_max)
    sched_n = compile_schedule([], cfg.e_max)
    b = cfg.b
    one = jnp.arange(20) == 9
    native = dict(mask=one, x=jnp.full(20, -8 * b - 10 * b), y=jnp.zeros(20), psi=jnp.zeros(20),
                  v=jnp.full(20, cfg.v0), alive=jnp.ones(20, bool), slot=jnp.full(20, 9, jnp.int32))
    f = jax.jit(make_rollout(cfg, pol, 15))
    _, a = f(jax.random.PRNGKey(0), jnp.int32(9), sched_j, None)
    _, n_ = f(jax.random.PRNGKey(0), jnp.int32(10), sched_n, native)
    a, n_ = tonp(a), tonp(n_)
    for k in ("x", "y", "psi", "v"):
        assert _bit(a["low"][k], n_["low"][k]), k            # every low-level frame, incl. the joiner's flight to its slot
    assert _bit(a["reward"][:, :10], n_["reward"][:, :10]) and _bit(a["E_used"], n_["E_used"])
    assert a["slot"][3, 9] == n_["slot"][3, 9] == 9 and a["alive"][3, 9]


def test_23d_n_max_independence():
    ev = [crosswind(5.0, 20.0), injury(8.0, 3), leave(14.0, 5)]
    outs = {}
    for nm in (9, 20):
        cfg = GeeseConfig(n_max=nm, T=40.0, init_mode="scatter")
        outs[nm] = tonp(run(cfg, rule_blind(cfg), 3, 9, ev)[1])
    a, b = outs[9], outs[20]
    for k in ("x", "y", "psi", "v", "P_mean", "reward", "save_mean", "slot", "alive", "E_used", "e_form_end"):
        assert _bit(a[k], b[k][:, :9]), k
    for k in ("x", "y", "P", "e_form", "r", "wind"):
        assert _bit(a["low"][k], b["low"][k][:, :, :9]), k
    assert _bit(a["D"], b["D"][:, :9, :9])


def test_23e_all_dead_no_nan():
    cfg = GeeseConfig(n_max=20, T=40.0, init_mode="scatter")
    ev = [leave(2.0 + 2 * i, i) for i in range(8)]
    for pol in (rule_blind(cfg), stay_policy(cfg)):
        st, rec = run(cfg, pol, 1, 8, ev)
        r = tonp(rec)
        leaves = jax.tree_util.tree_leaves(r) + jax.tree_util.tree_leaves(tonp(st))
        for lf in leaves:
            if np.issubdtype(lf.dtype, np.floating):
                assert not np.isnan(lf).any()
        assert r["n_alive"][-1] == 0 and not np.asarray(st.alive).any()
        assert np.isfinite(r["E_used"]).all()
    st, rec = run(cfg, rule_blind(cfg), 1, 0, n_dec=3)               # born empty
    assert not np.isnan(np.asarray(rec["reward"])).any()


def test_23f_vmap_matches_single():
    cfg = GeeseConfig(n_max=20, log_low=False)
    pol = rule_blind(cfg)
    fn = make_rollout(cfg, pol, 3)
    sched = compile_schedule([crosswind(1.0, 3.0)], cfg.e_max)
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    fb = jax.jit(jax.vmap(fn, in_axes=(0, None, None)))
    _, rb = fb(keys, jnp.int32(9), sched)
    rb = tonp(rb)
    assert rb["x"].shape == (256, 3, 20) and rb["D"].shape == (256, 3, 20, 20)
    fs = jax.jit(fn)
    for i in (0, 77, 255):
        _, rs = fs(keys[i], jnp.int32(9), sched)
        rs = tonp(rs)
        for k in ("x", "y", "reward", "P_mean", "D", "slot"):
            assert np.abs(rs[k].astype(float) - rb[k][i].astype(float)).max() < 1e-9, k


# --------------------------------------------------------------------------- #24
def test_24_obs_dims_and_clock():
    for mode, d_no_id, d_all in (("hier", 25, 40), ("continuous2d", 31, 46)):
        cfg = GeeseConfig(n_max=9, action_mode=mode, identity="nbr+self")
        assert obs_dim(cfg, identity=False) == d_no_id and obs_dim(cfg, identity=True) == d_all
        st = reset(cfg, jax.random.PRNGKey(0), 9, compile_schedule([], cfg.e_max))
        o = np.asarray(build_obs(cfg, st))
        assert o.shape == (9, d_all) and not np.isnan(o).any()
        lay = layout(cfg)
        assert lay["L0_end"][0] == d_no_id - 4
        assert np.abs(o[:, lay["identity"][0]:]).max() > 0                 # identity on
        cfg0 = GeeseConfig(n_max=9, action_mode=mode, identity="none")
        o0 = np.asarray(build_obs(cfg0, reset(cfg0, jax.random.PRNGKey(0), 9, compile_schedule([], cfg.e_max))))
        assert np.abs(o0[:, lay["identity"][0]:]).max() == 0               # shape unchanged, all zero
        # no clock channel: shifting the absolute time leaves the observation unchanged
        sh = st._replace(t=st.t + 1234.5, cmd_t=st.cmd_t + 1234.5, path_t0=st.path_t0 + 1234.5)
        assert np.array_equal(np.asarray(build_obs(cfg, sh)), o)
        cfgc = cfg.replace(obs_clock=True)
        stc = reset(cfgc, jax.random.PRNGKey(0), 9, compile_schedule([], cfg.e_max))
        oc = np.asarray(build_obs(cfgc, stc))
        assert oc.shape[1] == d_all + 1 and obs_dim(cfgc, identity=False) == d_no_id + 1
        occ2 = np.asarray(build_obs(cfgc, stc._replace(t=stc.t + 300.0)))
        assert not np.array_equal(oc, occ2)                               # clock only exists when switched on


def test_24_hier_equals_llc_into_continuous2d():
    cfg = GeeseConfig(n_max=9, T=30.0, init_mode="scatter", log_stride=1, log_D=False)
    ev = [crosswind(2.0, 20.0)]
    sched = compile_schedule(ev, cfg.e_max)
    tab = np.zeros((15, 9), np.int32)
    tab[8, 0] = 1                                            # one swap in the middle
    pol = table_policy(cfg, tab)
    fn = make_rollout(cfg, pol, 15)
    st_h, rec = jax.jit(fn)(jax.random.PRNGKey(4), jnp.int32(9), sched)
    rec = tonp(rec)
    a_seq = jnp.asarray(rec["low"]["a"])                     # (15, 20, n, 2) LLC outputs actually applied
    acts = jnp.asarray(rec["action_req"])

    @jax.jit
    def replay(st):
        def dec(st, xs):
            act, a20 = xs
            st, _ = resolve_swaps(cfg, st, act)
            st, _ = jax.lax.scan(lambda s, a: step_continuous(cfg, s, a), st, a20)
            return st._replace(dec=st.dec + 1,
                               lock=jnp.maximum(st.lock - 1, 0)), None
        st, _ = jax.lax.scan(dec, st, (acts, a_seq))
        return st
    st0 = reset(cfg, jax.random.PRNGKey(4), 9, sched)
    st_c = replay(st0)
    for k in ("x", "y", "psi", "v", "E_used", "R", "slot", "a_prev"):
        assert _bit(getattr(st_h, k), getattr(st_c, k)), k
    assert int(rec["swapped"].sum()) == 2


# --------------------------------------------------------------------------- #18 (environment side)
def test_18_D_ij():
    cfg = GeeseConfig(n_max=9, T=40.0, init_mode="scatter")
    st, rec = run(cfg, rule_blind(cfg), 0, 9, [crosswind(5.0, 20.0)])
    r = tonp(rec)
    D = r["D"]
    n = 9
    assert D.shape == (20, n, n)
    assert np.abs(D[:, np.arange(n), np.arange(n)]).max() == 0                      # diagonal
    prov = D.sum(2)                                                                    # provided by i
    recv = D.sum(1)                                                                    # received by j
    meas = r["save_mean"] * cfg.delta_h                                               # measured saving (P_solo*s)
    sel = np.abs(meas) > 1e-6
    assert (np.abs(recv - meas)[sel] / np.abs(meas)[sel]).max() < 1e-9                # sum_i D_ij == saving of j
    assert np.abs(recv - meas).max() < 1e-9
    B = recv - prov
    assert np.abs(B.sum(1)).max() < 1e-9                                               # sum B = 0
    # structural zeros in the steady V (no wind): nobody behind a bird -> its row is zero; nobody ahead -> column zero
    st, rec = run(cfg, rule_blind(cfg), 0, 9, n_dec=12)
    r0 = tonp(rec)
    d = 10
    slot = r0["slot"][d]
    tail = [i for i in range(n) if slot[i] >= 7]
    apex = int(np.argmin(slot))
    assert all(np.abs(r0["D"][d, i, :]).max() == 0 for i in tail)
    assert np.abs(r0["D"][d, :, apex]).max() == 0
    assert r0["D"][d].max() > 0


def test_18_two_bird_hand_calc():
    cfg = GeeseConfig(n_max=2, T=20.0, init_noise=0.0, use_events=False)
    st, rec = run(cfg, stay_policy(cfg), 0, 2)
    D = np.asarray(rec["D"])
    assert D.shape == (10, 2, 2)
    w = float(wake.wbar(np, 2.0, 0.875, cfg.v0, cfg))
    expect = cfg.delta_h * cfg.gain * cfg.m * wake.G * w / cfg.P_solo
    assert abs(D[3, 0, 1] - expect) / expect < 1e-9
    assert D[3, 1, 0] == 0 and D[3, 0, 0] == 0 and D[3, 1, 1] == 0
    assert abs(expect - 0.125 * cfg.delta_h) < 0.003 * cfg.delta_h                    # = the calibrated 12.5%


def test_18_log_D_flag_and_trajectory_shape(tmp_path):
    cfg = GeeseConfig(n_max=9, T=600.0)
    st, traj = run_episode(cfg, rule_blind(cfg), 0, 9)
    assert traj.global_fields["D"].shape == (300, 9, 9)
    cfg2 = GeeseConfig(n_max=9, T=20.0, log_D=False)
    _, traj2 = run_episode(cfg2, rule_blind(cfg2), 0, 9)
    assert "D" not in traj2.global_fields
    p = tmp_path / "t.npz"
    traj.save(p)
    t2 = Trajectory.load(p)
    assert np.array_equal(t2.global_fields["D"], traj.global_fields["D"])
    assert np.array_equal(t2.low["x"], traj.low["x"]) and t2.meta["N"] == 9
    assert traj.low["x"].shape == (300 * 4, 9)                                         # 0.5 s frames
    for k in ("slot", "lock", "action_req", "eff_action", "blocked", "swapped", "reward", "P_mean", "alive", "E_used"):
        assert traj.agent_fields[k].shape[:2] == (300, 9), k
    assert traj.global_fields["leader"].shape == (300,)
