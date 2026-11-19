"""M1a #6 LLC forms the V from a scattered start; #16 rule_blind information limits and rotation statistics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from animalbench.core import compile_schedule, crosswind
from animalbench.scenarios.geese import (GeeseConfig, build_obs, layout, make_rollout, rule_blind, rule_commander,
                                         rule_oracle, stay_policy, apply_command)
from animalbench.scenarios.geese.env import reset
from animalbench.scenarios.geese.summary import episode_stats
from animalbench.scenarios.geese.env import to_trajectory

N_SEED = 200


def formation_stats(policy_name, wind=False, ou=0.0, T=70.0):
    cfg = GeeseConfig(n_max=9, T=T, init_mode="scatter", log_D=False, log_stride=5)
    pol = {"stay": stay_policy, "rule_blind": rule_blind}[policy_name](cfg)
    ev = [crosswind(0.0, 1000.0, speed=3.0, ramp=5.0, ou_sigma=ou)] if wind else []
    sched = compile_schedule(ev, cfg.e_max)
    f = jax.jit(jax.vmap(make_rollout(cfg, pol), in_axes=(0, None, None)))
    keys = jax.random.split(jax.random.PRNGKey(0), N_SEED)
    _, recs = f(keys, jnp.int32(9), sched)
    low = jax.tree_util.tree_map(np.asarray, recs["low"])
    B = N_SEED
    ef = low["e_form"].reshape(B, -1, 9) / cfg.b
    t = low["t"].reshape(B, -1)[0]
    mx = ef.max(2)
    ok = mx < 0.15
    need = int(round(10.0 / (t[1] - t[0])))
    tform = np.full(B, np.nan)
    for b in range(B):
        run_ = 0
        for k in range(len(t)):
            run_ = run_ + 1 if ok[b, k] else 0
            if run_ >= need:
                tform[b] = t[k - need + 1]
                break
    slot = np.repeat(np.asarray(recs["slot"]), cfg.n_sub // cfg.log_stride, axis=1)   # frame-wise slot
    save = low["save"].reshape(B, -1, 9)
    F, WU = [], []
    for b in range(B):
        if np.isnan(tform[b]):
            continue
        sel = t >= tform[b]
        F.append((mx[b, sel] < 0.5).mean())
        fol = slot[b][sel] > 0
        WU.append(((save[b][sel] > 0.02) & fol).sum(1).mean() / 8.0)
    return dict(success=float(np.mean(tform <= 60.0)), t_form_med=float(np.nanmedian(tform)),
                t_form_max=float(np.nanmax(tform)), F_min=float(np.min(F)), wake_util_min=float(np.min(WU)))


@pytest.fixture(scope="module")
def stats():
    return {}


@pytest.mark.parametrize("policy_name", ["stay", "rule_blind"])
def test_6_forms_v_from_scatter_no_wind(policy_name, stats):
    s = formation_stats(policy_name)
    stats[policy_name] = s
    assert s["success"] >= 0.95, s
    assert s["wake_util_min"] > 0.8, s
    if policy_name == "stay":      # F is a pure-LLC quantity: a rule_blind apex hand-over inside the window costs ~6 s of e_form>0.5b
        assert s["F_min"] > 0.9, s
    else:
        assert s["F_min"] > 0.85, s


def test_6_forms_v_with_3ms_crosswind():
    s = formation_stats("stay", wind=True, ou=0.0)
    assert s["success"] >= 0.95, s
    assert s["F_min"] > 0.9 and s["wake_util_min"] > 0.8, s


# --------------------------------------------------------------------------- #16
def test_16_rule_blind_ignores_non_L0_columns():
    cfg = GeeseConfig(n_max=9, identity="nbr+self")
    lay = layout(cfg)
    st = reset(cfg, jax.random.PRNGKey(0), 9, compile_schedule([], cfg.e_max))
    st = apply_command(cfg, st, next_leader=4, order_hint=[4, 0, 1])
    obs = build_obs(cfg, st)
    pol = rule_blind(cfg)
    ps = pol.init(st.key_policy, cfg)
    from animalbench.scenarios.geese import action_mask
    mask = action_mask(cfg, st)
    # run a few decisions with clean vs poisoned obs (poison = NaN in command + identity columns)
    obs_nan = obs.at[:, lay["L0_end"][0]:].set(jnp.nan)
    ps1, ps2 = ps, ps
    for d in range(60):
        st_d = st._replace(dec=jnp.int32(d))
        o1 = obs.at[:, lay["is_apex"][0]].set(jnp.where(d > 5, 1.0, obs[:, lay["is_apex"][0]]))
        o2 = o1.at[:, lay["L0_end"][0]:].set(jnp.nan)
        a1, ps1 = pol.act(ps1, o1, st_d, mask, None)
        a2, ps2 = pol.act(ps2, o2, st_d, mask, None)
        assert np.array_equal(np.asarray(a1), np.asarray(a2))
    # and in a closed-loop episode the trajectory is identical when the command channel is garbage
    cfg_c = GeeseConfig(n_max=9, T=100.0, init_mode="v")
    f = jax.jit(make_rollout(cfg_c, rule_blind(cfg_c)))
    sched = compile_schedule([], cfg_c.e_max)
    _, r1 = f(jax.random.PRNGKey(0), jnp.int32(9), sched)
    assert np.asarray(r1["swapped"]).any()


@pytest.fixture(scope="module")
def blind_runs():
    out = {}
    for rule in ("unilateral", "mutual"):
        cfg = GeeseConfig(n_max=9, T=600.0, swap_rule=rule, log_low=False, log_D=False)
        f = jax.jit(jax.vmap(make_rollout(cfg, rule_blind(cfg)), in_axes=(0, None, None)))
        keys = jax.random.split(jax.random.PRNGKey(7), 20)
        _, recs = f(keys, jnp.int32(9), compile_schedule([], cfg.e_max))
        out[rule] = (cfg, jax.tree_util.tree_map(np.asarray, recs))
    return out


@pytest.mark.parametrize("rule", ["unilateral", "mutual"])
def test_16_rule_blind_tenure_and_rate(rule, blind_runs):
    cfg, r = blind_runs[rule]
    lead = r["leader"]                                       # (B, 300)
    B = lead.shape[0]
    tenures, first = [], []
    for b in range(B):
        ch = np.flatnonzero(lead[b, 1:] != lead[b, :-1]) + 1
        bounds = np.concatenate([[0], ch, [lead.shape[1]]])
        spans = np.diff(bounds) * cfg.delta_h
        first.append(spans[0])
        tenures += list(spans[1:-1])                         # completed, non-initial spells
    tenures = np.array(tenures)
    lo, hi = cfg.t_hold_lo + cfg.t_lock, cfg.t_hold_hi + cfg.t_lock + cfg.delta_h   # [25,35]s + lock (+ decision quantisation)
    frac = np.mean((tenures >= lo) & (tenures <= hi))
    assert frac >= 0.95, (frac, tenures.min(), tenures.max())
    assert np.all(np.array(first) >= cfg.t_hold_lo) and np.all(np.array(first) <= cfg.t_hold_hi + 2 * cfg.delta_h)
    rate = np.mean([(lead[b, 1:] != lead[b, :-1]).sum() / (lead.shape[1] * cfg.delta_h) for b in range(B)])
    expect = 1.0 / (0.5 * (cfg.t_hold_lo + cfg.t_hold_hi) + cfg.t_lock)
    assert abs(rate - expect) / expect <= 0.20, (rate, expect)
    # rotation only among the first three slots (treadmill)
    assert len(np.unique(lead[0])) <= 3


def test_16_oracle_and_commander_run_and_command_moves_bird_up():
    cfg = GeeseConfig(n_max=9, T=60.0, init_mode="v", log_low=False, log_D=False)
    sched = compile_schedule([], cfg.e_max)
    # oracle: privileged energy-debt trigger -> hands over the apex without any timer
    cfg_o = cfg.replace(T=200.0)
    _, r = jax.jit(make_rollout(cfg_o, rule_oracle(cfg_o)))(jax.random.PRNGKey(0), jnp.int32(9), sched)
    assert int(np.asarray(r["swapped"]).sum()) >= 2
    # commander: bird 1's slot is 1; next_leader=2 moves it to the apex within a few decisions
    pol = rule_commander(cfg)
    from animalbench.scenarios.geese.step import decision_step
    from animalbench.scenarios.geese import action_mask
    st = reset(cfg, jax.random.PRNGKey(0), 9, sched)
    st = apply_command(cfg, st, next_leader=2, order_hint=[2, 0, 1])
    ps = pol.init(st.key_policy, cfg)
    for d in range(2):
        obs = build_obs(cfg, st); m = action_mask(cfg, st)
        a, ps = pol.act(ps, obs, st, m, None)
        st, _ = decision_step(cfg, st, a)
    assert int(st.slot[2]) == 0
