"""High-level controllers (HLC): rule_blind (L0), rule_commander (L1), rule_oracle (upper bound), trivial policies.

Policy.act(pstate, obs, state, mask, key) -> (actions int32 (n_max,), pstate). rule_blind/rule_commander read only
obs (L0 columns, plus the command slice for commander); rule_oracle also reads the true state.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .obs import layout

STAY, BACK, FWD = 0, 1, 2


class Policy(NamedTuple):
    name: str
    init: Callable
    act: Callable


def _col(obs, lay, name):
    return obs[:, lay[name][0]]


def _blind_core(cfg, obs):
    """Shared by blind and commander. Returns pstate-free pieces from L0 columns only."""
    lay = layout(cfg)
    return _col(obs, lay, "is_apex") > 0.5, _col(obs, lay, "lock") <= 0.0, _col(obs, lay, "rank")


def rule_blind(cfg) -> Policy:
    n = cfg.n_max
    mutual = cfg.swap_rule == "mutual"
    lo, hi = cfg.t_hold_lo, cfg.t_hold_hi

    def init(key, cfg_=None):
        z = jnp.zeros((n,), cfg.jdtype)
        return dict(held=z, thold=z + lo, was_apex=jnp.zeros((n,), bool), cnt=jnp.zeros((n,), jnp.int32), key=key)

    def draw(key, i, c):
        k = jax.random.fold_in(jax.random.fold_in(key, i), c)
        return lo + (hi - lo) * jax.random.uniform(k, (), dtype=cfg.jdtype)

    def act(ps, obs, st, mask, key=None):
        is_apex, lock0, rank = _blind_core(cfg, obs)
        new_apex = is_apex & ~ps["was_apex"]
        thold_new = jax.vmap(draw, in_axes=(None, 0, 0))(ps["key"], jnp.arange(n), ps["cnt"])
        thold = jnp.where(new_apex, thold_new, ps["thold"])
        held = jnp.where(new_apex, 0.0, ps["held"])
        cnt = ps["cnt"] + new_apex.astype(jnp.int32)
        retreat = is_apex & lock0 & (held >= thold)
        a = jnp.where(retreat, BACK, STAY)
        if mutual:
            slot = jnp.round(rank * (n - 1)).astype(jnp.int32)
            a = jnp.where(~is_apex & lock0 & (slot >= 1) & (slot <= 2), FWD, a)
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY).astype(jnp.int32)
        held_next = jnp.where(is_apex, jnp.where(lock0, held + cfg.delta_h, held), 0.0)
        return a, dict(held=held_next, thold=thold, was_apex=is_apex, cnt=cnt, key=ps["key"])
    return Policy("rule_blind", init, act)


def rule_commander(cfg) -> Policy:
    base = rule_blind(cfg)
    lay = layout(cfg)

    def act(ps, obs, st, mask, key=None):
        a, ps2 = base.act(ps, obs, st, mask, key)
        cs = lay["command"][0]
        is_next = obs[:, cs + 1] > 0.5
        valid = obs[:, cs + 2] > 0.5
        is_apex, lock0, _ = _blind_core(cfg, obs)
        go = is_next & valid & ~is_apex & lock0 & mask[:, FWD]
        return jnp.where(go, FWD, a).astype(jnp.int32), ps2
    return Policy("rule_commander", base.init, act)


def rule_oracle(cfg) -> Policy:
    """Upper bound with privileged global energy: apex retreats when its cumulative energy exceeds the
    alive-mean by oracle_debt (P_solo*s); in mutual mode slots 1,2 volunteer whenever unlocked."""
    n = cfg.n_max
    mutual = cfg.swap_rule == "mutual"

    def init(key, cfg_=None):
        return dict()

    def act(ps, obs, st, mask, key=None):
        al = st.alive
        na = jnp.maximum(jnp.sum(al.astype(cfg.jdtype)), 1.0)
        E = jnp.where(al, st.E_used, 0.0) / cfg.P_solo
        debt = E - jnp.sum(E) / na
        unl = al & (st.lock == 0)
        a = jnp.where(unl & (st.slot == 0) & (debt > cfg.oracle_debt), BACK, STAY)
        if mutual:
            a = jnp.where(unl & (st.slot >= 1) & (st.slot <= 2), FWD, a)
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY).astype(jnp.int32)
        return a, ps
    return Policy("rule_oracle", init, act)


def stay_policy(cfg) -> Policy:
    return Policy("stay", lambda key, cfg_=None: dict(),
                  lambda ps, obs, st, mask, key=None: (jnp.zeros((cfg.n_max,), jnp.int32), ps))


def table_policy(cfg, table, respect_mask=True) -> Policy:
    """Open-loop policy: table (n_dec, n_max) of actions (masked to STAY when illegal unless respect_mask=False)."""
    table = jnp.asarray(table, jnp.int32)

    def act(ps, obs, st, mask, key=None):
        a = table[jnp.minimum(st.dec, table.shape[0] - 1)]
        if respect_mask:
            a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY)
        return a.astype(jnp.int32), ps
    return Policy("table", lambda key, cfg_=None: dict(), act)
