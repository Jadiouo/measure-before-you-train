"""Wrap a trained IPPO checkpoint (marl/ippo.py) as a geese `Policy` so it can be run by the scripted-agent machinery
(`make_rollout`) with scheduled interventions (I1 relabel / I2 immune).  Memory handling mirrors ippo.make_rollout:
the GRU state is zeroed before acting whenever the env raises `mem_reset` (I1-M / I1-RM)."""
from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

from ..scenarios.geese import GeeseConfig, Policy
from .ippo import NEG, load_ckpt, net_step, prep_obs


def load_marl_policy(ckpt_path, greedy=False, **cfg_kw):
    """-> (GeeseConfig with events enabled, Policy).  Only shared-parameter checkpoints (G=1) are supported."""
    ts, mcfg, _ = load_ckpt(ckpt_path)
    assert mcfg.shared, "shared-parameter checkpoints only"
    ecfg = dataclasses.replace(mcfg.env_cfg(), use_events=True, **cfg_kw)
    params = jax.tree_util.tree_map(lambda x: x[0], ts.params)          # drop the group axis (G=1)
    n, H, mem = ecfg.n_max, mcfg.hidden, mcfg.memory
    step = jax.vmap(partial(net_step, memory=mem), in_axes=(None, 0, 0))

    def init(key, cfg_=None):
        return dict(h=jnp.zeros((n, H), jnp.float32))

    def act(ps, obs, st, mask, key=None):
        h = jnp.where(st.mem_reset, 0.0, ps["h"])
        h2, lg, _ = step(params, h, prep_obs(ecfg, obs))
        lm = jnp.where(mask, lg, NEG)
        a = jnp.argmax(lm, -1) if greedy else jax.random.categorical(key, lm, axis=-1)
        return a.astype(jnp.int32), dict(h=h2)
    return ecfg, Policy("M15_marl", init, act)
