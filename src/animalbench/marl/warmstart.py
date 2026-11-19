"""Behaviour cloning (DAgger) of a scripted HLC teacher into the shared-parameter GRU policy of ippo.py.

The teacher is a `Policy(name, init, act)` from scenarios/geese/scripted.py that only uses L0 observations and its
own recurrent state (e.g. S2 anonymous state-driven rotation: own fatigue EMA from P_prev).  Teacher and student see the
same observation stream, so the teacher's state can be advanced along the STUDENT's trajectory (DAgger): the student
acts, the teacher labels.  Output: params in the ippo.py format (leading group axis G=1), loadable as
`train(..., init_params_=params)`.
"""
from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..core import compile_schedule
from ..scenarios.geese import action_mask, build_obs, decision_step, reset
from .ippo import (F32, NEG, MarlConfig, adam_init, adam_update, init_params, net_seq, net_step, obs_dim,
                   prep_obs)


def make_dagger_rollout(ecfg, mcfg: MarlConfig, teacher, n_envs):
    """f(params, key, beta) -> data.  beta = prob. that the TEACHER action (instead of the student's sample) is executed
    (per env, per episode).  Labels are always the teacher's actions.  Returns time-major arrays (T,B,A,..)."""
    A, T, H = mcfg.n_agents, ecfg.n_dec, mcfg.hidden
    sched = compile_schedule([], ecfg.e_max)

    def rollout(params, key, beta):
        p0 = jax.tree_util.tree_map(lambda x: x[0], params)
        k_reset, k_act, k_mix = jax.random.split(key, 3)
        st = jax.vmap(lambda k: reset(ecfg, k, jnp.int32(A), sched))(jax.random.split(k_reset, n_envs))
        ps = jax.vmap(lambda s: teacher.init(s.key_policy, ecfg))(st)
        use_teacher = jax.random.bernoulli(k_mix, beta, (n_envs,))
        h0 = jnp.zeros((n_envs, A, H), F32)

        def body(carry, t):
            st, ps, h = carry
            obs_full = jax.vmap(lambda s: build_obs(ecfg, s))(st)
            obs = prep_obs(ecfg, obs_full)
            mask = jax.vmap(lambda s: action_mask(ecfg, s))(st)
            h = jnp.where(st.mem_reset[:, None, None], 0.0, h)
            h2, lg, _ = jax.vmap(jax.vmap(partial(net_step, memory=mcfg.memory), in_axes=(None, 0, 0)),
                                 in_axes=(None, 0, 0))(p0, h, obs)
            lm = jnp.where(mask, lg, NEG)
            a_s = jax.random.categorical(jax.random.fold_in(k_act, t), lm, axis=-1).astype(jnp.int32)
            a_t, ps2 = jax.vmap(lambda p, o, s, m, k: teacher.act(p, o, s, m, k))(
                ps, obs_full, st, mask, jax.vmap(lambda s: jax.random.fold_in(s.key_policy, s.dec))(st))
            a_t = a_t.astype(jnp.int32)
            a_exec = jnp.where(use_teacher[:, None], a_t, a_s)
            out = dict(obs=obs, mask=mask, label=a_t, reset=st.mem_reset)
            st2, _ = jax.vmap(lambda s, x: decision_step(ecfg, s, x))(st, a_exec)
            return (st2, ps2, h2), out

        _, data = lax.scan(body, (st, ps, h0), jnp.arange(T))
        return data
    return rollout


def bc_loss(p, obs, mask, label, rflag, mcfg):
    """obs (S,T,D) ... -> mean CE of teacher labels under the masked student policy; also accuracy."""
    lg, _ = jax.vmap(partial(net_seq, memory=mcfg.memory, H=mcfg.hidden), in_axes=(None, 0, 0))(p, obs, rflag)
    lsm = jax.nn.log_softmax(jnp.where(mask, lg, NEG), -1)
    lp = jnp.take_along_axis(lsm, label[..., None], -1)[..., 0]
    # class-balanced: STAY dominates; weight each class by 1/(freq) (clipped) so BACK/FWD decisions are learned
    w = jnp.stack([(label == c).mean() for c in range(3)])
    wt = 1.0 / jnp.maximum(w, 0.02)
    wt = wt / (wt * w).sum()
    ce = -(lp * wt[label]).mean()
    acc = (jnp.argmax(jnp.where(mask, lg, NEG), -1) == label).mean()
    return ce, acc


def dagger_bc(ecfg, mcfg: MarlConfig, teacher, seed=0, iters=60, n_envs=64, epochs=4, n_mb=4, lr=1e-3,
              beta0_iters=4, log=print):
    """Returns (params, history).  Dataset aggregation: the last `keep` iterations are kept in a rolling buffer."""
    key = jax.random.PRNGKey(seed)
    k0, key = jax.random.split(key)
    params = init_params(k0, obs_dim(ecfg), mcfg)
    opt = adam_init(params)
    roll = jax.jit(make_dagger_rollout(ecfg, mcfg, teacher, n_envs))

    @jax.jit
    def update(params, opt, obs, mask, label, rflag, key):
        S = obs.shape[0]
        mb = S // n_mb

        def epoch(carry, k):
            params, opt = carry
            perm = jax.random.permutation(k, S)[: mb * n_mb].reshape(n_mb, mb)

            def step(c, idx):
                params, opt = c
                (l, acc), g = jax.value_and_grad(lambda p: bc_loss(jax.tree_util.tree_map(lambda x: x[0], p),
                                                                    obs[idx], mask[idx], label[idx], rflag[idx], mcfg),
                                                 has_aux=True)(params)
                params, opt, _ = adam_update(params, g, opt, lr, 1.0)
                return (params, opt), (l, acc)
            return lax.scan(step, (params, opt), perm)
        (params, opt), (ls, accs) = lax.scan(epoch, (params, opt), jax.random.split(key, epochs))
        return params, opt, ls.mean(), accs.mean()

    buf, hist = [], []
    A = mcfg.n_agents
    t0 = time.time()
    for i in range(iters):
        key, kr, ku = jax.random.split(key, 3)
        beta = 1.0 if i < beta0_iters else 0.0
        d = roll(params, kr, beta)
        seq = lambda x: jnp.moveaxis(x, 0, 2).reshape((n_envs * A,) + (x.shape[0],) + x.shape[3:])   # (S,T,..)
        rf = jnp.broadcast_to(d["reset"][:, :, None], d["label"].shape)
        item = dict(obs=seq(d["obs"]), mask=seq(d["mask"]), label=seq(d["label"]), rflag=seq(rf))
        buf.append(item)
        buf = buf[-8:]
        cat = {k: jnp.concatenate([b[k] for b in buf], 0) for k in item}
        params, opt, l, acc = update(params, opt, cat["obs"], cat["mask"], cat["label"], cat["rflag"], ku)
        hist.append((i, float(l), float(acc)))
        if i % 10 == 0 or i == iters - 1:
            log(f"bc it {i} loss {float(l):.4f} acc {float(acc):.4f} wall {time.time() - t0:.0f}s")
    return params, hist
