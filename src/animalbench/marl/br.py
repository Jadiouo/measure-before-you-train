"""M4br: best-response training of ONE IPPO GRU bird against scripted co-players (new file; marl/ippo.py untouched
except for two default-preserving MarlConfig fields).

The environment has n_max = 3 birds; bird `rl_id` is controlled by the network, the others by `policy`
(a scripted geese Policy returning actions for all birds; its output for `rl_id` is overwritten).  The learner sees
only its own sequence (A = 1, shared params, so MarlConfig(n_agents=1)); everything else (GRU, PPO update, GAE,
return normalisation, hyper-parameters) is ippo.py's, reused unchanged.
Diagnostics recorded per decision (all computed from what the scripted co-players actually do):
  ready_rl : a scripted apex asks BACK, the RL bird is its target, unlocked and not ahead of its fair share
             (= the norm says "your turn")
  turn_rl  : same without the fair-share condition (any scripted-apex BACK request aimed at the RL bird, unlocked)
  due_rl   : the RL bird is apex, unlocked and its tenure has reached the norm's retreat threshold
Take-over rate = P(RL picks FWD | ready_rl); retreat rate = P(RL picks BACK | due_rl).
"""
from __future__ import annotations

import json
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..core import compile_schedule
from ..scenarios.geese import action_mask, build_obs, decision_step, reset
from ..scenarios.geese.hlc import BACK, FWD
from . import ippo
from .ippo import F32, NEG, MarlConfig, net_step, prep_obs


def make_rollout_br(ecfg, mcfg1: MarlConfig, policy, rl_id=0, n_envs=None, greedy=False):
    B = mcfg1.n_envs if n_envs is None else n_envs
    A, T = ecfg.n_max, ecfg.n_dec
    H, mem = mcfg1.hidden, mcfg1.memory
    sched = compile_schedule([], ecfg.e_max)

    def act_net(params, h, obs):       # params (1,..), h (1,B,H), obs (1,B,D)
        return jax.vmap(jax.vmap(partial(net_step, memory=mem), in_axes=(None, 0, 0)), in_axes=(0, 0, 0))(params, h, obs)

    def rollout(params, key):
        k_reset, k_act = jax.random.split(key)
        st = jax.vmap(lambda k: reset(ecfg, k, jnp.int32(A), sched))(jax.random.split(k_reset, B))
        ps = jax.vmap(lambda s: policy.init(s.key_policy, ecfg))(st)
        h0 = jnp.zeros((1, B, H), F32)

        def body(carry, t):
            st, ps, h = carry
            obs_full = jax.vmap(lambda s: build_obs(ecfg, s))(st)
            obs = prep_obs(ecfg, obs_full)[:, rl_id]                                   # (B,D)
            mask = jax.vmap(lambda s: action_mask(ecfg, s))(st)                        # (B,A,3)
            mrl = mask[:, rl_id]
            h = jnp.where(st.mem_reset[:, None, None].transpose(1, 0, 2), 0.0, h)
            h2, lg, v = act_net(params, h, obs[None])
            lg, v = lg[0], v[0]
            lm = jnp.where(mrl, lg, NEG)
            if greedy:
                a_rl = jnp.argmax(lm, -1)
            else:
                a_rl = jax.random.categorical(jax.random.fold_in(k_act, t), lm, axis=-1)
            a_rl = a_rl.astype(jnp.int32)
            logp = jnp.take_along_axis(jax.nn.log_softmax(lm, -1), a_rl[:, None], -1)[:, 0]
            kk = jax.vmap(lambda s: jax.random.fold_in(s.key_policy, s.dec))(st)
            a_sc, ps2 = jax.vmap(lambda p_, o_, s_, m_, k_: policy.act(p_, o_, s_, m_, k_))(ps, obs_full, st, mask, kk)
            a_full = a_sc.at[:, rl_id].set(a_rl)
            unl_rl = st.lock[:, rl_id] == 0
            turn = ps2["ready_edge"][:, rl_id] & unl_rl           # includes the fair-share condition only via ready_edge
            pre_slot, pre_lock = st.slot, st.lock
            st2, rec = jax.vmap(lambda s, x: decision_step(ecfg, s, x))(st, a_full)
            out = dict(obs=obs[:, None], mask=mrl[:, None], act=a_rl[:, None], logp=logp[:, None], val=v[:, None],
                       rew=rec["reward"][:, rl_id].astype(F32)[:, None], reset=st.mem_reset,
                       slot3=pre_slot, n_swaps=rec["n_swaps"], rew3=rec["reward"].astype(F32),
                       ready=turn, due=ps2["due_apex"][:, rl_id] & unl_rl, a_rl_=a_rl)
            return (st2, ps2, h2), out

        (stT, psT, hT), data = lax.scan(body, (st, ps, h0), jnp.arange(T))
        hT = jnp.where(stT.mem_reset[:, None, None].transpose(1, 0, 2), 0.0, hT)
        obsT = prep_obs(ecfg, jax.vmap(lambda s: build_obs(ecfg, s))(stT))[:, rl_id]
        _, _, vT = act_net(params, hT, obsT[None])
        return data, vT[0][:, None], stT
    return rollout


def br_metrics(data, ecfg):
    f = lambda x: x.astype(F32)
    ready, due, a = data["ready"], data["due"], data["a_rl_"]
    div = lambda n, d: n / jnp.maximum(d, 1.0)
    return dict(takeover=div((f(a == FWD) * ready).sum(), ready.sum()), n_ready=ready.sum() / ready.shape[1],
                retreat=div((f(a == BACK) * due).sum(), due.sum()), n_due=due.sum() / due.shape[1],
                ret_rl=(f(data["rew"][..., 0]) * ecfg.delta_h).sum(0).mean(),
                ret_group=(f(data["rew3"]) * ecfg.delta_h).sum(0).mean(),
                swap_rate=data["n_swaps"].sum(0).mean() / ecfg.T)


def make_iteration_br(ecfg, mcfg1: MarlConfig, policy, rl_id=0):
    roll = make_rollout_br(ecfg, mcfg1, policy, rl_id)
    n_it = max(1, mcfg1.total_agent_steps // (mcfg1.n_envs * ecfg.n_dec))
    B = mcfg1.n_envs

    def iteration(ts, key):
        kr, ku = jax.random.split(key)
        data, last_v, _ = roll(ts.params, kr)
        mean_ba = jnp.broadcast_to(ts.rn[:, 0:1], (1, B)).reshape(B, 1)
        std_ba = (jnp.sqrt(ts.rn[:, 1:2]) + 1e-6) * jnp.ones((1, B))
        std_ba = std_ba.reshape(B, 1)
        val = data["val"] * std_ba + mean_ba
        lastv = last_v * std_ba + mean_ba
        adv, ret = ippo.gae(data["rew"] * mcfg1.reward_scale, val, lastv, mcfg1.gamma, mcfg1.lam)
        sh = True
        retg = ippo.seq_group(ret, sh)
        bm, bv = retg.mean((1, 2)), retg.var((1, 2))
        rate = jnp.maximum(1.0 / (ts.it.astype(F32) + 1.0), 0.02)
        rn = jnp.stack([ts.rn[:, 0] + rate * (bm - ts.rn[:, 0]), ts.rn[:, 1] + rate * (bv - ts.rn[:, 1])], -1)
        std_n = (jnp.sqrt(rn[:, 1]) + 1e-6)[:, None, None]
        ret_n = (retg - rn[:, 0][:, None, None]) / std_n
        sdata = dict(obs=ippo.seq_group(data["obs"], sh), mask=ippo.seq_group(data["mask"], sh),
                     act=ippo.seq_group(data["act"], sh), logp=ippo.seq_group(data["logp"], sh),
                     adv=ippo.seq_group(adv, sh), ret_n=ret_n,
                     reset=ippo.seq_group(data["reset"][:, :, None].repeat(1, 2), sh))
        frac = ts.it.astype(F32) / n_it
        lr = mcfg1.lr * (1.0 - frac * (1.0 - mcfg1.lr_final_frac))
        params, opt, aux = ippo.ppo_update(ts.params, ts.opt, sdata, lr, ku, mcfg1, 1.0)
        met = br_metrics(data, ecfg)
        met.update(aux)
        met["ev"] = 1.0 - jnp.var(ret - val) / (jnp.var(ret) + 1e-8)
        return ippo.TrainState(params, opt, rn, ts.it + 1), met
    iteration.n_it = n_it
    return iteration


def train_br(mcfg1: MarlConfig, policy, seed, out_dir, rl_id=0, log_every=1, verbose=True):
    ecfg = mcfg1.env_cfg().replace(n_max=3)
    os.makedirs(out_dir, exist_ok=True)
    it_fn = make_iteration_br(ecfg, mcfg1, policy, rl_id)
    step = jax.jit(it_fn)
    key = jax.random.PRNGKey(seed)
    k0, key = jax.random.split(key)
    ts = ippo.init_train(k0, ecfg, mcfg1)
    per_it = mcfg1.n_envs * ecfg.n_dec
    logf = open(os.path.join(out_dir, "log.jsonl"), "w")
    t0 = time.time()
    for i in range(it_fn.n_it):
        key, k = jax.random.split(key)
        ts, met = step(ts, k)
        m = {kk: float(v) for kk, v in met.items()}
        m.update(it=i, agent_steps=(i + 1) * per_it, wall=time.time() - t0)
        logf.write(json.dumps(m) + "\n"); logf.flush()
        if verbose and (i % max(1, it_fn.n_it // 15) == 0 or i == it_fn.n_it - 1):
            print(f"it {i}/{it_fn.n_it} ret_rl {m['ret_rl']:.1f} grp {m['ret_group']:.1f} takeover {m['takeover']:.2f} "
                  f"(n {m['n_ready']:.1f}) retreat {m['retreat']:.2f} swap {m['swap_rate']:.3f} ent {m['ent']:.2f} "
                  f"wall {m['wall']:.0f}s", flush=True)
    ippo.save_ckpt(os.path.join(out_dir, "final.ckpt"), ts, ecfg, mcfg1)
    logf.close()
    return ts, ecfg, it_fn
