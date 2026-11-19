"""IPPO in pure JAX for the geese `hier` environment (spec v0.3 4.2, v0.4 4.3).

Layout conventions
  * rollout data are time-major (T, B, A, ...)  [B envs, A agents = birds, T decisions]
  * the learner works on "groups": shared params -> G=1 group of S=B*A sequences; non-shared -> G=A groups of S=B
    sequences (agent i always uses network i, i.e. networks are tied to the body, not to the slot).
  * policy = obs encoder -> GRU (memory="gru") or tanh-MLP (memory="mlp") -> masked categorical + value head.
  * the value head predicts a running-normalised return (PopArt-free: only the target is normalised).
  * one rollout = one full episode per env (T = cfg.n_dec decisions); the episode end is treated as a time-limit
    truncation (bootstrap from V(last obs)) so that the criterion is the gamma-discounted infinite-horizon return of
    the small game (spec v0.4 4.3), not a finite-horizon one.  The env has no clock in the obs, so this is consistent.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import time
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..core import compile_schedule
from ..scenarios.geese import GeeseConfig, action_mask, build_obs, decision_step, layout, reset

F32 = jnp.float32
NEG = -1e9


@dataclass(frozen=True)
class MarlConfig:
    n_agents: int = 3
    n_envs: int = 64
    shared: bool = True
    memory: str = "gru"            # "gru" | "mlp"
    hidden: int = 64
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    lr_final_frac: float = 0.1
    epochs: int = 4
    n_mb: int = 4
    reward_scale: float = 1.0
    swap_rule: str = "unilateral"
    kappa: float = 0.0
    identity: str = "none"
    dtype: str = "float32"
    total_agent_steps: int = 20_000_000
    random_slots: bool = False     # M3 basin experiments: random initial bird->slot permutation
    vf_warmup_iters: int = 0       # warm start: first k iterations update ONLY the value head (policy+trunk frozen)
    reward_mode: str = "individual"  # M4br: individual | team | mixed (env reward_mode); default = unchanged behaviour
    mix_alpha: float = 0.5
    # survival-migration mode (default OFF; notes/impl/survival_ippo_prereg.md).  E0 cap = surv_eps * P_solo * T.
    survival: bool = False
    surv_alpha: float = 0.0
    surv_beta: float = 0.5
    surv_eps: float = 1.06
    surv_burst_m: float = 0.5
    surv_burst_p: float = 0.05
    surv_h: float = 0.05
    surv_shaping: str = "none"
    surv_obs_abs: bool = False        # v2 switches, see GeeseConfig (default OFF)
    surv_pw_spread: float = 0.0
    surv_pw_fixed: bool = False
    path_T: float = 3.0               # swap path duration (s); GeeseConfig default
    # centralised critic (MAPPO), default OFF (off = bit-identical to IPPO).  Actor unchanged (shared GRU, own obs).
    # Critic for bird i: its own GRU (separate params p["c"]) over concat(obs_i, obs_{i+1}, ..., obs_{i+A-1 mod A}) =
    # all birds' prepped observations, own first, so V_i is bird i's value given the global state (alpha=1: team value).
    central: bool = False
    # ablation (default OFF = bit-identical): separate critic GRU with the SAME architecture/param count as the MAPPO critic
    # (input width A*D), but fed only [own obs, zeros...] (zero-padded) -> decoupled from the actor trunk, no global information.
    separate_critic: bool = False

    @property
    def use_c(self):
        return self.central or self.separate_critic

    def env_cfg(self) -> GeeseConfig:
        if self.survival:
            return GeeseConfig(n_max=self.n_agents, dtype=self.dtype, log_low=False, log_D=False, use_events=False,
                               swap_rule=self.swap_rule, kappa=self.kappa, identity=self.identity,
                               random_slots=self.random_slots, obs_clock=True, E0_coef=self.surv_eps, E0_sigma=0.0,
                               survival=True, surv_alpha=self.surv_alpha, surv_beta=self.surv_beta,
                               surv_burst_m=self.surv_burst_m, surv_burst_p=self.surv_burst_p, surv_h=self.surv_h,
                               surv_shaping=self.surv_shaping,
                               surv_obs_abs=self.surv_obs_abs, surv_pw_spread=self.surv_pw_spread,
                               surv_pw_fixed=self.surv_pw_fixed, path_T=self.path_T)
        return GeeseConfig(n_max=self.n_agents, dtype=self.dtype, log_low=False, log_D=False, use_events=False,
                           swap_rule=self.swap_rule, kappa=self.kappa, identity=self.identity,
                           random_slots=self.random_slots, reward_mode=self.reward_mode, mix_alpha=self.mix_alpha)


def obs_dim(ecfg):
    return layout(ecfg)["L0_end"][0]          # L0 columns only (no command slice, no identity block)


def prep_obs(ecfg, obs):
    """(.., n, D_full) env obs -> (.., n, D) float32 network input (L0 columns, mild fixed rescaling)."""
    lay = layout(ecfg)
    o = obs[..., :lay["L0_end"][0]].astype(F32)
    pp = lay["P_prev"][0]
    ef = lay["e_form"][0]
    o = o.at[..., pp].set((o[..., pp] - 0.9) * 10.0)
    o = o.at[..., ef].set(jnp.clip(o[..., ef], -3.0, 3.0))
    return o


# ------------------------------------------------------------------------------------------------ network
def _orth(key, shape, scale):
    return jax.nn.initializers.orthogonal(scale, dtype=F32)(key, shape)


def init_net(key, D, mcfg: MarlConfig):
    H = mcfg.hidden
    k = jax.random.split(key, 8)
    p = dict(w1=_orth(k[0], (D, H), np.sqrt(2)), b1=jnp.zeros((H,), F32))
    if mcfg.memory == "gru":
        p.update(wi=_orth(k[1], (H, 3 * H), 1.0), wh=_orth(k[2], (H, 3 * H), 1.0),
                 bi=jnp.zeros((3 * H,), F32), bh=jnp.zeros((3 * H,), F32))
    else:
        p.update(w2=_orth(k[1], (H, H), np.sqrt(2)), b2=jnp.zeros((H,), F32))
    p.update(wpi=_orth(k[3], (H, 3), 0.01), bpi=jnp.zeros((3,), F32),
             wv=_orth(k[4], (H, 1), 1.0), bv=jnp.zeros((1,), F32))
    return p


def net_step(p, h, obs, memory):
    """One decision for one sequence. h:(H,), obs:(D,) -> (h', logits(3), v)."""
    x = jnp.tanh(obs @ p["w1"] + p["b1"])
    if memory == "gru":
        gi = x @ p["wi"] + p["bi"]
        gh = h @ p["wh"] + p["bh"]
        H = h.shape[-1]
        r = jax.nn.sigmoid(gi[:H] + gh[:H])
        z = jax.nn.sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = jnp.tanh(gi[2 * H:] + r * gh[2 * H:])
        h2 = (1 - z) * n + z * h
        y = h2
    else:
        y = jnp.tanh(x @ p["w2"] + p["b2"])
        h2 = h
    return h2, y @ p["wpi"] + p["bpi"], (y @ p["wv"] + p["bv"])[0]


def net_seq(p, obs, reset_flags, memory, H):
    """obs:(T,D), reset_flags:(T,) bool (zero h before step t) -> logits (T,3), v (T,)."""
    def f(h, x):
        o, rs = x
        h = jnp.where(rs, jnp.zeros_like(h), h)
        h2, lg, v = net_step(p, h, o, memory)
        return h2, (lg, v)
    _, (lg, v) = lax.scan(f, jnp.zeros((H,), F32), (obs, reset_flags))
    return lg, v


# ------------------------------------------------------------------------------------------------ Adam
class Adam(NamedTuple):
    m: object
    v: object
    t: jnp.ndarray


def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return Adam(z, z, jnp.zeros((), jnp.int32))


def adam_update(params, grads, opt, lr, max_norm, b1=0.9, b2=0.999, eps=1e-5):
    leaves = jax.tree_util.tree_leaves(grads)
    gn = jnp.sqrt(sum(jnp.sum(g.astype(F32) ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (gn + 1e-8))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    t = opt.t + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, opt.m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, opt.v, grads)
    c1 = 1 - b1 ** t.astype(F32)
    c2 = 1 - b2 ** t.astype(F32)
    new = jax.tree_util.tree_map(lambda p, m_, v_: p - lr * (m_ / c1) / (jnp.sqrt(v_ / c2) + eps), params, m, v)
    return new, Adam(m, v, t), gn


# ------------------------------------------------------------------------------------------------ grouping
def group(x, shared):
    """(B, A, ...) -> (G, S, ...)."""
    B, A = x.shape[:2]
    if shared:
        return x.reshape((1, B * A) + x.shape[2:])
    return jnp.moveaxis(x, 1, 0)


def ungroup(y, B, A, shared):
    if shared:
        return y.reshape((B, A) + y.shape[2:])
    return jnp.moveaxis(y, 0, 1)


def seq_group(x, shared):
    """(T, B, A, ...) -> (G, S, T, ...)."""
    return group(jnp.moveaxis(x, 0, 2), shared)


def central_obs(obs, own_only=False):
    """(..., A, D) -> (..., A, A*D): for each bird, all birds' observations with its own first.
    own_only (separate_critic ablation): own obs followed by zeros of the same width."""
    A = obs.shape[-2]
    if own_only:
        return jnp.concatenate([obs] + [jnp.zeros_like(obs)] * (A - 1), -1)
    return jnp.concatenate([jnp.roll(obs, -k, axis=-2) for k in range(A)], -1)


def init_params(key, D, mcfg: MarlConfig):
    G = 1 if mcfg.shared else mcfg.n_agents

    def one(k):
        p = init_net(k, D, mcfg)
        if mcfg.use_c:
            p["c"] = init_net(jax.random.fold_in(k, 99), D * mcfg.n_agents, mcfg)
        return p
    return jax.vmap(one)(jax.random.split(key, G))


def policy_probs(logits, mask):
    return jax.nn.softmax(jnp.where(mask, logits, NEG), -1)


# ------------------------------------------------------------------------------------------------ rollout
def make_rollout(ecfg: GeeseConfig, mcfg: MarlConfig, n_envs=None, n_dec=None, greedy=False):
    """f(params, key, dev_action=-1, dev_mode=0) -> (data, last_v, final_state).  One full episode per env.
    dev_mode 1: bird 0's action at decision 0 is forced to dev_action, then the policy continues;
    dev_mode 2: forced at decision 0, then bird 0 always STAYs (spec #12 one-shot deviation).  jit/vmap-safe."""
    B = mcfg.n_envs if n_envs is None else n_envs
    A, T = mcfg.n_agents, (ecfg.n_dec if n_dec is None else n_dec)
    sh, H, mem = mcfg.shared, mcfg.hidden, mcfg.memory
    sched = compile_schedule([], ecfg.e_max)

    cen = mcfg.use_c

    def cparams(params):
        return params["c"]

    def act_net(params, h, obs):
        return jax.vmap(jax.vmap(partial(net_step, memory=mem), in_axes=(None, 0, 0)), in_axes=(0, 0, 0))(params, h, obs)

    def rollout(params, key, dev_action=-1, dev_mode=0):
        k_reset, k_act = jax.random.split(key)
        st = jax.vmap(lambda k: reset(ecfg, k, jnp.int32(A), sched))(jax.random.split(k_reset, B))
        G = 1 if sh else A
        S = B * A if sh else B
        h0 = jnp.zeros((G, S, H), F32)
        if cen:
            h0 = (h0, jnp.zeros((G, S, H), F32))

        def body(carry, t):
            st, h = carry
            if cen:
                h, hc = h
            obs = prep_obs(ecfg, jax.vmap(lambda s: build_obs(ecfg, s))(st))            # (B,A,D)
            mask = jax.vmap(lambda s: action_mask(ecfg, s))(st)                         # (B,A,3)
            rs = st.mem_reset                                                           # (B,)
            hb = ungroup(h, B, A, sh)
            hb = jnp.where(rs[:, None, None], 0.0, hb)
            h_in = group(hb, sh)
            h2, lg, v = act_net(params, h_in, group(obs, sh))
            if cen:
                hcb = jnp.where(rs[:, None, None], 0.0, ungroup(hc, B, A, sh))
                hc2, _, vc = act_net(cparams(params), group(hcb, sh), group(central_obs(obs, mcfg.separate_critic), sh))
                v = vc
                h2 = (h2, hc2)
            lg = ungroup(lg, B, A, sh)
            v = ungroup(v, B, A, sh)
            lm = jnp.where(mask, lg, NEG)
            if greedy:
                a = jnp.argmax(lm, -1)
            else:
                a = jax.random.categorical(jax.random.fold_in(k_act, t), lm, axis=-1)
            a = a.astype(jnp.int32)
            logp = jnp.take_along_axis(jax.nn.log_softmax(lm, -1), a[..., None], -1)[..., 0]
            if dev_mode:
                forced = jnp.where(mask[:, 0, dev_action], dev_action, 0)
                a0 = jnp.where(t == 0, forced, jnp.where(dev_mode == 2, 0, a[:, 0]))
                a = a.at[:, 0].set(a0.astype(jnp.int32))
            pre = dict(slot=st.slot, lock=st.lock)
            st2, rec = jax.vmap(lambda s, x: decision_step(ecfg, s, x))(st, a)
            out = dict(obs=obs, mask=mask, act=a, logp=logp, val=v, logits=lg, rew=rec["reward"].astype(F32),
                       reset=rs, swapped=rec["swapped"], n_swaps=rec["n_swaps"], slot=pre["slot"], lock=pre["lock"])
            if ecfg.survival:
                out.update(arrived=rec["arrived"], cross=rec["cross"], e_frac=rec["e_frac_end"])
            return (st2, h2), out

        (stT, hT), data = lax.scan(body, (st, h0), jnp.arange(T))
        obsT = prep_obs(ecfg, jax.vmap(lambda s: build_obs(ecfg, s))(stT))
        if cen:
            hT, hcT = hT
        hb = ungroup(hT, B, A, sh)
        hb = jnp.where(stT.mem_reset[:, None, None], 0.0, hb)
        _, _, vT = act_net(params, group(hb, sh), group(obsT, sh))
        if cen:
            hcb = jnp.where(stT.mem_reset[:, None, None], 0.0, ungroup(hcT, B, A, sh))
            _, _, vT = act_net(cparams(params), group(hcb, sh), group(central_obs(obsT, mcfg.separate_critic), sh))
        vT = ungroup(vT, B, A, sh)
        if ecfg.survival:                    # the route ends: terminal state has value 0 (no time-limit bootstrap)
            vT = jnp.zeros_like(vT)
        return data, vT, stT
    return rollout


# ------------------------------------------------------------------------------------------------ GAE / update
def gae(rew, val, last_v, gamma, lam):
    """time-major (T,...) arrays, truncation bootstrap from last_v. Returns adv, ret."""
    def f(carry, x):
        adv_next, v_next = carry
        r, v = x
        delta = r + gamma * v_next - v
        adv = delta + gamma * lam * adv_next
        return (adv, v), adv
    _, adv = lax.scan(f, (jnp.zeros_like(last_v), last_v), (rew, val), reverse=True)
    return adv, adv + val


def ppo_loss(p, obs, mask, act, logp_old, adv, ret_n, rflag, mcfg: MarlConfig, cobs=None):
    """One group's minibatch: arrays (S,T,...). Returns scalar loss and aux."""
    seq = jax.vmap(partial(net_seq, memory=mcfg.memory, H=mcfg.hidden), in_axes=(None, 0, 0))
    lg, v = seq(p, obs, rflag)
    if mcfg.use_c:
        _, v = seq(p["c"], cobs, rflag)
    lm = jnp.where(mask, lg, NEG)
    lsm = jax.nn.log_softmax(lm, -1)
    logp = jnp.take_along_axis(lsm, act[..., None], -1)[..., 0]
    ratio = jnp.exp(logp - logp_old)
    an = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = -jnp.minimum(ratio * an, jnp.clip(ratio, 1 - mcfg.clip, 1 + mcfg.clip) * an).mean()
    vl = 0.5 * ((v - ret_n) ** 2).mean()
    pr = jnp.exp(lsm)
    ent = -jnp.where(mask, pr * lsm, 0.0).sum(-1).mean()
    loss = pg + mcfg.vf_coef * vl - mcfg.ent_coef * ent
    aux = dict(pg=pg, vl=vl, ent=ent, kl=((ratio - 1) - jnp.log(ratio)).mean(),
               clipfrac=(jnp.abs(ratio - 1) > mcfg.clip).mean())
    return loss, aux


def ppo_update(params, opt, sdata, lr, key, mcfg: MarlConfig, pi_on=1.0):
    """sdata: dict of grouped sequence arrays (G,S,T,...): obs, mask, act, logp, adv, ret_n, reset.
    Runs epochs x n_mb minibatch steps (minibatches split over the S axis). Returns params, opt, aux(mean)."""
    G, S = sdata["act"].shape[:2]
    mb = S // mcfg.n_mb
    keys = jax.random.split(key, mcfg.epochs)

    def one_group(p, batch):
        return ppo_loss(p, batch["obs"], batch["mask"], batch["act"], batch["logp"], batch["adv"], batch["ret_n"],
                        batch["reset"], mcfg, batch.get("cobs"))

    def total_loss(p, batch):
        ls, aux = jax.vmap(one_group)(p, batch)
        return ls.sum(), jax.tree_util.tree_map(jnp.mean, aux)

    def epoch(carry, k):
        params, opt = carry
        perm = jax.random.permutation(k, S)[: mb * mcfg.n_mb].reshape(mcfg.n_mb, mb)

        def mbstep(c, idx):
            params, opt = c
            batch = jax.tree_util.tree_map(lambda x: x[:, idx], sdata)
            (l, aux), g = jax.value_and_grad(total_loss, has_aux=True)(params, batch)
            g = {k: (v if k in ("wv", "bv", "c") else v * pi_on) for k, v in g.items()}     # pi_on=0: value head only
            params, opt, gn = adam_update(params, g, opt, lr, mcfg.max_grad_norm)
            aux = dict(aux, loss=l, gnorm=gn)
            return (params, opt), aux
        return lax.scan(mbstep, (params, opt), perm)

    (params, opt), aux = lax.scan(epoch, (params, opt), keys)
    return params, opt, jax.tree_util.tree_map(jnp.mean, aux)


class TrainState(NamedTuple):
    params: object
    opt: Adam
    rn: jnp.ndarray        # (G,2) running return mean / var
    it: jnp.ndarray


def init_train(key, ecfg, mcfg: MarlConfig) -> TrainState:
    params = init_params(key, obs_dim(ecfg), mcfg)
    G = 1 if mcfg.shared else mcfg.n_agents
    return TrainState(params, adam_init(params), jnp.stack([jnp.zeros((G,), F32), jnp.ones((G,), F32)], -1),
                      jnp.zeros((), jnp.int32))


def rollout_metrics(data, ecfg, T_sec):
    """Scalar diagnostics of a batch of episodes (time-major data)."""
    lock0 = data["lock"] == 0
    apex = data["slot"] == 0
    a = data["act"]
    f = lambda x: x.astype(F32)
    ret = (f(data["rew"]) * ecfg.delta_h).sum(0)                 # (B,A) P_solo*s per episode
    unl_apex = apex & lock0
    unl_fol = (data["slot"] >= 1) & lock0
    div = lambda n, d: n / jnp.maximum(d, 1.0)
    share = f(apex).mean(0)                                                # (B,A) per-episode lead share
    ent = -(jnp.where(share > 0, share * jnp.log(jnp.maximum(share, 1e-9)), 0.0)).sum(-1) / jnp.log(share.shape[-1])
    sv = {}
    if "arrived" in data:
        arr = f(data["arrived"]).sum(0)                                   # (B,A) 0/1 per episode
        sv = dict(arrival_rate=arr.mean(), all_arrive=(arr.min(-1) > 0.5).mean(),
                  efrac_final=(f(data["e_frac"])[-1] * arr).mean(), lead_share0=share[:, 0].mean())
    return dict(sv,
        max_lead_share=share.max(-1).mean(), min_lead_share=share.min(-1).mean(), lead_entropy=ent.mean(),
        ret=ret.mean(), swap_rate=data["n_swaps"].sum(0).mean() / T_sec,
        p_back_apex=div((f(a == 1) * unl_apex).sum(), unl_apex.sum()),
        p_stay_apex=div((f(a == 0) * unl_apex).sum(), unl_apex.sum()),
        p_back_non=div((f(a == 1) * unl_fol).sum(), unl_fol.sum()),
        p_fwd_fol=div((f(a == 2) * unl_fol).sum(), unl_fol.sum()))


def make_iteration(ecfg, mcfg: MarlConfig):
    roll = make_rollout(ecfg, mcfg)
    sh = mcfg.shared
    n_it = max(1, mcfg.total_agent_steps // (mcfg.n_envs * mcfg.n_agents * ecfg.n_dec))
    T_sec = ecfg.T

    def iteration(ts: TrainState, key):
        kr, ku = jax.random.split(key)
        data, last_v, _ = roll(ts.params, kr)
        # value head is normalised: convert to real scale for GAE
        # per-group stats broadcast: group index of (B,A) -> use ungroup of stats
        G = ts.rn.shape[0]
        B, A = mcfg.n_envs, mcfg.n_agents
        mean_ba = ungroup(jnp.broadcast_to(ts.rn[:, 0:1], (G, B * A if sh else B)), B, A, sh)
        std_ba = ungroup(jnp.broadcast_to(jnp.sqrt(ts.rn[:, 1:2]) + 1e-6, (G, B * A if sh else B)), B, A, sh)
        val = data["val"] * std_ba + mean_ba
        lastv = last_v * std_ba + mean_ba
        if ecfg.survival:
            lastv = jnp.zeros_like(lastv)
        warm = ts.it < mcfg.vf_warmup_iters          # critic warm-up: Monte-Carlo-ish targets (lam=1), 10x lr, head only
        lam_eff = jnp.where(warm, 1.0, mcfg.lam)
        adv, ret = gae(data["rew"] * mcfg.reward_scale, val, lastv, mcfg.gamma, lam_eff)
        # update return stats (EMA, rate decays to 0.02)
        retg = seq_group(ret, sh)                                      # (G,S,T)
        bm, bv = retg.mean((1, 2)), retg.var((1, 2))
        rate = jnp.maximum(1.0 / (ts.it.astype(F32) + 1.0), 0.02)
        rn = jnp.stack([ts.rn[:, 0] + rate * (bm - ts.rn[:, 0]), ts.rn[:, 1] + rate * (bv - ts.rn[:, 1])], -1)
        std_n = (jnp.sqrt(rn[:, 1]) + 1e-6)[:, None, None]
        ret_n = (retg - rn[:, 0][:, None, None]) / std_n
        sdata = dict(obs=seq_group(data["obs"], sh), mask=seq_group(data["mask"], sh), act=seq_group(data["act"], sh),
                     logp=seq_group(data["logp"], sh), adv=seq_group(adv, sh), ret_n=ret_n,
                     reset=seq_group(data["reset"][:, :, None].repeat(A, 2), sh))
        if mcfg.use_c:
            sdata["cobs"] = seq_group(central_obs(data["obs"], mcfg.separate_critic), sh)
        frac = ts.it.astype(F32) / n_it
        lr = mcfg.lr * (1.0 - frac * (1.0 - mcfg.lr_final_frac)) * jnp.where(warm, 10.0, 1.0)
        pi_on = (ts.it >= mcfg.vf_warmup_iters).astype(F32)
        params, opt, aux = ppo_update(ts.params, ts.opt, sdata, lr, ku, mcfg, pi_on)
        met = rollout_metrics(data, ecfg, T_sec)
        met.update(aux)
        met["finite"] = jnp.all(jnp.asarray([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(params)]))
        # explained variance of the (real-scale) value prediction
        met["ev"] = 1.0 - jnp.var(ret - val) / (jnp.var(ret) + 1e-8)
        return TrainState(params, opt, rn, ts.it + 1), met
    iteration.n_it = n_it
    return iteration


# ------------------------------------------------------------------------------------------------ checkpoint / loop
def save_ckpt(path, ts: TrainState, ecfg=None, mcfg=None, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = dict(ts=jax.tree_util.tree_map(np.asarray, ts), mcfg=dataclasses.asdict(mcfg) if mcfg else None,
                extra=extra)
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def load_ckpt(path):
    blob = pickle.load(open(path, "rb"))
    ts = jax.tree_util.tree_map(jnp.asarray, blob["ts"])
    mcfg = MarlConfig(**blob["mcfg"]) if blob.get("mcfg") else None
    return ts, mcfg, blob.get("extra")


def train(mcfg: MarlConfig, seed: int, out_dir: str, log_every=1, ckpt_every=50, verbose=True, max_iters=None,
          init_params_=None):
    """Runs the PPO loop (jit'ed iteration, python outer loop); writes log.jsonl + final.ckpt into out_dir."""
    ecfg = mcfg.env_cfg()
    os.makedirs(out_dir, exist_ok=True)
    it_fn = make_iteration(ecfg, mcfg)
    step = jax.jit(it_fn)
    n_it = it_fn.n_it if max_iters is None else max_iters
    key = jax.random.PRNGKey(seed)
    k0, key = jax.random.split(key)
    ts = init_train(k0, ecfg, mcfg)
    if init_params_ is not None:                       # warm start (BC policy); fresh Adam / return stats
        ip = dict(init_params_)
        if mcfg.use_c and "c" not in ip:             # BC gives actor only: keep the freshly initialised critic
            ip["c"] = ts.params["c"]
        ts = ts._replace(params=jax.tree_util.tree_map(jnp.asarray, ip), opt=adam_init(ip))
    per_it = mcfg.n_envs * mcfg.n_agents * ecfg.n_dec
    logf = open(os.path.join(out_dir, "log.jsonl"), "w")
    t0 = time.time()
    for i in range(n_it):
        key, k = jax.random.split(key)
        ts, met = step(ts, k)
        if i % log_every == 0 or i == n_it - 1:
            m = {kk: float(v) for kk, v in met.items()}
            m.update(it=i, agent_steps=(i + 1) * per_it, wall=time.time() - t0)
            logf.write(json.dumps(m) + "\n")
            logf.flush()
            if verbose and (i % max(1, n_it // 20) == 0 or i == n_it - 1):
                print(f"it {i}/{n_it} steps {m['agent_steps']:.2e} ret {m['ret']:.1f} swap {m['swap_rate']:.4f} "
                      f"pBackApex {m['p_back_apex']:.2f} ent {m['ent']:.3f} wall {m['wall']:.0f}s", flush=True)
        if ckpt_every and (i + 1) % ckpt_every == 0:
            save_ckpt(os.path.join(out_dir, "final.ckpt"), ts, ecfg, mcfg, dict(it=i + 1))
    save_ckpt(os.path.join(out_dir, "final.ckpt"), ts, ecfg, mcfg, dict(it=n_it, wall=time.time() - t0))
    logf.close()
    return ts, ecfg
