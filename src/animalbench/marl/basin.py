"""M3 basin experiments: episode-level leadership metrics, pre-registered classification, scripted benchmarks.

Classification thresholds are fixed here BEFORE looking at any M3 result (task spec, 2026-10-25):
  fixed    : mean-over-episodes max_lead_share >= 0.9  AND swap_rate < 0.005 /s
  rotation : mean-over-episodes min_lead_share >= 0.2  AND normalised lead entropy >= 0.8 AND swap_rate > 0.01 /s
  other    : everything else
Lead shares are computed PER EPISODE (random_slots=True makes the initial leader differ across episodes, so
per-bird averages over episodes would hide a fixed leader), then averaged over episodes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..core import compile_schedule
from ..scenarios.geese.env import make_rollout as env_rollout

THR = dict(fixed_share=0.9, fixed_swap=0.005, rot_min_share=0.2, rot_entropy=0.8, rot_swap=0.01)


def episode_metrics(slot, n_swaps, rew, delta_h=2.0, T_sec=600.0):
    """slot (T,B,A), n_swaps (T,B), rew (T,B,A) per-decision mean reward (P_solo units) -> dict of scalars."""
    slot, n_swaps, rew = np.asarray(slot), np.asarray(n_swaps), np.asarray(rew, dtype=np.float64)
    share = (slot == 0).mean(0)                                        # (B,A)
    A = share.shape[-1]
    ent = -(np.where(share > 0, share * np.log(np.maximum(share, 1e-12)), 0.0)).sum(-1) / np.log(A)
    ret = rew.sum(0) * delta_h                                         # (B,A) P_solo*s per episode
    lid = (slot == 0).argmax(-1)                                       # (T,B)
    ten = []
    for b in range(lid.shape[1]):
        run = 1
        for t in range(1, lid.shape[0]):
            if lid[t, b] == lid[t - 1, b]:
                run += 1
            else:
                ten.append(run); run = 1
        ten.append(run)
    return dict(swap_rate=float(n_swaps.sum(0).mean() / T_sec), max_lead_share=float(share.max(-1).mean()),
                min_lead_share=float(share.min(-1).mean()), lead_entropy=float(ent.mean()),
                median_tenure_s=float(np.median(ten) * delta_h),
                ret_bird=float(ret.mean()), ret_worst_bird=float(ret.min(-1).mean()),
                ret_ep_group_sd=float(ret.mean(-1).std()), n_episodes=int(slot.shape[1]))


def classify(m, thr=THR):
    if m["max_lead_share"] >= thr["fixed_share"] and m["swap_rate"] < thr["fixed_swap"]:
        return "fixed"
    if (m["min_lead_share"] >= thr["rot_min_share"] and m["lead_entropy"] >= thr["rot_entropy"]
            and m["swap_rate"] > thr["rot_swap"]):
        return "rotation"
    return "other"


def scripted_data(ecfg, policy, n_envs=64, seed=0, N=3):
    """Run a scripted Policy through the real environment (vmapped). Returns numpy dict slot/lock/n_swaps/rew (T,B,..)."""
    sched = compile_schedule([], ecfg.e_max)
    f = env_rollout(ecfg, policy)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    _, recs = jax.jit(jax.vmap(lambda k: f(k, jnp.int32(N), sched)))(keys)      # (B,T,..)
    g = lambda x: np.moveaxis(np.asarray(x), 0, 1)
    return dict(slot=g(recs["slot"]), lock=g(recs["lock"]), n_swaps=g(recs["n_swaps"]), rew=g(recs["reward"]))
