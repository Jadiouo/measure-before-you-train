"""Evaluation + pre-registered classification for the survival-migration IPPO experiment (notes/impl/survival_ippo_prereg.md).

Thresholds are frozen in the prereg BEFORE any training.  basin.py thresholds (THR) are reused unchanged; one new category
`strong_lead` is added with the explicit thresholds below."""
from __future__ import annotations

import jax
import numpy as np

from .basin import THR, classify as basin_classify, episode_metrics
from .ippo import MarlConfig, make_rollout

# New (frozen in the prereg): body 0 is the strong bird (energy cap x(1+h)).
STRONG = dict(share0_min=0.5, ratio_min=1.5, swap_min=0.005, fixed_body0_min=0.8)
TREADMILL_SWAP = 0.1      # /s, same as notes/impl/M3basin_decisions.md ("treadmill" = rotation with swap rate >= 0.1/s)


def survival_metrics(d, ecfg):
    """d = numpy dict from ippo.make_rollout (time-major).  Returns the basin metrics + per-body lead share + survival numbers."""
    m = episode_metrics(d["slot"], d["n_swaps"], d["rew"], ecfg.delta_h, ecfg.T)
    share = (d["slot"] == 0).mean(0)                                   # (B,A) per-episode lead share per body
    m["lead_share_by_bird"] = share.mean(0).tolist()
    arr = d["arrived"].sum(0).astype(float)                            # (B,A)
    m["arrival_rate"] = float(arr.mean())
    m["arrival_by_bird"] = arr.mean(0).tolist()
    m["all_arrive"] = float((arr.min(-1) > 0.5).mean())
    m["efrac_final"] = float((d["e_frac"][-1] * arr).mean())
    m["reward_sum"] = float(d["rew"].sum(0).mean())
    return m


def classify_survival(m):
    """fixed | treadmill | strong_lead | rotation | other  (evaluated in this order)."""
    base = basin_classify(m)
    s = m["lead_share_by_bird"]
    if base == "fixed":                      # a fixed leader that is (nearly) always body 0 = the extreme of strong_lead
        return "strong_lead" if s[0] >= STRONG["fixed_body0_min"] else "fixed"
    if base == "rotation" and m["swap_rate"] >= TREADMILL_SWAP:
        return "treadmill"
    if (s[0] >= STRONG["share0_min"] and s[0] >= STRONG["ratio_min"] * max(s[1:]) and m["swap_rate"] > STRONG["swap_min"]):
        return "strong_lead"
    return base                      # rotation | other


def evaluate(mcfg: MarlConfig, params, seed=777, n_envs=128, greedy=False):
    ecfg = mcfg.env_cfg()
    mm = MarlConfig(**{**mcfg.__dict__, "n_envs": n_envs})
    data, _, _ = jax.jit(make_rollout(ecfg, mm, greedy=greedy))(params, jax.random.PRNGKey(seed))
    d = {k: np.asarray(v) for k, v in data.items()}
    m = survival_metrics(d, ecfg)
    m["class"] = classify_survival(m)
    m["greedy"] = greedy
    return m
