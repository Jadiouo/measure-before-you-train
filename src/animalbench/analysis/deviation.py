"""One-shot deviation tests (spec v0.3 #12-#14).

Setup: N birds in a settled V (decision d0 = 30 -> t = 60 s), everybody else follows a fixed behaviour
('stay' = all stay, 'rule_blind' = every other bird runs rule_blind), the tested bird follows
  deviate=True : one request at d0 (apex: BACK, slot-1 bird: FWD), then STAY
  deviate=False: STAY all the time.
Advantage = (individual return with deviation) - (without), over H seconds, return = sum of interval reward * delta_h in
P_solo*s units (the reward is -P_eff/P_solo, so a positive advantage means the deviation pays).  Paired by seed
(same wind / initial state).
"""
from __future__ import annotations

import numpy as np

from ..core import crosswind
from ..scenarios.geese import GeeseConfig, rule_blind, stay_policy
from ..scenarios.geese.hlc import BACK, FWD, STAY, Policy
from .batch import run_batch

import jax.numpy as jnp


def wrap_policy(cfg, base: Policy, bird: int, d0: int, action: int, deviate: bool) -> Policy:
    def act(ps, obs, st, mask, key=None):
        a, ps2 = base.act(ps, obs, st, mask, key)
        mine = jnp.where(deviate & (st.dec == d0), action, STAY)
        mine = jnp.where(mask[bird, mine], mine, STAY)
        return a.at[bird].set(mine).astype(jnp.int32), ps2
    return Policy(f"{base.name}+dev", base.init, act)


def bootstrap_ci(x, B=2000, seed=0, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return tuple(np.percentile(m, q))


def advantage(others="stay", who="apex", seeds=range(100), N=9, d0=30, H=60.0, noisy=False, swap_rule="unilateral",
              **cfg_kw):
    """Returns dict(adv=(n_seeds,), ret_dev, ret_stay, mean, ci) for the tested bird."""
    nd = d0 + int(round(H / 2.0))
    cfg = GeeseConfig(n_max=N, T=2.0 * nd, swap_rule=swap_rule, log_low=False, log_D=False, init_mode="v",
                      **cfg_kw)
    base = stay_policy(cfg) if others == "stay" else rule_blind(cfg)
    bird, action = (0, BACK) if who == "apex" else (1, FWD)
    events = [crosswind(0.0, 1e4, speed=0.0, ramp=1.0, ou_sigma=0.5)] if noisy else []
    out = {}
    for dev in (False, True):
        pol = wrap_policy(cfg, base, bird, d0, action, dev)
        r = run_batch(cfg, pol, seeds, N, n_dec=nd, events=events, keep=("reward", "slot", "alive", "P_mean"))
        out[dev] = r
    ret = {k: out[k]["reward"][:, d0:, bird].sum(1) * cfg.delta_h for k in (False, True)}
    adv = ret[True] - ret[False]
    return dict(adv=adv, ret_dev=ret[True], ret_stay=ret[False], mean=float(adv.mean()),
                ci=bootstrap_ci(adv), cfg=cfg, recs=out)


def mean_individual_return(policy="stay", seeds=range(20), N=9, T=600.0, swap_rule="unilateral", **cfg_kw):
    """Mean over seeds and birds of the individual return sum(reward)*delta_h (P_solo*s) over a whole episode,
    plus the per-seed vector (used for the 'group mean is equal for fixed leader and rotation' check)."""
    cfg = GeeseConfig(n_max=N, T=T, swap_rule=swap_rule, log_low=False, log_D=False, init_mode="v", **cfg_kw)
    pol = stay_policy(cfg) if policy == "stay" else rule_blind(cfg)
    r = run_batch(cfg, pol, seeds, N, keep=("reward", "alive", "n_swaps"))
    ret = (r["reward"] * cfg.delta_h).sum(1)           # (seeds, N)
    return dict(per_seed=ret.mean(1), mean=float(ret.mean()), n_swaps=r["n_swaps"].sum(1), per_bird=ret)


def e0_advantage(E0_coef, seeds=range(20), N=9, d0=200, T=600.0, **cfg_kw):
    """Apex that has led since t=0 (others stay) vs the same bird retreating once at decision d0, with an energy cap
    E0 = E0_coef * P_solo * T.  Returns reward advantage (P_solo*s) and seconds alive until T for both."""
    nd = int(T / 2.0)
    cfg = GeeseConfig(n_max=N, T=T, log_low=False, log_D=False, init_mode="v", E0_coef=E0_coef, **cfg_kw)
    out = {}
    for dev in (False, True):
        pol = wrap_policy(cfg, stay_policy(cfg), 0, d0, BACK, dev)
        out[dev] = run_batch(cfg, pol, seeds, N, n_dec=nd, keep=("reward", "alive"))
    ret = {k: out[k]["reward"][:, d0:, 0].sum(1) * cfg.delta_h for k in out}
    alive = {k: out[k]["alive"][:, d0:, 0].sum(1) * cfg.delta_h for k in out}
    return dict(adv=ret[True] - ret[False], alive_gain=alive[True] - alive[False], alive_stay=alive[False],
                alive_dev=alive[True])
