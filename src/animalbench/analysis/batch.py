"""Run many seeds of the real environment (vmap over seeds, same `make_rollout` as `run_episode`) and return numpy
arrays with a leading seed axis.  CPU only; seeds are processed in chunks to bound memory."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..core import compile_schedule
from ..scenarios.geese import GeeseConfig, make_rollout

KEEP = ("leader", "slot", "alive", "swapped", "reward", "P_mean", "D", "n_swaps", "eff_action", "action_req", "lock",
        "phi", "E_used", "n_alive")


def run_batch(cfg: GeeseConfig, policy, seeds, N: int, n_dec=None, events=(), chunk=25, keep=KEEP,
              immune=None, init_override=None):
    """-> dict name -> array (len(seeds), n_dec, ...). `cfg.log_low` should be False for speed."""
    sched = compile_schedule(list(events), cfg.e_max)
    f = make_rollout(cfg, policy, n_dec)
    g = jax.jit(jax.vmap(lambda k: f(k, jnp.int32(N), sched, init_override, immune)))
    out = None
    seeds = list(seeds)
    for i in range(0, len(seeds), chunk):
        keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds[i:i + chunk]])
        _, recs = g(keys)
        part = {k: np.asarray(recs[k]) for k in keep if k in recs}
        if out is None:
            out = {k: [v] for k, v in part.items()}
        else:
            for k, v in part.items():
                out[k].append(v)
    return {k: np.concatenate(v, 0) for k, v in out.items()}
