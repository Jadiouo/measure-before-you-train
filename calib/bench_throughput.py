"""#25 throughput benchmark (report only): CPU, N=9, B=256, float32. Writes calib/out/throughput.json."""
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from animalbench.core import compile_schedule
from animalbench.scenarios.geese import GeeseConfig, make_rollout, rule_blind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def run(B=256, n_dec=10, N=9, n_max=9, dtype="float32"):
    cfg = GeeseConfig(n_max=n_max, dtype=dtype, log_D=False, log_low=False, use_events=False, init_mode="scatter")
    f = jax.jit(jax.vmap(make_rollout(cfg, rule_blind(cfg), n_dec), in_axes=(0, None, None)))
    sched = compile_schedule([], cfg.e_max)
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    jax.block_until_ready(f(keys, jnp.int32(N), sched)[1]["reward"])      # compile + warm
    t = time.perf_counter()
    jax.block_until_ready(f(keys, jnp.int32(N), sched)[1]["reward"])
    dt = time.perf_counter() - t
    agent_dec = B * N * n_dec
    return dict(B=B, N=N, n_max=n_max, dtype=dtype, decisions=n_dec, seconds=dt,
                agent_steps_per_s=agent_dec / dt,                       # 1 agent-step = 1 agent decision (0.5 Hz)
                low_level_agent_steps_per_s=agent_dec * cfg.n_sub / dt,
                env_decisions_per_s=B * n_dec / dt,
                note="agent-step = one high-level decision of one bird (20 physics steps); <5k/s means re-cut matrix (spec sec 7)")


if __name__ == "__main__":
    r = run()
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(os.path.join(OUT, "throughput.json"), "w"), indent=1)
    print(json.dumps(r))
