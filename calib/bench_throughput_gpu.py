"""GPU throughput benchmark (report only): N=9, B in {256,1024,4096}, float32, rule_blind.
Same workload as bench_throughput.py; writes calib/out/throughput_gpu.json. Run via gpujob with .venv-gpu."""
import json
import os
import time

import jax
import jax.numpy as jnp

from animalbench.core import compile_schedule
from animalbench.scenarios.geese import GeeseConfig, make_rollout, rule_blind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def run(B, n_dec=10, N=9, n_max=9, reps=3):
    cfg = GeeseConfig(n_max=n_max, dtype="float32", log_D=False, log_low=False, use_events=False, init_mode="scatter")
    f = jax.jit(jax.vmap(make_rollout(cfg, rule_blind(cfg), n_dec), in_axes=(0, None, None)))
    sched = compile_schedule([], cfg.e_max)
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    t0 = time.perf_counter()
    jax.block_until_ready(f(keys, jnp.int32(N), sched)[1]["reward"])
    compile_s = time.perf_counter() - t0
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(keys, jnp.int32(N), sched)[1]["reward"])
        ts.append(time.perf_counter() - t)
    dt = min(ts)
    steps = B * N * n_dec
    return dict(B=B, N=N, decisions=n_dec, seconds_best=dt, seconds_all=ts, compile_plus_first_s=compile_s,
                agent_steps_per_s=steps / dt, speedup_vs_cpu=steps / dt / 34245.2)


if __name__ == "__main__":
    dev = jax.devices()[0]
    res = [run(B) for B in (256, 1024, 4096)]
    ms = dev.memory_stats() or {}
    out = dict(device=str(dev), device_kind=dev.device_kind, jax=jax.__version__, jaxlib=jax.lib.__name__ and __import__("jaxlib").__version__,
               cpu_baseline_agent_steps_per_s=34245.23127537634, results=res,
               peak_bytes_in_use=ms.get("peak_bytes_in_use"), note="agent-step = one high-level decision of one bird")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "throughput_gpu.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
