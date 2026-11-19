"""Run one episode and write a trajectory file.   python scripts/run_episode.py --N 9 --T 600 --agent rule_blind
Prints S, apex handovers and timing; trajectory goes to runs/ (git-ignored)."""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp

from animalbench.core import compile_schedule
from animalbench.scenarios.geese import (GeeseConfig, make_rollout, rule_blind, rule_commander, rule_oracle,
                                         stay_policy)
from animalbench.scenarios.geese.env import to_trajectory
from animalbench.scenarios.geese.summary import episode_stats

POLICIES = dict(rule_blind=rule_blind, rule_commander=rule_commander, rule_oracle=rule_oracle,
                stay=stay_policy)


def run(N=9, T=600.0, agent="rule_blind", seed=0, n_max=None, swap_rule="unilateral", init_mode="v",
        out_dir=None, events=(), repeat=2, **cfg_kw):
    cfg = GeeseConfig(n_max=n_max or N, T=T, swap_rule=swap_rule, init_mode=init_mode, **cfg_kw)
    pol = POLICIES[agent](cfg)
    f = jax.jit(make_rollout(cfg, pol))
    sched = compile_schedule(list(events), cfg.e_max)
    key = jax.random.PRNGKey(seed)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        st, recs = f(key, jnp.int32(N), sched)
        jax.block_until_ready(recs["reward"])
        times.append(time.perf_counter() - t0)
    meta = dict(scenario="geese", agent=agent, seed=seed, N=N, n_max=cfg.n_max, T=T, dt=cfg.dt, delta_h=cfg.delta_h,
                stride=cfg.log_stride, gain=cfg.gain, P_solo=cfg.P_solo, b=cfg.b, swap_rule=swap_rule,
                init_mode=init_mode, jax=jax.__version__, units=dict(D="P_solo*s", reward="mean(-P_eff/P_solo)"))
    traj = to_trajectory(cfg, recs, meta, sched)
    stats = episode_stats(traj, cfg)
    n_low = cfg.n_dec * cfg.n_sub
    stats.update(compile_plus_run_s=times[0], run_s=times[-1], ms_per_low_step=1e3 * times[-1] / n_low,
                 ms_per_decision=1e3 * times[-1] / cfg.n_dec, n_dec=cfg.n_dec, n_low_steps=n_low)
    path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"traj_geese_{agent}_N{N}_T{int(T)}_seed{seed}.npz")
        traj.save(path)
        stats["traj_path"] = path
    return traj, stats, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=9)
    ap.add_argument("--T", type=float, default=600.0)
    ap.add_argument("--agent", default="rule_blind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap_rule", default="unilateral")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    _, st, _ = run(a.N, a.T, a.agent, a.seed, swap_rule=a.swap_rule, out_dir=a.out)
    st.pop("apex_tenures_s")
    print(json.dumps(st, indent=1))
