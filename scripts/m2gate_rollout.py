"""M2 gate: intervention rollouts (I1/I2) for scripted agents and the M1.5 learned policy.
One jit/vmap compile per agent; all conditions x seeds are batched together via a batched PerturbSchedule.
Writes <out>/<agent>.npz with leader/slot/alive/D(float32)/immune/id_owner for every condition (leading axes: cond, seed).
Run on GPU through gpujob (.venv-gpu) or on CPU (.venv).  Usage:
  python scripts/m2gate_rollout.py --out runs/m2gate/gpu --seeds 50 [--agents S1_treadmill,...]"""
import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from animalbench.analysis import known_answer as ka
from animalbench.core import compile_schedule, immune_on, relabel
from animalbench.marl.policy_adapter import load_marl_policy
from animalbench.scenarios.geese import GeeseConfig, SCRIPTED, make_rollout
from animalbench.scenarios.geese.scripted_id import REGISTRY_ID

T_EVENT = 300.0
CONDS = ["BASE", "I1_R", "I1_S", "I1_M", "I1_RM", "I2_F", "I2_S"]
SCRIPTED_ALL = {**SCRIPTED, **REGISTRY_ID}
AGENTS = ["S1_treadmill", "S2_state_rotation", "S2f_state_rotation_fair", "S3c_reciprocity_code", "S3p_reciprocity_punish",
          "N_cyc", "N_rnd", "N_fix", "S2f_ia", "N_cyc_ia", "N_rnd_ia", "M15_unilateral_shared_s0", "P_body", "P_body_codenoise", "P_code_lookup"]
M15_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "m15",
                        "unilateral_shared_s0", "final.ckpt")


def cond_events(cond, N, base):
    ident = list(range(N))
    ev = {"BASE": [], "I1_R": [relabel(T_EVENT)], "I1_S": [relabel(T_EVENT, perm=ident)],
          "I1_M": [relabel(T_EVENT, perm=ident, reset_mem=True)], "I1_RM": [relabel(T_EVENT, reset_mem=True)],
          "I2_F": [immune_on(T_EVENT, "last")], "I2_S": [immune_on(T_EVENT, "last", on=False)]}[cond]
    return list(base) + ev


def build(agent):
    if agent.startswith("M15_"):
        cfg, pol = load_marl_policy(M15_CKPT, log_D=True, log_low=False)
        return cfg, pol, 3, []
    mk, rule = SCRIPTED_ALL[agent]
    cfg = GeeseConfig(n_max=9, T=600.0, swap_rule=rule, log_low=False, log_D=True, init_mode="scatter")
    return cfg, mk(cfg), 9, list(ka.EVENTS)


def run_agent(agent, n_seeds, chunk=100, seed0=0):
    cfg, pol, N, base = build(agent)
    f = make_rollout(cfg, pol)
    g = jax.jit(jax.vmap(lambda k, s: f(k, jnp.int32(N), s)))
    scheds = [compile_schedule(cond_events(c, N, base), cfg.e_max) for c in CONDS]
    rows = [(ci, seed0 + s) for ci in range(len(CONDS)) for s in range(n_seeds)]
    keep = ("leader", "slot", "alive", "D", "immune", "id_owner", "swapped", "n_alive")
    parts = {k: [] for k in keep}
    for i in range(0, len(rows), chunk):
        sub = rows[i:i + chunk]
        keys = jnp.stack([jax.random.PRNGKey(s) for _, s in sub])
        sch = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *[scheds[ci] for ci, _ in sub])
        _, recs = g(keys, sch)
        for k in keep:
            parts[k].append(np.asarray(recs[k]))
    out = {}
    for k in keep:
        a = np.concatenate(parts[k], 0)
        a = a.astype(np.float32) if k == "D" else a
        out[k] = a.reshape((len(CONDS), n_seeds) + a.shape[1:])
    return out, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--agents", default=",".join(AGENTS))
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0, help="first PRNG seed (seeds seed0 .. seed0+seeds-1)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("devices:", jax.devices(), flush=True)
    for ag in a.agents.split(","):
        t = time.time()
        out, cfg = run_agent(ag, a.seeds, a.chunk, a.seed0)
        np.savez_compressed(os.path.join(a.out, f"{ag}.npz"), conds=np.array(CONDS), dh=cfg.delta_h, **out)
        print(f"{ag}: {time.time() - t:.1f}s  leader{out['leader'].shape}", flush=True)
