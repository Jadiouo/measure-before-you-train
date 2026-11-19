"""M4br experiment 2: PPO-collapse-bias controls. N=3, mutual, kappa=1, identity none, random_slots, shared GRU, 10M steps.
settings: ent05 (ent_coef .05) | ent10lr (ent_coef .1, lr x2) | team (reward_mode team).  Classification: marl/basin.py (pre-registered).
usage: run_m4br_exp2.py --cell ent05 --seed 0"""
import argparse, json, os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, numpy as np
import animalbench  # noqa
from animalbench.marl import MarlConfig, train
from animalbench.marl.ippo import make_rollout
from animalbench.marl.basin import episode_metrics, classify

CELLS = {"ent05": dict(ent_coef=0.05), "ent10lr": dict(ent_coef=0.1, lr=6e-4), "team": dict(reward_mode="team"),
         "base": dict()}
ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=float, default=1e7); ap.add_argument("--root", default="runs/m4br/exp2")
ap.add_argument("--kappa", type=float, default=1.0)   # extra (non-registered) check: kappa 8
a = ap.parse_args()
out = os.path.join(a.root, f"{a.cell}_s{a.seed}"); os.makedirs(out, exist_ok=True)
m = MarlConfig(swap_rule="mutual", kappa=a.kappa, shared=True, memory="gru", random_slots=True,
               total_agent_steps=int(a.steps), **CELLS[a.cell])
ecfg = m.env_cfg()


def evaluate(params, greedy=False, seed=777, n_envs=128):
    mm = MarlConfig(**{**m.__dict__, "n_envs": n_envs})
    d, _, _ = jax.jit(make_rollout(ecfg, mm, greedy=greedy))(params, jax.random.PRNGKey(seed))
    d = {k: np.asarray(v) for k, v in d.items()}
    r = episode_metrics(d["slot"], d["n_swaps"], d["rew"], ecfg.delta_h, ecfg.T)
    r["class"] = classify(r); r["greedy"] = greedy
    return r


ts, _ = train(m, 100 * a.seed + 7, out, ckpt_every=100)
res = dict(cell=a.cell, seed=a.seed, steps=a.steps, mcfg=m.__dict__, sampled=evaluate(ts.params), greedy=evaluate(ts.params, True))
json.dump(res, open(os.path.join(out, "eval.json"), "w"), indent=1)
print("FINAL", res["sampled"]["class"], {k: round(v, 4) for k, v in res["sampled"].items() if isinstance(v, float)}, flush=True)
