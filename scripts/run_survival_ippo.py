"""Survival-migration IPPO (N=3).  usage: run_survival_ippo.py --cell low|edge|high --alpha 0.5 --seed 0 --steps 2e7
Cells (eps = energy cap / solo-flight energy, m_g = burst strength; h = 0.05, p = 0.05, beta = 0.5):
  low (1.06, 0.5) | edge (1.02, 1.0) | high (0.98, 1.0).   Writes <root>/<cell>_a<alpha>_s<seed>/{log.jsonl,final.ckpt,eval.json}.
Always also evaluates the baseline policies of the SAME cell: untrained (random) network, and the 'stay' scripted fixed leader."""
import argparse, json, os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import jax, jax.numpy as jnp, numpy as np
import animalbench  # noqa
from animalbench.core import compile_schedule
from animalbench.marl import MarlConfig, train
from animalbench.marl.ippo import init_train
from animalbench.marl.survival_eval import evaluate, survival_metrics, classify_survival
from animalbench.scenarios.geese import make_rollout as env_rollout, stay_policy

CELLS = {"low": (1.06, 0.5), "edge": (1.02, 1.0), "high": (0.98, 1.0)}
ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True); ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=float, default=2e7)
ap.add_argument("--root", default="runs/survival"); ap.add_argument("--shaping", default="none")
ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--lam", type=float, default=0.95)
ap.add_argument("--baseline_only", action="store_true")
a = ap.parse_args()
eps, mg = CELLS[a.cell]
out = os.path.join(a.root, f"{a.cell}_a{a.alpha}_s{a.seed}"); os.makedirs(out, exist_ok=True)
m = MarlConfig(survival=True, surv_alpha=a.alpha, surv_eps=eps, surv_burst_m=mg, surv_shaping=a.shaping,
               swap_rule="mutual", shared=True, memory="gru", random_slots=True, gamma=a.gamma, lam=a.lam,
               total_agent_steps=int(a.steps))
ecfg = m.env_cfg()


def baselines():
    res = {}
    ts0 = init_train(jax.random.PRNGKey(12345), ecfg, m)               # untrained net ~ uniform over legal actions
    res["random"] = evaluate(m, ts0.params)
    f = env_rollout(ecfg, stay_policy(ecfg))
    sched = compile_schedule([], ecfg.e_max)
    keys = jax.random.split(jax.random.PRNGKey(777), 128)
    _, recs = jax.jit(jax.vmap(lambda k: f(k, jnp.int32(3), sched)))(keys)
    g = lambda x: np.moveaxis(np.asarray(x), 0, 1)
    d = dict(slot=g(recs["slot"]), n_swaps=g(recs["n_swaps"]), rew=g(recs["reward"]), arrived=g(recs["arrived"]),
             e_frac=g(recs["e_frac_end"]))
    res["stay_fixed_leader"] = survival_metrics(d, ecfg)
    return res


t0 = time.time()
base = baselines()
print("BASELINE random arrival", round(base["random"]["arrival_rate"], 3), "stay arrival", round(base["stay_fixed_leader"]["arrival_rate"], 3), flush=True)
if a.baseline_only:
    json.dump(dict(cell=a.cell, alpha=a.alpha, baselines=base), open(os.path.join(out, "baseline.json"), "w"), indent=1)
    raise SystemExit
ts, _ = train(m, 100 * a.seed + 7, out, ckpt_every=100)
res = dict(cell=a.cell, alpha=a.alpha, seed=a.seed, steps=a.steps, wall_s=time.time() - t0, device=str(jax.devices()[0]),
           mcfg=m.__dict__, baselines=base, sampled=evaluate(m, ts.params), greedy=evaluate(m, ts.params, greedy=True))
json.dump(res, open(os.path.join(out, "eval.json"), "w"), indent=1)
s = res["sampled"]
print("FINAL", s["class"], "arrival", round(s["arrival_rate"], 3), "random", round(base["random"]["arrival_rate"], 3),
      "lead_share", [round(x, 2) for x in s["lead_share_by_bird"]], "swap", round(s["swap_rate"], 4), "wall", round(res["wall_s"]), flush=True)
