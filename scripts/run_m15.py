"""M1.5: N=3 IPPO smoke test.  usage: run_m15.py --rule unilateral --shared 1 --seed 0 --steps 2e7 --out runs/m15/x
Trains, then evaluates the final checkpoint and writes eval.json next to log.jsonl."""
import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import animalbench  # noqa: F401,E402
from animalbench.marl import MarlConfig, load_ckpt, train  # noqa: E402
from animalbench.marl.evaluate import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rule", default="unilateral")
ap.add_argument("--shared", type=int, default=1)
ap.add_argument("--memory", default="gru")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=float, default=2e7)
ap.add_argument("--n_envs", type=int, default=64)
ap.add_argument("--out", required=True)
ap.add_argument("--ent", type=float, default=0.01)
ap.add_argument("--lr", type=float, default=3e-4)
a = ap.parse_args()
m = MarlConfig(swap_rule=a.rule, shared=bool(a.shared), memory=a.memory, total_agent_steps=int(a.steps),
               n_envs=a.n_envs, ent_coef=a.ent, lr=a.lr)
ts, ecfg = train(m, a.seed, a.out, ckpt_every=100)
res = evaluate(ecfg, m, ts.params)
json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("sampled",)}, indent=1, default=float)[:1500])
print(json.dumps(res["counterfactual"], indent=1))
