"""M3 basin / kappa experiments (N=3, IPPO shared GRU, identity none, random_slots).
usage: run_m3basin.py --cell A --seed 0 [--steps 1e7]
cells: A kappa0 random | B kappa0 BC-warm(S2) | C1/C2 kappa lo/hi random | D1/D2 same kappa BC-warm | E unilateral+kappa hi random
Writes runs/m3basin/<cell>_s<seed>/{log.jsonl,final.ckpt,eval.json,(bc_eval.json)}.  Thresholds: marl/basin.py (pre-registered)."""
import argparse, json, os, pickle, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
import animalbench  # noqa
from animalbench.marl import MarlConfig, train
from animalbench.marl.ippo import make_rollout
from animalbench.marl.basin import episode_metrics, classify, THR
from animalbench.marl.warmstart import dagger_bc
from animalbench.scenarios.geese import scripted as S

KLO, KHI = 1.0, 8.0     # kappa levels, set from runs/m3basin/kappa_threshold.json (see notes/impl/M3basin_decisions.md)
CELLS = {
    "A": dict(kappa=0.0, warm=False, rule="mutual"), "B": dict(kappa=0.0, warm=True, rule="mutual"),
    "C1": dict(kappa=KLO, warm=False, rule="mutual"), "C2": dict(kappa=KHI, warm=False, rule="mutual"),
    "D1": dict(kappa=KLO, warm=True, rule="mutual"), "D2": dict(kappa=KHI, warm=True, rule="mutual"),
    "E": dict(kappa=KHI, warm=False, rule="unilateral"),
}
WARM_VF_ITERS = 25
ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=float, default=1e7); ap.add_argument("--bc_iters", type=int, default=40)
ap.add_argument("--root", default="runs/m3basin")
a = ap.parse_args()
c = CELLS[a.cell]
out = os.path.join(a.root, f"{a.cell}_s{a.seed}")
os.makedirs(out, exist_ok=True)
m = MarlConfig(swap_rule=c["rule"], kappa=c["kappa"], shared=True, memory="gru", random_slots=True,
               total_agent_steps=int(a.steps), vf_warmup_iters=WARM_VF_ITERS if c["warm"] else 0)
ecfg = m.env_cfg()


def evaluate(params, tag, seed=777, n_envs=128, greedy=False):
    mm = MarlConfig(**{**m.__dict__, "n_envs": n_envs})
    d, _, _ = jax.jit(make_rollout(ecfg, mm, greedy=greedy))(params, jax.random.PRNGKey(seed))
    d = {k: np.asarray(v) for k, v in d.items()}
    r = episode_metrics(d["slot"], d["n_swaps"], d["rew"], ecfg.delta_h, ecfg.T)
    r["class"] = classify(r); r["greedy"] = greedy
    return r


init = None
if c["warm"]:
    t0 = time.time()
    # BC student is independent of kappa and of the cell (teacher uses only L0 obs; kappa only changes rewards),
    # so it is trained once per seed and cached; B/D1/D2 of the same seed start from the same BC policy.
    cache = os.path.join(a.root, "bc", f"bc_s{a.seed}.pkl")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        blob = pickle.load(open(cache, "rb"))
        init = jax.tree_util.tree_map(jnp.asarray, blob["params"]); hist = blob["hist"]
    else:
        teacher = S.state_rotation(ecfg, 0.94, 0.90)          # anonymous mutual state-driven rotation (L0 obs only)
        assert c["rule"] == "mutual"
        init, hist = dagger_bc(ecfg, m, teacher, seed=1000 + a.seed, iters=a.bc_iters)
        pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, init), hist=hist), open(cache, "wb"))
    bc = dict(sampled=evaluate(init, "bc"), greedy=evaluate(init, "bc", greedy=True), final_acc=hist[-1][2],
              final_loss=hist[-1][1], wall=time.time() - t0)
    json.dump(bc, open(os.path.join(out, "bc_eval.json"), "w"), indent=1)
    print("BC policy:", bc["sampled"]["class"], {k: round(v, 4) for k, v in bc["sampled"].items() if isinstance(v, float)}, flush=True)
ts, _ = train(m, 100 * a.seed + 7, out, ckpt_every=100, init_params_=init)
res = dict(cell=a.cell, seed=a.seed, kappa=c["kappa"], rule=c["rule"], warm=c["warm"], steps=a.steps,
           sampled=evaluate(ts.params, "final"), greedy=evaluate(ts.params, "final", greedy=True))
json.dump(res, open(os.path.join(out, "eval.json"), "w"), indent=1)
print("FINAL", res["sampled"]["class"], {k: round(v, 4) for k, v in res["sampled"].items() if isinstance(v, float)}, flush=True)
