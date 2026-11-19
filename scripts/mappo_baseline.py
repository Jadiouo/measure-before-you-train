"""random-init baseline reward at alpha=0.75 (edge cell), 4 keys, 512 eps seed 777."""
import sys, json
sys.argv = ["x", "none", "--alpha", "0.75"]
src = open("scripts/diag_survival_learnability.py").read().split('if a.mode == "upper":')[0]
exec(src)
m0 = mk(); ecfg = m0.env_cfg(); out = {}
for s in [12345, 1, 2, 3]:
    out[s] = eval_net(m0, init_train(jax.random.PRNGKey(s), ecfg, m0).params)
rs = [v["reward"] for v in out.values()]
res = dict(per_key=out, mean=float(np.mean(rs)), ci95_mean_of_keys=float(np.mean([v["ci95"] for v in out.values()])))
json.dump(res, open("calib/out/survival_mappo_baseline_a075.json", "w"), indent=1)
print(res["mean"], res["ci95_mean_of_keys"], rs)
