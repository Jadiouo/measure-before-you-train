"""vmap-over-seeds trainer == single-seed trainer (same per-seed keys): params agree after a few updates.
Subprocess (float32, no animalbench import: that package turns on x64 globally)."""
import json, os, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(ROOT, "calib/out/drone_bucket_l23.json")):
    pytest.skip("needs calib/out/drone_bucket_l23.json", allow_module_level=True)

CODE = r'''
import argparse, json, numpy as np, jax
import l3_mappo as M, l3_mappo_vmap as V, drone_l3_jax as J
name, seeds, U, envs = "N5_ar50_s0.15_p0.0_c10_h0.0", [0, 1], 3, 16
res, pv = V.train_cell(name, seeds, U, True, envs, log_every=100)
args = argparse.Namespace(lam=0.97, clip=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, epochs=4, n_mb=4)
env = J.Env(M.parse_cell_name(name), control=True)
upd = M.make_update(env, envs, args)
md = 0.0
for i, s in enumerate(seeds):
    key = jax.random.PRNGKey(100 * s + 7); key, kp = jax.random.split(key)
    p = M.init_params(kp, env); opt = M.adam_init(p)
    for u in range(U):
        key, ku = jax.random.split(key)
        p, opt, aux = upd(p, opt, ku, 5e-4 * (1.0 - 0.9 * u / max(U - 1, 1)))
    pi = jax.tree_util.tree_map(lambda x: x[i], pv)
    for a, b in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(pi)):
        md = max(md, float(np.abs(np.asarray(a) - np.asarray(b)).max()))
print(json.dumps(dict(maxdiff=md, n_eval=len(res[0]["eval"]["greedy"]["range"]), nseeds=len(res))))
'''


def test_vmap_matches_single():
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=os.path.join(ROOT, "scripts"))
    env.pop("JAX_ENABLE_X64", None)
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:]
    o = json.loads(r.stdout.strip().splitlines()[-1])
    assert o["maxdiff"] < 2e-4 and o["n_eval"] == 200 and o["nseeds"] == 2, o
