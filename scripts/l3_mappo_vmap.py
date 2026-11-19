"""Seeds-vmapped version of scripts/l3_mappo.py: S training seeds of one cell trained in ONE jitted/vmapped update.
Same per-seed key derivation as the single-seed script, same output JSON per (cell, seed) (read by l3_analyze.py).
usage: l3_mappo_vmap.py --cells c1,c2 --seeds 0,1,2 --updates 500 [--control] --outdir DIR"""
import argparse, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l3_mappo as M
import drone_l3_jax as J


def train_cell(cellname, seeds, updates, control, envs=256, lr0=5e-4, log_every=50, json_path="calib/out/drone_bucket_l23.json", outdir=None, none_bias=5.0):
    args = argparse.Namespace(lam=0.97, clip=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, epochs=4, n_mb=4)
    cell = M.parse_cell_name(cellname)
    env = J.Env(cell, control=control)
    keys, params = [], []
    for s in seeds:
        key = jax.random.PRNGKey(100 * s + 7)
        key, kp = jax.random.split(key)
        keys.append(key); params.append(M.init_params(kp, env, none_bias=none_bias))
    stack = lambda xs: jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)
    params = stack(params); keys = jnp.stack(keys)
    opt = jax.vmap(M.adam_init)(params)
    upd = M.make_update(env, envs, args)          # jitted single-seed update
    vupd = jax.jit(jax.vmap(upd.__wrapped__ if hasattr(upd, "__wrapped__") else upd, in_axes=(0, 0, 0, None)))
    logs = [[] for _ in seeds]
    t0 = time.time()
    for u in range(updates):
        sp = jax.vmap(jax.random.split)(keys)     # (S,2,2): [next key, ku]
        keys, ku = sp[:, 0], sp[:, 1]
        lr = lr0 * (1.0 - 0.9 * u / max(updates - 1, 1))
        params, opt, aux = vupd(params, opt, ku, lr)
        if u == 0:
            jax.block_until_ready(aux["range"]); t_first = time.time() - t0; t1 = time.time()
        if u % log_every == 0 or u == updates - 1:
            for i in range(len(seeds)):
                a = {k: float(v[i]) for k, v in aux.items()}; a.update(update=u, wall=time.time() - t0); logs[i].append(a)
            print(cellname, u, "range", [round(float(x), 4) for x in aux["range"]], "nswap", [round(float(x), 1) for x in aux["nswap"]], flush=True)
    jax.block_until_ready(aux["range"])
    spu = (time.time() - t1) / max(updates - 1, 1)
    k, B, w, ev = J.eval_instances(cell, 200)
    ev_run = M.make_eval(env, w, k, B, ev)
    sp = jax.vmap(jax.random.split)(keys); keys, ke = sp[:, 0], sp[:, 1]
    out = jax.vmap(ev_run)(params, ke)
    jd = json.load(open(json_path))
    jc = [c for j in jd["jobs"] for c in j["cells"] if J.cell_from_json(c) == cell]
    ref = M.reference_ranges(cell, jc[0])
    res = []
    for i, s in enumerate(seeds):
        r = dict(cell=M.cname(cell), control=control, seed=s, updates=updates, envs=envs, vmap_seeds=len(seeds), none_bias=none_bias, log=logs[i],
                 t_first_update=t_first, sec_per_update_all_seeds=spu, eval={}, ref=ref, opt=jc[0]["opt"], n_lp=jc[0]["n_lp"])
        for name, (rg, ns, unf) in out.items():
            assert not bool(np.asarray(unf).any())
            r["eval"][name] = dict(range=np.asarray(rg[i]).tolist(), nswap=np.asarray(ns[i]).tolist())
        res.append(r)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            json.dump(r, open(f"{outdir}/{M.cname(cell)}_s{s}.json", "w"))
    return res, params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--updates", type=int, default=500); ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--control", action="store_true"); ap.add_argument("--outdir", required=True)
    ap.add_argument("--none_bias", type=float, default=5.0)  # v1 value 5.0 = default; v2 prereg uses 2.0
    a = ap.parse_args()
    for c in a.cells.split(","):
        train_cell(c, [int(x) for x in a.seeds.split(",")], a.updates, a.control, a.envs, outdir=a.outdir, none_bias=a.none_bias)
