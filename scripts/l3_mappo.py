"""Decentralised-actor / centralised-critic PPO (MAPPO) on the L3 drone-bucket cells.  Pre-registration:
notes/impl/l3_mappo_prereg.md.

Actor  : one shared MLP (2x64 tanh) applied to each plane's L3-allowed local observation (own + chain neighbours p-1, p+1).
Critic : MLP (2x64 tanh) on the global state (all planes), one team value (reward = team range gain / B0, gamma = 1).
Train  : fresh instances every update (gen noise, gen events); final parameters (no checkpoint picking) are evaluated once
         on the SAME 200 instances / noise / events that scripts/drone_bucket_l23.py used for the rule tables (paired).
Usage  : python scripts/l3_mappo.py train --cell 'N5_ar50_s0.15_p0.0_c10_h0.0' --seed 0 --updates 300 --out .scratch/l3/x.json
         add --control for the positive-control variant (global obs + gate action).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drone_l3_jax as J            # noqa: E402
import drone_bucket_l23 as L23      # noqa: E402

F32 = jnp.float32
NEG = -1e9


def parse_cell_name(name):
    # N5_ar50_s0.15_p0.0_c10_h0.0   (noise kind+L glued; 'none0' / 'iid0')
    parts = name.split("_")
    N = int(parts[0][1:])
    kl = parts[1]
    kind = "".join(ch for ch in kl if ch.isalpha())
    Lc = int("".join(ch for ch in kl if ch.isdigit()) or 0)
    return J.Cell(N, kind, Lc, float(parts[2][1:]), float(parts[3][1:]), int(parts[4][1:]), float(parts[5][1:]))


def cname(cell):
    return f"N{cell.N}_{cell.kind}{cell.L}_s{cell.sigma}_p{cell.p_event}_c{cell.c}_h{cell.h}"


# ---------------------------------------------------------------------------------------------------------------------
# networks / Adam
# ---------------------------------------------------------------------------------------------------------------------
def _orth(key, shape, scale):
    return jax.nn.initializers.orthogonal(scale, dtype=F32)(key, shape)


def init_params(key, env, hidden=64, none_bias=5.0):
    k = jax.random.split(key, 8)
    D, Dc, A = env.obs_dim, env.cobs_dim, env.nact
    a = dict(w1=_orth(k[0], (D, hidden), np.sqrt(2)), b1=jnp.zeros(hidden, F32),
             w2=_orth(k[1], (hidden, hidden), np.sqrt(2)), b2=jnp.zeros(hidden, F32),
             w3=_orth(k[2], (hidden, A), 0.01), b3=jnp.zeros(A, F32).at[0].set(none_bias))
    c = dict(w1=_orth(k[3], (Dc, hidden), np.sqrt(2)), b1=jnp.zeros(hidden, F32),
             w2=_orth(k[4], (hidden, hidden), np.sqrt(2)), b2=jnp.zeros(hidden, F32),
             w3=_orth(k[5], (hidden, 1), 1.0), b3=jnp.zeros(1, F32))
    return dict(a=a, c=c)


def mlp(p, x):
    x = jnp.tanh(x @ p["w1"] + p["b1"])
    x = jnp.tanh(x @ p["w2"] + p["b2"])
    return x @ p["w3"] + p["b3"]


def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (z, z, jnp.zeros((), jnp.int32))


def adam_update(params, grads, opt, lr, max_norm, b1=0.9, b2=0.999, eps=1e-5):
    m, v, t = opt
    gn = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    sc = jnp.minimum(1.0, max_norm / (gn + 1e-8))
    grads = jax.tree_util.tree_map(lambda g: g * sc, grads)
    t = t + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    c1, c2 = 1 - b1 ** t.astype(F32), 1 - b2 ** t.astype(F32)
    new = jax.tree_util.tree_map(lambda p, m_, v_: p - lr * (m_ / c1) / (jnp.sqrt(v_ / c2) + eps), params, m, v)
    return new, (m, v, t), gn


# ---------------------------------------------------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------------------------------------------------
def rollout(env, params, inst, st0, key, noise_fn, greedy, nb=J.NB):
    def body(c, b):
        st, key = c
        t0 = b * J.DT
        o = env.obs(st, inst, t0)
        co = env.cobs(st, inst, t0)
        mk = env.masks(st, t0)
        logits = jnp.where(mk, mlp(params["a"], o), NEG)
        key, k1, k2 = jax.random.split(key, 3)
        act = jnp.argmax(logits, -1) if greedy else jax.random.categorical(k1, logits)
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits), act[..., None], -1)[..., 0]
        v = mlp(params["c"], co)[..., 0]
        dec = env.decode(st, inst, act, t0)
        active = st.alive
        st2, r, key = env.block(st, inst, t0, dec, noise_fn, k2)
        out = dict(o=o.astype(F32), co=co.astype(F32), mk=mk, act=act, lp=lp.astype(F32), v=v.astype(F32),
                   r=r.astype(F32) / J.B0 * 1.0, active=active)
        return (st2, key), out

    (st, _), data = lax.scan(body, (st0, key), jnp.arange(nb))
    return st, data


def gae(data, lam, gamma=1.0):
    v, r, act = data["v"], data["r"], data["active"].astype(F32)
    v_next = jnp.concatenate([v[1:], jnp.zeros_like(v[:1])], 0) * jnp.concatenate([act[1:], jnp.zeros_like(act[:1])], 0)

    delta = r + gamma * v_next - v
    # carry must also be cut when the next block is inactive
    nxt = jnp.concatenate([act[1:], jnp.zeros_like(act[:1])], 0)

    def f2(carry, x):
        d, nx = x
        adv = d + gamma * lam * nx * carry
        return adv, adv

    _, adv = lax.scan(f2, jnp.zeros_like(v[0]), (delta, nxt), reverse=True)
    adv = adv * act
    return adv, (adv + v) * act


def ppo_loss(params, d, adv, ret, clip, ent_coef, vf_coef):
    logits = jnp.where(d["mk"], mlp(params["a"], d["o"]), NEG)
    lsm = jax.nn.log_softmax(logits)
    lp = jnp.take_along_axis(lsm, d["act"][..., None], -1)[..., 0]
    nvalid = d["mk"].sum(-1)
    w = (d["active"][..., None] & (nvalid >= 2)).astype(F32)
    ratio = jnp.exp(lp - d["lp"])
    a = adv[..., None]
    pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip, 1 + clip) * a)
    pr = jnp.exp(lsm)
    ent = -(jnp.where(d["mk"], pr * lsm, 0.0)).sum(-1)
    den = w.sum() + 1e-8
    l_pi = (pg * w).sum() / den
    l_ent = (ent * w).sum() / den
    v = mlp(params["c"], d["co"])[..., 0]
    am = d["active"].astype(F32)
    l_v = (((v - ret) ** 2) * am).sum() / (am.sum() + 1e-8)
    kl = (((d["lp"] - lp) * w).sum() / den)
    cf = ((jnp.abs(ratio - 1) > clip) * w).sum() / den
    return l_pi - ent_coef * l_ent + vf_coef * l_v, dict(pi=l_pi, v=l_v, ent=l_ent, kl=kl, clipfrac=cf, nw=w.sum())


def make_update(env, n_envs, args):
    pool_p = env.cell.p_event

    def sample_inst(key):
        k1, k2, k3 = jax.random.split(key, 3)
        N = env.N
        h = env.cell.h
        if h > 0:
            kk = 1 + jax.random.uniform(k1, (n_envs, N), F32, -h, h)
            BB = J.B0 * (1 + jax.random.uniform(k2, (n_envs, N), F32, -h, h))
        else:
            kk = jnp.ones((n_envs, N), F32)
            BB = jnp.full((n_envs, N), J.B0, F32)
        on = jax.random.uniform(k3, (n_envs,)) < pool_p
        k4, k5 = jax.random.split(jax.random.fold_in(key, 7))
        plane = jax.random.randint(k4, (n_envs,), 0, N)
        te = jax.random.randint(k5, (n_envs,), int(0.25 * J.B0), int(0.75 * J.B0))
        return J.Inst(kk, BB, on, plane, te)

    nf = J.gen_noise_fn(env)

    @jax.jit
    def update(params, opt, key, lr):
        key, kI, kE, kR, kP = jax.random.split(key, 5)
        inst = sample_inst(kI)
        st0 = env.init_state(inst, key=kE)
        st, data = rollout(env, params, inst, st0, kR, nf, greedy=False)
        adv, ret = gae(data, args.lam)
        am = data["active"].astype(F32)
        mean = (adv * am).sum() / (am.sum() + 1e-8)
        std = jnp.sqrt((((adv - mean) ** 2) * am).sum() / (am.sum() + 1e-8)) + 1e-8
        advn = (adv - mean) / std

        def epoch(carry, ek):
            params, opt = carry
            perm = jax.random.permutation(ek, n_envs).reshape(args.n_mb, -1)

            def mb(carry, idx):
                params, opt = carry
                d = {k_: v_[:, idx] for k_, v_ in data.items()}
                (loss, aux), g = jax.value_and_grad(ppo_loss, has_aux=True)(
                    params, d, advn[:, idx], ret[:, idx], args.clip, args.ent_coef, args.vf_coef)
                params, opt, gn = adam_update(params, g, opt, lr, args.max_grad_norm)
                return (params, opt), aux

            (params, opt), aux = lax.scan(mb, (params, opt), perm)
            return (params, opt), jax.tree_util.tree_map(lambda x: x[-1], aux)

        (params, opt), aux = lax.scan(epoch, (params, opt), jax.random.split(kP, args.epochs))
        aux = jax.tree_util.tree_map(lambda x: x[-1], aux)
        aux["range"] = (st.dist / J.B0).mean()
        aux["nswap"] = st.nswap.mean()
        aux["unfinished"] = st.alive.mean()
        aux["ret0"] = (ret[0] * am[0]).mean()
        aux["v0"] = (data["v"][0]).mean()
        return params, opt, aux

    return update


def make_eval(env, w, k, B, ev):
    inst = J.inst_from_numpy(k, B, ev)
    nf = J.table_noise(None if w is None else jnp.asarray(w, F32))

    @jax.jit
    def run(params, key):
        out = {}
        for name, gr in (("greedy", True), ("sample", False)):
            st0 = env.init_state(inst)
            st, _ = rollout(env, params, inst, st0, key, nf, greedy=gr)
            out[name] = (st.dist / J.B0, st.nswap, st.alive.any())
        return out

    return run


# ---------------------------------------------------------------------------------------------------------------------
# reference rules (numpy, parameters from calib/out/drone_bucket_l23.json) on the same evaluation instances
# ---------------------------------------------------------------------------------------------------------------------
def rule_kwargs(rule, name):
    if rule == "fixed":
        return dict(policy="fixed")
    if rule == "rot":
        return dict(policy="rot", p1=int(name))
    if rule == "gate":
        d, dw = name.split("/")
        return dict(policy="gate", p1=int(d), p2=int(dw))
    if rule == "gate_est":
        d, dw, a = name.split("/")
        return dict(policy="gate", p1=int(d), p2=int(dw), p3=float(a))
    if rule in ("local", "local_s"):
        t, d, a = name.split("/")
        return dict(policy="local", p1=int(t), p2=int(d), p3=float(a), init="identity" if rule == "local" else "sorted")
    raise ValueError(rule)


def reference_ranges(cell, jcell, n_inst=200):
    k, B, w, ev = J.eval_instances(cell, n_inst)
    mvec, s_ref, ceff = J.cell_params(cell)
    n = k.shape[0]
    M = np.tile(mvec, (n, 1))
    ce = np.full(n, ceff)
    res = {}
    for rule, info in jcell["rules"].items():
        r, _ = L23.simulate(k, B, M, ce, w, ev, **rule_kwargs(rule, info["param"]))
        res[rule] = dict(param=info["param"], range=r.tolist(), json_range=info["range"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam", type=float, default=0.97)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n_mb", type=int, default=4)
    ap.add_argument("--none_bias", type=float, default=5.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--json", default="calib/out/drone_bucket_l23.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cell = parse_cell_name(args.cell)
    env = J.Env(cell, control=args.control)
    key = jax.random.PRNGKey(100 * args.seed + 7)
    key, kp = jax.random.split(key)
    params = init_params(kp, env, none_bias=args.none_bias)
    opt = adam_init(params)
    update = make_update(env, args.envs, args)
    log = []
    t0 = time.time()
    for u in range(args.updates):
        key, ku = jax.random.split(key)
        lr = args.lr * (1.0 - 0.9 * u / max(args.updates - 1, 1))
        params, opt, aux = update(params, opt, ku, lr)
        if u == 0:
            jax.block_until_ready(aux["range"])
            t_first = time.time() - t0
            t1 = time.time()
        if u % args.log_every == 0 or u == args.updates - 1:
            a = {k_: float(v_) for k_, v_ in aux.items()}
            a["update"] = u
            a["wall"] = time.time() - t0
            log.append(a)
            print(json.dumps({k_: round(v_, 5) if isinstance(v_, float) else v_ for k_, v_ in a.items()}), flush=True)
    jax.block_until_ready(aux["range"])
    t_train = time.time() - t0
    sec_per_update = (time.time() - t1) / max(args.updates - 1, 1)

    # ---- evaluation on the rule-table instances (paired)
    k, B, w, ev = J.eval_instances(cell, 200)
    ev_run = make_eval(env, w, k, B, ev)
    key, ke = jax.random.split(key)
    out = ev_run(params, ke)
    res = dict(cell=cname(cell), control=args.control, seed=args.seed, updates=args.updates, envs=args.envs,
               args=vars(args), log=log, t_first_update=t_first, sec_per_update=sec_per_update, t_train=t_train,
               episodes_per_update=args.envs,
               decisions_per_update=args.envs * J.NB, eval={})
    for name, (rng_, ns, unfinished) in out.items():
        assert not bool(unfinished), "eval episode not finished within NB blocks"
        res["eval"][name] = dict(range=np.asarray(rng_).tolist(), nswap=np.asarray(ns).tolist())
    # reference rules
    jd = json.load(open(args.json))
    jc = [c for j in jd["jobs"] for c in j["cells"] if J.cell_from_json(c) == cell]
    assert len(jc) == 1, f"cell not found in {args.json}"
    res["ref"] = reference_ranges(cell, jc[0])
    res["opt"] = jc[0]["opt"]
    res["n_lp"] = jc[0]["n_lp"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"))
    g = np.mean(res["eval"]["greedy"]["range"])
    print("EVAL greedy mean range", round(g, 5), "sample", round(float(np.mean(res["eval"]["sample"]["range"])), 5),
          "| ref:", {k_: round(float(np.mean(v_["range"])), 5) for k_, v_ in res["ref"].items()}, flush=True)


if __name__ == "__main__":
    main()
