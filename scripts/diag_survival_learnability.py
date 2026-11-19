"""Diagnostic (NOT a prereg experiment): why does edge-cell IPPO barely beat random init?  See notes/impl/survival_learnability_diag.md
  upper : mean alpha-reward of random-init nets / trained ckpt / scripted fixed / rot15 / thr15 (edge cell, alpha=0.5) with 95% CI
  train : one diagnostic training run with overridden hyper-parameters, then 512-episode eval.
usage: diag_survival_learnability.py upper|train --name X [--steps 2e7 --ent 0.01 --shaping none --gamma 1 --lr 3e-4]"""
import argparse, dataclasses, json, os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import jax, jax.numpy as jnp, numpy as np
import animalbench  # noqa
from animalbench.core import compile_schedule
from animalbench.marl import MarlConfig, train, load_ckpt
from animalbench.marl.ippo import init_train, make_rollout
from animalbench.marl.survival_eval import survival_metrics, classify_survival
from animalbench.scenarios.geese import make_rollout as env_rollout, stay_policy
from animalbench.scenarios.geese.hlc import rule_blind, Policy, STAY, BACK

ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("--name", default="x")
ap.add_argument("--steps", type=float, default=2e7); ap.add_argument("--ent", type=float, default=0.01)
ap.add_argument("--shaping", default="none"); ap.add_argument("--gamma", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--bc_hold", type=float, default=0); ap.add_argument("--vf_warm", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--central", action="store_true"); ap.add_argument("--separate_critic", action="store_true"); ap.add_argument("--v2", action="store_true")   # v2 env: path_T=2.0, surv_obs_abs, surv_pw_spread=0.05 (redrawn per episode)
a = ap.parse_args()
EPS, MG, ALPHA, NEP = 1.02, 1.0, a.alpha, 512


def mk(**kw):
    if a.v2:
        kw.update(path_T=2.0, surv_obs_abs=True, surv_pw_spread=0.05)
    return MarlConfig(survival=True, surv_alpha=ALPHA, surv_eps=EPS, surv_burst_m=MG, swap_rule="mutual", shared=True,
                      memory="gru", random_slots=True, total_agent_steps=int(kw.pop("steps", 2e7)), **kw)


def stats(rew_sum_ep, d, ecfg):
    """rew_sum_ep (B,A) episode alpha-reward per bird.  CI over episodes of the bird-mean."""
    x = rew_sum_ep.mean(1)
    m = survival_metrics(d, ecfg)
    return dict(reward=float(x.mean()), ci95=float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), ep_sd=float(x.std(ddof=1)),
                arrival=m["arrival_rate"], all_arrive=m["all_arrive"], efrac=m["efrac_final"], swap_rate=m["swap_rate"],
                tenure_s=m["median_tenure_s"], lead_share=m["lead_share_by_bird"], cls=classify_survival(m), n_ep=len(x))


def eval_net(mcfg, params, seed=777):
    ecfg = mcfg.env_cfg()
    mm = dataclasses.replace(mcfg, n_envs=NEP)
    data, _, _ = jax.jit(make_rollout(ecfg, mm))(params, jax.random.PRNGKey(seed))
    d = {k: np.asarray(v) for k, v in data.items()}
    return stats(d["rew"].sum(0), d, ecfg)


def eval_script(ecfg, pol, seed=777):
    f = env_rollout(ecfg, pol)
    sched = compile_schedule([], ecfg.e_max)
    _, recs = jax.jit(jax.vmap(lambda k: f(k, jnp.int32(3), sched)))(jax.random.split(jax.random.PRNGKey(seed), NEP))
    g = lambda x: np.moveaxis(np.asarray(x), 0, 1)
    d = dict(slot=g(recs["slot"]), n_swaps=g(recs["n_swaps"]), rew=g(recs["reward"]), arrived=g(recs["arrived"]), e_frac=g(recs["e_frac_end"]))
    return stats(d["rew"].sum(0), d, ecfg)


def thr_policy(ecfg, hold):
    """rot-`hold` plus 'lowest-energy bird never leads / never volunteers' (privileged, scripted approximation of thr15)."""
    base = rule_blind(ecfg)

    def act(ps, obs, st, mask, key=None):
        act0, ps2 = base.act(ps, obs, st, mask, key)
        al = st.alive
        ef = jnp.where(al, jnp.clip(st.E_rem, 0.0, None) / st.E0, 9.0)
        lowest = (jnp.arange(ecfg.n_max) == jnp.argmin(ef)) & (al.sum() > 1)
        unl = al & (st.lock == 0)
        a = jnp.where(lowest & (st.slot > 0) & (act0 == 2), STAY, act0)
        a = jnp.where(lowest & (st.slot == 0) & unl, BACK, a)
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY).astype(jnp.int32)
        return a, ps2
    return Policy("thr", base.init, act)


if a.mode == "upper":
    m0 = mk(); ecfg = m0.env_cfg(); out = {}
    for s in [12345, 1, 2, 3]:
        ts0 = init_train(jax.random.PRNGKey(s), ecfg, m0)
        out[f"random_init_key{s}"] = eval_net(m0, ts0.params)
    ts, mc, _ = load_ckpt("runs/survival/edge_a0.5_s0/final.ckpt")
    out["trained_s0_2e7"] = eval_net(mc, ts.params)
    out["script_fixed_leader"] = eval_script(ecfg, stay_policy(ecfg))
    for h in (15, 30):
        e2 = dataclasses.replace(ecfg, t_hold_lo=float(h), t_hold_hi=float(h))
        out[f"script_rot{h}"] = eval_script(e2, rule_blind(e2))
        out[f"script_thr{h}_approx"] = eval_script(e2, thr_policy(e2, h))
    json.dump(out, open("calib/out/survival_learnability_upper.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k:24s} R={v['reward']:.4f}±{v['ci95']:.4f} sd={v['ep_sd']:.3f} arr={v['arrival']:.3f} ef={v['efrac']:.4f} swap={v['swap_rate']:.3f} {v['cls']}", flush=True)
else:
    m = mk(steps=a.steps, ent_coef=a.ent, surv_shaping=a.shaping, gamma=a.gamma, lr=a.lr, vf_warmup_iters=a.vf_warm, central=a.central, separate_critic=a.separate_critic)
    out_dir = f"runs/survival_diag/{a.name}"; os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    init = None
    if a.bc_hold > 0:
        from animalbench.marl.warmstart import dagger_bc
        e2 = dataclasses.replace(m.env_cfg(), t_hold_lo=a.bc_hold, t_hold_hi=a.bc_hold)
        init, hist = dagger_bc(m.env_cfg(), m, rule_blind(e2), seed=1000 + a.seed, iters=60)   # teacher: scripted rot<hold>
        print("BC done acc", hist[-1], flush=True)
        r0 = eval_net(m, init); print("AFTER_BC", json.dumps(r0), flush=True)
    ts, _ = train(m, 100 * a.seed + 7, out_dir, ckpt_every=100, init_params_=init)
    r = eval_net(m, ts.params)
    L = [json.loads(l) for l in open(f"{out_dir}/log.jsonl")]
    av = lambda ls, k: float(np.mean([l[k] for l in ls]))
    first, last = L[:20], L[-20:]
    curve = {k: [av(first, k), av(last, k)] for k in ["ret", "arrival_rate", "efrac_final", "swap_rate", "vl", "ev", "ent", "gnorm", "kl", "clipfrac"]}
    res = dict(after_bc=(r0 if a.bc_hold > 0 else None), ent_by_decile=[float(np.mean([l["ent"] for l in L[i * len(L) // 10:(i + 1) * len(L) // 10]])) for i in range(10)], swap_by_decile=[float(np.mean([l["swap_rate"] for l in L[i * len(L) // 10:(i + 1) * len(L) // 10]])) for i in range(10)], name=a.name, args=vars(a), wall_s=time.time() - t0, eval=r, train_first20_last20=curve,
               ret_by_decile=[float(np.mean([l["ret"] for l in L[i * len(L) // 10:(i + 1) * len(L) // 10]])) for i in range(10)])
    json.dump(res, open(f"{out_dir}/diag.json", "w"), indent=1)
    print("DONE", a.name, json.dumps(r), flush=True)
