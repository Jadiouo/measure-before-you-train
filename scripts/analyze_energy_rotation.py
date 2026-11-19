"""Energy-conditional rotation analysis (notes/impl/survival_ablation_results.md section 2).  Post-hoc, descriptive.
Paired evaluation: every policy sees the same 512 initial states (reset keys = split(split(PRNGKey(777))[0], 512))."""
import os, sys, json, dataclasses, glob
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import jax, jax.numpy as jnp, numpy as np
from scipy import stats as st
import animalbench  # noqa
from animalbench.core import compile_schedule
from animalbench.marl import load_ckpt
from animalbench.marl.ippo import init_train, make_rollout
from animalbench.scenarios.geese import make_rollout as env_rollout
from animalbench.scenarios.geese.hlc import rule_blind

NEP = int(os.environ.get("NEP", 512)); EPS = 1e-6
ART = sys.argv[1] if len(sys.argv) > 1 else "calib/out/survival_ablation.json"


def run_net(mc, params):
    ecfg = mc.env_cfg(); mm = dataclasses.replace(mc, n_envs=NEP)
    data, _, _ = jax.jit(make_rollout(ecfg, mm))(params, jax.random.PRNGKey(777))
    return {k: np.asarray(v) for k, v in data.items()}, ecfg


def run_script(ecfg, pol):
    f = env_rollout(ecfg, pol); sched = compile_schedule([], ecfg.e_max)
    k_reset = jax.random.split(jax.random.PRNGKey(777))[0]
    _, recs = jax.jit(jax.vmap(lambda k: f(k, jnp.int32(3), sched)))(jax.random.split(k_reset, NEP))
    g = lambda x: np.moveaxis(np.asarray(x), 0, 1)
    return dict(slot=g(recs["slot"]), rew=g(recs["reward"]), e_frac=g(recs["e_frac_end"]), arrived=g(recs["arrived"]))


def analyse(d):
    slot, ef, rew = d["slot"], d["e_frac"], d["rew"]                    # (T,B,A)
    T, B, A = slot.shape
    lead = np.argmax(slot == 0, -1)                                    # (T,B) leader body (slot is pre-step)
    ef_pre = np.concatenate([np.ones((1, B, A)), ef[:-1]], 0)           # energy before decision t
    alive_pre = ef_pre > EPS
    out = {}
    # swaps: leader changes between t and t+1, old leader still alive (exclude death-forced changes)
    inc_best_f, inc_best_all, out_low, n_sw = 0, 0, 0, 0
    dE = []
    ti, bi = np.where((lead[1:] != lead[:-1]))
    for t, b in zip(ti, bi):                                           # swap happens in decision t; energies seen: ef_pre[t]
        o, n = lead[t, b], lead[t + 1, b]
        if not (ef[t, b, o] > EPS and ef[t, b, n] > EPS): continue
        al = ef_pre[t, b] > EPS
        if al.sum() < 3: continue
        e = ef_pre[t, b]
        others = [j for j in range(A) if j != o]
        other = [j for j in others if j != n][0]
        n_sw += 1
        inc_best_f += float(e[n] > e[other]) + 0.5 * float(e[n] == e[other])
        inc_best_all += float(e[n] == e.max())
        out_low += float(e[o] == e.min())
        dE.append(float(e[n] - e[other]))
    out["n_swaps_3alive"] = n_sw
    out["incoming_is_higher_of_two_followers"] = inc_best_f / max(n_sw, 1)    # chance 0.5
    out["incoming_is_global_max"] = inc_best_all / max(n_sw, 1)               # chance ~1/3 for non-random
    out["outgoing_is_lowest"] = out_low / max(n_sw, 1)
    out["mean_energy_in_minus_other_follower"] = float(np.mean(dE)) if dE else None
    # leader energy rank while 3 alive (every step)
    m3 = alive_pre.sum(-1) == 3
    el = np.take_along_axis(ef_pre, lead[..., None], -1)[..., 0]
    rank = (ef_pre > el[..., None]).sum(-1)                            # 0 = leader has highest energy
    out["leader_rank_dist_3alive"] = [float(((rank == r) & m3).sum() / max(m3.sum(), 1)) for r in range(3)]  # chance 1/3 each if indep
    out["swaps_per_s_note"] = "per-decision swap counts only among 3-alive swaps"
    # lead share by body; deaths
    share = (slot == 0).mean((0, 1)); out["lead_share_by_body"] = share.tolist()
    dead = ef <= EPS                                                    # (T,B,A)
    first = dead & ~np.concatenate([np.zeros((1, B, A), bool), dead[:-1]], 0)
    tt, bb, aa = np.where(first)
    was_leader = np.array([slot[t, b, a] == 0 for t, b, a in zip(tt, bb, aa)], float) if len(tt) else np.array([])
    out["n_deaths"] = int(len(tt)); out["death_rate_per_bird"] = float(len(tt) / (B * A))
    out["frac_deaths_while_leader"] = float(was_leader.mean()) if len(tt) else None
    lead_steps = (slot == 0)[ef_pre > EPS].sum() if False else float(((slot == 0) & alive_pre).sum())
    fol_steps = float(((slot != 0) & alive_pre).sum())
    out["death_hazard_per_1000_leader_steps"] = float(1000 * was_leader.sum() / lead_steps) if len(tt) else 0.0
    out["death_hazard_per_1000_follower_steps"] = float(1000 * (len(tt) - was_leader.sum()) / fol_steps) if len(tt) else 0.0
    # swaps per episode (all)
    out["swaps_per_episode"] = float((lead[1:] != lead[:-1]).sum(0).mean())
    # relation of lead share with relative energy: mean e_frac of birds when leading vs following (3 alive)
    out["mean_ef_when_leading"] = float(el[m3].mean()); out["mean_ef_when_following"] = float(((ef_pre.sum(-1) - el)[m3] / 2).mean())
    return out


def ep_reward(rew):                                                    # (T,B,A) -> (B,) bird-mean episode reward
    return rew.sum(0).mean(-1)


res, R = {}, {}
base = sorted(glob.glob("runs/survival_diag/mp_A_s?")) + sorted(glob.glob("runs/survival_diag/ab_E_s?"))
mA = None
for p in base:
    if not os.path.exists(p + "/final.ckpt"): continue
    ts, mc, _ = load_ckpt(p + "/final.ckpt"); mA = mA or mc
    d, ecfg = run_net(mc, ts.params); n = os.path.basename(p)
    res[n] = analyse(d); R[n] = ep_reward(d["rew"]); print(n, flush=True)
for s in [12345, 1, 2, 3]:
    ts0 = init_train(jax.random.PRNGKey(s), ecfg, mA)
    d, _ = run_net(mA, ts0.params); res[f"random_net_{s}"] = analyse(d); R[f"random_net_{s}"] = ep_reward(d["rew"])
e2 = dataclasses.replace(ecfg, t_hold_lo=30.0, t_hold_hi=30.0)
d = run_script(e2, rule_blind(e2)); res["rot30"] = analyse(d); R["rot30"] = ep_reward(d["rew"])
rnd = np.mean([R[f"random_net_{s}"] for s in [12345, 1, 2, 3]], 0)
R["random_net_mean"] = rnd


def pd(x, y):
    df = x - y; return dict(mean=float(df.mean()), ci95_episode=float(1.96 * df.std(ddof=1) / np.sqrt(len(df))))


paired = {}
for grp in ("mp_A_s", "ab_E_s"):
    names = [n for n in R if n.startswith(grp)]
    if not names: continue
    for ref in ("rot30", "random_net_mean"):
        per = {n: pd(R[n], R[ref]) for n in names}
        seed_means = np.array([per[n]["mean"] for n in names]); k = len(names)
        tci = float(st.t.ppf(0.975, k - 1) * seed_means.std(ddof=1) / np.sqrt(k)) if k > 1 else None
        pooled = np.mean([R[n] for n in names], 0) - R[ref]
        paired[f"{grp}* vs {ref}"] = dict(per_seed=per, seed_mean=float(seed_means.mean()), seed_level_t_ci95=tci,
                                         pooled_episode=dict(mean=float(pooled.mean()), ci95=float(1.96 * pooled.std(ddof=1) / np.sqrt(len(pooled)))))
    paired[f"{grp}* abs reward"] = {n: dict(mean=float(R[n].mean()), ci95=float(1.96 * R[n].std(ddof=1) / np.sqrt(NEP))) for n in names}
paired["rot30 abs"] = dict(mean=float(R["rot30"].mean())); paired["random_net_mean abs"] = dict(mean=float(rnd.mean()))
json.dump(dict(n_ep=NEP, policies=res, paired=paired), open(ART, "w"), indent=1)
print(json.dumps(paired, indent=1))
