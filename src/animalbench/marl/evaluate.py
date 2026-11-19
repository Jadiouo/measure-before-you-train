"""Post-training evaluation: action distributions, leadership, small-game comparison, counterfactual apex advantage."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..analysis import small_game as sg
from ..analysis.deviation import bootstrap_ci
from .ippo import MarlConfig, make_rollout, policy_probs

ACT = ("stay", "back", "fwd")


def _run(ecfg, mcfg, params, seed, n_envs, greedy, dev_action=-1, dev_mode=0):
    m = MarlConfig(**{**mcfg.__dict__, "n_envs": n_envs})
    f = jax.jit(make_rollout(ecfg, m, greedy=greedy), static_argnums=(2, 3))
    data, _, _ = f(params, jax.random.PRNGKey(seed), dev_action, dev_mode)
    return {k: np.asarray(v) for k, v in data.items()}


def equilibria(rule, gamma, N=3):
    """Pure symmetric equilibria (per-slot action tuples) of the small game with discount `gamma`."""
    p, cost = sg.load_params(N)
    g = sg.Game(N, p, cost, rule, gamma=gamma)
    return [tuple(int(x) for x in e[0]) for e in g.equilibria(tol=1e-9)]


def action_stats(d, ecfg):
    """Policy statistics on unlocked decisions by slot (mean masked probabilities and sampled-action frequencies)."""
    probs = np.asarray(policy_probs(jnp.asarray(d["logits"]), jnp.asarray(d["mask"])))
    unl = d["lock"] == 0
    out = {"visits": {}, "mean_probs": {}, "freq": {}}
    n_unl = unl.sum()
    for s in range(3):
        sel = unl & (d["slot"] == s)
        out["visits"][s] = float(sel.sum() / max(n_unl, 1))
        out["mean_probs"][s] = probs[sel].mean(0).tolist() if sel.any() else [float("nan")] * 3
        out["freq"][s] = [float((d["act"][sel] == a).mean()) if sel.any() else float("nan") for a in range(3)]
    # locked decisions are forced to stay by the env regardless of the policy: reported separately
    out["locked_frac"] = float(1 - unl.mean())
    return out


def leadership(d):
    T, B, A = d["slot"].shape
    apex = d["slot"] == 0
    lid = apex.argmax(-1)                                     # (T,B)
    changes = (lid[1:] != lid[:-1]).sum(0)                    # per episode
    share = apex.mean(0)                                      # (B,A)
    tenures = []
    for b in range(B):
        cur, run = lid[0, b], 1
        for t in range(1, T):
            if lid[t, b] == cur:
                run += 1
            else:
                tenures.append(run)
                cur, run = lid[t, b], 1
        tenures.append(run)
    return dict(leader_changes_per_ep=float(changes.mean()), max_lead_share=float(share.max(1).mean()),
                median_tenure_s=float(np.median(tenures) * 2.0), lead_share_mean=share.mean(0).tolist())


def tv_to_equilibrium(stats, eq_profiles):
    """Visit-weighted total variation between the learned per-slot policy (unlocked states) and the best matching
    pure equilibrium profile.  Returns (tv_min, tv_by_profile)."""
    res = {}
    for prof in eq_profiles:
        tv = 0.0
        for s in range(3):
            pr = np.array(stats["mean_probs"][s])
            tgt = np.zeros(3)
            tgt[prof[s]] = 1.0
            tv += stats["visits"][s] * 0.5 * np.abs(pr - tgt).sum()
        res[prof] = tv
    return min(res.values()), res


def counterfactual_apex(ecfg, mcfg, params, seed=123, n_envs=256, H_dec=30):
    """Advantage of the unlocked apex (bird 0, decision 0 of a fresh V) of forcing BACK instead of STAY, others follow
    the learned policy, paired (same env seed).  mode 'policy': bird 0 follows its policy afterwards; mode 'oneshot':
    bird 0 stays afterwards (spec #12 definition).  Returns for each: mean advantage in P_solo*s over H (=60 s) and
    gamma-discounted over the whole episode, with bootstrap CIs."""
    out = {}
    g = mcfg.gamma
    T = ecfg.n_dec
    disc = g ** np.arange(T)
    for name, mode in (("policy", 1), ("oneshot", 2)):
        r = {a: _run(ecfg, mcfg, params, seed, n_envs, False, a, mode)["rew"][:, :, 0] for a in (0, 1)}   # (T,B)
        short = {a: r[a][:H_dec].sum(0) * ecfg.delta_h for a in r}
        disc_r = {a: (r[a] * disc[:, None]).sum(0) * ecfg.delta_h for a in r}
        adv_s = short[1] - short[0]
        adv_d = disc_r[1] - disc_r[0]
        out[name] = dict(adv_H=float(adv_s.mean()), ci_H=[float(x) for x in bootstrap_ci(adv_s)],
                         adv_disc=float(adv_d.mean()), ci_disc=[float(x) for x in bootstrap_ci(adv_d)])
    return out


def evaluate(ecfg, mcfg, params, seed=777, n_envs=128, gamma_eq=0.99, cf=True):
    rule = mcfg.swap_rule
    res = {"rule": rule, "shared": mcfg.shared, "memory": mcfg.memory}
    eqs = equilibria(rule, gamma_eq)
    res["equilibria"] = [list(e) for e in eqs]
    for name, greedy in (("sampled", False), ("greedy", True)):
        d = _run(ecfg, mcfg, params, seed, n_envs, greedy)
        st = action_stats(d, ecfg)
        tv, tvs = tv_to_equilibrium(st, eqs)
        res[name] = dict(stats=st, leadership=leadership(d), tv=tv, tv_by_profile={str(k): v for k, v in tvs.items()},
                         swap_rate=float(d["n_swaps"].sum(0).mean() / ecfg.T),
                         ret=float((d["rew"].sum(0) * ecfg.delta_h).mean()))
        unl_apex = (d["lock"] == 0) & (d["slot"] == 0)
        unl_non = (d["lock"] == 0) & (d["slot"] >= 1)
        res[name]["p_back_apex"] = float((d["act"][unl_apex] == 1).mean()) if unl_apex.any() else float("nan")
        # P(back | non-apex): BACK is masked for slots 1,2 when N=3 (no slot behind), so this is 0 by construction
        res[name]["p_back_non"] = float((d["act"][unl_non] == 1).mean()) if unl_non.any() else float("nan")
        res[name]["p_fwd_fol"] = float((d["act"][unl_non] == 2).mean()) if unl_non.any() else float("nan")
    if cf:
        res["counterfactual"] = counterfactual_apex(ecfg, mcfg, params, n_envs=2 * n_envs)
    return res
