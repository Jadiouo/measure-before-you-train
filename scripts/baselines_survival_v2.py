"""Scripted baselines in the REAL v2 env (edge cell, N=3, mutual, random_slots): fixed, rot30 (rule_blind hold 30 s), rot16 (hold 16 s),
emax (every `hold` decisions the living bird with the highest ABSOLUTE remaining energy becomes leader).  No RL, CPU only.
Metric 'incoming_is_higher_energy' = at a 3-alive swap, P(new leader has more ABSOLUTE remaining energy than the other follower)
(chance 0.5; absolute energy = e_frac * E0_body; E0 of body 0 carries the (1+h) bonus; the v2 observation shows exactly this quantity /E_ref).
Output calib/out/survival_v2_baselines.json.
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/baselines_survival_v2.py"""
import dataclasses
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

import animalbench  # noqa
from animalbench.core import compile_schedule
from animalbench.marl import MarlConfig
from animalbench.scenarios.geese import make_rollout, stay_policy
from animalbench.scenarios.geese.hlc import Policy, rule_blind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "out", "survival_v2_baselines.json")
NEP = int(os.environ.get("NEP", 512))
EPS = 1e-6
STAY, BACK, FWD = 0, 1, 2


def v2_env(path_T=2.0, **kw):
    base = dict(survival=True, surv_alpha=1.0, surv_eps=1.02, surv_burst_m=1.0, surv_h=0.05, surv_burst_p=0.05,
                swap_rule="mutual", random_slots=True, surv_obs_abs=True, surv_pw_spread=0.05, path_T=path_T)
    base.update(kw)
    return MarlConfig(**base).env_cfg()


def gate_policy(cfg, hold_dec):
    """Energy-gated ring (the only energy-aware hand-over the env structure allows: the apex retreats to apex_target_slot, which alternates
    arms, so WHO takes over is fixed by the ring; the controllable part is WHEN).  After >= hold_dec decisions as leader, hand over
    as soon as the bird that would take over has MORE absolute energy than the leader (else keep leading and re-test every decision)."""
    from animalbench.scenarios.geese.state import apex_target_slot, occ_from_slot
    n = cfg.n_max

    def init(key, cfg_=None):
        return dict(held=jnp.zeros((), jnp.int32), last=jnp.full((), -1, jnp.int32))

    def act(ps, obs, st, mask, key=None):
        occ = occ_from_slot(cfg, st.slot, st.alive)
        leader = jnp.clip(occ[0], 0, n - 1)
        held = jnp.where(leader == ps["last"], ps["held"] + 1, 0)
        tslot = apex_target_slot(cfg, occ, st.apex_ctr)
        tgt = jnp.clip(jnp.where(tslot >= 0, occ[jnp.clip(tslot, 0, n - 1)], -1), 0, n - 1)
        go = (held >= hold_dec) & (tslot >= 0) & (st.E_rem[tgt] > st.E_rem[leader])
        i = jnp.arange(n)
        a = jnp.where(go & (i == leader), BACK, jnp.where(go & (i == tgt), FWD, STAY)).astype(jnp.int32)
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY)
        return a, dict(held=held, last=leader)
    return Policy("gate", init, act)


def lowest_gate_policy(cfg, hold_dec, check=None):
    """POST-HOC (obs-gate): same ring and min tenure as gate_policy, but the hand-over test uses ONLY what the leader's own observation
    contains: hand over when e_self < e_min_other (leader is the lowest-energy living bird).  The values are read from the leader's obs row
    (layout e_self / e_min_other).  `check` (optional callable) receives (obs-derived e_self, e_min_other, st-derived e_self, e_min_other, gate
    flags) via jax.debug.callback for the equivalence test; it never changes the decision."""
    from animalbench.scenarios.geese.state import apex_target_slot, occ_from_slot
    from animalbench.scenarios.geese.obs import layout
    n = cfg.n_max
    lay = layout(cfg)
    i_self, i_min = lay["e_self"][0], lay["e_min_other"][0]
    den = cfg.E0_coef * cfg.P_solo * cfg.T

    def init(key, cfg_=None):
        return dict(held=jnp.zeros((), jnp.int32), last=jnp.full((), -1, jnp.int32))

    def act(ps, obs, st, mask, key=None):
        occ = occ_from_slot(cfg, st.slot, st.alive)
        leader = jnp.clip(occ[0], 0, n - 1)
        held = jnp.where(leader == ps["last"], ps["held"] + 1, 0)
        tslot = apex_target_slot(cfg, occ, st.apex_ctr)
        tgt = jnp.clip(jnp.where(tslot >= 0, occ[jnp.clip(tslot, 0, n - 1)], -1), 0, n - 1)
        e_self, e_min = obs[leader, i_self], obs[leader, i_min]
        go = (held >= hold_dec) & (tslot >= 0) & (e_self < e_min)
        if check is not None:
            ef_ = jnp.clip(st.E_rem, 0.0, None) / den
            oth = st.alive & (jnp.arange(n) != leader)
            s_min = jnp.min(jnp.where(oth, ef_, 2.0))
            s_min = jnp.where(s_min > 1.5, 1.0, s_min)
            jax.debug.callback(check, e_self, e_min, ef_[leader], s_min, e_self < e_min, ef_[leader] < s_min,
                               (st.E_rem[leader] < jnp.min(jnp.where(oth, st.E_rem, jnp.inf))))
        i = jnp.arange(n)
        a = jnp.where(go & (i == leader), BACK, jnp.where(go & (i == tgt), FWD, STAY)).astype(jnp.int32)
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY)
        return a, dict(held=held, last=leader)
    return Policy("lowest_gate", init, act)


def run(ecfg, pol):
    f = make_rollout(ecfg, pol)
    sched = compile_schedule([], ecfg.e_max)
    keys = jax.random.split(jax.random.split(jax.random.PRNGKey(777))[0], NEP)
    _, recs = jax.jit(jax.vmap(lambda k: f(k, jnp.int32(3), sched)))(keys)
    g = lambda x: np.moveaxis(np.asarray(x), 0, 1)
    return dict(slot=g(recs["slot"]), rew=g(recs["reward"]), ef=g(recs["e_frac_end"]), arrived=g(recs["arrived"]), nsw=np.asarray(recs["n_swaps"]).sum(1))


def analyse(d, ecfg):
    slot, ef, rew = d["slot"], d["ef"], d["rew"]                       # (T,B,A)
    T, B, A = slot.shape
    E0 = np.full(A, ecfg.E0_coef * ecfg.P_solo * ecfg.T); E0[0] *= (1 + ecfg.surv_h)
    ef_pre = np.concatenate([np.ones((1, B, A)), ef[:-1]], 0)
    Eabs = ef_pre * E0[None, None, :]                                  # absolute remaining energy before decision t
    lead = np.argmax(slot == 0, -1)
    ti, bi = np.where(lead[1:] != lead[:-1])
    hi, n, gm, olow = 0.0, 0, 0.0, 0.0
    for t, b in zip(ti, bi):
        o, nw = lead[t, b], lead[t + 1, b]
        if not (ef[t, b, o] > EPS and ef[t, b, nw] > EPS) or (ef_pre[t, b] > EPS).sum() < 3:
            continue
        other = [j for j in range(A) if j not in (o, nw)][0]
        e = Eabs[t, b]
        hi += float(e[nw] > e[other]) + 0.5 * float(e[nw] == e[other]); n += 1
        gm += float(e[nw] == e.max()); olow += float(e[o] == e.min())
    ep = rew.sum(0).mean(-1)
    died = (ef <= EPS)[-1]
    # per-episode fraction of swaps going to the higher-energy follower (for episode-level spread)
    return dict(reward=float(ep.mean()), reward_ci95_episode=float(1.96 * ep.std(ddof=1) / np.sqrt(B)),
                all_arrive=float(d["arrived"].any(0).all(-1).mean()), any_death=float(died.any(-1).mean()),
                swaps_per_ep=float(d["nsw"].mean()), n_swaps_3alive=n,
                incoming_is_higher_energy=hi / max(n, 1), incoming_is_global_max=gm / max(n, 1), outgoing_is_lowest=olow / max(n, 1), lead_share_by_body=(slot == 0).mean((0, 1)).tolist()), ep


def main():
    res = {}
    for pt in (2.0, 3.0):
        ecfg = v2_env(pt)
        pols = {"fixed": stay_policy(ecfg)}
        for hold in (30.0, 16.0):
            e2 = dataclasses.replace(ecfg, t_hold_lo=hold, t_hold_hi=hold)
            pols[f"rot{int(hold)}"] = rule_blind(e2)
        for hd in (15, 8):
            pols[f"gate{hd * 2}s"] = gate_policy(ecfg, hd)
        cache = {}
        for name, pol in pols.items():
            e_use = dataclasses.replace(ecfg, t_hold_lo=float(name[3:]), t_hold_hi=float(name[3:])) if name in ("rot30", "rot16") else ecfg
            d = run(e_use, pol)
            r, ep = analyse(d, ecfg)
            cache[name] = ep
            res[f"pathT{pt}/{name}"] = r
            print(pt, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "lead_share_by_body"}, flush=True)
        for name in cache:
            if name != "rot30":
                df = cache[name] - cache["rot30"]
                res[f"pathT{pt}/{name}"]["minus_rot30_mean"] = float(df.mean())
                res[f"pathT{pt}/{name}"]["minus_rot30_ci95_episode"] = float(1.96 * df.std(ddof=1) / np.sqrt(len(df)))
    json.dump(dict(n_ep=NEP, env="edge eps=1.02 m_g=1 h=0.05 p=0.05 beta=0.5 N=3 mutual random_slots, surv_obs_abs, surv_pw_spread=0.05 redrawn",
                   results=res), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
