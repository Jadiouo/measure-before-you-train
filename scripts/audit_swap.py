"""Swap (leader rotation) audit.  Read-only w.r.t. the library; writes calib/out/swap_audit.json.
Sections (python scripts/audit_swap.py [trace|cost|kappa|edge|spread|all]):
  trace : one swap, N=3 mutual, per-step power curve of the 3 birds vs no swap; durations, min separation, max P/Ps.
          Also N=9 apex edge, to explain 0.8688 (N=3) vs 1.1948 (N=9).
  cost  : cost decomposition per bird (integral of P - step function), in P_solo*s and in follower-seconds (/0.125).
  kappa : N=3 T=600 scripted rotation of tenure tau vs fixed leader; P_eff = P(1+kappa*max(0,phi-phi0)) computed offline
          (kappa never feeds back into the state), break-even kappa* per tenure.
  edge  : survival edge cell (eps=1.02, m_g=1, h=.05, mutual, random slots): fixed vs rot30, paired; why rot30 wins.
  spread: energy-fraction spread among candidates at swap decisions (rot30), for h / bursts / per-bird power variants.
"""
import dataclasses
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
import jax
import jax.numpy as jnp
import numpy as np

import animalbench  # noqa
from animalbench.core import compile_schedule
from animalbench.marl import MarlConfig
from animalbench.scenarios.geese import GeeseConfig, make_rollout, stay_policy, table_policy
from animalbench.scenarios.geese.hlc import Policy, rule_blind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib", "out", "swap_audit.json")
STAY, BACK, FWD = 0, 1, 2


def rnd(x, k=5):
    return np.round(np.asarray(x, float), k).tolist()


# --------------------------------------------------------------------------------------------- trace / cost
def swap_run(N, pair, T0=15, T=100.0, rule="mutual", **kw):
    cfg = GeeseConfig(n_max=N, T=T, init_mode="v", log_D=False, log_stride=1, use_events=False, swap_rule=rule, **kw)
    nd = cfg.n_dec
    sched = compile_schedule([], cfg.e_max)
    tab = np.zeros((nd, N), np.int32)
    f, bk = pair
    if rule == "mutual":
        tab[T0, f] = BACK
        tab[T0, bk] = FWD
    else:
        tab[T0, f if f > 0 else 0] = BACK
    tab0 = np.zeros((nd, N), np.int32)
    res = {}
    for name, t in (("swap", tab), ("base", tab0)):
        f_ = make_rollout(cfg, table_policy(cfg, t), nd)
        _, r = jax.jit(f_)(jax.random.PRNGKey(1), jnp.int32(N), sched)
        res[name] = r
    return cfg, T0, res


def trace_section():
    cfg, T0, res = swap_run(3, (0, 1))
    Ps = cfg.P_solo
    lw = res["swap"]["low"]
    lb = res["base"]["low"]
    flat = lambda a: np.asarray(a).reshape(-1, *np.asarray(a).shape[2:])
    t = flat(lw["t"]); ts = T0 * cfg.delta_h
    P = flat(lw["P"]) / Ps; Pb = flat(lb["P"]) / Ps
    sv = flat(lw["save"]); ef = flat(lw["e_form"]) / cfg.b
    x = flat(lw["x"]); y = flat(lw["y"])
    sel = (t >= ts - 2) & (t <= ts + 14)
    tt = t[sel] - ts
    # slot after request: leader of bird i
    p_old = Pb[(t > ts - 5) & (t < ts)].mean(0)                    # steady before request
    p_new = P[(t > ts + 20) & (t < ts + 40)].mean(0)               # steady after settle
    # step reference (instant switch at request)
    step = np.where((t >= ts)[:, None], p_new[None], p_old[None])
    dt = cfg.dt
    win = (t >= ts) & (t < ts + 40)
    excess = ((P - step) * dt)[win]
    cost_bird = excess.sum(0)
    pos = np.clip(P - step, 0, None)[win].sum(0) * dt
    neg = np.clip(P - step, None, 0)[win].sum(0) * dt
    # time-resolved: cost accumulated in [0,3], [3,8], [8,40]
    seg = {}
    for a, b in ((0, 3), (3, 8), (8, 40)):
        m = (t >= ts + a) & (t < ts + b)
        seg[f"{a}-{b}s"] = rnd(((P - step) * dt)[m].sum(0))
    # time during which |P - p_new| > 0.01
    dev = np.abs(P - p_new[None]) > 0.01
    tdev = [float((t[dev[:, i] & (t >= ts)].max() - ts + dt) if (dev[:, i] & (t >= ts)).any() else 0.0) for i in range(3)]
    dist = np.sqrt((x[:, :, None] - x[:, None, :]) ** 2 + (y[:, :, None] - y[:, None, :]) ** 2) / cfg.b + np.eye(3) * 1e9
    msep = dist[(t >= ts) & (t < ts + 8)].min(axis=0)
    out = dict(
        cfg=dict(path_T=cfg.path_T, swap_lane_b=cfg.swap_lane, t_lock=cfg.t_lock, lock_dec=cfg.lock_dec, delta_h=cfg.delta_h,
                 gain=cfg.gain, P_solo=Ps),
        p_old=rnd(p_old), p_new=rnd(p_new),
        cost_per_bird=rnd(cost_bird), cost_total=float(cost_bird.sum()), cost_pos=rnd(pos), cost_neg=rnd(neg),
        cost_by_segment=seg, t_power_settle_s=tdev,
        t_eform_settle_s=float(max(
            (t[(np.maximum(ef[:, 0], ef[:, 1]) >= 0.15) & (t >= ts)].max() - ts + dt), 0.0)),
        max_P_over_Ps_during=rnd(P[(t >= ts) & (t < ts + 8)].max(0)),
        min_P_over_Ps_during=rnd(P[(t >= ts) & (t < ts + 8)].min(0)),
        min_sep_pairs_b={"0-1": float(msep[0, 1]), "0-2": float(msep[0, 2]), "1-2": float(msep[1, 2])},
        P_curve=dict(t_rel=rnd(tt, 2), P_swap=rnd(P[sel], 4), P_base=rnd(Pb[sel], 4), save_swap=rnd(sv[sel], 4)),
    )
    # lock check: run a request every decision, count intervals between swaps of the same bird
    cfg2 = GeeseConfig(n_max=3, T=200.0, init_mode="v", log_D=False, log_low=False, use_events=False, swap_rule="mutual")
    nd = cfg2.n_dec
    sched = compile_schedule([], cfg2.e_max)

    def spam(ps, obs, st, mask, key=None):
        a = jnp.where(st.slot == 0, BACK, jnp.where(st.slot >= 1, FWD, STAY)).astype(jnp.int32)
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY)
        return a, ps
    pol = Policy("spam", lambda key, cfg_=None: dict(), spam)
    _, r = jax.jit(make_rollout(cfg2, pol, nd))(jax.random.PRNGKey(2), jnp.int32(3), sched)
    sw = np.asarray(r["swapped"])                                  # (nd, 3)
    gaps = []
    for i in range(3):
        idx = np.where(sw[:, i])[0]
        gaps += list(np.diff(idx))
    out["lock_check"] = dict(min_gap_decisions_between_swaps_same_bird=int(min(gaps)), min_gap_s=float(min(gaps) * cfg2.delta_h),
                             n_swaps_in_200s_spam=int(np.asarray(r["n_swaps"]).sum()), t_lock_s=cfg2.t_lock)
    return out


def cost_compare():
    """N=3 vs N=9 apex edge, per-bird integral cost (same definition as slot_params intercept) and cal_swap group metric."""
    res = {}
    for N in (3, 9):
        cfg, T0, r = swap_run(N, (0, 1), T0=15, T=100.0)
        Ps = cfg.P_solo
        E1 = np.asarray(r["swap"]["E_used"]); E0 = np.asarray(r["base"]["E_used"])
        H = 20
        grp = ((E1[T0 + H - 1].sum() - E1[T0 - 1].sum()) - (E0[T0 + H - 1].sum() - E0[T0 - 1].sum())) / Ps
        per = ((E1[T0 + H - 1] - E1[T0 - 1]) - (E0[T0 + H - 1] - E0[T0 - 1])) / Ps       # vs the no-swap run (includes slope part)
        res[f"N{N}"] = dict(group_extra_40s=float(grp), per_bird_extra_vs_noswap_40s=rnd(per, 4))
    # slot-level steady powers => no-swap group-energy equality
    return res


def sens_section():
    """N=3 apex-edge group cost (40 s, vs no-swap run) vs hand-set path parameters."""
    out = {}
    for name, kw in (("base", {}), ("lane0", dict(swap_lane=0.0)), ("lane0.4", dict(swap_lane=0.4)), ("lane1.6", dict(swap_lane=1.6)),
                     ("lane-0.8", dict(swap_lane=-0.8)), ("pathT2", dict(path_T=2.0)), ("pathT5", dict(path_T=5.0)),
                     ("pathT8", dict(path_T=8.0))):
        cfg, T0, r = swap_run(3, (0, 1), **kw)
        Ps = cfg.P_solo
        E1 = np.asarray(r["swap"]["E_used"]); E0 = np.asarray(r["base"]["E_used"])
        H = 20
        per = ((E1[T0 + H - 1] - E1[T0 - 1]) - (E0[T0 + H - 1] - E0[T0 - 1])) / Ps
        lw = r["swap"]["low"]
        Pm = np.asarray(lw["P"]).reshape(-1, 3) / Ps
        out[name] = dict(group_extra_40s_minus_steady=float(per.sum() - (per[0] * 0)),
                         max_P_over_Ps=rnd(Pm.max(0), 3))
    return out


# --------------------------------------------------------------------------------------------- kappa
def make_rot_policy(tau_dec):
    def act(ps, obs, st, mask, key=None):
        fire = (st.dec > 0) & (st.dec % tau_dec == 0)
        a = jnp.where(fire & (st.slot == 0), BACK, jnp.where(fire & (st.slot >= 1) & (st.slot <= 2), FWD, STAY)).astype(jnp.int32)
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], 1)[:, 0], a, STAY)
        return a, ps
    return Policy("rot", lambda key, cfg_=None: dict(), act)


def kappa_section():
    cfg = GeeseConfig(n_max=3, T=600.0, init_mode="v", log_D=False, log_stride=1, use_events=False, swap_rule="mutual")
    nd = cfg.n_dec
    sched = compile_schedule([], cfg.e_max)
    taus_s = [8, 16, 30, 60, 120, 300, 10 ** 6]                 # last = fixed leader
    taus = jnp.asarray([int(s / cfg.delta_h) for s in taus_s[:-1]] + [10 ** 6], jnp.int32)

    def one(tau, key):
        return make_rollout(cfg, make_rot_policy(tau), nd)(key, jnp.int32(3), sched)[1]
    g = jax.jit(jax.vmap(one, in_axes=(0, None)))
    r = g(taus, jax.random.PRNGKey(5))
    Ps = cfg.P_solo
    P = np.asarray(r["low"]["P"]).reshape(len(taus_s), -1, 3) / Ps     # (tau, steps, bird)
    nsw = np.asarray(r["n_swaps"]).sum(1)
    dt = cfg.dt
    f = 1 - np.exp(-dt / cfg.tau_f)

    def eff_energy(Pk, kappa):
        phi = np.ones(3); E = np.zeros(3)
        for p in Pk:
            phi = phi + f * (p - phi)
            E += p * (1 + kappa * np.maximum(0.0, phi - cfg.phi0)) * dt
        return E
    rows = {}
    for i, s in enumerate(taus_s):
        E0_ = eff_energy(P[i], 0.0)
        E1_ = eff_energy(P[i], 1.0)
        rows[s] = dict(n_swaps=int(nsw[i]), E_kappa0=rnd(E0_, 3), E_kappa1=rnd(E1_, 3),
                       tot0=float(E0_.sum()), slope_per_kappa_tot=float(E1_.sum() - E0_.sum()),
                       max0=float(E0_.max()), max_slope=float(E1_.max() - E0_.max()))
    fx = rows[10 ** 6]
    table = {}
    for s in taus_s[:-1]:
        a = rows[s]["tot0"] - fx["tot0"]; b = rows[s]["slope_per_kappa_tot"] - fx["slope_per_kappa_tot"]
        am = rows[s]["max0"] - fx["max0"]; bm = rows[s]["max_slope"] - fx["max_slope"]
        table[s] = dict(n_swaps=rows[s]["n_swaps"], d_total_kappa0=a, d_total_slope=b,
                        kappa_star_total=(float(-a / b) if b < 0 else None),
                        d_maxbird_kappa0=am, d_maxbird_slope=bm,
                        kappa_star_maxbird=(float(-am / bm) if bm < 0 else None),
                        swap_cost_check=rows[s]["n_swaps"] * 0.868823)
    # kappa-grid totals for the report
    grid = {}
    for k in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        grid[str(k)] = {str(s): float(rows[s]["tot0"] + k * rows[s]["slope_per_kappa_tot"] - fx["tot0"] - k * fx["slope_per_kappa_tot"])
                        for s in taus_s[:-1]}
    return dict(per_tenure=table, fixed=fx, d_total_vs_fixed_by_kappa=grid,
                note="E in P_solo*s over 600 s, 3 birds summed; kappa enters only the reward (step.py:44-51), never E_used/E_rem")


# --------------------------------------------------------------------------------------------- survival edge
def surv_cfg(eps=1.02, mg=1.0, h=0.05, burst_p=0.05):
    m = MarlConfig(survival=True, surv_alpha=0.5, surv_eps=eps, surv_burst_m=mg, surv_h=h, surv_burst_p=burst_p,
                   swap_rule="mutual", random_slots=True)
    return m.env_cfg()


def surv_run(cfg, pol, nep, seed=777):
    f = make_rollout(cfg, pol)
    sched = compile_schedule([], cfg.e_max)
    keys = jax.random.split(jax.random.PRNGKey(seed), nep)
    _, recs = jax.jit(jax.vmap(lambda k: f(k, jnp.int32(3), sched)))(keys)
    return {k: np.asarray(recs[k]) for k in ("reward", "arrived", "e_frac_end", "n_swaps", "slot", "P_mean", "swapped")}


def rot_pol(cfg, hold=30.0):
    e2 = dataclasses.replace(cfg, t_hold_lo=hold, t_hold_hi=hold)
    return e2, rule_blind(e2)


def surv_stats(cfg, d):
    Ps = cfg.P_solo
    E0 = np.full(3, cfg.E0_coef * Ps * cfg.T); E0[0] *= (1 + cfg.surv_h)
    ef = d["e_frac_end"][:, -1, :]
    arr = d["arrived"].any(1)                                    # (B,3)
    Eused = (E0[None] * (1 - ef)) / Ps                           # dead birds: whole cap
    died = ef <= 0
    beta = cfg.surv_beta
    return dict(arr=arr, ef=ef, Eused=Eused, died=died, nsw=d["n_swaps"].sum(1),
                rew=d["reward"].sum(1).mean(1), arr_term=arr.mean(1), en_term=beta * (arr * ef).mean(1))


def edge_section(nep=1024):
    out = {}
    for name, kw in (("edge_bursts", dict()), ("edge_noburst", dict(mg=0.0)), ("edge_noburst_h0", dict(mg=0.0, h=0.0)),
                     ("edge_bursts_h0", dict(h=0.0))):
        cfg = surv_cfg(**kw)
        fx = surv_stats(cfg, surv_run(cfg, stay_policy(cfg), nep))
        e2, pol = rot_pol(cfg, 30.0)
        ro = surv_stats(cfg, surv_run(cfg, pol, nep))
        dE = ro["Eused"].sum(1) - fx["Eused"].sum(1)
        any_d_fx = fx["died"].any(1); any_d_ro = ro["died"].any(1)
        nodeath = ~any_d_fx & ~any_d_ro
        o = dict(
            n_ep=nep,
            fixed=dict(reward=float(fx["rew"].mean()), arrival=float(fx["arr"].mean()), p_any_death=float(any_d_fx.mean()),
                       group_E_Ps_s=float(fx["Eused"].sum(1).mean()), n_swaps=float(fx["nsw"].mean())),
            rot30=dict(reward=float(ro["rew"].mean()), arrival=float(ro["arr"].mean()), p_any_death=float(any_d_ro.mean()),
                       group_E_Ps_s=float(ro["Eused"].sum(1).mean()), n_swaps=float(ro["nsw"].mean())),
            d_reward=float((ro["rew"] - fx["rew"]).mean()),
            d_reward_ci95=float(1.96 * (ro["rew"] - fx["rew"]).std(ddof=1) / np.sqrt(nep)),
            d_arrival_term=float((ro["arr_term"] - fx["arr_term"]).mean()),
            d_energy_bonus_term=float((ro["en_term"] - fx["en_term"]).mean()),
            d_groupE_all=float(dE.mean()),
            n_nodeath_eps=int(nodeath.sum()),
            d_groupE_nodeath_eps=float(dE[nodeath].mean()) if nodeath.any() else None,
            swap_cost_pred_nodeath=float(ro["nsw"][nodeath].mean() * 0.868823) if nodeath.any() else None,
            fixed_death_by_leader_is_bird0=None,
        )
        out[name] = o
    return out


# --------------------------------------------------------------------------------------------- spread
def spread_stats(cfg, d, power_mult=None, mg_off=False):
    """energy-fraction spread among followers (next-leader candidates) at swap decisions."""
    Ps = cfg.P_solo
    E0 = np.full(3, cfg.E0_coef * Ps * cfg.T); E0[0] *= (1 + cfg.surv_h)
    ef = d["e_frac_end"]                                         # (B, nd, 3) end of decision
    if power_mult is not None:                                   # recompute from P (valid when no bursts): E_i = cumsum(mult_i*P_i*dh)
        use = np.cumsum(d["P_mean"] * Ps * cfg.delta_h * power_mult[None, None, :], axis=1)
        E0pm = E0 * np.ones(3)
        ef = np.clip(1 - use / E0pm[None, None, :], 0, None)
    slot = d["slot"]; sw = d["swapped"].any(2)                   # (B, nd)
    rows_f, rows_a, rows_abs = [], [], []
    B, nd = sw.shape
    for b in range(B):
        for t in np.where(sw[b])[0]:
            if t == 0:
                continue
            s = slot[b, t]                                       # slots at start of decision t
            if not (s >= 0).all():
                continue
            f1 = np.where(s == 1)[0]; f2 = np.where(s == 2)[0]
            if len(f1) == 0 or len(f2) == 0:
                continue
            e = ef[b, t - 1]
            rows_f.append(abs(e[f1[0]] - e[f2[0]]))
            rows_a.append(e.max() - e.min())
            rows_abs.append(abs(e[f1[0]] * E0[f1[0]] - e[f2[0]] * E0[f2[0]]) / Ps)
    rf, ra, rabs = map(np.asarray, (rows_f, rows_a, rows_abs))
    q = lambda a: dict(mean=float(a.mean()), median=float(np.median(a)), p90=float(np.quantile(a, .9)))
    return dict(n_swap_events=int(len(rf)), follower_gap_frac=q(rf), all_range_frac=q(ra), follower_gap_abs_Ps_s=q(rabs),
                frac_gap_lt_0p005=float((rf < 0.005).mean()))


def spread_section(nep=256):
    out = {}
    cases = (("A_edge_h.05_bursts1", dict(), None), ("B_h0_noburst", dict(mg=0.0, h=0.0), None),
             ("C_h0_bursts1", dict(h=0.0), None), ("D_h.05_noburst", dict(mg=0.0), None),
             ("E_h0_noburst_powermult_.95_1_1.05", dict(mg=0.0, h=0.0), np.array([1.05, 1.0, 0.95])),
             ("F_h.05_noburst_powermult_1.05_1_.95", dict(mg=0.0), np.array([1.05, 1.0, 0.95])))
    for name, kw, pm in cases:
        cfg = surv_cfg(**kw)
        e2, pol = rot_pol(cfg, 30.0)
        d = surv_run(cfg, pol, nep)
        out[name] = spread_stats(cfg, d, pm)
    # fixed-leader reference: gap leader vs follower at end
    cfg = surv_cfg(mg=0.0, h=0.0)
    d = surv_run(cfg, stay_policy(cfg), 64)
    ef = d["e_frac_end"][:, -1, :]
    out["ref_fixed_end_range_frac_h0_noburst"] = float((ef.max(1) - ef.min(1)).mean())
    return out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if what in ("trace", "all"):
        res["trace"] = trace_section(); res["cost_compare"] = cost_compare(); res["sens"] = sens_section()
    if what in ("kappa", "all"):
        res["kappa"] = kappa_section()
    if what in ("edge", "all"):
        res["edge"] = edge_section()
    if what in ("spread", "all"):
        res["spread"] = spread_section()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    show = {k: v for k, v in res.items()}
    if "trace" in show:
        show["trace"] = {k: v for k, v in show["trace"].items() if k != "P_curve"}
    print(json.dumps(show, indent=1)[:12000])


if __name__ == "__main__":
    main()
