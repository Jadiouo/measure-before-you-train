"""Survival-migration small game (N=3): fixed D, energy cap E0, random gusts, death = drop out, mixed reward
r_i = (1-alpha)*individual + alpha*team.  A numerical PRIOR for the N=3 IPPO runs (notes/impl/survival_game.md).

Units: energy in P_solo*s, time in decisions of dh = 2 s (same as ledger_game.py / small_game.py).
Parameter provenance (every number is read from a file, see `load_params`):
  p_lead = p[0], p_follow = p[1]      calib/out/small_game_params.json  ["3"]["p"]            (1.0, 0.875)
  swap cost (old leader / new leader / bystander)  small_game_params.json ["3"]["edges"]["1"]["cost"] (0.2796/0.1884/0.4008)
  c_swap_apex = 1.1948 P_solo*s       calib/out/swap_cost.json ["c_swap_apex"]  (other calibration, used only as sensitivity)
  dh = 2 s, T = 600 s -> D = 300     src/animalbench/scenarios/geese/config.py  GeeseConfig.delta_h, GeeseConfig.T
  t_lock = 8 s -> 4 decisions        GeeseConfig.t_lock  (shortest tenure; ledger_game.json uses the same lock_dec)
Everything else (gust law, E0 heterogeneity, N=2 / N=1 parameters, deviation menu) is an ASSUMPTION, see the .md.

Dynamics (vectorised over episodes, common random numbers across strategies/deviations):
  each step: effective leader pays p_lead*dh, the other living birds p_follow*dh (a lone bird pays p_lead*dh);
  every living bird is hit with prob pg by a gust of extra cost  m_g*dh*Exp(1); a change of effective leader between two
  living birds charges the measured swap cost (old/new/bystander; no bystander if only 2 are alive; a leader's death
  moves the successor up for free); energy <= 0 -> dead (energy 0, never counted again).
Strategies: fixed (lowest-index living bird leads; bird 0 is the strongest), rot (ring 0->1->2, handoff every tau
  decisions), thr (ring, but the lowest-energy living bird is skipped when >= 2 are alive).
One-shot deviations of bird i (others keep the strategy; the strategy state machine then continues untouched):
  refuse  : at the start of i's tenure i does not take over; the previous leader stays for the whole tenure (w = tau)
  early   : half way through its tenure i steps down; the previous leader covers the rest
  volunteer: a non-leader i takes the lead for a block of w decisions (extra tenure)
  fixed strategy: 'refuse'/'early' collapse into 'stepdown' (the leader hands the lead to the ring successor for w).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUT = os.path.join(ROOT, "calib", "out")
DH = 2.0
N = 3


def load_params(path_sg=None, path_sw=None):
    path_sg = path_sg or os.path.join(OUT, "small_game_params.json")
    path_sw = path_sw or os.path.join(OUT, "swap_cost.json")
    d = json.load(open(path_sg))["3"]
    c = d["edges"]["1"]["cost"]
    return dict(p_lead=float(d["p"][0]), p_follow=float(d["p"][1]), c_old=float(c[0]), c_new=float(c[1]),
                c_by=float(c[2]), c_swap_apex=float(json.load(open(path_sw))["c_swap_apex"]))


def load_params_pathT(path_T, path_sg=None, path_pt=None):
    """v2: same as load_params but the three swap costs (old leader / new leader / bystander) come from
    calib/out/swap_cost_pathT.json (calib/cal_swap_pathT.py) for the given path_T (s)."""
    pr = load_params(path_sg)
    path_pt = path_pt or os.path.join(OUT, "swap_cost_pathT.json")
    rows = {float(r["path_T"]): r for r in json.load(open(path_pt))["pathT"]}
    c = rows[float(path_T)]["c_old_new_bystander"]
    pr.update(c_old=float(c[0]), c_new=float(c[1]), c_by=float(c[2]), path_T=float(path_T))
    return pr


def make_mult(n, spread, seed):
    """v2 individual energy-use coefficient: (n,3) ~ U(1-spread, 1+spread), redrawn every episode (common random numbers)."""
    rng = np.random.default_rng(seed)
    return 1.0 + spread * (2.0 * rng.random((n, N)) - 1.0)


def make_gusts(n, D, pg, seed):
    """Common random numbers: (hit, size) arrays (n, D, 3); size is Exp(1), scaled by m_g later."""
    rng = np.random.default_rng(seed)
    hit = rng.random((n, D, N)) < pg
    size = rng.exponential(1.0, (n, D, N))
    return hit, size


def _next_alive(cur, alive, excl=None):
    """First k=1..3 with cand=(cur+k)%3 alive and not excluded (k=3 -> cur itself).  cur, excl: (n,) int."""
    n = cur.shape[0]
    idx = np.arange(n)
    out = cur.copy()
    done = np.zeros(n, bool)
    for k in (1, 2, 3):
        cand = (cur + k) % N
        ok = alive[idx, cand]
        if excl is not None:
            ok &= cand != excl
        take = ok & ~done
        out[take] = cand[take]
        done |= take
    return out


def simulate(strategy, E0, D, hit, size, pr, tau=15, m_g=0.0, h=0.1, cmult=1.0, dev=None, w=None, mult=None):
    """Run n episodes.  strategy in {"fixed","rot","thr"}; E0 = cap of a normal bird (P_solo*s), bird 0 has E0*(1+h).
    dev = None | dict(mode, i, t_d); w = block length of a deviation (default tau, or 15 for fixed).
    Returns dict(arrived (n,3) bool, efrac (n,3) remaining energy / own E0 (0 if dead), death_t (n,3) (D if alive),
    n_swaps (n,))."""
    n = hit.shape[0]
    idx = np.arange(n)
    cap = np.array([E0 * (1 + h), E0, E0])
    E = np.tile(cap, (n, 1))
    alive = np.ones((n, N), bool)
    death_t = np.full((n, N), D, np.int32)
    L = np.zeros(n, np.int64)                      # nominal leader
    prev = np.full(n, -1, np.int64)
    elapsed = np.zeros(n, np.int64)
    last_eff = np.full(n, -1, np.int64)
    n_sw = np.zeros(n, np.int64)
    lead_cnt = np.zeros((n, N), np.int64)                 # v2: decisions spent as effective leader (while >=2 alive)
    if w is None:
        w = 15 if strategy == "fixed" else tau
    started = np.zeros(n, bool)
    remaining = np.zeros(n, np.int64)
    pl, pf = pr["p_lead"], pr["p_follow"]
    for t in range(D):
        na = alive.sum(1)
        # ---- nominal-leader update: death of L (promote), or handoff after tau decisions
        dead_L = ~alive[idx, L]
        hand = (~dead_L) & (elapsed >= tau) & (strategy != "fixed")
        if strategy == "gate":                   # v2: ring successor takes over only if it has MORE absolute energy than the leader
            hand = hand & (E[idx, _next_alive(L, alive)] > E[idx, L])
        if dead_L.any() or hand.any():
            excl = None
            if strategy == "thr":
                en = np.where(alive, E, np.inf)
                m = en.argmin(1)
                excl = np.where(na >= 2, m, -1)
            newL = _next_alive(L, alive, excl)
            if strategy == "emax":
                newL = np.where(alive, E, -np.inf).argmax(1)
            chg = dead_L | hand
            prev = np.where(hand, L, np.where(dead_L, -1, prev))
            L = np.where(chg, newL, L)
            elapsed = np.where(chg, 0, elapsed)
        # ---- deviation trigger
        if dev is not None and t >= dev["t_d"]:
            i, mode = dev["i"], dev["mode"]
            can = (~started) & alive[:, i]
            if mode in ("refuse", "stepdown"):
                trig = can & (L == i) & ((elapsed == 0) if strategy != "fixed" else True)
                ww = np.full(n, w)
            elif mode == "early":
                trig = can & (L == i) & (elapsed >= tau // 2)
                ww = np.maximum(tau - elapsed, 1)
            elif mode == "volunteer":
                trig = can & (L != i)
                ww = np.full(n, w)
            else:
                raise ValueError(mode)
            started |= trig
            remaining = np.where(trig, ww, remaining)
        # ---- effective leader
        eff = L.copy()
        if dev is not None:
            i, mode = dev["i"], dev["mode"]
            act = (remaining > 0) & alive[:, i]
            if mode == "volunteer":
                eff = np.where(act, i, eff)
            else:
                rep = np.where((prev >= 0) & (prev != i) & alive[idx, np.clip(prev, 0, 2)], prev,
                               _next_alive(np.full(n, i), alive & (np.arange(N)[None, :] != i)))
                eff = np.where(act & (L == i), rep, eff)
            remaining = np.where(remaining > 0, remaining - 1, 0)
        # ---- costs
        isl = np.arange(N)[None, :] == eff[:, None]
        base = np.where(isl, pl, pf)
        base = np.where((na == 1)[:, None], pl, base)
        cost = base * DH
        lead_cnt += isl * (na >= 2)[:, None] * alive
        sw = (last_eff >= 0) & (eff != last_eff) & alive[idx, np.clip(last_eff, 0, 2)] & (na >= 2)
        if sw.any():
            n_sw += sw
            old = np.arange(N)[None, :] == last_eff[:, None]
            byst = ~isl & ~old & alive
            cost = cost + sw[:, None] * cmult * (old * pr["c_old"] + isl * pr["c_new"] + byst * pr["c_by"])
        if mult is not None:
            cost = cost * mult
        gust = hit[:, t, :] * size[:, t, :] * (m_g * DH)
        E = E - np.where(alive, cost + gust, 0.0)
        newly = alive & (E <= 0.0)
        if newly.any():
            death_t = np.where(newly, t, death_t)
            alive = alive & ~newly
            E = np.where(newly, 0.0, E)
        last_eff = np.where(alive[idx, eff], eff, -1)
        elapsed = elapsed + 1
    efrac = np.where(alive, E / cap[None, :], 0.0)
    return dict(arrived=alive.copy(), efrac=efrac, death_t=death_t, n_swaps=n_sw, lead_cnt=lead_cnt)


def utilities(res, alpha, beta=0.5):
    """(n,3) mixed reward.  individual = arrived_i + beta*efrac_i ; team = mean_j arrived_j + beta*mean_j efrac_j."""
    ind = res["arrived"].astype(float) + beta * res["efrac"]
    team = ind.mean(1, keepdims=True)
    return (1 - alpha) * ind + alpha * team


def summary(res):
    a, e = res["arrived"], res["efrac"]
    return dict(team_arrival=float(a.mean()), all_arrive=float(a.all(1).mean()), indiv_arrival=a.mean(0).tolist(),
                mean_energy=float(e.mean()), n_swaps=float(res["n_swaps"].mean()))


def deviations(strategy, tau, D, fracs=(0.1, 0.4, 0.7, 0.9), rule="unilateral"):
    """rule="unilateral": every deviation is executable (a bird that steps down / volunteers is obeyed; the maximal
    deviation power).  rule="mutual": a swap needs both sides, so only 'refuse' (not taking over) is executable; the
    fixed strategy then has no executable one-shot deviation at all (-> trivially an equilibrium, as in ledger_game)."""
    out = []
    for f in fracs:
        t_d = int(f * D)
        if strategy == "fixed":
            if rule == "mutual":
                continue
            out.append(dict(mode="stepdown", i=0, t_d=t_d))
            for i in (1, 2):
                out.append(dict(mode="volunteer", i=i, t_d=t_d))
        else:
            for i in range(N):
                for mode in (("refuse",) if rule == "mutual" else ("refuse", "early", "volunteer")):
                    out.append(dict(mode=mode, i=i, t_d=t_d))
    return out


Z_BONF = 3.2                    # two-sided 0.05 / ~36 deviations (Bonferroni)
EPS_LIST = (0.002, 0.005, 0.02)  # practical tolerances in utility units (arrival prob + beta*energy fraction)


def classify(gains_mean, gains_se, z=Z_BONF, eps=0.005):
    """gains_*: arrays over deviations (own-utility gain of deviating, paired with common random numbers).
    NOT_EQ     : some deviation gains significantly more than eps  (lower bound > eps)
    EQ_STRICT  : every upper bound <= 0
    EQ_EPS     : every upper bound <= eps (a deviation may gain, but provably less than eps)
    UNDECIDED  : otherwise (the best deviation's CI contains both < eps and > eps; the difference is inside the CI)."""
    gm, gs = np.asarray(gains_mean, float), np.asarray(gains_se, float)
    if gm.size == 0:
        return "EQ_STRICT", -1
    lo, hi = gm - z * gs, gm + z * gs
    if lo.max() > eps:
        return "NOT_EQ", int(np.argmax(lo))
    if hi.max() <= 1e-12:
        return "EQ_STRICT", int(np.argmax(gm))
    if hi.max() <= eps:
        return "EQ_EPS", int(np.argmax(gm))
    return "UNDECIDED", int(np.argmax(hi))


def eval_cell(strategy, tau, E0, D, hit, size, pr, m_g, h, cmult, alphas, betas, devs=None, rule="unilateral", mult=None):
    """Baseline + all deviations for one (strategy, tau, env) cell.  Returns dict with per-(alpha,beta) classification."""
    base = simulate(strategy, E0, D, hit, size, pr, tau=tau, m_g=m_g, h=h, cmult=cmult, mult=mult)
    devs = devs if devs is not None else deviations(strategy, tau, D, rule=rule)
    n = hit.shape[0]
    gm = {(a, b): [] for a in alphas for b in betas}
    gs = {(a, b): [] for a in alphas for b in betas}
    for dv in devs:
        r = simulate(strategy, E0, D, hit, size, pr, tau=tau, m_g=m_g, h=h, cmult=cmult, dev=dv, mult=mult)
        for a in alphas:
            for b in betas:
                d = (utilities(r, a, b) - utilities(base, a, b))[:, dv["i"]]
                gm[(a, b)].append(float(d.mean()))
                gs[(a, b)].append(float(d.std(ddof=1) / math.sqrt(n)))
    out = dict(summary=summary(base), n_dev=len(devs), per={})
    for a in alphas:
        for b in betas:
            m, e = np.array(gm[(a, b)]), np.array(gs[(a, b)])
            lab, k = classify(m, e)
            ent = dict(label=lab, labels_by_eps={str(x): classify(m, e, eps=x)[0] for x in EPS_LIST})
            if m.size:
                ent.update(max_gain=float(m.max()), lo_max=float((m - Z_BONF * e).max()), hi_max=float((m + Z_BONF * e).max()),
                           worst_dev=devs[k], worst_gain=float(m[k]), worst_se=float(e[k]))
            out["per"][f"{a}|{b}"] = ent
    return out
