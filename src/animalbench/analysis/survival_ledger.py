"""Ledger-reciprocity (repeated-game) version of the N=3 survival-migration game.  See notes/impl/survival_game.md
section "帳本互惠".  Reuses parameters, gusts, utilities and classify() of survival_game.py (nothing there is changed).

Question: at alpha=0 (selfish reward), if the OTHER birds follow a public lead-time ledger and punish defaulters only with
actions that exist in the environment, is ledger-rotation a (one-shot-deviation) equilibrium?

The only environment-internal punishment (assumption, see .md): under swap_rule="mutual" a swap needs both birds, so the
others can simply NOT take over the lead from a debtor.  The debtor then keeps leading (cost p_lead - p_follow = 0.125*dh
per decision more than a follower); punishers stay followers (cheaper), so the punishment is free for them.  There is NO
position-based punishment in N=3 (both follower slots have the same p_follow) and NO punishment at all under the
"unilateral" swap rule (the leader can step down alone).

Ledger strategy "LR" (public information only)
  u_i  : lead decisions of bird i (carried across trips, shifted so that min over living birds = 0 at the end of a trip)
  first leader of a trip / every handoff (after tau decisions, or when the leader dies): the living bird with the lowest u,
        ties broken in ring order after the current leader (the current leader last).  If that is the current leader itself
        it simply keeps leading ("the others refuse to take over from a debtor") -- no swap, no swap cost.
  fine : a bird that REFUSES its tenure (or steps down EARLY) gets u_i -= lam (decisions).  Volunteering is not fined.
Trips: K trips, bird energy is refilled at each trip start (stop-over; assumption), death is by default for one trip only
  (perm_death=False) or permanent (perm_death=True, sensitivity).  Value of bird i = sum_k delta^k U_i,k, with the survival
  game's mixed utility U (alpha, beta).  One-shot deviations at trip j, decision t_d, of bird i; CRN (paired) gains.
"""
from __future__ import annotations

import math

import numpy as np

from . import survival_game as sv

DH, N = sv.DH, sv.N


def _pick(u, L, alive):
    """argmin over living birds of u (ties -> ring order after L, L itself last).  u (n,3) int, L (n,)."""
    off = (np.arange(N)[None, :] - L[:, None] - 1) % N
    key = np.where(alive, u * N + off, 10 ** 9)
    return key.argmin(1)


def simulate_trip(E0, D, hit, size, pr, u0, alive0, tau=15, lam=0, m_g=0.0, h=0.05, cmult=1.0, dev=None, w=None):
    """One trip of the ledger strategy.  u0 (n,3) int, alive0 (n,3) bool.  dev = None | dict(mode,i,t_d).
    Returns dict(arrived, efrac, u_end (n,3) shifted to min 0, alive_end (n,3), n_swaps, lead (n,3) lead decisions)."""
    n = hit.shape[0]
    idx = np.arange(n)
    cap = np.array([E0 * (1 + h), E0, E0])
    E = np.tile(cap, (n, 1))
    alive = alive0.copy()
    u = u0.astype(np.int64).copy()
    lead = np.zeros((n, N), np.int64)
    w = tau if w is None else w
    L = _pick(u, np.full(n, N - 1), alive)          # first leader: lowest u, ties -> ring order from bird 0
    prev = np.full(n, -1, np.int64)
    elapsed = np.zeros(n, np.int64)
    last_eff = np.full(n, -1, np.int64)
    n_sw = np.zeros(n, np.int64)
    started = np.zeros(n, bool)
    remaining = np.zeros(n, np.int64)
    pl, pf = pr["p_lead"], pr["p_follow"]
    for t in range(D):
        na = alive.sum(1)
        dead_L = ~alive[idx, L]
        hand = (~dead_L) & (elapsed >= tau)
        if dead_L.any() or hand.any():
            nxt = _pick(u, L, alive)
            chg = (dead_L | hand) & (nxt != L)
            prev = np.where(hand & chg, L, np.where(dead_L, -1, prev))
            L = np.where(dead_L | hand, nxt, L)
            elapsed = np.where(dead_L | hand, 0, elapsed)
        if dev is not None and t >= dev["t_d"]:
            i, mode = dev["i"], dev["mode"]
            can = (~started) & alive[:, i]
            if mode == "refuse":
                trig = can & (L == i) & (elapsed == 0)
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
            if mode in ("refuse", "early"):
                u[:, i] -= lam * trig                      # public fine for defaulting
        eff = L.copy()
        if dev is not None:
            i, mode = dev["i"], dev["mode"]
            act = (remaining > 0) & alive[:, i]
            if mode == "volunteer":
                eff = np.where(act, i, eff)
            else:
                others = alive & (np.arange(N)[None, :] != i)
                rep = np.where((prev >= 0) & (prev != i) & alive[idx, np.clip(prev, 0, 2)], prev,
                               sv._next_alive(np.full(n, i), others))
                eff = np.where(act & (L == i), rep, eff)
            remaining = np.where(remaining > 0, remaining - 1, 0)
        isl = np.arange(N)[None, :] == eff[:, None]
        base = np.where(isl, pl, pf)
        base = np.where((na == 1)[:, None], pl, base)
        cost = base * DH
        sw = (last_eff >= 0) & (eff != last_eff) & alive[idx, np.clip(last_eff, 0, 2)] & (na >= 2)
        if sw.any():
            n_sw += sw
            old = np.arange(N)[None, :] == last_eff[:, None]
            byst = ~isl & ~old & alive
            cost = cost + sw[:, None] * cmult * (old * pr["c_old"] + isl * pr["c_new"] + byst * pr["c_by"])
        gust = hit[:, t, :] * size[:, t, :] * (m_g * DH)
        E = E - np.where(alive, cost + gust, 0.0)
        ok = alive[idx, eff]
        u[idx, eff] += ok
        lead[idx, eff] += ok
        newly = alive & (E <= 0.0)
        if newly.any():
            alive = alive & ~newly
            E = np.where(newly, 0.0, E)
        last_eff = np.where(alive[idx, eff], eff, -1)
        elapsed = elapsed + 1
    efrac = np.where(alive, E / cap[None, :], 0.0)
    um = np.where(alive, u, 10 ** 9).min(1, keepdims=True)
    u_end = u - np.where(um < 10 ** 9, um, 0)
    return dict(arrived=alive.copy(), efrac=efrac, u_end=u_end, alive_end=alive.copy(), n_swaps=n_sw, lead=lead)


def run_trips(K, E0, D, gusts, pr, state, lam, tau=15, m_g=0.0, h=0.05, cmult=1.0, perm_death=False, dev=None, dev_trip=None,
              start=0):
    """Trips start..K-1 from `state` = (u, alive) at the start of trip `start`.  dev applies only in trip dev_trip.
    Returns (list of trip results, list of (u, alive) at the start of each trip start..K-1)."""
    u, alive = state
    res, starts = [], []
    for k in range(start, K):
        starts.append((u.copy(), alive.copy()))
        hit, size = gusts[k]
        r = simulate_trip(E0, D, hit, size, pr, u, alive, tau=tau, lam=lam, m_g=m_g, h=h, cmult=cmult,
                          dev=dev if k == dev_trip else None)
        res.append(r)
        u = r["u_end"]
        alive = r["alive_end"] if perm_death else np.ones_like(alive)
    return res, starts


def value(res, alpha, delta, beta=0.5, k0=0):
    """(n,3) discounted sum_k delta^(k0+k) U_k over the given trip results (k0 = absolute index of the first one)."""
    return sum((delta ** (k0 + k)) * sv.utilities(r, alpha, beta) for k, r in enumerate(res))


Z_MAX = 3.6
MODES = ("refuse", "early", "volunteer")


def bonf_z(m):
    """two-sided 0.05 Bonferroni over m deviations (normal quantile via erfc bisection; scipy-free)."""
    p = 0.025 / max(m, 1)
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) / 2 > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def dev_trips(K):
    """deviation trips: all of them for K<=3, else 0, 2, K//2, K-2, K-1."""
    return list(range(K)) if K <= 3 else sorted({0, 2, K // 2, K - 2, K - 1})


def eval_ledger_cell(K, E0, D, gusts, pr, m_g, lam, alphas, deltas, tau=15, h=0.05, beta=0.5, cmult=1.0, perm_death=False,
                     fracs=(0.1, 0.4, 0.7, 0.9)):
    """Baseline + all one-shot deviations of one bird in one trip.  Returns dict with summary and, per (alpha, delta),
    classification over (a) all deviation trips (b) non-final trips only (c) final trip only."""
    n = gusts[0][0].shape[0]
    N0 = (np.zeros((n, N), np.int64), np.ones((n, N), bool))
    base, starts = run_trips(K, E0, D, gusts, pr, N0, lam, tau, m_g, h, cmult, perm_death)
    summ = dict(team_arrival_trip0=float(base[0]["arrived"].mean()), all_arrive_trip0=float(base[0]["arrived"].all(1).mean()),
                team_arrival_mean=float(np.mean([r["arrived"].mean() for r in base])),
                all_arrive_mean=float(np.mean([r["arrived"].all(1).mean() for r in base])),
                lead_share=(sum(r["lead"] for r in base).mean(0) / (K * D)).tolist(),
                n_swaps_per_trip=float(np.mean([r["n_swaps"].mean() for r in base])))
    gm = {(a, d): [] for a in alphas for d in deltas}
    gs = {(a, d): [] for a in alphas for d in deltas}
    meta = []
    for j in dev_trips(K):
        for mode in MODES:
            for i in range(N):
                for f in fracs:
                    dv = dict(mode=mode, i=i, t_d=int(f * D))
                    res, _ = run_trips(K, E0, D, gusts, pr, starts[j], lam, tau, m_g, h, cmult, perm_death, dv, j, start=j)
                    meta.append(dict(trip=j, mode=mode, i=i, t_d=dv["t_d"]))
                    for a in alphas:
                        for d in deltas:
                            diff = (value(res, a, d, beta, j) - value(base[j:], a, d, beta, j))[:, i]
                            gm[(a, d)].append(float(diff.mean()))
                            gs[(a, d)].append(float(diff.std(ddof=1) / math.sqrt(n)))
    z = bonf_z(len(meta))
    final = np.array([m["trip"] == K - 1 for m in meta])
    out = dict(summary=summ, n_dev=len(meta), z=z, per={})
    for a in alphas:
        for d in deltas:
            m, e = np.array(gm[(a, d)]), np.array(gs[(a, d)])
            ent = {}
            for name, sel in (("all", np.ones_like(final)), ("nonfinal", ~final), ("final", final)):
                if not sel.any():
                    ent[name] = None
                    continue
                lab, k = sv.classify(m[sel], e[sel], z=z)
                ks = np.flatnonzero(sel)[k]
                ent[name] = dict(label=lab, labels_by_eps={str(x): sv.classify(m[sel], e[sel], z=z, eps=x)[0] for x in sv.EPS_LIST},
                                 max_gain=float(m[sel].max()), lo_max=float((m[sel] - z * e[sel]).max()),
                                 hi_max=float((m[sel] + z * e[sel]).max()), worst_dev=meta[ks], worst_gain=float(m[ks]),
                                 worst_se=float(e[ks]))
            out["per"][f"{a}|{d}"] = ent
    return out
