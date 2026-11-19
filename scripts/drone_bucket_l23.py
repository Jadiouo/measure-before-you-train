"""Difficulty ladder L2 (random disturbances) and L3 (local information only) on the L1 energy-bucket model.

L2: per-step consumption multiplier w_it (per plane, independent across planes): iid or AR(1) gusts, sigma in
    {5,15,30}%, plus an optional event (prob p per episode: one plane's k rises 20% permanently at a random mid-flight time).
    Rules see only remaining battery (and, for the *_est variants, an EMA of their own observed burn ratio).
    Upper bound = hindsight LP: whole noise sequence known, block-time-indexed fluid MILP (opt_hind).  LOOSE bound.
L3: each plane sees only its own and its chain neighbours' (position index p+-1) remaining battery; swaps are decided
    pairwise (odd-even transposition "bubble" rule).  No central sorting; initial layout = arbitrary (identity).
Reuses scripts/drone_bucket_l1.py (position tables, LP, helpers).
Run (CPU only):
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/drone_bucket_l23.py [--quick]
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drone_bucket_l1 as L1
import drone_bucket as OLD

B0 = L1.B0
NS = [5, 10]
CS = [1, 10]
HS = [0.0, 0.15]
F_IND = 0.4
TMAX = 2400
EVENT_GAIN = 1.2
EPS_M = 0.003           # positions whose multipliers differ by less than this are treated as equal by the local rule
BLOCK = 20
# noise configs: (kind, L, sigma)
NOISES = [("none", 0, 0.0)] + [("iid", 0, s) for s in (0.05, 0.15, 0.30)] + \
         [("ar", Lc, s) for Lc in (10, 50) for s in (0.05, 0.15, 0.30)]
PEVENTS = [0.0, 0.2]
TAUS = L1.TAUS
DELTAS = L1.DELTAS
DWELLS = [0, 40]
EST_DELTAS = [2, 5, 10, 20, 40]
EST_ALPHAS = [0.02, 0.1]
LOC_TAUS = [5, 10, 20]
LOC_DELTAS = [2, 5, 10, 20]
LOC_ALPHAS = [0.0, 0.05]       # 0 = nominal rem/k (no estimate)
OUT_DIR = ".scratch/drone_bucket/l23_jobs"


# ---------------------------------------------------------------- disturbances
def gen_noise(seed, I, N, kind, Lc, sigma):
    """w[t,i,n] >= 0.1, mean ~1.  iid or stationary AR(1) with correlation length Lc (phi=exp(-1/Lc))."""
    if kind == "none":
        return None
    r = np.random.default_rng(seed)
    eps = r.standard_normal((TMAX, I, N)).astype(np.float32)
    if kind == "ar":
        phi = np.float32(np.exp(-1.0 / Lc)); q = np.float32(np.sqrt(1 - phi * phi))
        e = np.empty_like(eps); e[0] = eps[0]
        for t in range(1, TMAX):
            e[t] = phi * e[t - 1] + q * eps[t]
        eps = e
    return np.maximum(1 + np.float32(sigma) * eps, np.float32(0.1))


def gen_event(seed, I, N, p):
    r = np.random.default_rng(seed)
    on = r.random(I) < p
    plane = r.integers(0, N, I)
    te = r.integers(int(0.25 * B0), int(0.75 * B0), I)
    return on, plane, te


# ---------------------------------------------------------------- simulator
def simulate(k, B, m, ceff, w, ev, policy, p1=0, p2=0, p3=0.0, init="sorted"):
    """Vectorised over instances. w:(TMAX,I,N) or None; ev=(on,plane,te) or None.
    policy: fixed | rot(tau=p1) | gate(delta=p1, dwell=p2, alpha=p3; alpha 0 -> nominal rem/k) |
            local(tau=p1, delta=p2, alpha=p3).  Returns range/B0, #swap steps."""
    I, N = k.shape
    ar = np.arange(I)
    rem = B.copy()
    pos = np.tile(np.arange(N), (I, 1)) if init == "identity" else L1.sorted_assign(B / k, m)
    mcls = m[0]
    dist = np.zeros(I); alive = np.ones(I, bool); nswap = np.zeros(I)
    kmul = np.ones((I, N)); rho = np.ones((I, N)); last = np.full(I, -10 ** 6)
    PA, PB = np.triu_indices(N, 1)
    t = 0
    while alive.any():
        if t >= TMAX:
            raise RuntimeError("TMAX too small")
        if ev is not None:
            trig = ev[0] & (ev[2] == t)
            if trig.any():
                kmul[ar[trig], ev[1][trig]] = EVENT_GAIN
        part = np.zeros((I, N), bool)
        if t > 0:
            if policy == "rot" and t % p1 == 0:
                pos = (pos - 1) % N
                part[:] = True
            elif policy == "gate":
                rd = rem / (k * (rho if p3 > 0 else 1.0))
                ld = (pos == 0).argmax(1)
                best = np.where(pos == 0, -np.inf, rd).argmax(1)
                go = ((rd[ar, best] - rd[ar, ld]) > p1) & alive & ((t - last) >= p2)
                if go.any():
                    ig = ar[go]; a = ld[go]; b = best[go]
                    pa = pos[ig, a].copy(); pos[ig, a] = pos[ig, b]; pos[ig, b] = pa
                    part[ig, a] = True; part[ig, b] = True
                    last[ig] = t
            elif policy == "local" and t % p1 == 0:
                rd = rem / (k * (rho if p3 > 0 else 1.0))
                occ = np.argsort(pos, axis=1)
                for p in range((t // p1) % 2, N - 1, 2):
                    mp, mq = mcls[p], mcls[p + 1]
                    if abs(mp - mq) < EPS_M:
                        continue
                    a = occ[:, p]; b = occ[:, p + 1]
                    if mp > mq:
                        go = (rd[ar, a] + p2 < rd[ar, b])
                    else:
                        go = (rd[ar, b] + p2 < rd[ar, a])
                    go &= alive
                    if go.any():
                        ig = ar[go]; aa = a[go]; bb = b[go]
                        pos[ig, aa] = p + 1; pos[ig, bb] = p
                        part[ig, aa] = True; part[ig, bb] = True
        nswap += part.any(1)
        mpl = np.take_along_axis(m, pos, 1)
        f = np.where(part, 1.0, mpl)
        wt = 1.0 if w is None else w[t]
        cons = k * kmul * wt * f + ceff[:, None] * k * part
        if p3 > 0 and policy in ("gate", "local"):
            obs = kmul * wt
            rho = np.where(part, rho, (1 - p3) * rho + p3 * obs)
        new = rem - cons
        dead = alive & (new <= 0).any(axis=1)
        frac = np.where(cons > 0, rem / cons, np.inf).min(axis=1)
        dist += np.where(alive, np.where(dead, frac, 1.0), 0.0)
        rem = np.where(alive[:, None], new, rem)
        alive &= ~dead
        t += 1
    return dist / B0, nswap


# ---------------------------------------------------------------- hindsight upper bound
def opt_hind(k, B, m, ceff, w, ev_i, delta=BLOCK, tl=5.0):
    """Hindsight fluid MILP: noise sequence w[:,n] (T,N) and event (plane,te)|None known in advance.
    Block-time-indexed: block j (delta steps) has fraction u_j of the horizon flown (nonincreasing), plane i spends share
    x_jig of the block in position class g.  Energy: sum_j delta*W_ji*k_i*m_g*x_jig + swap overhead <= B_i, where
    W_ji = mean over block j of w*kmul (block means -> intra-block timing is NOT exploited, so this is slightly BELOW a strict
    hindsight bound; see the block-size check in the notes).  Swap cost as in L1 (z/f ever-occupied classes).
    Horizon = first step at which some plane could not have flown further even at the cheapest position.
    Returns (bound, incumbent, timed_out)."""
    N = len(k)
    wk = np.ones((TMAX, N)) if w is None else w.astype(float)
    wk = wk * k[None, :]
    if ev_i is not None:
        pl, te = ev_i
        wk[te:, pl] *= EVENT_GAIN
    uniq, inv, cnt = np.unique(m, return_inverse=True, return_counts=True)
    G = len(uniq)
    cum = np.cumsum(wk, axis=0) * uniq.min()
    Ti = (cum < B[None, :]).sum(0)           # steps plane i can fly at best
    H = int(Ti.min()) + 1
    if delta is None:          # one block over the whole horizon: 'slow-information' reference, NOT a hindsight bound
        delta = H
    J = int(np.ceil(H / delta))
    pad = J * delta - H
    wkp = np.vstack([wk[:H], np.tile(wk[H - 1], (pad, 1))]) if pad else wk[:H]
    W = wkp.reshape(J, delta, N).mean(1)       # (J,N): mean of k_i*w*kmul
    b = B / B0
    nx = J * N * G
    iu, ix = 0, J
    iz = ix + nx; jf = iz + N * G
    nv = jf + N * G
    X = lambda j, i, g: ix + (j * N + i) * G + g
    Z = lambda i, g: iz + i * G + g
    F = lambda i, g: jf + i * G + g
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = [0]

    def add(ent, l, h):
        for c_, v_ in ent:
            rows.append(r[0]); cols.append(c_); vals.append(v_)
        lo.append(l); hi.append(h); r[0] += 1

    for j in range(J):
        for i in range(N):
            add([(X(j, i, g), 1.0) for g in range(G)] + [(iu + j, -1.0)], 0, 0)
        for g in range(G):
            add([(X(j, i, g), 1.0) for i in range(N)] + [(iu + j, -float(cnt[g]))], 0, 0)
        for i in range(N):
            for g in range(G):
                add([(X(j, i, g), 1.0), (Z(i, g), -1.0)], -np.inf, 0)
        if j + 1 < J:
            add([(iu + j + 1, 1.0), (iu + j, -1.0)], -np.inf, 0)
    for i in range(N):
        for g in range(G):
            add([(F(i, g), 1.0), (Z(i, g), -1.0)], -np.inf, 0)
        add([(F(i, g), 1.0) for g in range(G)], 1, 1)
    for g in range(G):
        add([(F(i, g), 1.0) for i in range(N)], cnt[g], cnt[g])
    for i in range(N):
        ent = []
        for g in range(G):
            e = k[i] * ((1 - uniq[g]) + ceff) / B0
            ent += [(Z(i, g), e), (F(i, g), -e)]
            ent += [(X(j, i, g), delta * W[j, i] * uniq[g] / B0) for j in range(J)]
        add(ent, -np.inf, b[i])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r[0], nv))
    cost = np.zeros(nv); cost[:J] = -delta / B0
    integ = np.zeros(nv); integ[iz:iz + N * G] = 1
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), np.ones(nv)), options={"time_limit": tl, "mip_rel_gap": 1e-5})
    if res.x is None:
        return np.nan, np.nan, True
    D = float(-res.fun)
    bound = D
    to = res.status != 0
    if to and getattr(res, "mip_dual_bound", None) is not None and np.isfinite(res.mip_dual_bound):
        bound = float(-res.mip_dual_bound)
    return bound, D, to


# ---------------------------------------------------------------- one job = (N, noise cfg, event prob)
def rule_grid():
    g = {"fixed": [("-", dict(policy="fixed"))]}
    g["rot"] = [(str(t), dict(policy="rot", p1=t)) for t in TAUS]
    g["gate"] = [(f"{d}/{dw}", dict(policy="gate", p1=d, p2=dw)) for d in DELTAS for dw in DWELLS]
    g["gate_est"] = [(f"{d}/{dw}/{a}", dict(policy="gate", p1=d, p2=dw, p3=a))
                     for d in EST_DELTAS for dw in DWELLS for a in EST_ALPHAS]
    g["local"] = [(f"{t}/{d}/{a}", dict(policy="local", p1=t, p2=d, p3=a, init="identity"))
                  for t in LOC_TAUS for d in LOC_DELTAS for a in LOC_ALPHAS]
    g["local_s"] = [(f"{t}/{d}/{a}", dict(policy="local", p1=t, p2=d, p3=a, init="sorted"))
                    for t in LOC_TAUS for d in LOC_DELTAS for a in LOC_ALPHAS]
    return g


def run_job(args):
    N, ni, pev, n_inst, n_lp, tl = args
    kind, Lc, sigma = NOISES[ni]
    t0 = time.time()
    r, _, _ = L1.ratio_table(N, "V", "actual")
    r2 = L1.two_plane_best(); scale = 0.40 / (1 - r2)
    mvec = L1.power_mult(r, F_IND, scale)
    s_ref = float(1 - mvec[1:].mean())
    grid = rule_grid()
    cells = []
    for h in HS:
        trivial = (kind == "none" and pev == 0.0)
        n = 1 if (h == 0 and trivial) else n_inst
        k, B = OLD.sample(1000 * N + int(round(h * 100)), n, N, h)
        seed = 10 ** 6 + 1000 * N + 10 * ni + int(round(h * 100))
        w = gen_noise(seed, n, N, kind, Lc, sigma)
        ev = gen_event(seed + 7, n, N, pev) if pev > 0 else None
        M = np.tile(mvec, (n, 1))
        for c in CS:
            ce = np.full(n, c * s_ref)
            rules = {}
            for rule, plist in grid.items():
                d = {}
                for name, kw in plist:
                    kw = dict(kw)
                    d[name] = simulate(k, B, M, ce, w, ev, **kw)[0]
                rules[rule] = d
            # hindsight bound on the first n_lp instances
            nl = min(n, n_lp)
            opt = np.full(nl, np.nan); nto = 0; inc = np.full(nl, np.nan)
            for i in range(nl):
                wi = None if w is None else w[:, i, :]
                evi = None if (ev is None or not ev[0][i]) else (int(ev[1][i]), int(ev[2][i]))
                if trivial:
                    o_, x_, to = L1.opt_fluid(k[i], B[i], mvec, ce[i], tl=tl)
                else:
                    o_, x_, to = opt_hind(k[i], B[i], mvec, ce[i], wi, evi, tl=tl)
                opt[i], inc[i] = o_, x_
                nto += int(to)
            rng = np.random.default_rng(11)
            cell = {"N": N, "noise": kind, "L": Lc, "sigma": sigma, "p_event": pev, "c": c, "h": h,
                    "n_inst": n, "n_lp": nl, "opt": float(np.nanmean(opt)), "opt_incumbent": float(np.nanmean(inc)),
                    "opt_timeouts": nto, "rules": {}}
            for rule, d in rules.items():
                pm = {p: float(v.mean()) for p, v in d.items()}
                pbest = max(pm, key=pm.get)
                v = d[pbest]
                g = (opt - v[:nl]) / opt
                cell["rules"][rule] = {"param": pbest, "range": float(v.mean()), "range_lp_subset": float(v[:nl].mean()),
                                       "gap": float(np.nanmean(g)), "gap_ci95": L1.boot_ci(g[~np.isnan(g)], rng),
                                       "n_events": int(ev[0].sum()) if ev is not None else 0}
            glob = {x: cell["rules"][x]["range"] for x in ("fixed", "rot", "gate", "gate_est")}
            bg = max(glob, key=glob.get)
            cell["best_global"] = bg
            cell["best_global_gap"] = cell["rules"][bg]["gap"]
            cell["best_global_gap_ci95"] = cell["rules"][bg]["gap_ci95"]
            cell["local_loss_vs_gate"] = float(1 - cell["rules"]["local"]["range"] / cell["rules"][bg]["range"])
            cells.append(cell)
    out = {"N": N, "noise": kind, "L": Lc, "sigma": sigma, "p_event": pev, "ratios": r.tolist(),
           "s_ref": s_ref, "time_s": time.time() - t0, "cells": cells}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(f"{OUT_DIR}/{N}_{ni}_{pev}.json", "w"))
    return out


def run_coarse(args):
    """Supplementary: re-simulate each cell's best params, compare to the J=1 (whole-horizon) LP reference."""
    N, ni, pev, n_inst, n_lp, tl = args
    kind, Lc, sigma = NOISES[ni]
    r, _, _ = L1.ratio_table(N, "V", "actual")
    scale = 0.40 / (1 - L1.two_plane_best()); mvec = L1.power_mult(r, F_IND, scale); s_ref = float(1 - mvec[1:].mean())
    grid = rule_grid()
    saved = json.load(open(f"{OUT_DIR}/{N}_{ni}_{pev}.json"))
    out = []
    for cell in saved["cells"]:
        h, c = cell["h"], cell["c"]
        trivial = (kind == "none" and pev == 0.0)
        n = 1 if (h == 0 and trivial) else n_inst
        k, B = OLD.sample(1000 * N + int(round(h * 100)), n, N, h)
        seed = 10 ** 6 + 1000 * N + 10 * ni + int(round(h * 100))
        w = gen_noise(seed, n, N, kind, Lc, sigma)
        ev = gen_event(seed + 7, n, N, pev) if pev > 0 else None
        M = np.tile(mvec, (n, 1)); ce = np.full(n, c * s_ref)
        nl = min(n, n_lp)
        opt = np.full(nl, np.nan); nto = 0
        for i in range(nl):
            wi = None if w is None else w[:, i, :]
            evi = None if (ev is None or not ev[0][i]) else (int(ev[1][i]), int(ev[2][i]))
            o_, x_, to = opt_hind(k[i], B[i], mvec, ce[i], wi, evi, delta=None, tl=tl)
            opt[i] = o_; nto += int(to)
        rec = {"N": N, "noise": kind, "L": Lc, "sigma": sigma, "p_event": pev, "c": c, "h": h, "opt_coarse": float(np.nanmean(opt)), "timeouts": nto, "gaps": {}}
        for rule in ("fixed", "rot", "gate", "gate_est", "local", "local_s"):
            name = cell["rules"][rule]["param"]
            kw = dict([x for x in grid[rule] if x[0] == name][0][1])
            v = simulate(k, B, M, ce, w, ev, **kw)[0][:nl]
            rec["gaps"][rule] = float(np.nanmean((opt - v) / opt))
        out.append(rec)
    return out


def block_check():
    """How much does the block size matter?  Same instances, delta in {20,5,1}."""
    r, _, _ = L1.ratio_table(5, "V", "actual")
    scale = 0.40 / (1 - L1.two_plane_best()); m = L1.power_mult(r, F_IND, scale); sref = float(1 - m[1:].mean())
    rows = []
    for kind, Lc, sg in [("iid", 0, 0.30), ("ar", 50, 0.30)]:
        k, B = OLD.sample(5015, 6, 5, 0.15)
        w = gen_noise(99, 6, 5, kind, Lc, sg)
        for dl in (20, 5, 1):
            v = [opt_hind(k[i], B[i], m, 1 * sref, w[:, i, :], None, delta=dl, tl=60)[0] for i in range(6)]
            rows.append({"noise": kind, "sigma": sg, "delta": dl, "mean_bound": float(np.mean(v))})
    return rows


def resolve_out(out, quick, default, root="."):
    """--out not given + --quick -> .scratch/quick/<name> (never clobber the paper result files)."""
    import os
    if out is not None: return out
    if not quick: return default
    p = os.path.join(root, ".scratch/quick", os.path.basename(default))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    print("[--quick] no --out given; writing to", p, "(not the paper file", default + ")", flush=True)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--blockcheck", action="store_true")
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--nlp", type=int, default=40)
    ap.add_argument("--ninst", type=int, default=200)
    ap.add_argument("--tl", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="N,noise_index,p_event")
    a = ap.parse_args()
    a.out = resolve_out(a.out, a.quick, 'calib/out/drone_bucket_l23.json')
    if a.blockcheck:
        print(json.dumps(block_check(), indent=1)); sys.exit()
    n_inst = 20 if a.quick else a.ninst
    n_lp = 4 if a.quick else a.nlp
    jobs = [(N, ni, pev, n_inst, n_lp, a.tl) for N in NS for ni in range(len(NOISES)) for pev in PEVENTS]
    if a.only:
        n, ni, pv = a.only.split(","); jobs = [j for j in jobs if (j[0], j[1], j[2]) == (int(n), int(ni), float(pv))]
    jobs.sort(key=lambda j: (-j[0], -j[1]))
    t0 = time.time()
    if a.coarse:
        import multiprocessing as mp
        res = []
        with mp.get_context("spawn").Pool(4) as p:
            for r_ in p.imap_unordered(run_coarse, jobs):
                res += r_; print(len(res), flush=True)
        json.dump(res, open("calib/out/drone_bucket_l23_coarse.json", "w")); print("done", time.time() - t0); sys.exit()
    if a.bench:
        r = run_job(jobs[0]); print(r["time_s"], [(c["c"], c["h"], c["opt"], c["best_global_gap"]) for c in r["cells"]]); sys.exit()
    import multiprocessing as mp
    done = []
    with mp.get_context("spawn").Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(run_job, jobs)):
            done.append(r)
            print(i + 1, len(jobs), r["N"], r["noise"], r["L"], r["sigma"], r["p_event"], f"{r['time_s']:.0f}s",
                  f"total {time.time()-t0:.0f}s", flush=True)
    done.sort(key=lambda r: (r["N"], r["noise"], r["L"], r["sigma"], r["p_event"]))
    json.dump(dict(config=dict(B0=B0, n_inst=n_inst, n_lp=n_lp, tl=a.tl, f_ind=F_IND, block=BLOCK, noises=NOISES,
                               p_events=PEVENTS, event_gain=EVENT_GAIN, eps_m=EPS_M), jobs=done), open(a.out, "w"))
    print("done", f"{time.time()-t0:.0f}s")
