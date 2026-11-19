"""Difficulty ladder L1: per-position wake savings (horseshoe tables) in the energy-bucket model.

Position p (0 = leader) has a power multiplier m_p = 1 - f_ind * scale * (1 - D_p/D_solo)
(D_p/D_solo from the all-superposition horseshoe model; scale makes the best two-plane follower's
induced drag drop 40%).  Plane i at position p burns k_i * m_p per step; swap participants burn k_i
(no wake) + c * s_ref * k_i, with s_ref = mean follower saving (so c is "c average-follower steps").
Metric: team range / B0 = distance when the first battery hits 0.
Rules: fixed, rot (rotate whole line by one), gate (leader<->richest), sort (rank by remaining range),
pair (cost-aware best single swap).  Upper bound: fluid MILP, see opt_fluid().
Run (CPU only):
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/drone_bucket_l1.py [--quick|--anchor]
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horseshoe_formation as H
import drone_bucket as OLD

B0 = 1000.0
NS = [3, 5, 7, 10]
FORMS = ["V", "echelon"]
SETTINGS = ["actual", "optimal"]
FS = [0.3, 0.4, 0.5]
CS = [0, 1, 3, 10]
HS = [0.0, 0.05, 0.15, 0.30]   # 0.05 is an extra level (transition region), not in the task spec
RC, KIND = 0.075, "rankine"
TAUS = [10, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640]
DELTAS = [1, 2, 5, 10, 20, 40, 80]
SORT_TAUS = [5, 10, 20, 40, 80, 160]
SORT_DELTAS = [0, 5, 20]
PAIR_TAUS = [5, 10, 20, 40]
PAIR_DELTAS = [1, 3, 10, 30]
O_GRID = np.round(np.arange(-0.3, 0.6001, 0.025), 4)
OUT_DIR = ".scratch/drone_bucket/l1_jobs"


# ---------------------------------------------------------------- position tables
def two_plane_best():
    return min(float(H.ratios(np.array([[0, 0], [8.0, 1 - o]]), RC, KIND, "all")[1]) for o in O_GRID)


def ratio_table(N, form, setting):
    if setting == "actual":
        o, dx = 0.0, 3.0
    else:
        dx = 8.0
        o = min(O_GRID, key=lambda o: H.formation_eval(form, N, o, dx, RC, KIND)[1].mean())
    return H.formation_eval(form, N, float(o), dx, RC, KIND)[1], float(o), dx


def power_mult(r, f, scale):
    return np.round(1.0 - f * scale * (1.0 - r), 10)


# ---------------------------------------------------------------- simulator
def _take(m, pos):
    return np.take_along_axis(m, pos, 1)


def sorted_assign(rd, m):
    """largest remaining range -> position with largest multiplier (least saving); all move (initial layout)."""
    I, N = rd.shape
    ar = np.arange(I)[:, None]
    pl = np.argsort(-rd, axis=1, kind="stable")
    ps = np.argsort(-m, axis=1, kind="stable")
    pos = np.empty((I, N), int)
    pos[ar, pl] = ps
    return pos


def sort_target(pos, rd, m):
    """Sorted assignment (rd desc -> m desc) with the minimum number of planes moved."""
    I, N = pos.shape
    ar = np.arange(I)[:, None]
    mpl = _take(m, pos)
    ms = -np.sort(-m, axis=1)
    rank = np.argsort(np.argsort(-rd, axis=1, kind="stable"), axis=1)
    tm = np.take_along_axis(ms, rank, 1)
    mover = np.abs(tm - mpl) > 1e-9
    op = np.argsort(np.where(mover, -tm, np.inf), axis=1, kind="stable")
    ov = np.argsort(np.where(mover, -mpl, np.inf), axis=1, kind="stable")
    posv = np.take_along_axis(pos, ov, 1)
    nm = mover.sum(1)
    valid = np.arange(N)[None, :] < nm[:, None]
    newpos = pos.copy()
    newpos[ar, op] = np.where(valid, posv, newpos[ar, op])
    return newpos, mover


def best_pair(pos, rem, k, m, ceff, PA, PB):
    """Per instance, the single transposition with the largest predicted bottleneck-range gain
    (static assignment for the rest of the flight, swap-step cost charged). Returns (j, gain)."""
    I, N = pos.shape
    ar = np.arange(I)
    mpl = _take(m, pos)
    t0 = rem / (k * mpl)
    P = t0.min(1)
    mA, mB = mpl[:, PA], mpl[:, PB]
    kA, kB = k[:, PA], k[:, PB]
    ce = ceff[:, None]
    tA = (rem[:, PA] - kA * (1 + ce - mB)) / (kA * mB)
    tB = (rem[:, PB] - kB * (1 + ce - mA)) / (kB * mA)
    T = np.repeat(t0[:, None, :], len(PA), axis=1)
    idx = np.arange(len(PA))
    T[:, idx, PA] = tA
    T[:, idx, PB] = tB
    gain = T.min(2) - P[:, None]
    j = gain.argmax(1)
    return j, gain[ar, j]


def simulate(k, B, m, ceff, policy, p1=0, p2=0, init="sorted", lose=True):
    """Vectorised over instances. k,B,m:(I,N); m[:,p]=power multiplier at position p (p=0 leader);
    ceff:(I,) swap overhead in units of k_i. Returns range / B0."""
    I, N = k.shape
    ar = np.arange(I)
    rem = B.copy()
    pos = np.tile(np.arange(N), (I, 1)) if init == "identity" else sorted_assign(B / k, m)
    dist = np.zeros(I)
    alive = np.ones(I, bool)
    PA, PB = np.triu_indices(N, 1)
    nswap = np.zeros(I)
    t, tmax = 0, int(8 * B0)
    while alive.any() and t < tmax:
        part = np.zeros((I, N), bool)
        if policy == "rot" and t > 0 and t % p1 == 0:
            pos = (pos - 1) % N
            part[:] = True
        elif policy == "gate" and t > 0:
            rd = rem / k
            ld = (pos == 0).argmax(1)
            best = np.where(pos == 0, -np.inf, rd).argmax(1)
            go = ((rd[ar, best] - rd[ar, ld]) > p1) & alive
            if go.any():
                ig = ar[go]; a = ld[go]; b = best[go]
                pa = pos[ig, a].copy(); pos[ig, a] = pos[ig, b]; pos[ig, b] = pa
                part[ig, a] = True; part[ig, b] = True
        elif policy == "sort" and t > 0 and t % p1 == 0:
            rd = rem / k
            mpl = _take(m, pos)
            inv = ((rd[:, :, None] - rd[:, None, :]) > p2) & (mpl[:, :, None] < mpl[:, None, :] - 1e-9)
            go = inv.any((1, 2)) & alive
            if go.any():
                newpos, mover = sort_target(pos, rd, m)
                pos = np.where(go[:, None], newpos, pos)
                part = mover & go[:, None]
        elif policy == "pair" and t > 0 and t % p1 == 0:
            j, g = best_pair(pos, rem, k, m, ceff, PA, PB)
            go = (g > p2) & alive
            if go.any():
                ig = ar[go]; a = PA[j[go]]; b = PB[j[go]]
                pa = pos[ig, a].copy(); pos[ig, a] = pos[ig, b]; pos[ig, b] = pa
                part[ig, a] = True; part[ig, b] = True
        nswap += part.any(1)
        f = np.where(part & lose, 1.0, _take(m, pos))
        cons = k * f + ceff[:, None] * k * part
        new = rem - cons
        dead = alive & (new <= 0).any(axis=1)
        frac = np.where(cons > 0, rem / cons, np.inf).min(axis=1)
        dist += np.where(alive, np.where(dead, frac, 1.0), 0.0)
        rem = np.where(alive[:, None], new, rem)
        alive &= ~dead
        t += 1
    return dist / B0, nswap


# ---------------------------------------------------------------- fluid upper bound
def opt_fluid(k, B, m, ceff, lose=True, tl=5.0, integer=True):
    """Fluid upper bound on the range ratio for ANY rotation schedule.
    Positions with equal multiplier are merged into classes g (multiplicity n_g).  x_ig = distance plane i
    spends in class g (units of B0).  Time-share constraints: sum_g x_ig = D (a plane is somewhere),
    sum_i x_ig = n_g D (every slot is filled).  Energy: k_i sum_g m_g x_ig + swap overhead <= B_i/B0.
    Swap-cost approximation: z_ig=1 if plane i ever occupies class g; a plane that occupies n_i classes needs at
    least n_i - 1 position changes; each change into class g costs k_i((1-m_g)*lose + ceff)/B0 (wake lost in
    the swap step + c*s_ref).  The initial class f_ig is free (f continuous: a relaxation, Sum_g f_ig=1,
    Sum_i f_ig=n_g, f<=z).  Ignored: swap-step distance, integer steps, and the fact that one swap moves two
    planes at once (a swap is charged to both movers, as in the simulator).  Revisiting a class is free here
    -> upper bound.  When HiGHS stops at the time limit the dual bound is used (still an upper bound)."""
    N = len(k)
    b = B / B0
    uniq, inv, cnt = np.unique(m, return_inverse=True, return_counts=True)
    G = len(uniq)
    nx = N * G
    iD, ix, iz, jf = 0, 1, 1 + nx, 1 + 2 * nx
    nv = 1 + 3 * nx
    X = lambda i, g: ix + i * G + g
    Z = lambda i, g: iz + i * G + g
    F = lambda i, g: jf + i * G + g
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, l, h):
        nonlocal r
        for c_, v_ in entries:
            rows.append(r); cols.append(c_); vals.append(v_)
        lo.append(l); hi.append(h); r += 1

    mmin = uniq.min()
    U = b / (k * mmin)                      # max conceivable D for plane i
    Dub = float(U.min())
    for i in range(N):
        add([(X(i, g), 1.0) for g in range(G)] + [(iD, -1.0)], 0, 0)
    for g in range(G):
        add([(X(i, g), 1.0) for i in range(N)] + [(iD, -float(cnt[g]))], 0, 0)
    for i in range(N):
        for g in range(G):
            add([(X(i, g), 1.0), (Z(i, g), -Dub)], -np.inf, 0)
            add([(F(i, g), 1.0), (Z(i, g), -1.0)], -np.inf, 0)
    for i in range(N):
        add([(F(i, g), 1.0) for g in range(G)], 1, 1)
    for g in range(G):
        add([(F(i, g), 1.0) for i in range(N)], cnt[g], cnt[g])
    for i in range(N):
        ent = []
        for g in range(G):
            e = k[i] * ((1 - uniq[g]) * (1.0 if lose else 0.0) + ceff) / B0
            ent += [(X(i, g), k[i] * uniq[g]), (Z(i, g), e), (F(i, g), -e)]
        add(ent, -np.inf, b[i])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    cost = np.zeros(nv); cost[iD] = -1
    integ = np.zeros(nv)
    if integer:
        integ[iz:iz + nx] = 1
    ub = np.ones(nv); ub[iD] = Dub; ub[ix:ix + nx] = Dub
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), ub), options={"time_limit": tl, "mip_rel_gap": 1e-5})
    if res.x is None:
        return np.nan, np.nan, False
    D = float(res.x[iD])
    bound = D
    to = res.status != 0
    if to and getattr(res, "mip_dual_bound", None) is not None and np.isfinite(res.mip_dual_bound):
        bound = float(-res.mip_dual_bound)
    return bound, D, to


# ---------------------------------------------------------------- aggregation helpers
def boot_ci(x, rng, nb=2000):
    x = np.asarray(x)
    if len(x) < 2:
        return [float(x.mean())] * 2
    mm = rng.choice(len(x), (nb, len(x)))
    bs = x[mm].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def make_instances(N, tables, n_inst, scale):
    """Stack instances over (f, c, h) cells. tables: ratio vector (N,). Returns arrays and cell keys."""
    ks, Bs, Ms, Cs, keys = [], [], [], [], []
    for f in FS:
        m_vec = power_mult(tables, f, scale)
        s_ref = float(1 - m_vec[1:].mean())
        for c in CS:
            for h in HS:
                n = 1 if h == 0 else n_inst
                k, B = OLD.sample(1000 * N + int(round(h * 100)), n, N, h)
                ks.append(k); Bs.append(B)
                Ms.append(np.tile(m_vec, (n, 1)))
                Cs.append(np.full(n, c * s_ref))
                keys += [(f, c, h)] * n
    return (np.vstack(ks), np.vstack(Bs), np.vstack(Ms), np.concatenate(Cs), keys)


def run_job(args):
    N, form, setting, scale, n_inst, tl = args
    t0 = time.time()
    r, o_used, dx = ratio_table(N, form, setting)
    k, B, M, ce, keys = make_instances(N, r, n_inst, scale)
    I = len(k)
    res = {}
    res["fixed"] = {"-": simulate(k, B, M, ce, "fixed")[0]}
    res["rot"] = {str(t): simulate(k, B, M, ce, "rot", t)[0] for t in TAUS}
    res["gate"] = {str(d): simulate(k, B, M, ce, "gate", d)[0] for d in DELTAS}
    res["sort"] = {f"{t}/{d}": simulate(k, B, M, ce, "sort", t, d)[0] for t in SORT_TAUS for d in SORT_DELTAS}
    res["pair"] = {f"{t}/{d}": simulate(k, B, M, ce, "pair", t, d)[0] for t in PAIR_TAUS for d in PAIR_DELTAS}
    tsim = time.time() - t0
    opt = np.full(I, np.nan); optx = np.full(I, np.nan); nto = 0
    cache = {}
    for i in range(I):
        # identical instances (homogeneous cells) appear once, so no caching needed
        o_, x_, to = opt_fluid(k[i], B[i], M[i], ce[i], tl=tl)
        opt[i], optx[i] = o_, x_
        nto += int(to)
    topt = time.time() - t0 - tsim
    rng = np.random.default_rng(7)
    keyarr = np.array(keys, float)
    cells = []
    for f in FS:
        for c in CS:
            for h in HS:
                sel = np.where((keyarr[:, 0] == f) & (keyarr[:, 1] == c) & (keyarr[:, 2] == h))[0]
                o = opt[sel]
                cell = {"N": N, "form": form, "setting": setting, "f_ind": f, "c": c, "h": h,
                        "n_inst": len(sel), "opt": float(np.nanmean(o)),
                        "opt_incumbent": float(np.nanmean(optx[sel])),
                        "opt_timeouts": int(np.isnan(o).sum()), "rules": {}}
                best = None
                for rule, d in res.items():
                    pm = {p: float(v[sel].mean()) for p, v in d.items()}
                    pbest = max(pm, key=pm.get)
                    v = d[pbest][sel]
                    g = (o - v) / o
                    cell["rules"][rule] = {"param": pbest, "range": float(v.mean()), "gap": float(np.nanmean(g)),
                                           "gap_ci95": boot_ci(g[~np.isnan(g)], rng),
                                           "gap_min": float(np.nanmin(g)), "frac_inst_gap_gt2": float(np.nanmean(g > 0.02))}
                    if best is None or v.mean() > best[1]:
                        best = (rule, v.mean())
                cell["best_rule"] = best[0]
                cell["best_gap"] = cell["rules"][best[0]]["gap"]
                cell["best_gap_ci95"] = cell["rules"][best[0]]["gap_ci95"]
                cells.append(cell)
    out = {"N": N, "form": form, "setting": setting, "overlap_o": o_used, "dx": dx,
           "ratios": r.tolist(), "time_sim_s": tsim, "time_opt_s": topt, "opt_timeouts_total": nto,
           "cells": cells}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(f"{OUT_DIR}/{N}_{form}_{setting}.json", "w"))
    return out


# ---------------------------------------------------------------- anchors
def anchors(scale):
    out = {}
    # A. simulator + LP with uniform follower saving vs the OLD two-level code and the OLD result file
    old = json.load(open("calib/out/drone_bucket.json"))["cells"]
    rows = []
    for N, s, c, h in [(5, 0.2, 0, 0.15), (5, 0.2, 3, 0.15), (10, 0.3, 10, 0.30), (3, 0.1, 1, 0.15), (7, 0.3, 3, 0.30)]:
        k, B = OLD.sample(1000 * N + int(round(h * 100)), 200, N, h)
        I = len(k)
        m = np.tile(np.r_[1.0, np.full(N - 1, 1 - s)], (I, 1))
        ce = np.full(I, c * s)
        oc = [x for x in old if x["N"] == N and x["s"] == s and x["c"] == c and abs(x["h"] - h) < 1e-9][0]
        row = {"N": N, "s": s, "c": c, "h": h}
        # exact simulator equivalence at fixed parameters
        for pol, p in [("fixed", 0), ("rot", 40), ("gate", 10)]:
            a, _ = simulate(k, B, m, ce, pol, p, init="identity")
            b, _ = OLD.simulate(k, B, s, c, {"rot": "tail"}.get(pol, pol), p)
            row[f"sim_maxdiff_{pol}"] = float(np.abs(a - b).max())
        # best-parameter means vs stored old results
        fx = simulate(k, B, m, ce, "fixed", init="identity")[0].mean()
        rt = max(simulate(k, B, m, ce, "rot", t, init="identity")[0].mean() for t in OLD.TAUS)
        gt = max(simulate(k, B, m, ce, "gate", d, init="identity")[0].mean() for d in OLD.DELTAS)
        row.update(new_fixed=float(fx), old_fixed=oc["fixed"], new_rot=float(rt), old_tail=oc["tail"],
                   new_gate=float(gt), old_gate=oc["gate"])
        nopt = np.array([opt_fluid(k[i], B[i], m[i], ce[i])[0] for i in range(40)])
        oopt = np.array([OLD.opt_ratio(k[i], B[i], s, c) for i in range(40)])
        row.update(new_opt_40=float(nopt.mean()), old_opt_40=float(oopt.mean()),
                   opt_rel_diff_max=float(np.max(np.abs(nopt / oopt - 1))), opt_rel_diff_mean=float(np.mean(nopt / oopt - 1)))
        rows.append(row)
    out["uniform_vs_old"] = rows
    # B. theory: homogeneous, free swaps -> range = 1/mean(m) (real tables) and 1/(1-s(N-1)/N) (uniform)
    rows = []
    for N in NS:
        for form in FORMS:
            for setting in SETTINGS:
                r, _, _ = ratio_table(N, form, setting)
                m = power_mult(r, 0.5, scale)
                k = np.ones(N); B = np.full(N, B0)
                th = 1 / m.mean()
                o, _, _ = opt_fluid(k, B, m, 0.0, lose=False)
                kk = np.ones((1, N)); BB = np.full((1, N), B0)
                sim = simulate(kk, BB, m[None], np.zeros(1), "rot", 1, lose=False)[0][0]
                rows.append({"N": N, "form": form, "setting": setting, "theory": th, "lp": o, "sim_rot_tau1": float(sim)})
    out["theory_real_tables"] = rows
    out["theory_max_rel_err_lp"] = max(abs(x["lp"] / x["theory"] - 1) for x in rows)
    out["theory_max_rel_err_sim"] = max(abs(x["sim_rot_tau1"] / x["theory"] - 1) for x in rows)
    return out


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
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--only", default=None, help="N,form,setting for a single job")
    ap.add_argument("--tl", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = resolve_out(a.out, a.quick, 'calib/out/drone_bucket_l1.json')
    t0 = time.time()
    r2 = two_plane_best()
    scale = 0.40 / (1 - r2)
    print(f"two-plane best ratio {r2:.4f}, scale {scale:.4f}", flush=True)
    if a.anchor:
        an = anchors(scale)
        json.dump(an, open("calib/out/drone_bucket_l1_anchor.json", "w"), indent=1)
        print(json.dumps(an["uniform_vs_old"], indent=1)); print(an["theory_max_rel_err_lp"], an["theory_max_rel_err_sim"])
        sys.exit()
    n_inst = 20 if a.quick else 200
    jobs = [(N, f, s, scale, n_inst, a.tl) for N in NS for f in FORMS for s in SETTINGS]
    if a.only:
        n, f, s = a.only.split(","); jobs = [j for j in jobs if (j[0], j[1], j[2]) == (int(n), f, s)]
    jobs.sort(key=lambda j: -j[0])
    import multiprocessing as mp
    done = []
    with mp.get_context("spawn").Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(run_job, jobs)):
            done.append(r)
            print(i + 1, len(jobs), r["N"], r["form"], r["setting"], f"sim {r['time_sim_s']:.0f}s opt {r['time_opt_s']:.0f}s "
                  f"timeouts {r['opt_timeouts_total']}", f"total {time.time()-t0:.0f}s", flush=True)
    an = anchors(scale) if not a.quick else None
    done.sort(key=lambda r: (r["N"], r["form"], r["setting"]))
    json.dump(dict(config=dict(B0=B0, n_inst=n_inst, scale=scale, two_plane_best_ratio=r2, rc=RC, kind=KIND,
                               taus=TAUS, deltas=DELTAS, sort_taus=SORT_TAUS, sort_deltas=SORT_DELTAS,
                               pair_taus=PAIR_TAUS, pair_deltas=PAIR_DELTAS, tl=a.tl),
                   anchor=an, jobs=done), open(a.out, "w"))
    print("done", f"{time.time()-t0:.0f}s")
