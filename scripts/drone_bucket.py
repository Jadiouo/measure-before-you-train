"""Energy-bucket toy model for drone formation rotation (pure numpy + scipy MILP for the optimum).

Question: how much better than a rule-based time rotation is the optimal rotation?
Model (discrete time, 1 distance unit per step):
  position 0 = leader burns k_i; follower burns (1-s)*k_i. Battery B_i = B0*(1+u_i), k_i = 1+v_i.
  swap step: participants lose the wake (burn k_i) and pay extra c*s*k_i.   (c in units of "s-steps")
  tail-rotation swap: all N planes participate; leader<->plane swap: 2 planes participate.
Metric: distance when the first battery hits 0, divided by B0 (single-plane range at k=1).
Optimum (policy 4): fluid MILP, see opt_ratio().  Run:
  .venv/bin/python scripts/drone_bucket.py [--quick]
"""
import argparse, json, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import milp, LinearConstraint, Bounds

B0 = 1000.0
NS = [3, 5, 7, 10]
SS = [0.1, 0.2, 0.3]
CS = [0, 1, 3, 10]
HS = [0.0, 0.05, 0.15, 0.30]
TAUS = [5, 10, 20, 40, 80, 160, 320, 640]
DELTAS = [1, 2, 5, 10, 20, 40, 80]
SIGMA = 0.10          # per-step consumption-measurement noise in the hidden-k variant


def sample(seed, n, N, h):
    r = np.random.default_rng(seed)
    k = 1 + r.uniform(-h, h, (n, N)) if h > 0 else np.ones((n, N))
    B = B0 * (1 + r.uniform(-h, h, (n, N))) if h > 0 else np.full((n, N), B0)
    return k, B


def simulate(k, B, s, c, policy, param=0, lose=True, hidden_sigma=None, seed=0):
    """Vectorised over instances. policy: fixed | tail | pair | gate.
    hidden_sigma=None -> gate sees true k; else gate uses k_hat from noisy own consumption."""
    I, N = k.shape
    rem = B.copy()
    order = np.tile(np.arange(N), (I, 1))
    lastled = np.zeros((I, N))
    lastled[:, 0] = 0
    dist = np.zeros(I)
    alive = np.ones(I, bool)
    ar = np.arange(I)
    rng = np.random.default_rng(seed + 12345)
    sum_obs = np.zeros((I, N)); sum_f = np.zeros((I, N))
    t = 0
    tmax = int(6 * B0)
    while alive.any() and t < tmax:
        part = np.zeros((I, N), bool)
        if policy == "tail" and t > 0 and t % param == 0:
            order = np.roll(order, -1, axis=1)
            part[:] = True
        elif policy == "pair" and t > 0 and t % param == 0:
            ll = np.take_along_axis(lastled, order, axis=1)   # last-led time by position
            ll[:, 0] = np.inf
            j = ll.argmin(axis=1)
            a, b = order[ar, 0].copy(), order[ar, j].copy()
            order[ar, 0], order[ar, j] = b, a
            part[ar, a] = True; part[ar, b] = True
        elif policy == "gate" and t > 0:
            if hidden_sigma is None:
                kk = k
            else:
                kk = np.where(sum_f > 0, sum_obs / np.maximum(sum_f, 1e-12), 1.0)
            rd = rem / kk
            ld = order[:, 0]
            rdo = np.take_along_axis(rd, order, axis=1)
            rdo[:, 0] = -np.inf
            j = rdo.argmax(axis=1)
            best = order[ar, j]
            go = (rd[ar, best] - rd[ar, ld]) > param
            if go.any():
                a, b = ld[go].copy(), best[go].copy()
                ig = ar[go]
                order[ig, 0], order[ig, j[go]] = b, a
                part[ig, a] = True; part[ig, b] = True
        lead = np.zeros((I, N), bool)
        lead[ar, order[:, 0]] = True
        nowake = lead | (part if lose else False)
        f = np.where(nowake, 1.0, 1.0 - s)
        cons = k * f + c * s * k * part
        lastled[ar, order[:, 0]] = t
        if hidden_sigma is not None:
            obs = k * f * (1 + hidden_sigma * rng.standard_normal((I, N)))
            use = ~part
            sum_obs += np.where(use, obs, 0); sum_f += np.where(use, f, 0)
        new = rem - cons
        dead = alive & (new <= 0).any(axis=1)
        frac = np.where(cons > 0, rem / cons, np.inf).min(axis=1)
        dist += np.where(alive, np.where(dead, frac, 1.0), 0.0)
        rem = np.where(alive[:, None], new, rem)
        alive &= ~dead
        t += 1
    return dist / B0, rem.min(axis=1)


def opt_ratio(k, B, s, c, lose=True, tl=10.0):
    """Fluid-optimal range ratio (upper bound for any rotation schedule).
    Plane i is led for distance T_i; it burns k_i[(1-s)D + s T_i] plus swap overhead.
    Minimum-swap structure: leaders form a chain a_0->a_1->...->a_m (one contiguous block each,
    m+1 = #leaders).  Plane entering as non-first leader pays (lose+c)*s*k_i, plane leaving as
    non-last leader pays c*s*k_i.  Binary y_i (leads), f_i (first), l_i (last) -> MILP (HiGHS).
    Approximation: fluid (continuous D, T_i); swap-step distance and integer steps ignored,
    and a plane never leads twice (more swaps never help here)."""
    N = len(k)
    b = B / B0
    kk = k
    en = (1.0 if lose else 0.0) + c
    # vars: D, T(N), y(N), f(N), l(N)
    nv = 1 + 4 * N
    cost = np.zeros(nv); cost[0] = -1
    A, lo, hi = [], [], []
    for i in range(N):
        # kk(1-s)D + kk s T_i + en*s*kk*(y_i-f_i)/B0 + c*s*kk*(y_i-l_i)/B0 <= b_i
        row = np.zeros(nv)
        row[0] = kk[i] * (1 - s); row[1 + i] = kk[i] * s
        row[1 + N + i] = (en + c) * s * kk[i] / B0
        row[1 + 2 * N + i] = -en * s * kk[i] / B0
        row[1 + 3 * N + i] = -c * s * kk[i] / B0
        A.append(row); lo.append(-np.inf); hi.append(b[i])
        row = np.zeros(nv); row[1 + i] = 1; row[1 + N + i] = -4.0   # T_i <= M y_i
        A.append(row); lo.append(-np.inf); hi.append(0)
        row = np.zeros(nv); row[1 + 2 * N + i] = 1; row[1 + N + i] = -1
        A.append(row); lo.append(-np.inf); hi.append(0)
        row = np.zeros(nv); row[1 + 3 * N + i] = 1; row[1 + N + i] = -1
        A.append(row); lo.append(-np.inf); hi.append(0)
    row = np.zeros(nv); row[0] = -1; row[1:1 + N] = 1          # sum T = D
    A.append(row); lo.append(0); hi.append(0)
    row = np.zeros(nv); row[1 + 2 * N:1 + 3 * N] = 1; A.append(row); lo.append(1); hi.append(1)
    row = np.zeros(nv); row[1 + 3 * N:1 + 4 * N] = 1; A.append(row); lo.append(1); hi.append(1)
    integ = np.zeros(nv); integ[1 + N:] = 1
    ub = np.full(nv, 1.0); ub[0] = 4.0; ub[1:1 + N] = 4.0
    res = milp(cost, constraints=LinearConstraint(np.array(A), lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), ub), options={"time_limit": tl, "mip_rel_gap": 1e-7})
    if res.x is None:
        return np.nan
    return float(res.x[0])


def boot_ci(x, rng, nb=2000):
    x = np.asarray(x)
    m = rng.choice(len(x), (nb, len(x))); bs = x[m].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def run_cell(args):
    N, s, c, h, n_inst, hidden = args
    seed = 1000 * N + int(round(h * 100))
    k, B = sample(seed, n_inst, N, h)
    out = {"N": N, "s": s, "c": c, "h": h}
    rng = np.random.default_rng(7)
    sw = lambda pol, p, **kw: simulate(k, B, s, c, pol, p, seed=seed, **kw)
    fixed, fixed_rem = sw("fixed", 0)
    tails = {t: sw("tail", t) for t in TAUS}
    pairs = {t: sw("pair", t) for t in TAUS}
    gates = {d: sw("gate", d) for d in DELTAS}
    bt = max(TAUS, key=lambda t: tails[t][0].mean())
    bp = max(TAUS, key=lambda t: pairs[t][0].mean())
    bg = max(DELTAS, key=lambda d: gates[d][0].mean())
    opt = np.array([opt_ratio(k[i], B[i], s, c) for i in range(n_inst)])
    out.update(
        fixed=float(fixed.mean()),
        tail=float(tails[bt][0].mean()), tail_tau=bt,
        pair=float(pairs[bp][0].mean()), pair_tau=bp,
        gate=float(gates[bg][0].mean()), gate_delta=bg,
        opt=float(np.nanmean(opt)), opt_nan=int(np.isnan(opt).sum()),
        tail_minrem=float(tails[bt][1].mean() / B0), opt_note="fluid MILP upper bound")
    g_tail = (opt - tails[bt][0]) / opt
    g_pair = (opt - pairs[bp][0]) / opt
    g_gate = (opt - gates[bg][0]) / opt
    for nm, g in (("gap_tail", g_tail), ("gap_pair", g_pair), ("gap_gate", g_gate)):
        out[nm] = float(np.nanmean(g)); out[nm + "_ci95"] = boot_ci(g[~np.isnan(g)], rng)
    out["gap_tail_frac_inst_gt2pct"] = float(np.nanmean(g_tail > 0.02))
    # hidden-k variant: (2) tail unchanged (never looks at k); (3) gate with k_hat
    hid = {}
    for sg in (0.0, SIGMA):
        gh = {d: sw("gate", d, hidden_sigma=sg) for d in DELTAS}
        bd = max(DELTAS, key=lambda d: gh[d][0].mean())
        hid[str(sg)] = dict(gate_hidden=float(gh[bd][0].mean()), delta=bd,
                            diff_vs_tail=float((gh[bd][0] - tails[bt][0]).mean() / tails[bt][0].mean()),
                            diff_vs_gate_known=float((gh[bd][0] - gates[bg][0]).mean() / gates[bg][0].mean()))
    out["hidden"] = hid
    return out


def anchor():
    rows = []
    for N in NS:
        for s in SS:
            k, B = sample(1, 1, N, 0.0)
            th = 1 / (1 - s * (N - 1) / N)
            o = opt_ratio(k[0], B[0], s, 0, lose=False)
            kk, BB = sample(1, 50, N, 0.0)
            sim1, _ = simulate(kk, BB, s, 0, "tail", 1, lose=False)
            sim_l, _ = simulate(kk, BB, s, 0, "tail", 20, lose=True)
            fx, _ = simulate(kk, BB, s, 0, "fixed", 0)
            rows.append(dict(N=N, s=s, theory=th, milp_nolose=o, sim_tau1_nolose=float(sim1.mean()),
                             sim_tau20_lose=float(sim_l.mean()), fixed=float(fx.mean()),
                             err_milp=abs(o / th - 1), err_sim=abs(sim1.mean() / th - 1)))
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
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = resolve_out(a.out, a.quick, 'calib/out/drone_bucket.json')
    n_inst = 20 if a.quick else 200
    t0 = time.time()
    anc = anchor(); print("anchor max err milp/sim:", max(r["err_milp"] for r in anc), max(r["err_sim"] for r in anc), flush=True)
    cells = [(N, s, c, h, n_inst, True) for N in NS for s in SS for c in CS for h in HS]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(4) as p:
        res = []
        for i, r in enumerate(p.imap_unordered(run_cell, cells)):
            res.append(r)
            if i % 12 == 0: print(i, len(cells), f"{time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda r: (r["N"], r["s"], r["c"], r["h"]))
    json.dump(dict(config=dict(B0=B0, n_inst=n_inst, taus=TAUS, deltas=DELTAS, sigma=SIGMA), anchor=anc, cells=res),
              open(a.out, "w"), indent=1)
    print("done", f"{time.time()-t0:.0f}s")
