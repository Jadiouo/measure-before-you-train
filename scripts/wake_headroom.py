"""Wake-surfing headroom test: best traditional controller vs privileged oracle in a 2D section model.
Gap (privileged - best traditional) in percentage points of total power = upper bound on RL benefit.
Run: systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/wake_headroom.py [--quick]
See notes/impl/wake_headroom.md for every self-chosen parameter and its reason.
"""
from __future__ import annotations
import json, os, sys, time, itertools, argparse
import numpy as np
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ---------------- constants ----------------
DT = 0.25; TSIM = 600.0; NSTEP = int(TSIM / DT)
TAU_A = 1.0; AMAX = 0.03; ZETA = 0.9; SIG_ACT = 0.01     # actuator lag [s], accel limit [b/s^2], act. noise per step
TAU_GPS = 2.0                                              # GPS error correlation time [s]
SMAX = 0.08; LAMBDA_ROLL = 0.01; S_FLOOR = -0.15; DXF = 3.0; RC = 0.075
N_TUNE = 24; N_EVAL = 64
ROLL_NOISE = {"low": 0.05, "high": 0.20}                   # in units of |dM| at 0.2 b lateral offset
DRIFT_CASES = [("amp0", 0.0, 5.0)] + [(f"amp{a}_tau{t}", a, float(t)) for a in (0.05, 0.1, 0.2) for t in (5, 30)] \
    + [("sine", 0.0, 0.0), ("step", 0.0, 0.0)]
BIAS = [0.0, 0.04, 0.08, 0.12]    # outward (y) setpoint bias tuning grid [b]
GPS_SIG = (0.02, 0.04, 0.08); POW_SIG = (0.005, 0.02)

# ---------------- table from horseshoe model ----------------
def build_table():
    import horseshoe_formation as hf
    g = np.round(np.arange(-0.6, 0.6001, 0.01), 4)
    Y, Z = np.meshgrid(g + 0.0, g, indexing="ij")
    ysc = np.arange(0.5, 1.3, 0.005)       # raw lateral scan to find raw optimum along z=0
    def upw(yc, dz):
        yc = np.asarray(yc, float); dz = np.asarray(dz, float)
        px = np.full(yc.shape + (hf.NQ,), DXF); py = yc[..., None] + hf.YQ
        P = np.stack([px, py, np.broadcast_to(dz[..., None], px.shape)], -1)
        a = np.array([hf.L, -hf.S/2, 0.]); b_ = np.array([0., -hf.S/2, 0.]); c = np.array([0., hf.S/2, 0.]); d = np.array([hf.L, hf.S/2, 0.])
        w = hf.seg_w(P, a, b_, RC, "lamb") + hf.seg_w(P, b_, c, RC, "lamb") + hf.seg_w(P, c, d, RC, "lamb")
        w = w * hf.sgn(RC, "lamb")
        return (w * hf.WQ).sum(-1), (w * hf.YQ * hf.WQ).sum(-1)     # mean upwash, roll-moment proxy
    raw0, _ = upw(ysc, np.zeros_like(ysc))
    y0 = ysc[np.argmax(raw0)]
    U, Mr = upw(y0 + Y, Z)
    sraw = U / hf.W_SELF                      # = 1 - D/Dsolo (fractional induced-drag saving, raw)
    # iterate: net = k*sraw - LAMBDA*Mn^2, Mn relative to M at net optimum
    i0 = np.unravel_index(np.argmax(sraw), sraw.shape); k = SMAX / sraw[i0]
    for _ in range(4):
        Mopt = Mr[i0]; ig = np.argmin(abs(g)); Mref = abs(Mr[ig - 10, ig] - Mopt)   # |dM| at 0.1 b inboard
        Mn = (Mr - Mopt) / Mref
        net = np.maximum(k * sraw - LAMBDA_ROLL * Mn ** 2, S_FLOOR)
        i0 = np.unravel_index(np.argmax(net), net.shape); k = (SMAX + LAMBDA_ROLL * Mn[i0] ** 2) / sraw[i0]
    # recentre grid so that the net optimum is at (0,0)
    cy, cz = g[i0[0]], g[i0[1]]
    Mopt = Mr[i0]; Mn = (Mr - Mopt) / Mref
    net = np.maximum(k * sraw - LAMBDA_ROLL * Mn ** 2, S_FLOOR)
    return dict(g=g, S=net, M=Mn, cy=cy, cz=cz, k=k, y0=y0, Mref=Mref, sraw_peak=float(sraw.max()), S_min=float(net.min()))

TAB = None
def tab():
    global TAB
    if TAB is None:
        t = build_table(); g = t["g"]
        # shift so optimum at 0: resample on a new grid by interpolation
        t["gs"] = g
        t["S"] = shifted(t["S"], g, t["cy"], t["cz"]); t["M"] = shifted(t["M"], g, t["cy"], t["cz"])
        dM = np.gradient(t["M"], g, g)
        t["dMy"], t["dMz"] = dM
        # curvature of S at the optimum (per axis)
        i = len(g) // 2
        t["cy_curv"] = -(t["S"][i + 5, i] - 2 * t["S"][i, i] + t["S"][i - 5, i]) / 0.05 ** 2
        t["cz_curv"] = -(t["S"][i, i + 5] - 2 * t["S"][i, i] + t["S"][i, i - 5]) / 0.05 ** 2
        TAB = t
    return TAB

def shifted(A, g, cy, cz):
    out = np.empty_like(A)
    for a, y in enumerate(g):
        out[a] = [interp_raw(A, g, y + cy, z + cz) for z in g]
    return out

def interp_raw(A, g, y, z):
    return float(interp(A, g, np.array([y]), np.array([z]))[0])

def interp(A, g, y, z):
    h = g[1] - g[0]; n = len(g)
    fy = np.clip((y - g[0]) / h, 0, n - 1.000001); fz = np.clip((z - g[0]) / h, 0, n - 1.000001)
    iy = fy.astype(int); iz = fz.astype(int); ty = fy - iy; tz = fz - iz
    return (A[iy, iz] * (1 - ty) * (1 - tz) + A[iy + 1, iz] * ty * (1 - tz) + A[iy, iz + 1] * (1 - ty) * tz + A[iy + 1, iz + 1] * ty * tz)

# ---------------- scenario noise (shared across controllers/params; indexed by seed) ----------------
def make_scenario(drift, gps_sig, seed0, nseed, act_on=True):
    name, A, tau = drift
    rng = np.random.default_rng(seed0)
    n = NSTEP + 1
    t = np.arange(n) * DT
    d = np.zeros((n, nseed, 2))
    if A > 0:
        phi = np.exp(-DT / tau); s = A * np.sqrt(1 - phi ** 2)
        d[0] = rng.normal(0, A, (nseed, 2)); eps = rng.normal(0, 1, (n, nseed, 2))
        for k in range(1, n): d[k] = phi * d[k - 1] + s * eps[k]
    elif name == "sine":
        ph = rng.uniform(0, 2 * np.pi, (nseed, 2)); amp = np.array([0.15, 0.10])
        d = amp * np.sin(2 * np.pi * t[:, None, None] / 120.0 + ph[None])
    elif name == "step":
        sg = rng.choice([-1.0, 1.0], (nseed, 2)); amp = np.array([0.20, 0.10])
        d = np.where(t[:, None, None] >= 300.0, 1.0, 0.0) * sg[None] * amp
    gphi = np.exp(-DT / TAU_GPS)
    gn = np.zeros((n, nseed, 2)); ge = rng.normal(0, 1, (n, nseed, 2)); gn[0] = gps_sig * ge[0]
    for k in range(1, n): gn[k] = gphi * gn[k - 1] + gps_sig * np.sqrt(1 - gphi ** 2) * ge[k]
    return dict(d=d, gps=gn, an=rng.normal(0, 1, (n, nseed, 2)) * (SIG_ACT if act_on else 0.0),
                pn=rng.normal(0, 1, (n, nseed)), mn=rng.normal(0, 1, (n, nseed)))

def loopTF(om, Om):
    s = 1j * Om
    return om ** 2 / (TAU_A * s ** 3 + s ** 2 + 2 * ZETA * om * s + om ** 2)

# ---------------- batched rollout ----------------
def rollout(ctrl, par, sc, nseed, pow_sig=0.0, roll_sig=0.0):
    """par: dict of arrays shape (P,). Batch = P*nseed (param-major). Returns per-batch mean net saving, neg-time fraction."""
    T = tab(); g = T["gs"]
    P = len(next(iter(par.values()))); B = P * nseed
    sidx = np.tile(np.arange(nseed), P); pidx = np.repeat(np.arange(P), nseed)
    pr = {k: v[pidx] for k, v in par.items()}
    om = pr["om"]
    p = np.zeros((B, 2)); v = np.zeros((B, 2)); a = np.zeros((B, 2))
    acc = np.zeros(B); neg = np.zeros(B)
    if ctrl == "kf":
        phi = np.exp(-DT / pr["taum"]); qv = pr["qs"] ** 2
        dh = np.zeros((B, 2)); P00 = qv.copy(); P11 = qv.copy(); P01 = np.zeros(B)
    if ctrl == "esc":
        th = np.zeros((B, 2)); lp = np.zeros(B); Om = np.stack([2 * np.pi * pr["f"], 2 * np.pi * pr["f"] * 1.63], 1)
        G = loopTF(om[:, None], Om); gain = np.abs(G); ph = np.angle(G)
        kk = 2 * pr["lam"][:, None] / (pr["a"][:, None] * gain * np.array([T["cy_curv"], T["cz_curv"]])[None])
        alp = DT * 2 * np.pi * pr["f"] / 4.0
    d = sc["d"]
    for k in range(NSTEP):
        dk = d[k][sidx]; e = p - dk
        S = interp(T["S"], g, e[:, 0], e[:, 1]); acc += S; neg += (S < 0)
        pm = p + sc["gps"][k][sidx]
        if ctrl == "fixed":
            r = pr['by'][:, None] * np.array([1.0, 0.0]); acmd = om[:, None] ** 2 * (r - pm) - 2 * ZETA * om[:, None] * v
        elif ctrl == "priv":      # true own position + true vortex position/velocity
            dv = (d[k] - d[max(k - 1, 0)])[sidx] / DT
            tgt = pr['w'][:, None] * (dk + pr['lead'][:, None] * dv) + pr['by'][:, None] * np.array([1.0, 0.0]); acmd = om[:, None] ** 2 * (tgt - p) - 2 * ZETA * om[:, None] * (v - pr['w'][:, None] * dv)
        elif ctrl == "priv_gps":  # true vortex position/velocity, own position from GPS
            dv = (d[k] - d[max(k - 1, 0)])[sidx] / DT
            tgt = pr['w'][:, None] * (dk + pr['lead'][:, None] * dv) + pr['by'][:, None] * np.array([1.0, 0.0]); acmd = om[:, None] ** 2 * (tgt - pm) - 2 * ZETA * om[:, None] * (v - pr['w'][:, None] * dv)
        elif ctrl == "kf":
            dh = phi[:, None] * dh; P00 = phi ** 2 * P00 + qv * (1 - phi ** 2); P11 = phi ** 2 * P11 + qv * (1 - phi ** 2); P01 = phi ** 2 * P01
            Mm = interp(T["M"], g, e[:, 0], e[:, 1]) + roll_sig * sc["mn"][k][sidx]
            ey = pm - dh
            h = interp(T["M"], g, ey[:, 0], ey[:, 1])
            Hy = -interp(T["dMy"], g, ey[:, 0], ey[:, 1]); Hz = -interp(T["dMz"], g, ey[:, 0], ey[:, 1])
            PH0 = P00 * Hy + P01 * Hz; PH1 = P01 * Hy + P11 * Hz
            Sv = Hy * PH0 + Hz * PH1 + max(roll_sig, 1e-4) ** 2
            K0 = PH0 / Sv; K1 = PH1 / Sv; inn = Mm - h
            dh[:, 0] += K0 * inn; dh[:, 1] += K1 * inn
            P00n = P00 - K0 * PH0; P01n = P01 - K0 * PH1; P11n = P11 - K1 * PH1
            P00, P01, P11 = P00n, P01n, P11n
            dh = np.clip(dh, -0.5, 0.5)
            acmd = om[:, None] ** 2 * (dh + pr['by'][:, None] * np.array([1.0, 0.0]) - pm) - 2 * ZETA * om[:, None] * v
        elif ctrl == "esc":
            Sm = S + pow_sig * sc["pn"][k][sidx]
            lp += alp * (Sm - lp)
            tt = k * DT
            dem = np.sin(Om * tt + ph)
            th += DT * kk * (Sm - lp)[:, None] * dem
            th = np.clip(th, -0.6, 0.6)
            r = th + pr["a"][:, None] * np.sin(Om * tt)
            acmd = om[:, None] ** 2 * (r - pm) - 2 * ZETA * om[:, None] * v
        acmd = np.clip(acmd, -AMAX, AMAX)
        a += DT / TAU_A * (acmd - a) + sc["an"][k][sidx]
        v += a * DT; p += v * DT
    return (acc / NSTEP).reshape(P, nseed), (neg / NSTEP).reshape(P, nseed)

# ---------------- parameter grids ----------------
def grid(ctrl):
    if ctrl == "fixed":
        c = list(itertools.product([0.05, 0.1, 0.2, 0.4, 0.8], BIAS))
        return dict(om=np.array([x[0] for x in c]), by=np.array([x[1] for x in c]))
    if ctrl in ("priv", "priv_gps"):
        oms = [0.3, 0.6, 1.0, 1.5] if ctrl == "priv" else [0.1, 0.2, 0.4, 0.8, 1.2]
        c = list(itertools.product(oms, BIAS, [0.0, 0.5, 1.0], [0.0, 3.0]))
        return dict(om=np.array([x[0] for x in c]), by=np.array([x[1] for x in c]), w=np.array([x[2] for x in c]), lead=np.array([x[3] for x in c]))
    if ctrl == "kf":
        c = list(itertools.product([0.4, 0.8], [5.0, 30.0], [0.003, 0.02, 0.05, 0.1, 0.2, 0.4], BIAS))
        return dict(om=np.array([x[0] for x in c]), taum=np.array([x[1] for x in c]), qs=np.array([x[2] for x in c]), by=np.array([x[3] for x in c]))
    if ctrl == "esc":
        c = list(itertools.product([0.4, 0.8], [0.02, 0.05, 0.1], [0.01, 0.025, 0.05], [0.01, 0.03, 0.1]))
        return dict(om=np.array([x[0] for x in c]), a=np.array([x[1] for x in c]), f=np.array([x[2] for x in c]), lam=np.array([x[3] for x in c]))

def job(args):
    ctrl, drift, gps, pw, rl, act_on, ntune, neval, tag = args
    t0 = time.time()
    di = [x[0] for x in DRIFT_CASES].index(drift[0]) if drift[0] in [x[0] for x in DRIFT_CASES] else 0
    base = 1000 * (di + 1) + int(gps * 1000)
    par = grid(ctrl)
    sc_t = make_scenario(drift, gps, base + 1, ntune, act_on)
    sv, _ = rollout(ctrl, par, sc_t, ntune, pw, rl)
    tune_mean = sv.mean(1); b = int(np.argmax(tune_mean))
    best = {k: float(v[b]) for k, v in par.items()}
    sc_e = make_scenario(drift, gps, base + 7777, neval, act_on)
    sv2, ng = rollout(ctrl, {k: v[b:b + 1] for k, v in par.items()}, sc_e, neval, pw, rl)
    return dict(ctrl=ctrl, drift=drift[0], gps=gps, pow=pw, roll=rl, tag=tag, best=best, tune_score=float(tune_mean[b]),
                eval_seed=sv2[0].tolist(), neg_frac=float(ng.mean()), secs=time.time() - t0)

def build_jobs(quick):
    nt, ne = (6, 12) if quick else (N_TUNE, N_EVAL)
    drifts = DRIFT_CASES if not quick else [DRIFT_CASES[0], DRIFT_CASES[5], DRIFT_CASES[7]]
    gpss = GPS_SIG if not quick else (0.04,)
    J = []
    for dr in drifts:
        J.append(("priv", dr, 0.0, 0.0, 0.0, True, nt, ne, ""))
        for gs in gpss:
            J.append(("fixed", dr, gs, 0.0, 0.0, True, nt, ne, ""))
            J.append(("priv_gps", dr, gs, 0.0, 0.0, True, nt, ne, ""))
            for rl in ROLL_NOISE: J.append(("kf", dr, gs, 0.0, ROLL_NOISE[rl], True, nt, ne, rl))
            for pw in POW_SIG: J.append(("esc", dr, gs, pw, 0.0, True, nt, ne, str(pw)))
    # anchor: no drift, no noise at all (also no actuator noise)
    d0 = DRIFT_CASES[0]
    for c in ("fixed", "priv", "priv_gps", "kf", "esc"):
        J.append((c, d0, 0.0, 0.0, 0.0, False, nt, ne, "anchor"))
    return J

def resolve_out(out, quick, default, root="."):
    """--out not given + --quick -> .scratch/quick/<name> (never clobber the paper result files)."""
    import os
    if out is not None: return out
    if not quick: return default
    p = os.path.join(root, ".scratch/quick", os.path.basename(default))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    print("[--quick] no --out given; writing to", p, "(not the paper file", default + ")", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out = resolve_out(a.out, a.quick, os.path.join(ROOT, "calib/out/wake_headroom.json"), root=ROOT)
    T = tab()
    print("table: k=%.4f Smax=%.4f Smin=%.4f cy=%.3f curv(y,z)=%.2f,%.2f sraw_peak=%.3f" % (T["k"], T["S"].max(), T["S"].min(), T["cy"], T["cy_curv"], T["cz_curv"], T["sraw_peak"]), flush=True)
    J = build_jobs(a.quick); print(len(J), "jobs", flush=True)
    ctx = mp.get_context("spawn"); res = []; t0 = time.time()
    with ctx.Pool(min(a.workers, 4)) as pool:
        for i, r in enumerate(pool.imap_unordered(job, J, chunksize=1)):
            res.append(r)
            if i % 10 == 0: print(i, len(J), "%.0fs" % (time.time() - t0), flush=True)
    meta = dict(DT=DT, T=TSIM, TAU_A=TAU_A, AMAX=AMAX, ZETA=ZETA, SIG_ACT=SIG_ACT, TAU_GPS=TAU_GPS, SMAX=SMAX, LAMBDA_ROLL=LAMBDA_ROLL,
                RC=RC, DXF=DXF, ROLL_NOISE=ROLL_NOISE, N_TUNE=N_TUNE, N_EVAL=N_EVAL, quick=a.quick,
                table=dict(k=T["k"], Smin=float(T["S"].min()), cy=float(T["cy"]), cz=float(T["cz"]), Mref=float(T["Mref"]), curv=[T["cy_curv"], T["cz_curv"]], sraw_peak=T["sraw_peak"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(meta=meta, runs=res), open(a.out, "w"))
    if not a.quick: analyze(a.out)
    print("done %.0fs" % (time.time() - t0))

def analyze(path, nboot=2000):
    D = json.load(open(path)); R = D["runs"]
    idx = {}
    for r in R: idx[(r["ctrl"], r["drift"], r["gps"], r["pow"], r["roll"], r["tag"] == "anchor")] = r
    rng = np.random.default_rng(0)
    def boot(x):
        x = np.asarray(x); m = rng.integers(0, len(x), (nboot, len(x))); b = x[m].mean(1)
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    cells = []
    for dr in [d[0] for d in DRIFT_CASES]:
        for gs in GPS_SIG:
            for pw in POW_SIG:
                for rl in ROLL_NOISE:
                    cand = {"fixed": idx[("fixed", dr, gs, 0.0, 0.0, False)],
                            "kf": idx[("kf", dr, gs, 0.0, ROLL_NOISE[rl], False)],
                            "esc": idx[("esc", dr, gs, pw, 0.0, False)]}
                    bestc = max(cand, key=lambda c: cand[c]["tune_score"])     # picked on tuning seeds, not eval seeds
                    pv = idx[("priv", dr, 0.0, 0.0, 0.0, False)]; pg = idx[("priv_gps", dr, gs, 0.0, 0.0, False)]
                    sv = {c: np.array(cand[c]["eval_seed"]) for c in cand}
                    gap = (np.array(pv["eval_seed"]) - sv[bestc]) * 100; gapg = (np.array(pg["eval_seed"]) - sv[bestc]) * 100
                    cells.append(dict(drift=dr, gps=gs, pow=pw, roll=rl, best=bestc,
                        sav={c: float(sv[c].mean() * 100) for c in sv}, priv=float(np.mean(pv["eval_seed"]) * 100), priv_gps=float(np.mean(pg["eval_seed"]) * 100),
                        gap=float(gap.mean()), gap_ci=boot(gap), gap_gpsprivileged=float(gapg.mean()), gap_gpsprivileged_ci=boot(gapg),
                        neg_frac={c: cand[c]["neg_frac"] for c in cand} | {"priv": pv["neg_frac"]},
                        priv_minus_each=float(min((np.array(pv["eval_seed"]) - sv[c]).mean() * 100 for c in sv))))
    anchors = {r["ctrl"]: float(np.mean(r["eval_seed"]) * 100) for r in R if r["tag"] == "anchor"}
    D["cells"] = cells; D["anchors"] = anchors
    json.dump(D, open(path, "w"))
    return D

if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze(os.path.join(ROOT, "calib/out/wake_headroom.json"))
    else:
        main()
