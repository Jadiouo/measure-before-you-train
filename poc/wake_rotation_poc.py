#!/usr/bin/env python3
"""Wake / rotation proof-of-concept for spec 01 (v0.1). CPU + NumPy only.

Q1 wake shape (spec 2.3/2.4), Q2 sign of r_LF (spec 3.3/3.4), Q3 E0 rotation pressure (spec 2.7).
Usage:  .venv/bin/python poc/wake_rotation_poc.py [--quick]
Outputs: poc/out/*.png, poc/out/results.json, poc/out/log.txt
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "w")
RES = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


# ---------------------------------------------------------------- Q1: wake model
G = 9.81
BASE = dict(m=1.3, b=1.3, v=14.0, rho=1.2, SbCD=0.006 * 0.4, Ppro=4.0,
            x_d=40.0, rc0=0.1, kc=0.02)


def powers(p):
    m, b, v, rho = p["m"], p["b"], p["v"], p["rho"]
    Pind = 2 * (m * G) ** 2 / (np.pi * rho * v * b ** 2)
    Ppar = 0.5 * rho * v ** 3 * p["SbCD"]
    return Pind, Ppar, p["Ppro"], Pind + Ppar + p["Ppro"]


def wbar(dx_b, dy_b, p, nq=32):
    """Mean upwash (m/s) on follower at (dx, dy) in units of b. Spec 2.3. gain not applied."""
    dx_b, dy_b = np.broadcast_arrays(np.asarray(dx_b, float), np.asarray(dy_b, float))
    m, b, v, rho = p["m"], p["b"], p["v"], p["rho"]
    dx, dy = dx_b * b, dy_b * b
    b0 = np.pi / 4 * b
    G0 = m * G / (rho * v * b0)
    Gam = G0 * np.exp(-dx / (p["x_d"] * b))
    rc = p["rc0"] * b + p["kc"] * dx
    th = (np.arange(nq) + 0.5) * np.pi / nq
    sk = -(b / 2) * np.cos(th)
    om = np.sin(th) ** 2; om /= om.sum()
    out = np.zeros_like(dx)
    for s_, w_ in zip(sk, om):
        y = dy + s_
        wv = 0.0
        for s in (+1, -1):
            yy = y - s * b0 / 2
            wv = wv + s * Gam / (2 * np.pi) * yy / (yy ** 2 + rc ** 2)
        out = out + w_ * wv
    out = np.where((dx_b > 0.5) & (dx_b < 12.0), out, 0.0)
    return out


def save_frac(dx_b, dy_b, p, gw=1.0):
    """Fractional power saving relative to solo (positive = saving). gw * m g W / P_solo."""
    W = np.maximum(wbar(dx_b, dy_b, p), -0.2 * p["v"])
    return gw * p["m"] * G * W / powers(p)[3]


def calibrate(p, target=0.125, dxref=2.0):
    dys = np.linspace(0, 3, 601)
    s = save_frac(dxref, dys, p, 1.0)
    gw = target / s.max()
    return gw, dys[s.argmax()]


PARAMS = (("m", "m"), ("b", "b"), ("P_pro", "Ppro"), ("x_d", "x_d"), ("r_c0", "rc0"), ("k_c", "kc"))


def q1(quick):
    say("\n=== Q1 wake shape ===")
    p = dict(BASE)
    Pind, Ppar, Ppro, Ptot = powers(p)
    say(f"P_ind={Pind:.2f} P_par={Ppar:.2f} P_pro={Ppro:.2f} P_solo={Ptot:.2f} W")
    gw, dystar = calibrate(p)
    say(f"calibrated wake_gain={gw:.3f}, optimal d_y at d_x=2b: {dystar:.3f} b")
    dxs = np.linspace(0.55, 12, 230); dys = np.linspace(-3, 3, 241)
    DX, DY = np.meshgrid(dxs, dys, indexing="ij")
    S = save_frac(DX, DY, p, gw) * 100
    # checks
    checks = {}
    checks["gain"] = float(gw); checks["dy_star_dx2"] = float(dystar)
    checks["S_behind_dx2_pct"] = float(save_frac(2.0, 0.0, p, gw) * 100)
    checks["S_behind_min_pct"] = float(S[:, 120].min()); checks["S_behind_max_pct"] = float(S[:, 120].max())
    checks["behind_all_negative_dx<=?"] = None
    neg = dxs[S[:, 120] < 0]
    checks["behind_negative_range_b"] = [float(neg.min()), float(neg.max())] if len(neg) else None
    checks["symmetry_max_abs_err_pct"] = float(np.abs(S - S[:, ::-1]).max())
    best = S.max(axis=1); bestdy = np.abs(dys[S.argmax(axis=1)])
    checks["best_S_monotone_decreasing_from_dx0.55"] = bool(np.all(np.diff(best) <= 1e-9))
    checks["best_S_at_dx"] = {str(d): float(save_frac(d, np.linspace(0, 3, 601), p, gw).max() * 100)
                              for d in (0.55, 1.0, 1.5, 2, 3, 4, 6, 9, 11.9)}
    checks["best_dy_at_dx"] = {str(d): float(np.linspace(0, 3, 601)[save_frac(d, np.linspace(0, 3, 601), p, gw).argmax()])
                               for d in (0.55, 1.0, 1.5, 2, 3, 4, 6, 9, 11.9)}
    # location of the core-edge: first dy where S crosses zero (behind -> side)
    z = save_frac(2.0, np.linspace(0, 3, 601), p, gw)
    checks["zero_crossing_dy_dx2"] = float(np.linspace(0, 3, 601)[np.argmax(z > 0)])
    say(json.dumps(checks, indent=1))

    # sensitivity
    sens = {}
    for name, key in PARAMS:
        for f in (0.7, 1.3):
            q = dict(BASE); q[key] = BASE[key] * f
            g2, dy2 = calibrate(q)
            sg = save_frac(DX, DY, q, g2) * 100
            bst = sg.max(axis=1)
            beh = sg[:, 120]
            sens[f"{name}x{f}"] = dict(
                gain=float(g2), dy_star=float(dy2), P_solo=float(powers(q)[3]),
                monotone=bool(np.all(np.diff(bst) <= 1e-9)),
                behind_dx2_pct=float(save_frac(2.0, 0.0, q, g2) * 100),
                symm_err=float(np.abs(sg - sg[:, ::-1]).max()),
                best_S_dx1_5=float(save_frac(1.5, np.linspace(0, 3, 601), q, g2).max() * 100),
                best_S_dx6=float(save_frac(6.0, np.linspace(0, 3, 601), q, g2).max() * 100),
                raw_S_at_gain1_dx2=float(save_frac(2.0, dy2, q, 1.0) * 100))
            say(f"  {name}x{f}: " + json.dumps(sens[f'{name}x{f}']))
    RES["q1"] = dict(checks=checks, sens=sens)

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    im = ax[0].pcolormesh(dys, dxs, np.clip(S, -40, 20), cmap="RdBu_r", vmin=-30, vmax=30, shading="auto")
    ax[0].contour(dys, dxs, S, levels=[0], colors="k", linewidths=0.8)
    ax[0].set_xlabel("d_y / b (lateral)"); ax[0].set_ylabel("d_x / b (behind)")
    ax[0].set_title("Saving S(d_x,d_y) %  (black = 0)"); ax[0].invert_yaxis(); plt.colorbar(im, ax=ax[0])
    for d in (1.0, 1.5, 2, 3, 4, 6, 9):
        ax[1].plot(dys, save_frac(d, dys, p, gw) * 100, label=f"d_x={d}b")
    ax[1].axhline(0, color="k", lw=.5); ax[1].set_xlabel("d_y / b"); ax[1].set_ylabel("S %"); ax[1].legend(fontsize=8)
    ax[1].set_title("Lateral slices"); ax[1].set_ylim(-45, 20)
    for d in (0.0, 0.5, 0.875, 1.5):
        ax[2].plot(dxs, save_frac(dxs, d, p, gw) * 100, label=f"d_y={d}b")
    ax[2].plot(dxs, best, "k--", label="best over d_y")
    ax[2].set_xlabel("d_x / b"); ax[2].set_ylabel("S %"); ax[2].legend(fontsize=8); ax[2].set_title("Longitudinal decay"); ax[2].set_ylim(-45, 20)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "q1_wake_shape.png"), dpi=110); plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
    for ii, (name, key) in enumerate(PARAMS):
        for f, ls in ((0.7, ":"), (1.0, "-"), (1.3, "--")):
            q = dict(BASE); q[key] = BASE[key] * f
            g2, _ = calibrate(q)
            ax[ii].plot(dys, save_frac(2.0, dys, q, g2) * 100, ls, label=f"{name}x{f}")
        ax[ii].set_title(f"{name} +-30%: slice d_x=2b (gain recalibrated)", fontsize=9); ax[ii].legend(); ax[ii].set_xlabel("d_y/b"); ax[ii].set_ylim(-45, 20)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "q1_sensitivity.png"), dpi=110); plt.close()
    return p, gw, dystar


# ---------------------------------------------------------------- V slots
def v_slots(n, dxs, dys, p, gw):
    """Per-slot saving fraction and lead flag for compact V of n birds. Slot 0 = apex, k odd=left arm, even=right."""
    pos = [(0.0, 0.0)]
    for k in range(1, n):
        a = (k + 1) // 2; side = 1 if k % 2 else -1
        pos.append((-a * dxs, side * a * dys))   # x forward-positive
    sav = np.zeros(n); lead = np.zeros(n, bool)
    for i in range(n):
        tot = 0.0; ahead = False
        for j in range(n):
            if j == i: continue
            ddx = pos[j][0] - pos[i][0]; ddy = pos[i][1] - pos[j][1]   # j ahead of i if ddx>0
            if 0 < ddx < 5.0 and abs(ddy) < 2.0: ahead = True
            if ddx > 0: tot += float(wbar(ddx, ddy, p))
        W = max(tot, -0.2 * p["v"])
        sav[i] = gw * p["m"] * G * W / powers(p)[3]
        lead[i] = not ahead
    return sav, lead


# ---------------------------------------------------------------- discrete rotation engine
def simulate(N, T, strat, rng, sav, lead_flag, theta=0.02, T_rot=30, p_swap=0.2,
             away_u=None, mean_out=15.0, energy=None, min_bout=5):
    """1 s steps. Energy in units of P_solo*s. energy=(coef, sigma, rho_thr) or None (infinite reserve)."""
    order = list(rng.permutation(N))
    alive = np.ones(N, bool)
    away = np.zeros(N, bool)
    if away_u is not None:
        away = rng.random(N) < away_u
    q_ret = 1.0 / mean_out
    with np.errstate(divide="ignore", invalid="ignore"):
        q_out = (np.zeros(N) if away_u is None else
                 np.where(away_u > 0, away_u / (np.maximum(1 - away_u, 1e-9) * mean_out), 0.0))
    if energy is not None and np.isfinite(energy[0]):
        E0 = energy[0] * T * (1 + energy[1] * rng.standard_normal(N)); rho_thr = energy[2]
        use_e = True
    else:
        E0 = np.full(N, np.inf); use_e = False; rho_thr = None
    E = E0.copy()
    lead_t = np.zeros(N); fol_t = np.zeros(N); alive_t = np.zeros(N); used = np.zeros(N)
    nrot = 0; last_leader = None; bout_start = 0; death_t = np.full(N, np.nan)
    for t in range(T):
        # away dynamics
        if away_u is not None:
            for i in range(N):
                if not alive[i]: continue
                if away[i]:
                    if rng.random() < q_ret:
                        away[i] = False; order.remove(i); order.append(i)   # rejoin at tail
                else:
                    if rng.random() < q_out[i]: away[i] = True
        present = [b for b in order if alive[b] and not away[b]]
        # rotation
        if strat == "timed" and t > 0 and t % T_rot == 0 and present:
            order.remove(present[0]); order.append(present[0])
        elif strat == "random" and len(present) > 1 and rng.random() < p_swap:
            a, b = rng.choice(len(present), 2, replace=False)
            ia, ib = order.index(present[a]), order.index(present[b])
            order[ia], order[ib] = order[ib], order[ia]
        elif strat == "energy" and use_e and len(present) > 1 and t - bout_start >= min_bout:
            tf = max(1 - t / T, 1e-6)
            rho = {b: (E[b] / E0[b]) / tf for b in present}
            ld = present[0]
            if rho[ld] < rho_thr:
                best = max(present[1:], key=lambda b: rho[b])
                order.remove(best); order.insert(order.index(ld), best)  # best takes apex
                order.remove(ld); order.append(ld)                       # old leader to tail
        present = [b for b in order if alive[b] and not away[b]]
        if present:
            if present[0] != last_leader:
                if last_leader is not None and alive[last_leader]: nrot += 1
                last_leader = present[0]; bout_start = t
        P = np.ones(N)    # solo power by default (away / alone)
        for k, b in enumerate(present):
            P[b] = 1.0 - sav[k]
            if lead_flag[k]: lead_t[b] += 1
            elif sav[k] > theta: fol_t[b] += 1
        for b in range(N):
            if alive[b]:
                alive_t[b] += 1; used[b] += P[b]
                if use_e:
                    E[b] -= P[b]
                    if E[b] < -1e-9:
                        alive[b] = False; death_t[b] = t; order.remove(b)
    return dict(lam=lead_t / np.maximum(alive_t, 1), f=fol_t / np.maximum(alive_t, 1),
                alive_t=alive_t, used=used, nrot=nrot, dead=int((~alive).sum()))


def pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def gini(x):
    x = np.asarray(x, float); n = len(x)
    if n == 0 or x.mean() == 0: return np.nan
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * n * n * x.mean()))


def summ(v):
    v = np.array([x for x in v if not np.isnan(x)])
    if len(v) == 0: return dict(n=0)
    return dict(n=int(len(v)), med=float(np.median(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))


def run_cfg(N, nseed, sav_all, lead_all, base_seed, **kw):
    rs, lam_m, f_m, nrot = [], [], [], []
    for s in range(nseed):
        rng = np.random.default_rng(base_seed + s)
        kk = dict(kw)
        if "u_fn" in kk:
            kk["away_u"] = kk.pop("u_fn")(rng, N)
        o = simulate(N, kk.pop("T", 600), kk.pop("strat"), rng, sav_all[:N], lead_all[:N], **kk)
        rs.append(pearson(o["lam"], o["f"])); lam_m.append(o["lam"].mean()); f_m.append(o["f"].mean()); nrot.append(o["nrot"])
    d = summ(rs); d["frac_pos"] = float(np.mean([r > 0 for r in rs if not np.isnan(r)])) if d["n"] else None
    d["mean_f"] = float(np.mean(f_m)); d["mean_lam"] = float(np.mean(lam_m)); d["n_nan"] = int(sum(np.isnan(rs)))
    d["rot_per_min"] = float(np.mean(nrot) / (kw.get("T", 600) / 60)); d["_rs"] = rs
    return d


def beta_u(ubar, kappa):
    def fn(rng, N):
        if kappa is None: return np.full(N, ubar)
        return rng.beta(ubar * kappa, (1 - ubar) * kappa, N)
    return fn


def q2(p, gw, dystar, quick):
    say("\n=== Q2 r_LF sign ===")
    nseed = 30 if quick else 200
    sav, lead = v_slots(14, 2.0, dystar, p, gw)
    say("slot saving (%) for 14-bird V:", np.round(sav * 100, 1).tolist())
    say("slot lead flags:", lead.astype(int).tolist(), "; slots with saving<=2%:", [int(k) for k in np.where(sav <= 0.02)[0] if k > 0])
    RES["q2_slots"] = dict(sav_pct=(sav * 100).tolist(), lead=lead.astype(int).tolist())
    res = {}
    strategies = [("a_fixed", dict(strat="fixed")), ("b_random", dict(strat="random")),
                  ("c_timed30", dict(strat="timed", T_rot=30)), ("c_timed10", dict(strat="timed", T_rot=10)),
                  ("c_timed60", dict(strat="timed", T_rot=60)),
                  ("d_energy_E0.93_s0.05", dict(strat="energy", energy=(0.93, 0.05, 0.99))),
                  ("d_energy_E1.0_s0.05", dict(strat="energy", energy=(1.0, 0.05, 0.99)))]
    for N in (9, 14):
        for name, kw in strategies:
            r = run_cfg(N, nseed, sav, lead, 1000, **kw)
            res[f"N{N}_{name}"] = r
            say(f"N={N:2d} {name:24s} r_LF med={r.get('med', float('nan')):+.3f} [5%={r.get('p5', float('nan')):+.3f}, 95%={r.get('p95', float('nan')):+.3f}] "
                f"valid={r['n']} frac_pos={r['frac_pos']} mean_f={r['mean_f']:.2f} mean_lam={r['mean_lam']:.3f}")
    # straggler variant: timed rotation + latent per-bird away fraction
    say("-- straggler variant (timed 30 s rotation; away fraction u~Beta(mean,kappa); kappa=inf means identical u)")
    grid = {}
    ubars = (0.1, 0.3, 0.5, 0.7); kappas = (None, 20, 5, 2)
    for N in (9, 14):
        for ub in ubars:
            for kp in kappas:
                r = run_cfg(N, nseed, sav, lead, 5000, strat="timed", T_rot=30, u_fn=beta_u(ub, kp))
                grid[f"N{N}_u{ub}_k{kp}"] = r
                say(f"N={N:2d} ubar={ub} kappa={kp}: r med={r.get('med', float('nan')):+.3f} [{r.get('p5', float('nan')):+.3f},{r.get('p95', float('nan')):+.3f}] "
                    f"frac_pos={r['frac_pos']} mean_f={r['mean_f']:.2f}")
    # random-swap strategy too, heterogeneous case
    for N in (9, 14):
        r = run_cfg(N, nseed, sav, lead, 7000, strat="random", u_fn=beta_u(0.5, 2))
        res[f"N{N}_b_random+straggle_u0.5_k2"] = r
        say(f"N={N:2d} random+straggle(u=.5,k=2): r med={r['med']:+.3f} [{r['p5']:+.3f},{r['p95']:+.3f}] frac_pos={r['frac_pos']} mean_f={r['mean_f']:.2f}")
    for d in list(res.values()) + list(grid.values()):
        d.pop("_rs", None)
    RES["q2"] = res; RES["q2_straggle"] = grid

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    names = [n for n, _ in strategies] + ["b_random+straggle_u0.5_k2"]
    # re-run compactly for distribution plot (N=9 only, same seeds)
    data = []
    for name, kw in strategies:
        data.append(run_cfg(9, nseed, sav, lead, 1000, **kw)["_rs"])
    data.append(run_cfg(9, nseed, sav, lead, 7000, strat="random", u_fn=beta_u(0.5, 2))["_rs"])
    data.append(run_cfg(9, nseed, sav, lead, 5000, strat="timed", T_rot=30, u_fn=beta_u(0.5, 2))["_rs"])
    names.append("c_timed30+straggle_u0.5_k2")
    ax[0].boxplot([[x for x in d if not np.isnan(x)] for d in data], whis=(5, 95), showfliers=False)
    ax[0].set_xticks(range(1, len(names) + 1)); ax[0].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax[0].axhline(0.78, color="g", ls="--", label="Voelkl 0.78"); ax[0].axhline(0, color="k", lw=.5)
    ax[0].set_ylabel("r_LF (N=9, whiskers 5-95%)"); ax[0].legend(); ax[0].set_title("Q2: r_LF by strategy")
    M = np.array([[grid[f"N9_u{ub}_k{kp}"].get("med", np.nan) for kp in kappas] for ub in ubars])
    im = ax[1].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax[1].set_xticks(range(len(kappas))); ax[1].set_xticklabels(["inf(homog.)", "20", "5", "2(very het.)"])
    ax[1].set_yticks(range(len(ubars))); ax[1].set_yticklabels(ubars)
    for i in range(len(ubars)):
        for j in range(len(kappas)):
            ax[1].text(j, i, f"{M[i, j]:+.2f}\nf={grid[f'N9_u{ubars[i]}_k{kappas[j]}']['mean_f']:.2f}", ha="center", va="center", fontsize=8)
    ax[1].set_xlabel("heterogeneity of away fraction (Beta kappa)"); ax[1].set_ylabel("mean away fraction")
    ax[1].set_title("median r_LF, straggler variant, timed 30 s, N=9"); plt.colorbar(im, ax=ax[1])
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "q2_rLF.png"), dpi=110); plt.close()


def q3(p, gw, dystar, quick):
    say("\n=== Q3 E0 rotation pressure (energy strategy, N=9, T=600 s) ===")
    nseed = 15 if quick else 60
    sav, lead = v_slots(9, 2.0, dystar, p, gw)
    mean_pwr = np.mean(1 - sav)
    say(f"mean per-bird power in V (units of P_solo): {mean_pwr:.3f}; leader=1.0; mean follower={np.mean(1 - sav[1:]):.3f}")
    coefs = [0.85, 0.90, 0.93, 0.97, 1.0, np.inf]; sigs = [0.0, 0.05, 0.15]
    out = {}
    def cell(coef, sg, thr, N=9):
        nr, gi, dead = [], [], []
        for s in range(nseed):
            rng = np.random.default_rng(9000 + s)
            o = simulate(N, 600, "energy", rng, sav, lead, energy=(coef, sg, thr))
            nr.append(o["nrot"] / N / (600 / 60)); dead.append(o["dead"])
            gi.append(gini(o["used"] / np.maximum(o["alive_t"], 1)))
        return float(np.mean(nr)), float(np.nanmean(gi)), float(np.mean(dead)), float(np.mean(np.array(dead) > 0))
    for thr in (0.99, 0.95, 0.995):
        R = np.zeros((len(sigs), len(coefs), 4))
        for i, sg in enumerate(sigs):
            for j, c in enumerate(coefs):
                R[i, j] = cell(c, sg, thr)
        out[f"thr{thr}"] = dict(coefs=[str(c) for c in coefs], sigmas=sigs,
                                rot_per_bird_min=R[:, :, 0].tolist(), gini=R[:, :, 1].tolist(),
                                n_dead=R[:, :, 2].tolist(), frac_runs_with_death=R[:, :, 3].tolist())
        say(f"-- rho_thr={thr}")
        for i, sg in enumerate(sigs):
            say(f"   sigma={sg}: " + " | ".join(f"E0={c}: rot={R[i, j, 0]:.3f} G={R[i, j, 1]:.3f} dead={R[i, j, 2]:.2f}" for j, c in enumerate(coefs)))
    # no-rotation reference (fixed leader) death count
    refd = {}
    for c in (0.85, 0.90, 0.93, 0.97, 1.0):
        d = []
        for s in range(nseed):
            rng = np.random.default_rng(9000 + s)
            o = simulate(9, 600, "fixed", rng, sav, lead, energy=(c, 0.05, 0.99)); d.append(o["dead"])
        refd[str(c)] = float(np.mean(d))
    say("fixed-leader (never rotate), sigma=0.05, mean dead:", refd)
    out["fixed_leader_dead"] = refd; out["mean_power"] = float(mean_pwr)
    RES["q3"] = out
    o = out["thr0.99"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    labs = ["0.85", "0.90", "0.93", "0.97", "1.0", "inf"]
    for a, key, ttl in zip(ax, ("rot_per_bird_min", "gini", "n_dead"), ("rotations / bird / min", "energy Gini", "# birds exhausted (of 9)")):
        M = np.array(o[key]); im = a.imshow(M, cmap="viridis", aspect="auto")
        a.set_xticks(range(6)); a.set_xticklabels(labs); a.set_yticks(range(3)); a.set_yticklabels(sigs)
        for i in range(3):
            for j in range(6): a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        a.set_xlabel("E0_coef"); a.set_ylabel("E0_sigma"); a.set_title(ttl + " (rho_thr=0.99)"); plt.colorbar(im, ax=a)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "q3_phase.png"), dpi=110); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    t0 = time.time()
    p, gw, dystar = q1(a.quick)
    q2(p, gw, dystar, a.quick)
    q3(p, gw, dystar, a.quick)
    json.dump(RES, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    say(f"\ndone in {time.time() - t0:.0f}s; outputs in {OUT}")
