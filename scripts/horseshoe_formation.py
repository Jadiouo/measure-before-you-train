"""Horseshoe-vortex superposition model for fixed-wing formation induced drag.

Model (all lengths in wingspans b=1, numpy only):
- Each wing = one horseshoe vortex: bound segment of length s=pi/4*b (elliptic-loading
  vortex spacing) at x_j, two trailing legs at y_j +- s/2 running to x_j+L (L=300b).
  Planar (z=0), no wake roll-up, no wake decay, equal circulation for all wings.
- Biot-Savart per finite segment, with core regularisation on the distance h to the line:
  Rankine   factor = min(h^2/rc^2, 1);  Lamb-Oseen factor = 1-exp(-1.2564 h^2/rc^2).
- Mean upwash on wing i from wing j: sampled at nq Gauss-Chebyshev points across span,
  weight sin^2(theta) (= elliptic loading weight; same nodes as geese/wake.py quad_nodes).
- Induced drag ratio (standard small-perturbation approx, INFERENCE as applied here):
      D_i/D_solo = 1 - w_up_i / w_self,   w_self = Gamma0/(2 b)  (uniform downwash, elliptic wing)
  because D = integral Gamma(y)(w_self - w_up(y)) dy / V with Gamma elliptic.
  Source: Prandtl lifting line; Maskew NASA CR-151974 / Bangash et al. AIAA-2006-ish use same idea
  (exact citation not re-verified here).
Run: systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 .venv/bin/python scripts/horseshoe_formation.py
"""
from __future__ import annotations
import json, itertools, os
import numpy as np

S = np.pi / 4          # vortex spacing / b
L = 300.0
NQ = 24
_th = (np.arange(NQ) + 0.5) * np.pi / NQ
YQ = -0.5 * np.cos(_th)
WQ = np.sin(_th) ** 2
WQ = WQ / WQ.sum()
W_SELF = 1.0 / 2.0     # Gamma0=1, b=1


def seg_w(P, A, B, rc, kind):
    """z-velocity at points P (...,3) from unit-circulation segment A->B."""
    r1 = P - A
    r2 = P - B
    r0 = B - A
    cr = np.cross(r1, r2)
    cr2 = (cr ** 2).sum(-1)
    l0 = np.linalg.norm(r0)
    h2 = cr2 / l0 ** 2
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    dot = (r0 * (r1 / n1[..., None] - r2 / n2[..., None])).sum(-1)
    if kind == "rankine":
        f = np.minimum(h2 / rc ** 2, 1.0)
    else:
        f = 1.0 - np.exp(-1.2564 * h2 / rc ** 2)
    cr2s = np.maximum(cr2, 1e-18)
    v = cr[..., 2] / cr2s * dot / (4 * np.pi) * f
    return v


def horseshoe_w(px, py, rc, kind):
    """z velocity at (px,py) (relative to wing origin, x downstream) for Gamma=1, sign-fixed so
    that mid-span downstream is downwash (negative)."""
    P = np.stack([px, py, np.zeros_like(px)], -1)
    a = np.array([L, -S / 2, 0.]); b_ = np.array([0., -S / 2, 0.])
    c = np.array([0., S / 2, 0.]); d = np.array([L, S / 2, 0.])
    w = seg_w(P, a, b_, rc, kind) + seg_w(P, b_, c, rc, kind) + seg_w(P, c, d, rc, kind)
    return w


_sign = None
def sgn(rc, kind):
    w = horseshoe_w(np.array([5.0]), np.array([0.0]), rc, kind)[0]
    return -1.0 if w > 0 else 1.0   # multiply to make downwash negative


def kernel(dx, dy, rc, kind):
    """Mean (elliptic-weighted) upwash on a wing located (dx downstream, dy lateral) of a source wing."""
    dx = np.asarray(dx, float); dy = np.asarray(dy, float)
    px = dx[..., None] + 0 * YQ
    py = dy[..., None] + YQ
    w = horseshoe_w(px, py, rc, kind) * sgn(rc, kind)
    return (w * WQ).sum(-1)    # upwash positive (z-up), downwash negative


def ratios(pos, rc, kind, mode="all"):
    """pos (N,2) = (x, y). mode: all | ahead | nearest. Returns D_i/D_solo."""
    pos = np.asarray(pos, float)
    N = len(pos)
    dx = pos[:, None, 0] - pos[None, :, 0]   # [i,j] = i behind j
    dy = pos[:, None, 1] - pos[None, :, 1]
    K = kernel(dx, dy, rc, kind)
    np.fill_diagonal(K, 0.0)
    ahead = dx > 1e-9
    if mode == "all":
        M = np.ones((N, N), bool); np.fill_diagonal(M, False)
    elif mode == "ahead":
        M = ahead
    else:
        M = np.zeros((N, N), bool)
        dist = np.hypot(dx, dy)
        for i in range(N):
            idx = np.where(ahead[i])[0]
            if len(idx):
                M[i, idx[np.argmin(dist[i, idx])]] = True
    up = (K * M).sum(1)
    return 1.0 - up / W_SELF


def layout(form, N, dx, dy):
    """leader at origin; followers positions (x downstream, y lateral)."""
    P = [(0.0, 0.0)]
    if form == "V":
        k = 1
        while len(P) < N:
            P.append((k * dx, k * dy))
            if len(P) < N: P.append((k * dx, -k * dy))
            k += 1
    elif form == "echelon":
        for k in range(1, N): P.append((k * dx, k * dy))
    elif form == "line":
        for k in range(1, N): P.append((k * dx, 0.0))
    return np.array(P)


def role_labels(form, N):
    if form == "V":
        lab = ["leader"]; k = 1
        while len(lab) < N:
            lab.append(f"row{k}_R"); 
            if len(lab) < N: lab.append(f"row{k}_L")
            k += 1
        return lab
    return ["leader"] + [f"follower{k}" for k in range(1, N)]


def formation_eval(form, N, o, dx, rc, kind, mode="all"):
    dy = 0.0 if form == "line" else (1.0 - o)
    pos = layout(form, N, dx, dy)
    return pos, ratios(pos, rc, kind, mode)


def main():
    out = {"meta": {"model": "planar horseshoe superposition, b=1, vortex spacing pi/4, nq=%d" % NQ,
                    "ratio": "D_i/D_solo = 1 - mean_upwash/w_self, w_self=Gamma0/(2b)", "L": L}}
    cores = [0.05, 0.075, 0.1]
    kinds = ["rankine", "lamb_oseen"]

    # ---- calibration ----
    cal = {}
    # (1) Maskew 3 wing V, x/b=3, y/b=0.89
    c1 = {}
    for kind in kinds:
        for rc in cores:
            pos = layout("V", 3, 3.0, 0.89)
            r = ratios(pos, rc, kind, "all")
            ra = ratios(pos, rc, kind, "ahead")
            c1[f"{kind}_rc{rc}"] = {"all": r.tolist(), "ahead_only": ra.tolist()}
    cal["maskew_3wing_V_x3_y0.89"] = {"target_follower": 0.21, "limit_far": 0.20, "results": c1}
    # (2) two-wing lateral scan at x=3, and x scan
    ys = np.round(np.arange(0.6, 1.4001, 0.01), 3)
    c2 = {}
    for kind, rc in [("rankine", 0.075), ("lamb_oseen", 0.075), ("rankine", 0.05), ("rankine", 0.1)]:
        row = []
        for y in ys:
            r = ratios(np.array([[0, 0], [3.0, y]]), rc, kind, "ahead")
            row.append(float(r[1]))
        row = np.array(row)
        ib = int(np.argmin(row))
        best = row[ib]; half = 1 - 0.5 * (1 - best)   # ratio at which saving is halved
        # outward offset (y larger than best) where ratio crosses 'half'
        out_off = None
        for k in range(ib, len(ys)):
            if row[k] >= half: out_off = float(ys[k] - ys[ib]); break
        in_off = None
        for k in range(ib, -1, -1):
            if row[k] >= half: in_off = float(ys[ib] - ys[k]); break
        c2[f"{kind}_rc{rc}"] = {"best_y_over_b": float(ys[ib]), "best_overlap": float(1 - ys[ib]),
                                "best_ratio": float(best), "half_saving_offset_outward": out_off,
                                "half_saving_offset_inward": in_off,
                                "ratio_at": {str(y): float(row[list(ys).index(y)]) for y in [0.89, 1.0, 1.05, 1.1]}}
    cal["two_wing_lateral_scan_x3"] = {"maskew_reference": {"0.89": 0.21, "1.0": 0.28, "1.05": 0.49, "~1.09": "~0.6"},
                                       "results": c2}
    # (2b) best over x too
    c2b = {}
    for kind, rc in [("rankine", 0.075)]:
        best = (9, None)
        for x in np.arange(0.5, 10.01, 0.25):
            for y in np.arange(0.6, 1.2, 0.01):
                v = ratios(np.array([[0, 0], [x, y]]), rc, kind, "ahead")[1]
                if v < best[0]: best = (float(v), (float(x), float(y)))
        c2b[f"{kind}_rc{rc}"] = {"min_ratio": best[0], "at_x_y": best[1]}
        c2b["ratio_vs_x_at_y0.89"] = {str(x): float(ratios(np.array([[0, 0], [x, 0.89]]), rc, kind, "ahead")[1])
                                      for x in [1, 2, 3, 5, 8, 15, 40]}
    cal["two_wing_global_opt"] = c2b
    # (3) Maskew 0.2+0.8/N vs this model at same settings far downstream (V, y=0.89)
    c3 = {}
    for N in [3, 5, 9]:
        pos = layout("V", N, 3.0, 0.89)
        r = ratios(pos, 0.075, "rankine", "all")
        c3[str(N)] = {"mean_model": float(r.mean()), "maskew_0.2+0.8/N": 0.2 + 0.8 / N}
    cal["mean_vs_maskew_eq8_V_x3_y0.89"] = c3
    out["calibration"] = cal

    # ---- formation sweep ----
    o_grid = np.round(np.arange(-0.3, 0.6001, 0.025), 4)
    dx_grid = np.arange(1.0, 8.001, 0.25)
    PRACT = {"o": 0.0, "dx": 3.0}
    res = {}
    for N in [3, 5, 9]:
        for form in ["V", "echelon", "line"]:
            og = [0.0] if form == "line" else o_grid
            for kind in kinds:
                for rc in cores:
                    best = (9, None)
                    for o in og:
                        for dx in dx_grid:
                            _, r = formation_eval(form, N, o, dx, rc, kind, "all")
                            m = r.mean()
                            if m < best[0]: best = (float(m), (float(o), float(dx)))
                    for label, (o, dx) in [("optimal", best[1]), ("practical", (PRACT["o"] if form != "line" else 0.0, PRACT["dx"]))]:
                        pos, r_all = formation_eval(form, N, o, dx, rc, kind, "all")
                        _, r_ah = formation_eval(form, N, o, dx, rc, kind, "ahead")
                        _, r_ne = formation_eval(form, N, o, dx, rc, kind, "nearest")
                        lab = role_labels(form, N)
                        fol = np.arange(1, N)
                        ib = fol[np.argmin(r_all[1:])]; iw = fol[np.argmax(r_all[1:])]
                        d_ne = r_ne - r_ah
                        d_ne_all = r_ne - r_all
                        sav_ah = 1 - r_ah; sav_ne = 1 - r_ne
                        rec = {
                            "N": N, "form": form, "setting": label, "kind": kind, "rc": rc,
                            "overlap_o": o, "dx_over_b": dx, "mean_ratio": float(r_all.mean()),
                            "mean_ratio_ahead_only": float(r_ah.mean()),
                            "mean_ratio_nearest_only": float(r_ne.mean()),
                            "best_follower": {"idx": int(ib), "label": lab[ib], "ratio": float(r_all[ib])},
                            "worst_follower": {"idx": int(iw), "label": lab[iw], "ratio": float(r_all[iw])},
                            "nearest_vs_ahead": {"mean_abs_err_ratio": float(np.abs(d_ne).mean()),
                                                 "max_abs_err_ratio": float(np.abs(d_ne).max()),
                                                 "fleet_mean_err_ratio": float(d_ne.mean()),
                                                 "mean_saving_rel_err_pct": float(100 * (sav_ne.mean() - sav_ah.mean()) / max(abs(sav_ah.mean()), 1e-9))},
                            "nearest_vs_all": {"fleet_mean_err_ratio": float(d_ne_all.mean())},
                        }
                        if kind == "rankine" and rc == 0.075 or (kind == "lamb_oseen" and rc == 0.075):
                            rec["per_position"] = [{"label": lab[i], "x": float(pos[i, 0]), "y": float(pos[i, 1]),
                                                    "ratio_all": float(r_all[i]), "ratio_ahead": float(r_ah[i]),
                                                    "ratio_nearest": float(r_ne[i])} for i in range(N)]
                        res[f"N{N}_{form}_{label}_{kind}_rc{rc}"] = rec
    out["sweep"] = res
    os.makedirs("calib/out", exist_ok=True)
    with open("calib/out/horseshoe_formation.json", "w") as f:
        json.dump(out, f, indent=1)
    print("done", len(res))


if __name__ == "__main__":
    main()
