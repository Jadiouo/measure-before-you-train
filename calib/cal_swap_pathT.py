"""Swap cost vs path_T (N=3, apex edge, mutual).  Per-bird cost = intercept of dE_b(t) = slope*t + cost on t in [16 s, 60 s]
after the request (same definition as analysis/slot_params.py -> small_game_params.json), P_solo*s.
Also: max/min P/Ps during the swap, min pair separation (b), time until both swappers have e_form < 0.15 b.
Writes calib/out/swap_cost_pathT.json.   python -m calib.cal_swap_pathT"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from audit_swap import swap_run  # noqa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "swap_cost_pathT.json")


def one(path_T, T0=15):
    cfg, T0, r = swap_run(3, (0, 1), T0=T0, T=140.0, path_T=path_T)
    Ps, dh, dt = cfg.P_solo, cfg.delta_h, cfg.dt
    E1 = np.asarray(r["swap"]["E_used"]); E0 = np.asarray(r["base"]["E_used"])          # (dec, 3), end of decision
    dE = (E1 - E0) / Ps
    t = (np.arange(E1.shape[0]) - T0 + 1) * dh
    sel = t >= 16.0
    cost, slope = [], []
    for b in range(3):
        A = np.stack([t[sel], np.ones(sel.sum())], 1)
        (a, c), *_ = np.linalg.lstsq(A, dE[sel, b], rcond=None)
        cost.append(float(c)); slope.append(float(a))
    lw = r["swap"]["low"]
    fl = lambda a: np.asarray(a).reshape(-1, *np.asarray(a).shape[2:])
    tt = fl(lw["t"]); ts = T0 * dh
    P = fl(lw["P"]) / Ps; x = fl(lw["x"]); y = fl(lw["y"]); ef = fl(lw["e_form"]) / cfg.b
    win = (tt >= ts) & (tt < ts + 10)
    dist = np.sqrt((x[:, :, None] - x[:, None, :]) ** 2 + (y[:, :, None] - y[:, None, :]) ** 2) / cfg.b + np.eye(3) * 1e9
    msep = dist[win].min(axis=0)
    bad = (np.maximum(ef[:, 0], ef[:, 1]) >= 0.15) & (tt >= ts)
    return dict(path_T=path_T, c_swap_total=float(sum(cost)), c_old_new_bystander=cost, slope=slope,
                maxP_over_Ps=np.round(P[win].max(0), 3).tolist(), minP_over_Ps=np.round(P[win].min(0), 3).tolist(),
                min_sep_b=float(msep.min()), min_sep_pairs_b={"0-1": float(msep[0, 1]), "0-2": float(msep[0, 2]), "1-2": float(msep[1, 2])},
                t_eform_settle_s=float(tt[bad].max() - ts + dt) if bad.any() else 0.0,
                max_eform_b_during=float(ef[win].max()), sep_radius_b=cfg.sep_radius,
                P_curve_t=np.round(tt[win][::5] - ts, 2).tolist(), P_curve=np.round(P[win][::5], 4).tolist())


if __name__ == "__main__":
    res = {"pathT": [one(pt) for pt in (1.0, 1.5, 2.0, 3.0)],
           "note": "N=3 mutual apex swap, no wind; cost = intercept definition of slot_params.py; t_lock=8 s and delta_h=2 s unchanged"}
    json.dump(res, open(OUT, "w"), indent=1)
    for r in res["pathT"]:
        print({k: v for k, v in r.items() if not k.startswith("P_curve")})
