"""Measure the abstract-game inputs from the real environment (no wind, N-bird V, unilateral):
  p[s]            steady power fraction P/P_solo of slot s (mean of the stay run)
  cost[k][b]      transient energy cost of bird b (indexed by its slot BEFORE the swap) when edge (front(k), k) swaps,
                  in P_solo*s, defined as the intercept of the linear fit  dE_b(t) = (p_new - p_old) * t + cost  over
                  t in [16 s, 60 s] after the request (so 'cost' is relative to an instantaneous switch at the request).
  Python: python -m animalbench.analysis.slot_params  ->  calib/out/small_game_params.json
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..scenarios.geese import GeeseConfig, table_policy
from .batch import run_batch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "calib", "out")


def front_of(k):
    return 0 if k <= 2 else k - 2


def measure(N, d_s=20, n_dec=70, fit_from=8):
    cfg = GeeseConfig(n_max=N, T=2.0 * n_dec, log_low=False, log_D=False, init_mode="v", use_events=False)
    keep = ("P_mean", "E_used", "slot")
    z = np.zeros((n_dec, N), np.int32)
    base = run_batch(cfg, table_policy(cfg, z), [0], N, n_dec=n_dec, keep=keep, chunk=1)
    p = base["P_mean"][0, 30:60].mean(0)                       # bird i sits in slot i
    out = dict(p=p.tolist(), edges={})
    for k in range(1, N):
        f = front_of(k)
        tab = z.copy()
        if f == 0 and k == 1:
            tab[d_s, 0] = 1
        elif f == 0:
            tab[d_s, k] = 2
        else:
            tab[d_s, f] = 1
        r = run_batch(cfg, table_policy(cfg, tab), [0], N, n_dec=n_dec, keep=keep, chunk=1)
        dE = (r["E_used"][0] - base["E_used"][0]) / cfg.P_solo          # (dec, N)
        t = (np.arange(n_dec) - d_s + 1) * cfg.delta_h
        sel = np.arange(n_dec) >= d_s + fit_from
        cost, slope = [], []
        for b in range(N):
            A = np.stack([t[sel], np.ones(sel.sum())], 1)
            (a, c), *_ = np.linalg.lstsq(A, dE[sel, b], rcond=None)
            slope.append(float(a)); cost.append(float(c))
        out["edges"][str(k)] = dict(front=f, cost=cost, slope=slope,
                                    swap_slots=[int(f), int(k)], total=float(np.sum(cost)))
    return out


def main():
    res = {str(N): measure(N) for N in (3, 4, 5, 9)}
    res["note"] = ("cost[b] indexed by pre-swap slot (= bird id in this run); positive = extra energy; "
                   "slope should equal p_new - p_old for swapping birds and ~0 for bystanders")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "small_game_params.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    r = main()
    for N in ("3", "4", "5"):
        print(N, "p", np.round(r[N]["p"], 4))
        for k, e in r[N]["edges"].items():
            print("  edge", e["swap_slots"], "cost", np.round(e["cost"], 3), "slope", np.round(e["slope"], 3), "tot", round(e["total"], 3))
