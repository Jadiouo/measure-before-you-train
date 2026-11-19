"""Survival small game v2 (notes/impl/survival_game.md, section 'v2'): cheaper swap (path_T sweep, calib/out/swap_cost_pathT.json),
observable heterogeneity (h=0.05, optional +-5% per-bird energy-use multiplier redrawn every episode), energy-aware strategies.
CPU only.  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/sweep_survival_v2.py [--n 3000]
output: calib/out/survival_game_v2.json"""
import argparse
import json
import os
import time

import numpy as np

from animalbench.analysis import survival_game as sv

ALPHAS = (0.0, 0.5, 1.0)
BETAS = (0.5,)
D = 300
PG = 0.05
CELLS = {"low": (1.06, 0.5), "edge": (1.02, 1.0), "high": (0.98, 1.0)}      # (eps, m_g)
PATHS = (1.0, 1.5, 2.0, 3.0)
HET = {"h": 0.0, "h+mult5": 0.05}                                            # name -> multiplier spread (h=0.05 in both)
STRATS_ALL = [("fixed", 15), ("rot", 4), ("rot", 8), ("rot", 15), ("thr", 15), ("emax", 4), ("emax", 8), ("emax", 15)]


STRATS_GATE = [("gate", 4), ("gate", 8), ("gate", 15)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--gate_only", type=int, default=0, help="append only the gate strategies to the existing json")
    a = ap.parse_args()
    hit, size = sv.make_gusts(a.n, D, PG, 12345)                              # same gust stream as the v1 sweep
    path = os.path.join(sv.OUT, "survival_game_v2.json")
    rec, t0 = [], time.time()
    STRATS = STRATS_GATE if a.gate_only else STRATS_ALL
    if a.gate_only:
        rec = [r for r in json.load(open(path))["records"] if r["strategy"] != "gate"]
    paths = PATHS[:1] if a.quick else PATHS
    for pt in paths:
        pr = sv.load_params_pathT(pt)
        for hname, spread in HET.items():
            mult = sv.make_mult(a.n, spread, 777) if spread > 0 else None
            for cname, (eps, mg) in CELLS.items():
                E0 = eps * D * sv.DH
                for strat, tau in STRATS:
                    for rule in ("unilateral", "mutual"):
                        o = sv.eval_cell(strat, tau, E0, D, hit, size, pr, mg, 0.05, 1.0, ALPHAS, BETAS, rule=rule, mult=mult)
                        if rule == "unilateral":      # baseline-only quantities (same for both rules)
                            b = sv.simulate(strat, E0, D, hit, size, pr, tau=tau, m_g=mg, h=0.05, mult=mult)
                            ls = b["lead_cnt"].sum(0) / max(b["lead_cnt"].sum(), 1)
                            o["U_team"] = float(sv.utilities(b, 1.0, 0.5)[:, 0].mean())
                            o["lead_share"] = ls.tolist()
                            o["death_any"] = float((~b["arrived"]).any(1).mean())
                            o["c_swap"] = pr["c_old"] + pr["c_new"] + pr["c_by"]
                        rec.append(dict(path_T=pt, het=hname, cell=cname, eps=eps, m_g=mg, strategy=strat, tau=tau, rule=rule, **o))
                print(pt, hname, round(time.time() - t0), "s", flush=True)
        json.dump(dict(meta=dict(n=a.n, D=D, alphas=ALPHAS, betas=BETAS, cells=CELLS, h=0.05, mult_spread=HET, z=sv.Z_BONF,
                                 eps_list=sv.EPS_LIST, tau_unit="decisions of 2 s", seconds=time.time() - t0), records=rec), open(path, "w"))


if __name__ == "__main__":
    main()
