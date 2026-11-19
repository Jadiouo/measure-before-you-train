"""Grid run of the N=3 survival-migration small game (src/animalbench/analysis/survival_game.py;
notes/impl/survival_game.md).  CPU only (numpy).  usage:
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/sweep_survival.py [--n 4000] [--quick 1]
output: calib/out/survival_game.json   (written every 10 cells and at the end)"""
import argparse
import json
import os
import time

import numpy as np

from animalbench.analysis import survival_game as sv

ALPHAS = (0.0, 0.25, 0.5, 1.0)
BETAS = (0.5, 0.0, 1.0, 2.0)
STRATS = [("fixed", 15), ("rot", 4), ("rot", 15), ("rot", 30), ("thr", 15)]
D = 300                     # GeeseConfig.T = 600 s / delta_h = 2 s
PG = 0.05
EPS_GRID = [round(x, 2) for x in np.arange(0.92, 1.101, 0.02)]
MG_GRID = [0.0, 0.5, 1.0, 2.0]


def cells(quick):
    out = []
    for mg in MG_GRID:
        for eps in EPS_GRID:
            out.append(dict(h=0.05, m_g=mg, eps=eps, cmult=1.0, rule="unilateral"))
    if quick:
        return out[:3]
    for h in (0.0, 0.10):
        for mg in (0.5, 1.0):
            for eps in EPS_GRID:
                out.append(dict(h=h, m_g=mg, eps=eps, cmult=1.0, rule="unilateral"))
    for cm in (0.5, 1.375, 2.0):                    # 1.375 = c_swap_apex 1.1948 / 0.8688 (swap_cost.json vs small_game_params)
        for eps in EPS_GRID:
            out.append(dict(h=0.05, m_g=1.0, eps=eps, cmult=cm, rule="unilateral"))
    for eps in EPS_GRID:                            # mutual swap rule: only 'refuse' is executable
        for mg in (0.5, 1.0):
            out.append(dict(h=0.05, m_g=mg, eps=eps, cmult=1.0, rule="mutual"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--quick", type=int, default=0)
    a = ap.parse_args()
    pr = sv.load_params()
    path = os.path.join(sv.OUT, "survival_game.json")
    rec, t0 = [], time.time()
    cs = cells(a.quick)
    hit, size = sv.make_gusts(a.n, D, PG, 12345)    # common random numbers across ALL cells / strategies / deviations
    for ci, c in enumerate(cs):
        E0 = c["eps"] * D * sv.DH
        for strat, tau in STRATS:
            o = sv.eval_cell(strat, tau, E0, D, hit, size, pr, c["m_g"], c["h"], c["cmult"], ALPHAS, BETAS, rule=c["rule"])
            rec.append(dict(cell=c, strategy=strat, tau=tau, **o))
        if ci % 10 == 9 or ci == len(cs) - 1:
            meta = dict(n_episodes=a.n, D=D, dh=sv.DH, p_gust=PG, alphas=ALPHAS, betas=BETAS, z_bonferroni=sv.Z_BONF,
                        eps_list=sv.EPS_LIST, params=pr, gust_seed=12345,
                        sources=dict(p_lead_p_follow="calib/out/small_game_params.json [3].p",
                                     swap_cost_old_new_bystander="calib/out/small_game_params.json [3].edges[1].cost",
                                     c_swap_apex="calib/out/swap_cost.json c_swap_apex (sensitivity cmult=1.375)",
                                     dh_T_tlock="src/animalbench/scenarios/geese/config.py delta_h=2.0, T=600.0, t_lock=8.0 (-> tau_min 4)"),
                        note="eps = E0/(D*dh*P_solo); m_g = mean gust size in decisions of solo cost; h = bird-0 energy bonus; "
                             "label keys 'alpha|beta'; cmult scales swap costs; rule unilateral|mutual", seconds=time.time() - t0)
            json.dump(dict(meta=meta, records=rec), open(path, "w"))
            print(ci + 1, "/", len(cs), "cells", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
