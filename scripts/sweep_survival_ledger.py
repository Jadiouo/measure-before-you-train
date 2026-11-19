"""Grid run of the ledger-reciprocity survival game (src/animalbench/analysis/survival_ledger.py; notes/impl/survival_game.md).
CPU only.  usage: CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python scripts/sweep_survival_ledger.py [--n 3000]
output: calib/out/survival_ledger.json (written after every cell)"""
import argparse
import json
import os
import time

from animalbench.analysis import survival_game as sv
from animalbench.analysis import survival_ledger as sl

ALPHAS = (0.0, 0.25)
DELTAS = (0.9, 0.99)
KS = (1, 3, 10)
LAMS = (0, 15, 30, 60)                       # fine in decisions; tau = 15
GRID = [dict(kind="riskless", eps=1.06, m_g=0.5), dict(kind="riskless", eps=1.10, m_g=1.0),
        dict(kind="risky", eps=1.02, m_g=1.0), dict(kind="risky", eps=0.98, m_g=1.0)]
D, PG, TAU = 300, 0.05, 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--quick", type=int, default=0)
    a = ap.parse_args()
    pr = sv.load_params()
    path = os.path.join(sv.OUT, "survival_ledger.json")
    gusts = [sv.make_gusts(a.n, D, PG, 777 + k) for k in range(max(KS))]       # CRN across all cells / lam / deviations
    rec, t0 = [], time.time()
    jobs = [(K, g, lam, pd) for K in KS for g in GRID for lam in LAMS for pd in ((False, True) if K == 3 else (False,))]
    if a.quick:
        jobs = jobs[:2]
    for K, g, lam, pd in jobs:
        o = sl.eval_ledger_cell(K, g["eps"] * D * sv.DH, D, gusts, pr, g["m_g"], lam, ALPHAS, DELTAS, tau=TAU, perm_death=pd)
        rec.append(dict(K=K, lam=lam, perm_death=pd, **g, **o))
        meta = dict(n_episodes=a.n, D=D, tau=TAU, p_gust=PG, h=0.05, beta=0.5, alphas=ALPHAS, deltas=DELTAS, lams=LAMS,
                    eps_list=sv.EPS_LIST, params=pr, gust_seed=777, seconds=time.time() - t0)
        json.dump(dict(meta=meta, records=rec), open(path, "w"))
        print(f"{len(rec)}/{len(jobs)} K={K} lam={lam} pd={pd} {g['kind']} eps={g['eps']} mg={g['m_g']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
