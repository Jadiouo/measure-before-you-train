"""Parameter sweep of the ledger-state small game (analysis/ledger_game.py; notes/impl/M2pre_commander_ledger.md).

For every cell: one-shot-deviation gaps of the ledger rotation strategy under swap_rule="mutual", kappa=0, for several
discount factors; plus its long-run per-bird cost vs the mutual fixed leader.  N=3 grid (exact), N=4 apex-edge-only cells.
usage: python scripts/sweep_ledger.py [--n4 1]     (CPU; run with a MemoryMax)   output: calib/out/ledger_game.json"""
import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

from animalbench.analysis import small_game as sg
from animalbench.analysis.ledger_game import LedgerGame, fixed_leader_cost, fixed_leader_kappa_cost, simulate_kappa

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GAMMAS = (0.95, 0.99, 0.995, 0.9995)
TOL = 1e-6


def cell(N, p, cost, lock, theta, lam, cmult, gammas=GAMMAS, apex_only=False, max_states=2_000_000):
    lg = LedgerGame(N, p, cost, lock_dec=lock, theta=theta, lam=lam, cmult=cmult, apex_only=apex_only)
    try:
        lg.enumerate_states(max_states=max_states)
    except MemoryError:
        return dict(N=N, lock=lock, theta=theta, lam=lam, cmult=cmult, skipped="too many states")
    lg.sigma_chain()
    out = dict(N=N, lock_dec=lock, t_lock_s=lock * 2, theta=theta, lam=lam, cmult=cmult, n_states=lg.nS, per_gamma={})
    for g in gammas:
        r = lg.analyse(g, tol=TOL)
        out["per_gamma"][str(g)] = {k: r[k] for k in ("eps_all", "eps_unsat", "eps_on_path", "spe", "spe_unsat", "spe_on_path",
                                                    "frac_states_violating")}
        out["per_gamma"][str(g)]["worst_path"] = r["worst_path"]
    op = lg.on_path()
    c = lg.avg_cost(op)
    fx = fixed_leader_cost(p, lg.dh)
    out["cost_per_bird"] = c.tolist()
    out["cost_mean"] = float(c.mean())
    out["fixed_mean"] = fx["mean"]
    out["rel_cost_vs_fixed_mean"] = float(c.mean() / fx["mean"] - 1)
    out["rel_cost_vs_fixed_leader_bird"] = float(c.mean() / fx["leader"] - 1)
    out["rel_cost_vs_fixed_follower_bird"] = float(c.mean() / fx["follower"] - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n4", type=int, default=1)
    ap.add_argument("--n3", type=int, default=1)
    ap.add_argument("--out", default="ledger_game.json")
    a = ap.parse_args()
    t0 = time.time()
    res = dict(note="eps = max one-shot-deviation gain (P_solo*s); *_unsat excludes subgames whose ledger is at the clip Lc; "
                    "spe_* uses tol 1e-6. cost in P_solo*s per bird per decision (dh=2 s).", n3=[], n4=[])
    p3, c3 = sg.load_params(3)
    for lock in ((2, 4, 8) if a.n3 else ()):
        for k in (1, 2, 3):
            for lam in (0, 1, 2, 3):
                for cm in (0.5, 1.0, 2.0):
                    r = cell(3, p3, c3, lock, k * lock, lam, cm)
                    res["n3"].append(r)
                    pg = r["per_gamma"]
                    print(f"N3 lock{lock} th{k * lock} lam{lam} cm{cm}: states {r['n_states']} "
                          + " ".join(f"g{g}:{pg[g]['eps_unsat']:.4f}/{pg[g]['eps_on_path']:.4f}" for g in pg)
                          + f"  relcost {100 * r['rel_cost_vs_fixed_mean']:.2f}%  [{time.time() - t0:.0f}s]", flush=True)
    # kappa: cost of the baseline profile under fatigue (not re-solved)
    lg = LedgerGame(3, p3, c3, lock_dec=4, theta=8, lam=2)
    if a.n3:
        res["kappa_cost_N3_lock4_th8"] = {str(k): dict(ledger=simulate_kappa(lg, k), fixed_mean=fixed_leader_kappa_cost(p3, k))
                                      for k in (0.0, 0.5, 1.0, 2.0, 4.0)}
    json.dump(res, open(os.path.join(ROOT, "calib", "out", a.out), "w"), indent=1, default=float)
    if a.n4:
        p4, c4 = sg.load_params(4)
        for lock in (2, 4):
            for th in (lock, 2 * lock, 3 * lock):
                for lam in (0, 2):
                    r = cell(4, p4, c4, lock, th, lam, 1.0, gammas=(0.99, 0.9995), apex_only=True, max_states=1_500_000)
                    res["n4"].append(r)
                    if "skipped" in r:
                        print("N4", lock, th, lam, r["skipped"], flush=True)
                        continue
                    pg = r["per_gamma"]
                    print(f"N4(apex-only) lock{lock} th{th} lam{lam}: states {r['n_states']} "
                          + " ".join(f"g{g}:{pg[g]['eps_unsat']:.4f}/{pg[g]['eps_on_path']:.4f}" for g in pg)
                          + f" relcost {100 * r['rel_cost_vs_fixed_mean']:.2f}%  [{time.time() - t0:.0f}s]", flush=True)
                    json.dump(res, open(os.path.join(ROOT, "calib", "out", a.out), "w"), indent=1, default=float)
    json.dump(res, open(os.path.join(ROOT, "calib", "out", a.out), "w"), indent=1, default=float)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
