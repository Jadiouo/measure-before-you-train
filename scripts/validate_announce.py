"""Validation of the announced-rotation executor (scenarios/geese/commander.py): mutual, kappa=0, scripted_commander.
Compares no announcement (stay policy = fixed leader, the mutual equilibrium MARL found) with announced fair rotation.
usage: python scripts/validate_announce.py [--seeds 3]    output: calib/out/announce_validation.json  (CPU, MemoryMax)"""
import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

import animalbench  # noqa: F401
from animalbench.core.perturbation import leave
from animalbench.scenarios.geese import GeeseConfig, stay_policy
from animalbench.scenarios.geese.commander import announce_stats, run_announce_episode, scripted_commander

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--Ts", default="600")
    a = ap.parse_args()
    out = {}
    t0 = time.time()
    for N in (3, 9):
        for T in [float(x) for x in a.Ts.split(",")]:
            cfg = GeeseConfig(n_max=N, swap_rule="mutual", T=T, log_low=False, log_D=False)
            conds = dict(fixed=None, announce=scripted_commander(30.0), announce_comply50=scripted_commander(30.0))
            rows = {k: [] for k in conds}
            for sd in range(a.seeds):
                for name, cm in conds.items():
                    cp = 0.5 if name.endswith("50") else 1.0
                    st, tr, info = run_announce_episode(cfg, stay_policy(cfg), cm, seed=sd, N=N, comply_p=cp)
                    s = announce_stats(tr, cfg, N)
                    s["ticks"] = info["ticks"]
                    rows[name].append(s)
                    print(N, T, sd, name, {k: s[k] for k in ("handover_rate_per_s", "max_share_dev", "cost_mean")},
                          f"[{time.time() - t0:.0f}s]", flush=True)
            agg = {}
            base = np.mean([r["cost_mean"] for r in rows["fixed"]])
            for name, rs in rows.items():
                agg[name] = dict(handover_rate_per_s=float(np.mean([r["handover_rate_per_s"] for r in rs])),
                                 max_share_dev=float(np.mean([r["max_share_dev"] for r in rs])),
                                 lead_share=np.mean([r["lead_share"] for r in rs], 0).round(3).tolist(),
                                 n_swaps=float(np.mean([r["n_swaps"] for r in rs])),
                                 cost_mean=float(np.mean([r["cost_mean"] for r in rs])),
                                 cost_max_bird=float(np.mean([max(r["cost_per_bird"]) for r in rs])),
                                 cost_vs_fixed_pct=float(100 * (np.mean([r["cost_mean"] for r in rs]) / base - 1)))
            out[f"N{N}_T{int(T)}"] = agg
    # event-triggered re-plan: a leave event -> second commander call (budget 3)
    cfg = GeeseConfig(n_max=5, swap_rule="mutual", T=600.0, log_low=False, log_D=False)
    st, tr, info = run_announce_episode(cfg, stay_policy(cfg), scripted_commander(30.0), seed=0, N=5,
                                        events=[leave(200.0, "random")])
    out["event_replan_N5"] = dict(n_calls=len(info["calls"]), calls=info["calls"])
    json.dump(out, open(os.path.join(ROOT, "calib", "out", "announce_validation.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:3500])


if __name__ == "__main__":
    main()
