"""Minimal episode summaries used by scripts/tests (full metric suite belongs to M1a stage 2)."""
from __future__ import annotations

import numpy as np


def episode_stats(traj, cfg) -> dict:
    a = traj.agent_fields
    g = traj.global_fields
    Ps = cfg.P_solo
    dh = cfg.delta_h
    alive = a["alive"].astype(bool)
    Pm = a["P_mean"]                                  # per decision, units of P_solo
    E_group = float((Pm * alive).sum() * dh)          # P_solo*s
    E_solo = float(alive.sum() * dh)                  # every bird flying solo at P_solo
    low = traj.low
    S = 1.0 - E_group / E_solo
    out = dict(S=S, E_group=E_group, E_solo=E_solo)
    if "P_nowake" in low:                              # airspeed-adjusted counterfactual (no wake, same speed)
        Pn = low["P_nowake"].sum(); P = low["P"].sum()
        out["S_nowake_same_speed"] = float(1.0 - P / Pn)
    lead = np.asarray(g["leader"])
    out["handovers"] = int((lead[1:] != lead[:-1]).sum())
    out["handover_rate_per_s"] = out["handovers"] / (len(lead) * dh)
    # tenures of completed apex spells (seconds)
    ch = np.flatnonzero(lead[1:] != lead[:-1]) + 1
    bounds = np.concatenate([[0], ch, [len(lead)]])
    ten = np.diff(bounds) * dh
    out["apex_tenures_s"] = ten.tolist()
    share = np.array([(lead == i).mean() for i in range(cfg.n_max)])
    out["lead_support"] = int((share >= 0.05).sum())
    out["n_swaps_total"] = int(np.asarray(a["swapped"]).sum() // 2)
    return out
