"""M4br exp1a: two ledger reciprocators + one scripted deviator (bird 0).  kappa=1, N=3, mutual, random_slots.
Per-bird return (P_solo*s per 600 s episode), paired against the all-reciprocator baseline on the same seeds."""
import os, json, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import animalbench  # noqa
from animalbench.scenarios.geese import GeeseConfig
from animalbench.scenarios.geese.scripted_ledger import ledger_reciprocator, DEV_NAMES
from animalbench.marl.basin import scripted_data, episode_metrics, classify

KAPPA = float(os.environ.get("KAPPA", 1.0))
cfg = GeeseConfig(n_max=3, dtype="float32", log_low=False, log_D=False, use_events=False, swap_rule="mutual",
                  kappa=KAPPA, random_slots=True)
NE = 192
CONFIGS = [("self", 0), ("identity", 2), ("identity", 4), ("identity", 1000)]


def per_bird(pol):
    d = scripted_data(cfg, pol, n_envs=NE, seed=11)
    ret = d["rew"].sum(0) * cfg.delta_h                       # (B,A)
    m = episode_metrics(d["slot"], d["n_swaps"], d["rew"], cfg.delta_h, cfg.T)
    sw = d["n_swaps"].sum(0)
    return ret, m, sw


out = {}
for ver, lam in CONFIGS:
    key = f"{ver}_lam{lam}"
    base, mb, _ = per_bird(ledger_reciprocator(cfg, ver, lam))
    row = {"baseline": dict(ret_bird=float(base.mean()), cls=classify(mb), swap_rate=mb["swap_rate"],
                            min_share=mb["min_lead_share"])}
    for dk, dn in DEV_NAMES.items():
        r, m, sw = per_bird(ledger_reciprocator(cfg, ver, lam, dev_id=0, dev_kind=dk))
        diff = r[:, 0] - base[:, 0]
        oth = r[:, 1:].mean(1) - base[:, 1:].mean(1)
        grp = r.mean(1) - base.mean(1)
        se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
        row[dn] = dict(dev_ret=float(r[:, 0].mean()), dev_gain=float(diff.mean()), dev_gain_se=se(diff),
                       others_change=float(oth.mean()), group_change=float(grp.mean()), swap_rate=m["swap_rate"],
                       dev_lead_share=float(m["max_lead_share"] if False else 0.0), cls=classify(m))
        print(key, dn, {k: round(v, 2) if isinstance(v, float) else v for k, v in row[dn].items()}, flush=True)
    out[key] = row
    json.dump(out, open(f"runs/m4br/exp1/deviations_kappa{KAPPA:g}.json", "w"), indent=1)
