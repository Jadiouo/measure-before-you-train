"""Estimate the kappa at which scripted rotation (S2 anonymous, mutual, several thresholds) beats the fixed leader
in group return (N=3, random_slots).  Same seeds => paired difference.  Writes runs/m3basin/kappa_threshold.json."""
import json, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import animalbench  # noqa
from animalbench.scenarios.geese import GeeseConfig
from animalbench.scenarios.geese import scripted as S
from animalbench.marl.basin import scripted_data, episode_metrics

KAPPAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
TH = [(0.93, 0.89), (0.94, 0.90), (0.95, 0.91), (0.96, 0.92)]
out = {}
for k in KAPPAS:
    cfg = GeeseConfig(n_max=3, dtype="float32", log_low=False, log_D=False, use_events=False, swap_rule="mutual",
                      kappa=k, random_slots=True)
    def ep_ret(pol):
        d = scripted_data(cfg, pol, 64, 0)
        return (d["rew"].sum(0) * cfg.delta_h).mean(-1), episode_metrics(d["slot"], d["n_swaps"], d["rew"])
    rf, mf = ep_ret(S.null_fix(cfg))
    row = dict(fixed=float(rf.mean()), rot={})
    for hi, lo in TH:
        rr, mr = ep_ret(S.state_rotation(cfg, hi, lo))
        diff = rr - rf
        row["rot"][f"{hi}/{lo}"] = dict(ret=float(rr.mean()), adv=float(diff.mean()), adv_se=float(diff.std() / 8),
                                       swap_rate=mr["swap_rate"], tenure=mr["median_tenure_s"],
                                       worst=mr["ret_worst_bird"])
    row["fixed_worst"] = mf["ret_worst_bird"]
    row["best_adv"] = max(v["adv"] for v in row["rot"].values())
    out[str(k)] = row
    print(k, "fixed", round(row["fixed"], 2), "best rot adv", round(row["best_adv"], 2), flush=True)
json.dump(out, open("runs/m3basin/kappa_threshold.json", "w"), indent=1)
