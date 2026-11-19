"""K=1 (single trip), alpha=0: gain of 'refuse' (bird 1) vs deviation time t_d and fine lam -- shows where the in-trip
punishment works and where the end of the trip makes it uncollectable.  output: calib/out/survival_ledger_td.json"""
import json
import math
import os

import numpy as np

from animalbench.analysis import survival_game as sv
from animalbench.analysis import survival_ledger as sl

D, n, TAU = 300, 3000, 15
pr = sv.load_params()
g = [sv.make_gusts(n, D, 0.05, 777)]
st = (np.zeros((n, 3), np.int64), np.ones((n, 3), bool))
out = []
for kind, eps, mg in (("riskless", 1.06, 0.5), ("risky", 1.02, 1.0), ("risky", 0.98, 1.0)):
    for lam in (0, 15, 30, 60):
        base, _ = sl.run_trips(1, eps * D * 2, D, g, pr, st, lam, TAU, mg, 0.05)
        for td in (30, 60, 120, 180, 240, 270, 285):
            for i in (1,):
                r, _ = sl.run_trips(1, eps * D * 2, D, g, pr, st, lam, TAU, mg, 0.05, dev=dict(mode="refuse", i=i, t_d=td), dev_trip=0)
                dif = (sv.utilities(r[0], 0.0, 0.5) - sv.utilities(base[0], 0.0, 0.5))[:, i]
                out.append(dict(kind=kind, eps=eps, m_g=mg, lam=lam, t_d=td, i=i, gain=float(dif.mean()),
                                se=float(dif.std(ddof=1) / math.sqrt(n))))
json.dump(out, open(os.path.join(sv.OUT, "survival_ledger_td.json"), "w"), indent=0)
for o in out:
    print(o["kind"], o["eps"], "lam", o["lam"], "t_d", o["t_d"], f"{o['gain']:+.4f} +-{o['se']:.4f}")
