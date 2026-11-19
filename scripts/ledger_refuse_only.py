"""Same evaluation as sweep_survival_ledger.py but with ONLY the 'refuse' deviation (the one the ledger fine targets), lam=30,
to separate 'is defaulting deterred' from the other deviation modes.  output: calib/out/survival_ledger_refuse_only.json"""
import json
import os

from animalbench.analysis import survival_game as sv
from animalbench.analysis import survival_ledger as sl

D, n = 300, 3000
pr = sv.load_params()
gusts = [sv.make_gusts(n, D, 0.05, 777 + k) for k in range(10)]
sl.MODES = ("refuse",)
rec = []
for K in (3, 10):
    for kind, eps, mg in (("riskless", 1.06, 0.5), ("risky", 1.02, 1.0), ("risky", 0.98, 1.0)):
        for lam in (0, 30):
            o = sl.eval_ledger_cell(K, eps * D * 2, D, gusts, pr, mg, lam, (0.0, 0.25), (0.9, 0.99))
            rec.append(dict(K=K, lam=lam, kind=kind, eps=eps, m_g=mg, **o))
            e = o["per"]["0.0|0.99"]["nonfinal"]
            print(K, lam, eps, "nonfinal", e["label"], f"max {e['max_gain']:+.4f} hi {e['hi_max']:+.4f}", e["worst_dev"], flush=True)
json.dump(rec, open(os.path.join(sv.OUT, "survival_ledger_refuse_only.json"), "w"))
