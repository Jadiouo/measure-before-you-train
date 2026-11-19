"""Print the tables of calib/out/survival_ledger.json.  usage: .venv/bin/python scripts/summarize_survival_ledger.py [perm]"""
import json
import os
import sys

from animalbench.analysis import survival_game as sv

d = json.load(open(os.path.join(sv.OUT, "survival_ledger.json")))
perm = len(sys.argv) > 1
SH = {"EQ_STRICT": "EQ-strict", "EQ_EPS": "EQ-eps", "NOT_EQ": "NOT", "UNDECIDED": "undec"}


def cell(e):
    if e is None:
        return "   -   "
    return f"{SH[e['label']]} {e['max_gain']:+.4f}(hi{e['hi_max']:+.4f})"


for r in d["records"]:
    if r["perm_death"] != perm:
        continue
    print(f"K={r['K']:2d} lam={r['lam']:2d} {r['kind']:8s} eps={r['eps']} mg={r['m_g']} arrive0={r['summary']['team_arrival_trip0']:.3f} "
          f"allarr={r['summary']['all_arrive_mean']:.3f} nDev={r['n_dev']} z={r['z']:.2f}")
    for k, ent in r["per"].items():
        print(f"   a|d={k:9s} all: {cell(ent['all'])} | nonfinal: {cell(ent['nonfinal'])} | final: {cell(ent['final'])}")
