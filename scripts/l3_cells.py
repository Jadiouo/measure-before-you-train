"""Cell sets for the L3 MAPPO validation (notes/impl/l3_mappo_prereg.md).  Pure function of calib/out/drone_bucket_l23.json.
A  : the 23 'rot-selected' cells (bound trustworthy, rot gap > 2%)           (selection depends on a rule's result)
B  : all other trustworthy-bound N=5, h=0 cells of the same JSON (14)       (not selected by rot; every N=5 h=0 trusted cell not in A)
CTRL: three A cells used for the positive control (MAPPO-G).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL = ["N5_ar50_s0.15_p0.0_c10_h0.0", "N5_ar10_s0.3_p0.0_c10_h0.0", "N5_ar50_s0.05_p0.0_c1_h0.0"]


def timed_out(c):
    return c["opt_timeouts"] > 0 or abs(c["opt"] - c["opt_incumbent"]) > 1e-9 * max(1, abs(c["opt"]))


def name(c):
    return f"N{c['N']}_{c['noise']}{c['L']}_s{c['sigma']}_p{c['p_event']}_c{c['c']}_h{c['h']}"


def load(path=None):
    d = json.load(open(path or os.path.join(ROOT, "calib/out/drone_bucket_l23.json")))
    rows = [c for j in d["jobs"] for c in j["cells"]]
    A = [c for c in rows if not timed_out(c) and c["rules"]["rot"]["gap"] > 0.02]
    an = {name(c) for c in A}
    B = [c for c in rows if not timed_out(c) and c["N"] == 5 and c["h"] == 0 and name(c) not in an]
    return A, B


if __name__ == "__main__":
    A, B = load()
    print(len(A), len(B))
    print("A", [name(c) for c in A])
    print("B", [name(c) for c in B])
