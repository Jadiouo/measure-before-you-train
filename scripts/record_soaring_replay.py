"""Record one episode of the soaring simulation (round 2 world) for the portfolio replay.

Records only: re-runs simulate2 from scripts/soaring_headroom2.py (unchanged) with the tuned parameters
stored in calib/out/soaring_headroom2.json, on the same 60 evaluation worlds and the same simulator seed
(99) as the headroom test, and keeps the trajectory of ONE world (table index) for two conditions:
  L0 = every glider on its own (no broadcast), L1 = broadcast of thermals found (shared rule).
Check: the per-world Emin of the batch run must equal Emin_seeds in the JSON (max abs diff printed).
The world index is the one whose (L1 - L0) Emin difference is the median over the 60 worlds
(fixed by rule, not chosen by eye).
Run: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 .venv/bin/python scripts/record_soaring_replay.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import soaring_headroom2 as S2  # noqa: E402
from soaring_headroom import gen_tables, SIG, A, T_STEPS  # noqa: E402
HMAX = S2.HMAX  # round-2 value (3200), not the round-1 constant

CELL = "mid|long|W3.0|low"
N = 5
STRIDE = 10  # seconds per recorded frame


def main():
    d = json.load(open(os.path.join(ROOT, "calib/out/soaring_headroom2.json")))
    r = next(x for x in d["results"] if x["name"] == CELL)
    rho, Lbar, Wk = r["rho"], r["Lbar"], r["Wk"]
    sig, sigr = SIG["low"]
    pri = dict(rho=rho, Lbar=Lbar, Wk=Wk)
    PP = r["tuned"][f"N{N}"]
    te = gen_tables(np.arange(0, 60), rho, Lbar, Wk)
    ref = r["cells"][str(N)]

    # 1) batch run without trace: reproduce JSON, choose world
    res = {}
    for lab, share, comm in (("L0", False, 0.0), ("L1", True, S2.INF)):
        out = S2.simulate2(te, N, "rule", PP[lab], comm, sig, sigr, pri, 99, share)
        sm = S2.summarise(out)
        res[lab] = sm
        diff = np.abs(sm["Emin"] - np.array(ref[lab]["Emin_seeds"])).max()
        print(lab, "Emin mean", sm["Emin"].mean(), "max|diff vs JSON|", diff)
        assert diff < 1e-9, "does not reproduce JSON"
    delta = res["L1"]["Emin"] - res["L0"]["Emin"]
    order = np.argsort(delta)
    k = int(order[len(order) // 2])
    print("median world", k, "delta Emin", delta[k], "all deltas min/med/max", delta.min(), np.median(delta), delta.max())

    # 2) traced run, keep world k
    rec = {}
    for lab, share, comm in (("L0", False, 0.0), ("L1", True, S2.INF)):
        frames = []

        def trace(step, mode, h, w_air, wf, px, py, cur, cx, cy, s, frames=frames):
            if step % STRIDE:
                return
            on = s[k] > 0.05
            frames.append(dict(
                a=[[int(round(x % A)), int(round(y % A)), int(round(hh)), int(m)]
                   for x, y, hh, m in zip(px[k], py[k], h[k], mode[k])],
                t=[[int(round(x % A)), int(round(y % A)), round(float(v), 2)]
                   for x, y, v in zip(cx[k][on], cy[k][on], s[k][on])]))

        S2.simulate2(te, N, "rule", PP[lab], comm, sig, sigr, pri, 99, share, trace=trace)
        rec[lab] = frames
    meta = dict(cell=CELL, N=N, world=k, sim_seed=99, stride_s=STRIDE, A=A, T=T_STEPS, HMAX=HMAX,
                params=dict(rho=rho, Lbar=Lbar, Wk=Wk, vario_sigma=sig, wing_sigma=sigr),
                emin={lab: float(res[lab]["Emin"][k]) for lab in res},
                amin={lab: float(res[lab]["Amin"][k]) for lab in res},
                landed={lab: bool(res[lab]["fail"][k]) for lab in res},
                batch_mean_emin={lab: float(res[lab]["Emin"].mean()) for lab in res},
                delta_emin_all_worlds=dict(min=float(delta.min()), median=float(np.median(delta)), max=float(delta.max())),
                source="calib/out/soaring_headroom2.json results[mid|long|W3.0|low].tuned.N5 (L0,L1); cells.5.L0/L1.Emin_seeds")
    outp = os.path.join(ROOT, "web/portfolio/data/soaring_replay.json")
    json.dump(dict(meta=meta, L0=rec["L0"], L1=rec["L1"]), open(outp, "w"), separators=(",", ":"))
    print("wrote", outp, os.path.getsize(outp), "bytes")


if __name__ == "__main__":
    main()
