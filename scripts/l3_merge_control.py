"""Rebuild calib/out/l3_mappo_control.json (v1) and l3_mappo_control_v2.json from the per-seed raw outputs in
calib/out/l3_raw/{ctrl,v2_ctrl}{500,1000,2000}/ (same computation as `l3_analyze.py control <dir>`).
Default output dir is .scratch/l3_merge_check/ so the paper files in calib/out are never overwritten.
usage: python scripts/l3_merge_control.py [--raw calib/out/l3_raw] [--out .scratch/l3_merge_check]"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l3_analyze import load_cell, f_cell, ci_cell  # noqa: E402
import numpy as np  # noqa: E402


def analyze_dir(d):
    cells = {}
    for f in sorted(glob.glob(os.path.join(d, "*_s[0-9].json"))):
        cells.setdefault(os.path.basename(f).rsplit("_s", 1)[0], []).append(f)
    res = []
    for name, fs in cells.items():
        c = load_cell(fs)
        S, n = c["M"].shape
        allS, allI = np.arange(S), np.arange(n)
        F = f_cell(c, allS, allI); ci = ci_cell(c)
        per = [float(f_cell(c, np.array([s]), allI)) for s in range(S)]
        res.append(dict(cell=name, seeds=S, F_G=float(F), ci95=ci, per_seed=per, R_M=float(c["M"].mean()),
                        R_L=float(c["L"].mean()), R_gate=float(c["G"].mean()), RL=c["RL"], RG=c["RG"],
                        pass_cell=bool(F >= 0.5 and ci[0] > 0)))
    n_pass = sum(r["pass_cell"] for r in res)
    return dict(cells=res, n_pass=n_pass, gate_pass=n_pass >= 2 and len(res) == 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="calib/out/l3_raw")
    ap.add_argument("--out", default=".scratch/l3_merge_check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    v1 = {str(U): analyze_dir(os.path.join(a.raw, f"ctrl{U}")) for U in (500, 1000, 2000)}
    v2 = {f"U{U}": analyze_dir(os.path.join(a.raw, f"v2_ctrl{U}")) for U in (500, 1000, 2000)}
    for name, obj in (("l3_mappo_control.json", v1), ("l3_mappo_control_v2.json", v2)):
        p = os.path.join(a.out, name); json.dump(obj, open(p, "w"), indent=1); print("wrote", p)
