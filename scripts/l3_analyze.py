"""Analysis of l3_mappo.py outputs per notes/impl/l3_mappo_prereg.md (control gate: per-cell F_G; set-level F_S for L3).
usage: l3_analyze.py control <dir> [--out file.json]"""
import glob, json, os, sys
import numpy as np


def load_cell(files):
    rs = [json.load(open(f)) for f in files]
    ref = rs[0]["ref"]
    mean = {k: np.array(v["range"]) for k, v in ref.items()}
    RL = max(("rot", "local"), key=lambda k: mean[k].mean())
    RG = max(("fixed", "rot", "gate", "gate_est"), key=lambda k: mean[k].mean())
    M = np.stack([np.array(r["eval"]["greedy"]["range"]) for r in rs])   # (seeds, inst)
    return dict(cell=rs[0]["cell"], M=M, L=mean[RL], G=mean[RG], fixed=mean["fixed"], RL=RL, RG=RG)


def f_cell(c, si, ii):
    m = c["M"][np.ix_(si, ii)].mean(); l = c["L"][ii].mean(); g = c["G"][ii].mean()
    return (m - l) / (g - l)


def ci_cell(c, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    S, n = c["M"].shape
    out = []
    for _ in range(nb):
        si = rng.integers(0, S, S); ii = rng.integers(0, n, n) if n > 1 else np.zeros(1, int)
        out.append(f_cell(c, si, ii))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


if __name__ == "__main__":
    mode, d = sys.argv[1], sys.argv[2]
    if mode == "control":
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
        out = dict(cells=res, n_pass=sum(r["pass_cell"] for r in res), gate_pass=sum(r["pass_cell"] for r in res) >= 2 and len(res) == 3)
        print(json.dumps(out, indent=1))
        if len(sys.argv) > 4:
            json.dump(out, open(sys.argv[4], "w"), indent=1)
