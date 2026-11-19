"""I2 count-noise analysis on the EXISTING 50-episode JSON (no new rollout).  For agents whose true I2 answer is 'tolerate'
(S1, S2, S3c, S2f_ia, N_cyc_ia, N_rnd_ia) take the per-episode sham-corrected effects ey = dy_F - dy_S, el = dls_F - dls_S and
estimate (a) the episode-level false-positive rate (tolerate agent labelled non-tolerate) with a bootstrap CI,
(b) the same under a label-swap permutation null (F and S exchanged per episode => true effect 0),
(c) the hit rate of the block rule (mean over m episodes, same 0.5/0.2 thresholds) as a function of m, and the smallest m reaching 0.90 / 0.95.
python scripts/m2gate_i2_bootstrap.py  ->  calib/out/intervention_i2_bootstrap.json"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 0.2
AGENTS = ["S1_treadmill", "S2_state_rotation", "S3c_reciprocity_code", "S2f_ia", "N_cyc_ia", "N_rnd_ia"]
MS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
B = 4000


def tol(ey, el):
    return abs(ey) < TOL and abs(el) < TOL


def block_hit(dyF, dyS, dlF, dlS, m, rng, swap):
    n = len(dyF); h = 0
    for _ in range(B):
        i = rng.integers(0, n, m)
        a, b, c, d = dyF[i], dyS[i], dlF[i], dlS[i]
        if swap:
            sw = rng.random(m) < 0.5
            a, b = np.where(sw, b, a), np.where(sw, a, b)
            c, d = np.where(sw, d, c), np.where(sw, c, d)
        h += tol(a.mean() - b.mean(), c.mean() - d.mean())
    return h / B


def main():
    R = json.load(open(os.path.join(ROOT, "calib", "out", "intervention_confusion.json")))["agents"]
    rng = np.random.default_rng(20261030)
    out = {}
    for ag in AGENTS:
        f = [x for x in R[ag]["per_seed"]["i2"] if x["y_pre"] > 0 and x["ls_pre"] >= 2]
        dyF, dyS = np.array([x["dy_F"] for x in f]), np.array([x["dy_S"] for x in f])
        dlF, dlS = np.array([x["dls_F"] for x in f]), np.array([x["dls_S"] for x in f])
        fp = np.array([not tol(a - b, c - d) for a, b, c, d in zip(dyF, dyS, dlF, dlS)])
        bs = [rng.choice(fp, len(fp)).mean() for _ in range(B)]
        sw = np.array([[not tol(*((a - b, c - d) if rng.random() < .5 else (b - a, d - c))) for a, b, c, d in zip(dyF, dyS, dlF, dlS)] for _ in range(400)])
        d = dict(n_valid=len(f), mean_ey=float(np.mean(dyF - dyS)), mean_el=float(np.mean(dlF - dlS)), sd_ey=float(np.std(dyF - dyS)), sd_el=float(np.std(dlF - dlS)),
                 episode_fp_rate=float(fp.mean()), episode_fp_ci95=[float(x) for x in np.quantile(bs, [.025, .975])],
                 episode_fp_rate_swap_null=float(sw.mean()),
                 block_hit_bootstrap={m: block_hit(dyF, dyS, dlF, dlS, m, rng, False) for m in MS},
                 block_hit_swap_null={m: block_hit(dyF, dyS, dlF, dlS, m, rng, True) for m in MS})
        for thr in (0.90, 0.95):
            for nm in ("block_hit_bootstrap", "block_hit_swap_null"):
                ok = [m for m in MS if d[nm][m] >= thr]
                d[f"min_m_{nm}_{int(thr * 100)}"] = ok[0] if ok else None
        out[ag] = d
        print(ag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("block_hit")})
    out["min_m_all_agents_90_bootstrap"] = max(v["min_m_block_hit_bootstrap_90"] or 999 for v in out.values() if isinstance(v, dict))
    out["min_m_all_agents_95_bootstrap"] = max(v["min_m_block_hit_bootstrap_95"] or 999 for v in out.values() if isinstance(v, dict))
    json.dump(out, open(os.path.join(ROOT, "calib", "out", "intervention_i2_bootstrap.json"), "w"), indent=1)
    print("min m (all agents) for 90%:", out["min_m_all_agents_90_bootstrap"], " for 95%:", out["min_m_all_agents_95_bootstrap"])


if __name__ == "__main__":
    main()
