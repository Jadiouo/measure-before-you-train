"""Diagnostic for the group-gain hypothesis (alpha = 0): per-deviation own gain, split into the arrival part (beta=0) and
the energy part, for a few cells.  Does 'refusing a tenure' ever LOSE for the refuser because a dead companion makes
it work more?  usage: .venv/bin/python scripts/diag_survival.py   output: calib/out/survival_game_diag.json"""
import json
import os

import numpy as np

from animalbench.analysis import survival_game as sv

D, PG, N_EP = 300, 0.05, 6000
CELLS = [dict(h=0.05, m_g=1.0, eps=e) for e in (0.96, 1.0, 1.04)] + [dict(h=0.0, m_g=1.0, eps=e) for e in (0.98, 1.0, 1.04)]


def main():
    pr = sv.load_params()
    hit, size = sv.make_gusts(N_EP, D, PG, 777)          # different seed from the grid run
    out = []
    for c in CELLS:
        E0 = c["eps"] * D * sv.DH
        for strat, tau in (("rot", 15), ("fixed", 15)):
            base = sv.simulate(strat, E0, D, hit, size, pr, tau=tau, m_g=c["m_g"], h=c["h"])
            P_dead = float((~base["arrived"]).mean())
            rows = []
            for dv in sv.deviations(strat, tau, D):
                r = sv.simulate(strat, E0, D, hit, size, pr, tau=tau, m_g=c["m_g"], h=c["h"], dev=dv)
                row = dict(dv)
                for tag, beta in (("arrival", 0.0), ("total", 0.5)):
                    g = (sv.utilities(r, 0.0, beta) - sv.utilities(base, 0.0, beta))[:, dv["i"]]
                    row[tag] = float(g.mean())
                    row[tag + "_se"] = float(g.std(ddof=1) / np.sqrt(N_EP))
                # effect on the OTHER birds' arrival probability (the 'harm to the group')
                oth = [j for j in range(3) if j != dv["i"]]
                row["others_arrival_change"] = float((r["arrived"][:, oth].astype(float) - base["arrived"][:, oth]).mean())
                rows.append(row)
            out.append(dict(cell=c, strategy=strat, tau=tau, frac_not_arriving=P_dead, summary=sv.summary(base), devs=rows))
            by = {}
            for row in rows:
                by.setdefault(row["mode"], []).append(row)
            print(c, strat, "team %.3f" % sv.summary(base)["team_arrival"])
            for m, rr in by.items():
                print("   %-9s own gain: arrival mean %+.4f [min %+.4f max %+.4f]  total mean %+.4f  others' arrival %+.4f" % (
                    m, np.mean([x["arrival"] for x in rr]), min(x["arrival"] for x in rr), max(x["arrival"] for x in rr),
                    np.mean([x["total"] for x in rr]), np.mean([x["others_arrival_change"] for x in rr])))
    json.dump(dict(n_episodes=N_EP, seed=777, D=D, p_gust=PG, results=out), open(os.path.join(sv.OUT, "survival_game_diag.json"), "w"))


if __name__ == "__main__":
    main()
