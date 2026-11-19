"""p_follow sensitivity of the survival game (alpha 0 / 0.25, beta 0.5, h 0.05).  New output only:
calib/out/survival_game_pfollow.json.  usage: systemd-run ... .venv/bin/python scripts/pfollow_survival.py [--n 2000]"""
import argparse
import json
import os
import time

import numpy as np

from animalbench.analysis import survival_game as sv

D, PG = 300, 0.05
PF = (0.875, 0.75, 0.6, 0.5)
STRATS = [("fixed", 15), ("rot", 4), ("rot", 15), ("rot", 30), ("thr", 15)]
EPS = [round(x, 2) for x in np.arange(0.92, 1.101, 0.02)]
MG = (0.0, 0.5, 1.0, 2.0)
DIAG = [(1.0, 1.0), (1.02, 1.0), (0.98, 0.5)]        # (eps, m_g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    hit, size = sv.make_gusts(a.n, D, PG, 12345)
    base_pr = sv.load_params()
    rec, diag, t0 = [], [], time.time()
    for pf in PF:
        pr = dict(base_pr, p_follow=pf)
        for mg in MG:
            for eps in EPS:
                E0 = eps * D * sv.DH
                for strat, tau in STRATS:
                    o = sv.eval_cell(strat, tau, E0, D, hit, size, pr, mg, 0.05, 1.0, (0.0, 0.25), (0.5,))
                    rec.append(dict(p_follow=pf, m_g=mg, eps=eps, strategy=strat, tau=tau, **o))
        for eps, mg in DIAG:
            E0 = eps * D * sv.DH
            for strat, tau in (("rot", 15), ("rot", 30), ("thr", 15)):
                b = sv.simulate(strat, E0, D, hit, size, pr, tau=tau, m_g=mg, h=0.05)
                rows = []
                for dv in sv.deviations(strat, tau, D):
                    if dv["mode"] != "refuse":
                        continue
                    r = sv.simulate(strat, E0, D, hit, size, pr, tau=tau, m_g=mg, h=0.05, dev=dv)
                    i = dv["i"]
                    oth = [j for j in range(3) if j != i]
                    rows.append(dict(i=i, t_d=dv["t_d"],
                                     own_arrival=float((r["arrived"][:, i].astype(float) - b["arrived"][:, i]).mean()),
                                     others_arrival=float((r["arrived"][:, oth].astype(float) - b["arrived"][:, oth]).mean()),
                                     own_gain_a0=float((sv.utilities(r, 0.0)[:, i] - sv.utilities(b, 0.0)[:, i]).mean())))
                diag.append(dict(p_follow=pf, eps=eps, m_g=mg, strategy=strat, tau=tau, team_arrival=sv.summary(b)["team_arrival"],
                                 refuse_mean=dict(own_arrival=float(np.mean([x["own_arrival"] for x in rows])),
                                                  others_arrival=float(np.mean([x["others_arrival"] for x in rows])),
                                                  own_gain_a0=float(np.mean([x["own_gain_a0"] for x in rows]))),
                                 refuse_rows=rows))
        print("p_follow", pf, round(time.time() - t0), "s", flush=True)
    json.dump(dict(n_episodes=a.n, D=D, p_gust=PG, h=0.05, beta=0.5, p_follow_grid=PF, eps_grid=EPS, mg_grid=MG,
                   note="p_follow also used for N=2 followers; N=1 solo 1.0; swap costs unchanged; labels per 'alpha|beta'",
                   records=rec, diag=diag),
              open(os.path.join(sv.OUT, "survival_game_pfollow.json"), "w"))


if __name__ == "__main__":
    main()
