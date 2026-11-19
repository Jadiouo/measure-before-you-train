"""Extension of pfollow_survival.py to low eps (cheap following moves the at-risk band to eps < 0.92).
Merges key 'ext' into calib/out/survival_game_pfollow.json."""
import json, os
import numpy as np
from animalbench.analysis import survival_game as sv

D, PG, N = 300, 0.05, 2000
hit, size = sv.make_gusts(N, D, PG, 12345)
base = sv.load_params()
path = os.path.join(sv.OUT, "survival_game_pfollow.json")
d = json.load(open(path))
ext, diag = [], []
for pf in (0.75, 0.6, 0.5):
    pr = dict(base, p_follow=pf)
    # leader-only cost 1.0, followers pf: mean per-bird rate under rotation = (1+2pf)/3 ; scan eps around it
    for mg in (1.0, 2.0):
        for eps in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
            E0 = eps * D * sv.DH
            for strat, tau in (("fixed", 15), ("rot", 4), ("rot", 15), ("rot", 30), ("thr", 15)):
                o = sv.eval_cell(strat, tau, E0, D, hit, size, pr, mg, 0.05, 1.0, (0.0, 0.25), (0.5,))
                ext.append(dict(p_follow=pf, m_g=mg, eps=eps, strategy=strat, tau=tau, **o))
                if strat != "fixed" and 0.3 <= o["summary"]["team_arrival"] <= 0.97 and tau != 4:
                    b = sv.simulate(strat, E0, D, hit, size, pr, tau=tau, m_g=mg, h=0.05)
                    rows = []
                    for dv in sv.deviations(strat, tau, D):
                        if dv["mode"] != "refuse":
                            continue
                        r = sv.simulate(strat, E0, D, hit, size, pr, tau=tau, m_g=mg, h=0.05, dev=dv)
                        i = dv["i"]; oth = [j for j in range(3) if j != i]
                        rows.append((float((r["arrived"][:, i].astype(float) - b["arrived"][:, i]).mean()),
                                     float((r["arrived"][:, oth].astype(float) - b["arrived"][:, oth]).mean()),
                                     float((sv.utilities(r, 0.0)[:, i] - sv.utilities(b, 0.0)[:, i]).mean())))
                    m = np.mean(rows, 0)
                    diag.append(dict(p_follow=pf, m_g=mg, eps=eps, strategy=strat, tau=tau, team_arrival=o["summary"]["team_arrival"],
                                     own_arrival=m[0], others_arrival=m[1], own_gain_a0=m[2]))
    print("done", pf, flush=True)
d["ext"] = dict(note="low-eps extension, mg in {1,2}, eps 0.55-0.90", records=ext, diag=diag)
json.dump(d, open(path, "w"))
