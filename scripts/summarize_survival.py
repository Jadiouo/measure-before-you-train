"""Print the tables used in notes/impl/survival_game.md from calib/out/survival_game.json.
usage: .venv/bin/python scripts/summarize_survival.py [h] [m_g]  (defaults: all main-grid cells)"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
d = json.load(open(os.path.join(ROOT, "calib", "out", "survival_game.json")))
R = d["records"]
S = {"EQ_STRICT": "EQs", "EQ_EPS": "EQe", "NOT_EQ": "NO", "UNDECIDED": "??"}


def name(r):
    return r["strategy"] + (str(r["tau"]) if r["strategy"] != "fixed" else "")


def sel(**kw):
    return [r for r in R if all(r["cell"][k] == v for k, v in kw.items())]


def main_table(h, mg, cm=1.0, rule="unilateral", beta="0.5"):
    print(f"\n== h={h} m_g={mg} cmult={cm} rule={rule} beta={beta}  (label per alpha 0/.25/.5/1; EQs strict, EQe eps=0.005, NO, ??)")
    for r in sel(h=h, m_g=mg, cmult=cm, rule=rule):
        pass
    eps_vals = sorted({r["cell"]["eps"] for r in sel(h=h, m_g=mg, cmult=cm, rule=rule)})
    for e in eps_vals:
        for r in sel(h=h, m_g=mg, cmult=cm, rule=rule, eps=e):
            s = r["summary"]
            labs = " ".join(S[r["per"][f"{a}|{beta}"]["label"]] for a in (0.0, 0.25, 0.5, 1.0))
            print(f"eps={e:.2f} {name(r):7s} team={s['team_arrival']:.3f} all={s['all_arrive']:.3f} "
                  f"ind={[round(x, 2) for x in s['indiv_arrival']]} E={s['mean_energy']:.3f} sw={s['n_swaps']:.0f} | {labs}")


def _main():
    if len(sys.argv) > 2:
        main_table(float(sys.argv[1]), float(sys.argv[2]))
    else:
        for r in R[:0]:
            pass
        for h, mg in sorted({(r["cell"]["h"], r["cell"]["m_g"]) for r in R}):
            main_table(h, mg)


def regions():
    """alpha=0 equilibria of rotation-type strategies: where is any rotation tau an equilibrium?"""
    print("\n== alpha=0 rotation/threshold: label at eps_tol 0.005 | 0.002 ; max own gain (utility units) ; beta=0.5")
    for h, mg, cm, rule in sorted({(r["cell"]["h"], r["cell"]["m_g"], r["cell"]["cmult"], r["cell"]["rule"]) for r in R}):
        rows = []
        for r in sel(h=h, m_g=mg, cmult=cm, rule=rule):
            if r["strategy"] == "fixed":
                continue
            e = r["per"]["0.0|0.5"]
            rows.append((r["cell"]["eps"], name(r), e["label"], e["labels_by_eps"]["0.002"], e.get("max_gain", 0), e.get("lo_max", 0),
                         r["summary"]["team_arrival"], r["per"]["0.0|0.0"]["label"]))
        strict = [x for x in rows if x[2] == "EQ_STRICT" and x[6] >= 0.5]
        eqe = [x for x in rows if x[2] in ("EQ_STRICT", "EQ_EPS") and x[6] >= 0.5]
        eq002 = [x for x in rows if x[3] in ("EQ_STRICT", "EQ_EPS") and x[6] >= 0.5]
        print(f"h={h} mg={mg} cm={cm} {rule}: strict={[(x[0], x[1]) for x in strict]}")
        print(f"   eqE(0.005, team>=.5): {sorted({(x[0], x[1]) for x in eqe})}")
        print(f"   eqE(0.002, team>=.5): {sorted({(x[0], x[1]) for x in eq002})}")
        print(f"   beta=0 (arrival only) eq: {sorted({(x[0], x[1]) for x in rows if x[7] in ('EQ_STRICT', 'EQ_EPS')})}")


if __name__ == "__main__":
    _main()
    regions()
