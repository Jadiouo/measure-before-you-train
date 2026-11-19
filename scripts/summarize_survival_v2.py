"""Print the v2 small-game tables from calib/out/survival_game_v2.json."""
import json
import os
import sys
from animalbench.analysis import survival_game as sv

R = json.load(open(os.path.join(sv.OUT, "survival_game_v2.json")))["records"]
SH = {"EQ_STRICT": "EQs", "EQ_EPS": "EQe", "NOT_EQ": "NO", "UNDECIDED": "?"}


def pick(**kw):
    return [r for r in R if all(r[k] == v for k, v in kw.items())]


def name(r):
    return r["strategy"] + (str(r["tau"]) if r["strategy"] != "fixed" else "")


def table(pt, het, cell, rule="unilateral"):
    rows = pick(path_T=pt, het=het, cell=cell, rule=rule)
    print(f"\n### path_T={pt} het={het} cell={cell} rule={rule}")
    print("| 策略 | 全隊到達 | 任一死亡 | U_team | 鳥0領頭占比 | swaps | a=0 | a=0.5 | a=1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    base = {name(r): r for r in pick(path_T=pt, het=het, cell=cell, rule="unilateral")}
    for r in rows:
        b = base[name(r)]
        s = r["summary"]
        cells = []
        for a in ("0.0", "0.5", "1.0"):
            e = r["per"][f"{a}|0.5"]
            cells.append(f"{SH[e['label']]} {e.get('max_gain', 0):+.3f}")
        print(f"| {name(r)} | {s['all_arrive']:.3f} | {b['death_any']:.3f} | {b['U_team']:.4f} | {b['lead_share'][0]:.2f} | {s['n_swaps']:.1f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    pts = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [3.0, 2.0, 1.5]
    for pt in pts:
        for het in ("h", "h+mult5"):
            for cell in ("low", "edge", "high"):
                table(pt, het, cell)
