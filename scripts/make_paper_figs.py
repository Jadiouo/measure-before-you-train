"""Paper figures and the paper's number table.

Reads ONLY calib/out/*.json (Fig. 0 is pure drawing; the two constants that are not in calib/out are marked
CONST_* below and documented in docs/canonical_numbers.md).  Writes
  paper/figures/fig*.pdf  +  same-name .png (png is for the later web page)
  paper/figures/numbers.json   (every number quoted in paper/main.tex that comes from a computation)
Run:  .venv/bin/python scripts/make_paper_figs.py
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "calib/out")
FIG = os.path.join(ROOT, "paper/figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
BLUE, VERM, ORANGE, SKY, GREEN, GREY, BLACK, YEL = "#0072B2", "#D55E00", "#E69F00", "#56B4E9", "#009E73", "#8a8a8a", "#000000", "#F0E442"
LIGHT = "#d9d9d9"

# constants not stored in calib/out (see docs/canonical_numbers.md, rows "pred_*")
CONST_PRED_FRAC_L1 = 0.30        # git show 79b4dbc:.scratch/drone_bucket/PREDICTION_L1.md, item 4 ("約 30%")
CONST_PRED_GRID_L1 = 576         # same file, header line (grid of 576 cells when the prediction was written)
CONST_PRED_RANGE_L1 = (15, 45)   # same file, item 4 ("區間 15-45%")
CONST_GPU_S_PER_SETTING = 200    # notes/impl/l3_mappo_results.md "實測成本": U=2000, 5 seeds, vmap, about 198-210 s per setting (log, not in calib/out)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 100, "savefig.dpi": 200, "font.family": "DejaVu Sans",
})


def load(name):
    return json.load(open(os.path.join(OUT, name + ".json")))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)


def timed_out(c):
    return c["opt_timeouts"] > 0 or abs(c["opt"] - c["opt_incumbent"]) > 1e-9 * max(1.0, abs(c["opt"]))


# ================================================================== numbers
def compute_numbers():
    N = {}
    # ---- rotation scheduling
    b0 = load("drone_bucket")["cells"]
    tail = [100 * c["gap_tail"] for c in b0]
    gate = [100 * c["gap_gate"] for c in b0]
    N["L0_cells"] = len(b0)
    N["L0_timerot_gt2"] = sum(v > 2 for v in tail)
    N["L0_timerot_max"] = max(tail)
    N["L0_timerot_median"] = statistics.median(tail)
    N["L0_battery_gt2"] = sum(v > 2 for v in gate)
    N["L0_battery_gt1"] = sum(v > 1 for v in gate)
    N["L0_battery_max"] = max(gate)
    N["L0_battery_median"] = statistics.median(gate)
    N["L0_hetero_cells"] = sum(1 for c in b0 if c["h"] > 0)
    N["L0_timerot_gt2_hetero"] = sum(1 for c in b0 if c["h"] > 0 and c["gap_tail"] > 0.02)
    N["L0_timerot_gt2_homog"] = sum(1 for c in b0 if c["h"] == 0 and c["gap_tail"] > 0.02)
    cells1 = [c for j in load("drone_bucket_l1")["jobs"] for c in j["cells"]]
    best1 = [100 * c["best_gap"] for c in cells1]
    N["L1_cells"] = len(cells1)
    N["L1_best_gt2"] = sum(v > 2 for v in best1)
    N["L1_best_gt1"] = sum(v > 1 for v in best1)
    N["L1_best_max"] = max(best1)
    N["L1_best_median"] = statistics.median(best1)
    N["L1_timeout_cells"] = sum(timed_out(c) for c in cells1)
    gate1 = [100 * c["rules"]["gate"]["gap"] for c in cells1]
    N["L1_gate_gt2"], N["L1_gate_max"] = sum(v > 2 for v in gate1), max(gate1)
    hi1 = [100 * c["best_gap"] for c in cells1 if abs(c["h"]) >= 0.15]
    N["L1_h15plus_cells"], N["L1_h15plus_max"], N["L1_h15plus_gt2"] = len(hi1), max(hi1), sum(v > 2 for v in hi1)
    jobs1 = load("drone_bucket_l1")["jobs"]
    N["L1_cpu_s_sim"], N["L1_cpu_s_opt"] = sum(j["time_sim_s"] for j in jobs1), sum(j["time_opt_s"] for j in jobs1)
    N["L1_rule_counts"] = {k: sum(1 for c in cells1 if c["best_rule"] == k) for k in ("fixed", "rot", "gate", "sort", "pair")}
    N["L1_by_h"] = {str(h): 100 * statistics.mean(c["best_gap"] for c in cells1 if c["h"] == h) for h in sorted({c["h"] for c in cells1})}
    over = [c for c in cells1 if c["best_gap"] > 0.02]
    assert len(over) == 1
    c = over[0]
    N["L1_over_cell_bound_timed_out"] = bool(timed_out(c))
    N["L1_over_cell_gap_dual"] = 100 * c["best_gap"]
    N["L1_over_cell_gap_incumbent"] = 100 * (1 - (1 - c["best_gap"]) * c["opt"] / c["opt_incumbent"])
    cells23 = [c for j in load("drone_bucket_l23")["jobs"] for c in j["cells"]]
    fl = [timed_out(c) for c in cells23]
    N["L23_cells"] = len(cells23)
    N["L23_cpu_s"] = sum(j["time_s"] for j in load("drone_bucket_l23")["jobs"])
    N["L23_sigma0_cells"] = sum(1 for c in cells23 if c["sigma"] == 0)
    N["cpu_hours_L1_L23"] = (N["L1_cpu_s_sim"] + N["L1_cpu_s_opt"] + N["L23_cpu_s"]) / 3600
    N["cpu_s_per_setting_L1_L23"] = (N["L1_cpu_s_sim"] + N["L1_cpu_s_opt"] + N["L23_cpu_s"]) / (len(cells1) + len(cells23))
    N["L1_solver_share"] = N["L1_cpu_s_opt"] / (N["L1_cpu_s_sim"] + N["L1_cpu_s_opt"])
    N["gpu_s_per_setting_train5"] = CONST_GPU_S_PER_SETTING
    N["L23_timeout_cells"] = sum(fl)
    N["L23_trusted_cells"] = len(cells23) - sum(fl)
    gg = [100 * c["best_global_gap"] for c in cells23]
    N["L23_global_gt2"] = sum(v > 2 for v in gg)
    N["L23_global_gt2_trusted"] = sum(1 for v, f in zip(gg, fl) if v > 2 and not f)
    N["L23_global_max_trusted"] = max(v for v, f in zip(gg, fl) if not f)
    ok23 = [c for c in cells23 if not timed_out(c)]
    bad = [c for c in ok23 if c["rules"]["rot"]["gap"] > 0.02]
    bl = [100 * min(c["rules"]["rot"]["gap"], c["rules"]["local"]["gap"]) for c in bad]
    gl = [100 * c["best_global_gap"] for c in bad]
    df = [b - g for b, g in zip(bl, gl)]
    N["L3_n"] = len(bad)
    N["L3_n_n5ar_h0"] = sum(1 for c in bad if c["N"] == 5 and c["noise"] == "ar" and c["h"] == 0)
    N["L3_n_other"] = len(bad) - N["L3_n_n5ar_h0"]
    N["L3_local_min"], N["L3_local_max"] = min(bl), max(bl)
    N["L3_global_min"], N["L3_global_max"] = min(gl), max(gl)
    N["L3_diff_min"], N["L3_diff_max"], N["L3_diff_median"] = min(df), max(df), statistics.median(df)
    N["L3_local_gt2"] = sum(v > 2 for v in bl)
    N["L3_global_gt2"] = sum(v > 2 for v in gl)
    N["L3_rot_gt2_trusted_of_trusted"] = f"{len(bad)}/{len(ok23)}"
    N["L3_rot_gt2_all"] = sum(1 for c in cells23 if c["rules"]["rot"]["gap"] > 0.02)
    # ---- wake surfing
    w = load("wake_headroom")["cells"]
    fair = [c["gap_gpsprivileged"] for c in w]
    perf = [c["gap"] for c in w]
    N["wake_cells"] = len(w)
    N["wake_fair_max"] = max(fair)
    N["wake_fair_ci_hi_max"] = max(c["gap_gpsprivileged_ci"][1] for c in w)
    N["wake_fair_gt2"] = sum(v > 2 for v in fair)
    N["wake_fair_min"] = min(fair)
    N["wake_fair_gt1"] = sum(v > 1 for v in fair)
    N["wake_fair_neg_cells"] = sum(v < 0 for v in fair)
    N["wake_fair_neg_ci_cells"] = sum(1 for c in w if c["gap_gpsprivileged_ci"][1] < 0)
    N["wake_fair_neg_all_kf"] = all(c["best"] == "kf" for c in w if c["gap_gpsprivileged"] < 0)
    N["wake_perfect_max"] = max(perf)
    N["wake_perfect_gt2"] = sum(v > 2 for v in perf)
    N["wake_perfect_gt1"] = sum(v > 1 for v in perf)
    N["wake_perfect_cilo_gt2"] = sum(1 for c in w if c["gap_ci"][0] > 2)
    N["wake_random_drift_cells"] = sum(1 for c in w if c["drift"].startswith("amp"))
    N["wake_random_drift_perfect_max"] = max(c["gap"] for c in w if c["drift"].startswith("amp"))
    N["wake_best_counts"] = {k: sum(1 for c in w if c["best"] == k) for k in ("kf", "fixed", "esc")}
    # ---- soaring
    s3 = load("soaring_headroom3")["results"]
    rows = []
    for x in s3:
        for n, c in x["cells"].items():
            g = c["gaps"]["Emin"]
            rows.append(dict(cell=x["name"].rsplit("|", 1)[0], N=int(n), floor=bool(c["floor_cell"]), landed=c["landed_L1"],
                             gap=g["L2-max(L15a,L15b)"], g_l2l1=g["L2-L1"], g_l15a=g["L15a-L1"], g_l15b=g["L15b-L1"]))
    N["soar_rows"] = rows
    n5 = [r for r in rows if r["N"] == 5]
    nf5 = [r for r in n5 if not r["floor"]]
    N["soar_cells"] = len(rows)
    N["soar_n5_cells"] = len(n5)
    N["soar_n5_nonfloor"] = len(nf5)
    top = max(nf5, key=lambda r: r["gap"][0])
    N["soar_top_cell"] = top["cell"]
    N["soar_top_gap"], N["soar_top_lo"], N["soar_top_hi"] = top["gap"]
    N["soar_nonfloor_cilo_gt2"] = sum(1 for r in nf5 if r["gap"][1] > 2)
    N["soar_all_cilo_gt2"] = sum(1 for r in rows if r["gap"][1] > 2)
    N["soar_l2l1_cilo_gt2"] = sum(1 for r in rows if r["g_l2l1"][1] > 2)
    N["soar_l2l1_informative"] = sorted([[r["cell"], r["g_l2l1"][0], r["g_l2l1"][1], r["g_l2l1"][2]] for r in nf5], key=lambda t: t[1])
    N["soar_l2l1_informative_min"], N["soar_l2l1_informative_max"] = min(r["g_l2l1"][0] for r in nf5), max(r["g_l2l1"][0] for r in nf5)
    N["soar_l2l1_informative_lo_min"], N["soar_l2l1_informative_lo_max"] = min(r["g_l2l1"][1] for r in nf5), max(r["g_l2l1"][1] for r in nf5)
    N["soar_gap_vs_sameinfo_informative_min"], N["soar_gap_vs_sameinfo_informative_max"] = min(r["gap"][0] for r in nf5), max(r["gap"][0] for r in nf5)
    N["soar_gap_min"] = min(r["gap"][0] for r in rows)
    N["soar_gap_max"] = max(r["gap"][0] for r in rows)
    N["soar_l15a_min"] = min(r["g_l15a"][0] for r in rows)
    N["soar_l15a_max"] = max(r["g_l15a"][0] for r in rows)
    N["soar_l15a_wins"] = sum(1 for r in rows if r["g_l15a"][0] > 0)
    recov = [100 * r["g_l15b"][0] / r["g_l2l1"][0] for r in rows if abs(r["g_l2l1"][0]) > 1e-9]
    N["soar_recov_b_min"], N["soar_recov_b_max"] = min(recov), max(recov)
    N["soar_recov_b_ge89"] = sum(1 for v in recov if v >= 89)
    N["soar_recov_b_n"] = len(recov)
    s2 = load("soaring_headroom2")["results"]
    ann = [-x["cells"]["5"]["gap_vs_L1_Emin"]["L0"][0] for x in s2 if "|long|" in x["name"]]
    N["soar_announce_gain_min"], N["soar_announce_gain_max"], N["soar_announce_gain_n"] = min(ann), max(ann), len(ann)
    # ---- L3 MAPPO positive control
    for tag, fn, keys in (("v1", "l3_mappo_control", ("500", "1000", "2000")), ("v2", "l3_mappo_control_v2", ("U500", "U1000", "U2000"))):
        d = load(fn)
        N[f"ctrl_{tag}"] = {k: dict(n_pass=d[k]["n_pass"], cells=[dict(cell=c["cell"], F=c["F_G"], ci=c["ci95"], seeds=c["per_seed"],
                                                                         n_seeds=c["seeds"], passed=c["pass_cell"]) for c in d[k]["cells"]])
                            for k in keys}
    v2 = N["ctrl_v2"]["U2000"]["cells"]
    N["ctrl_v2_U2000_collapsed_runs"] = None  # needs entropy logs (not in calib/out); see results note
    N["ctrl_v2_U2000_neg_runs"] = sum(1 for c in v2 for f in c["seeds"] if f < -1)
    N["ctrl_v2_U2000_runs"] = sum(len(c["seeds"]) for c in v2)
    # ---- birds (appendix)
    b = load("survival_v2_seeds")
    N["bird_A2_reward"], N["bird_A2_reward_ci"] = b["A2"]["reward_seedlevel"]
    N["bird_A2_minus_rot30"], N["bird_A2_minus_rot30_ci"] = b["A2"]["minus_rot30_seedlevel"]
    N["bird_A2_incoming_max"] = b["A2"]["incoming_max_seedlevel"][0]
    N["bird_gate30s"] = b["_refs"]["gate30s"]
    N["bird_rot30"] = b["_refs"]["rot30"]
    N["bird_A2_n_sustain"] = b["A2"]["n_sustain"]
    # ---- prediction (constants)
    N["pred_frac_L1"] = CONST_PRED_FRAC_L1
    N["pred_grid_L1"] = CONST_PRED_GRID_L1
    N["pred_range_L1"] = list(CONST_PRED_RANGE_L1)
    # ---- cross-checks against notes/impl/portfolio_numbers.md (page numbers that were already audited)
    assert (N["L0_timerot_gt2"], N["L0_battery_gt2"], N["L1_cells"], N["L1_best_gt2"]) == (132, 4, 768, 1)
    assert round(N["L0_battery_max"], 2) == 2.22 and round(N["L1_best_median"], 3) == 0.045
    assert (N["L3_n"], N["L3_n_n5ar_h0"], N["L23_timeout_cells"], N["L23_global_gt2"]) == (23, 20, 62, 23)
    assert (round(N["L3_diff_min"], 2), round(N["L3_diff_max"], 2), round(N["L3_diff_median"], 2)) == (0.52, 2.17, 1.41)
    assert round(N["wake_fair_max"], 2) == 1.77 and N["wake_cells"] == 108
    assert round(N["soar_top_gap"], 1) == 2.5 and round(N["soar_top_lo"], 1) == 0.1 and round(N["soar_top_hi"], 1) == 5.1
    assert N["soar_nonfloor_cilo_gt2"] == 0
    # numbers added after the internal review r1 (notes/review/closed_door_paper_r1.md)
    assert (N["L1_gate_gt2"], round(N["L1_gate_max"], 2)) == (4, 2.33)
    assert (N["L1_h15plus_cells"], N["L1_h15plus_gt2"], round(N["L1_h15plus_max"], 2)) == (384, 0, 0.59)
    assert (N["wake_fair_neg_cells"], N["wake_fair_neg_ci_cells"], N["wake_fair_neg_all_kf"]) == (34, 22, True)
    assert N["soar_l2l1_cilo_gt2"] == 10 and N["soar_cells"] == 11
    assert [round(t[1], 1) for t in N["soar_l2l1_informative"]] == [5.9, 6.8, 10.0, 13.9]
    assert (round(N["soar_l2l1_informative_lo_min"], 1), round(N["soar_l2l1_informative_lo_max"], 1)) == (4.0, 11.3)
    assert (round(N["soar_gap_vs_sameinfo_informative_min"], 1), round(N["soar_gap_vs_sameinfo_informative_max"], 1)) == (-0.8, 2.5)
    assert N["L23_sigma0_cells"] == 16 and N["wake_random_drift_cells"] == 84 and round(N["wake_random_drift_perfect_max"], 2) == 1.55
    assert round(N["cpu_hours_L1_L23"], 1) == 9.8 and round(N["cpu_s_per_setting_L1_L23"]) == 38 and N["L1_cells"] + N["L23_cells"] == 928
    return N


# ================================================================== helpers for drawing
def plane(ax, x, y, s=1.0, color=BLUE, ang=90):
    """A small delta-wing glyph pointing along `ang` degrees."""
    pts = np.array([[0, 1.0], [0.75, -0.7], [0, -0.25], [-0.75, -0.7]]) * s
    t = np.deg2rad(ang - 90)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ax.add_patch(Polygon(pts @ R.T + [x, y], closed=True, fc=color, ec="white", lw=0.6, zorder=5))


def box(ax, x, y, w, h, fc="white", ec=GREY, lw=1.0, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=2))


# ================================================================== Fig 0: concept
def fig0():
    fig, ax = plt.subplots(figsize=(7.0, 5.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(0.5, 97.6, "A learner can add little if a simple rule is already\nclose to the best possible plan", fontsize=11.5, fontweight="bold", va="center", linespacing=1.25)
    B = {1: (0.5, 49.5), 2: (51.0, 49.5), 3: (0.5, 2.0), 4: (51.0, 2.0)}
    Wd, Ht = 48.5, 41.0
    titles = {1: "1  The leader works hardest", 2: "2  Its battery empties first", 3: "3  A rule picks the next leader", 4: "4  Headroom: room to improve"}
    caps = {1: "Followers ride the leader's wake and use less\npower. The leader saves nothing.",
            2: "The mission ends when the first battery is\nempty, so the leader sets the team's range.",
            3: "Swapping spreads the load, but a swap costs\nenergy. A rule says when and who.",
            4: "If the gap is under 2%, even a perfect learner\ncould add little: training is not worth it."}
    for k, (x0, y0) in B.items():
        box(ax, x0, y0, Wd, Ht, fc="#fafafa", ec=LIGHT)
        ax.text(x0 + 1.5, y0 + Ht - 3.6, titles[k], fontsize=9.5, fontweight="bold", va="center")
        ax.text(x0 + 1.5, y0 + 5.2, caps[k], fontsize=7.8, va="center", linespacing=1.3)
    for (xa, ya, xb, yb, rad) in ((49.6, 70, 50.9, 70, 0), (49.6, 22, 50.9, 22, 0)):
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=12, color=GREY, lw=1.4, zorder=6))
    ax.plot([75, 75, 25], [49.4, 47.0, 47.0], color=GREY, lw=1.4, zorder=6)
    ax.add_patch(FancyArrowPatch((25, 47.0), (25, 43.6), arrowstyle="-|>", mutation_scale=12, color=GREY, lw=1.4, zorder=6))
    # panel 1
    x0, y0 = B[1]
    cx, cy = x0 + 20, y0 + 19.5
    ax.add_patch(Polygon([[cx - 1.5, cy + 6.5], [cx + 1.5, cy + 6.5], [cx + 10, cy - 8.5], [cx - 10, cy - 8.5]], closed=True, fc=SKY, alpha=0.25, ec="none", zorder=3))
    plane(ax, cx, cy + 8.5, 3.4, VERM)
    for dx, dy in ((-7.5, 1.0), (7.5, 1.0), (-14.5, -6.5), (14.5, -6.5)):
        plane(ax, cx + dx, cy + dy, 3.4, BLUE)
    ax.text(cx + 5.2, cy + 10.5, "leader", fontsize=9, color=VERM, fontweight="bold", va="center")
    ax.text(x0 + 38.5, cy - 0.5, "followers", fontsize=9, color=BLUE, fontweight="bold", va="center", ha="center")
    # panel 2
    x0, y0 = B[2]
    bx0 = x0 + 8
    for i, lv in enumerate([0.42, 0.70, 0.78, 0.83, 0.88]):
        x = bx0 + i * 6.8
        ax.add_patch(Rectangle((x, y0 + 12), 4.6, 16, fc="white", ec=GREY, lw=1.0, zorder=4))
        ax.add_patch(Rectangle((x, y0 + 12), 4.6, 16 * lv, fc=VERM if i == 0 else BLUE, ec="none", zorder=5))
    ax.text(bx0 + 2.3, y0 + 30.4, "leader", fontsize=9, color=VERM, fontweight="bold", ha="center")
    ax.text(x0 + 24.5, y0 + 10.0, "battery level (schematic)", fontsize=7.5, color="#555", ha="center", va="center")
    # panel 3
    x0, y0 = B[3]
    px = x0 + 2.5
    for j, (h1, h2) in enumerate((("Time rule", "swap leaders every so often"), ("Battery rule", "hand the lead to the fullest battery"))):
        yy = y0 + 24.5 - j * 11.5
        box(ax, px, yy, Wd - 5.0, 9.2, fc="white", ec=BLUE, lw=1.3, r=1.0)
        ax.text(px + 1.6, yy + 6.5, h1, fontsize=9, fontweight="bold", color=BLUE, va="center")
        ax.text(px + 1.6, yy + 2.8, h2, fontsize=8.3, va="center")
    # panel 4
    x0, y0 = B[4]
    qx, L = x0 + 3.0, Wd - 8.0
    xv = lambda v: qx + (v - 0.90) / 0.10 * L
    ax.text(qx, y0 + 31.0, "best possible plan (with hindsight)", fontsize=8.3, va="center")
    ax.add_patch(Rectangle((qx, y0 + 25.0), L, 3.8, fc=GREEN, ec="none", zorder=4))
    ax.text(qx, y0 + 21.0, "simple rule", fontsize=8.3, va="center")
    ax.add_patch(Rectangle((qx, y0 + 15.0), xv(0.985) - qx, 3.8, fc=BLUE, ec="none", zorder=4))
    ax.add_patch(Rectangle((xv(0.985), y0 + 15.0), xv(1.0) - xv(0.985), 3.8, fc="none", ec=VERM, lw=1.3, hatch="////", zorder=5))
    ax.plot([xv(0.98)] * 2, [y0 + 12.5, y0 + 29.8], color=BLACK, lw=1.1, ls=(0, (3, 2)), zorder=6)
    ax.text(xv(0.98) - 0.7, y0 + 11.2, "2% line", fontsize=8.3, ha="right", va="center")
    ax.text(x0 + Wd - 1.5, y0 + 21.0, "headroom", fontsize=8.5, color=VERM, fontweight="bold", ha="right", va="center")
    ax.text(99.5, 0.0, "illustrative drawing, not data (panel 4 axis zoomed to 90-100%)", fontsize=7.2, color="#666", ha="right", va="center")
    save(fig, "fig0_concept")


def soar_label(cell):
    dens, life, w = cell.split("|")
    return f"{ {'sparse': 'sparse', 'mid': 'medium'}[dens] } thermals, {life}-lived, {'strong' if w == 'W3.0' else 'weak'}"


# ================================================================== Fig 1: waffle prediction vs outcome
def fig1(N):
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 3.5))
    nrow, ncol = 24, 32
    total = nrow * ncol
    assert total == N["L1_cells"]
    npred = round(N["pred_frac_L1"] * total)
    nobs = N["L1_best_gt2"]
    hit_obs = 17 * ncol + 22
    lo, hi = N["pred_range_L1"]
    for ax, k, title, sub in (
            (axs[0], npred, "What we predicted", f"about {int(N['pred_frac_L1'] * 100)}% above the 2% line\n(range {lo} to {hi}%): about {npred}\nof {total}, extrapolated from {N['pred_grid_L1']}"),
            (axs[1], nobs, "What we found", f"{nobs} of {total} settings\nabove the 2% line\n(gap {N['L1_best_max']:.2f}%)")):
        ax.set_xlim(0, ncol)
        ax.set_ylim(nrow, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        for i in range(total):
            r, c = divmod(i, ncol)
            hit = i < k if ax is axs[0] else i == hit_obs
            ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88, fc=VERM if hit else "#e6e6e6", ec="none"))
        ax.text(0, 1.36, title, fontsize=10.5, fontweight="bold", va="bottom", transform=ax.transAxes)
        ax.text(0, 1.03, sub, fontsize=8.2, va="bottom", linespacing=1.25, transform=ax.transAxes)
    r, c = divmod(hit_obs, ncol)
    axs[1].annotate("the one setting", xy=(c + 0.5, r + 1.0), xytext=(c - 4, r + 5.2), fontsize=9, color=VERM, fontweight="bold",
                    ha="center", va="top", arrowprops=dict(arrowstyle="-|>", color=VERM, lw=1.2))
    fig.suptitle("We predicted room in about 30% of settings; we found it in 1 of 768", x=0.01, y=0.995, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Each square is one scheduling setting; positions are arbitrary. Coloured: gap to the best possible plan above 2%.\nThe prediction was written for a 576-setting grid; one more level of drone differences made it 768 (Appendix).",
             fontsize=6.9, color="#444", va="bottom")
    fig.subplots_adjust(top=0.66, bottom=0.12, wspace=0.05)
    save(fig, "fig1_prediction")


# ================================================================== Fig 2: distance to the 2% line (three narrow figures)
def _title(fig, text):
    fig.suptitle(text, x=0.01, y=0.995, ha="left", va="top", fontsize=9, fontweight="bold", linespacing=1.2)


def fig2a(N):
    b0 = load("drone_bucket")["cells"]
    cells1 = [c for j in load("drone_bucket_l1")["jobs"] for c in j["cells"]]
    fig, ax = plt.subplots(figsize=(4.4, 1.95))
    rng = np.random.default_rng(0)
    rows = [(f"Time rule, basic\n{N['L0_timerot_gt2']} of {N['L0_cells']} beyond", [100 * c["gap_tail"] for c in b0], ORANGE),
            (f"Battery rule, basic\n{N['L0_battery_gt2']} of {N['L0_cells']} beyond", [100 * c["gap_gate"] for c in b0], BLUE),
            (f"Best of five rules, richer\n{N['L1_best_gt2']} of {N['L1_cells']} beyond", [100 * c["best_gap"] for c in cells1], BLUE)]
    for i, (lab, vals, col) in enumerate(rows):
        y = 2 - i
        vals = np.array(vals)
        ax.scatter(vals, y + rng.uniform(-0.28, 0.28, len(vals)), s=6, color=col, alpha=0.45, lw=0, zorder=3)
        ax.plot([np.median(vals)] * 2, [y - 0.36, y + 0.36], color=BLACK, lw=1.8, zorder=4)
    ax.axvline(2, color=BLACK, ls=(0, (4, 3)), lw=1.1, zorder=2)
    ax.text(2.15, 2.62, "2% line", fontsize=7.5, va="bottom")
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.8)
    ax.set_xlim(-0.3, 13)
    ax.set_ylim(-0.6, 2.95)
    ax.set_xlabel("Gap to the best possible plan, % of its value (black tick: median)", fontsize=7.8)
    _title(fig, "Simple rules sit inside the 2% line in almost every setting;\nonly the time rule often does not")
    fig.subplots_adjust(top=0.78, left=0.3)
    save(fig, "fig2a_rotation")


def fig2b(N):
    w = load("wake_headroom")["cells"]
    groups = [("No or random drift\n(84 settings)", [c for c in w if c["drift"].startswith("amp")]),
              ("Leader weaves\n(12 settings)", [c for c in w if c["drift"] == "sine"]),
              ("Leader steps\n(12 settings)", [c for c in w if c["drift"] == "step"])]
    assert sum(len(g[1]) for g in groups) == N["wake_cells"]
    fig, ax = plt.subplots(figsize=(4.4, 1.95))
    for i, (lab, cs) in enumerate(groups):
        y = len(groups) - 1 - i
        fair = [c["gap_gpsprivileged"] for c in cs]
        perf = [c["gap"] for c in cs]
        ax.plot([min(perf), max(perf)], [y - 0.17] * 2, color=GREY, lw=5, solid_capstyle="butt", alpha=0.7, zorder=3)
        ax.plot(statistics.median(perf), y - 0.17, "o", ms=4.5, color="white", mec="#444", mew=1.0, zorder=5)
        ax.plot([min(fair), max(fair)], [y + 0.17] * 2, color=BLUE, lw=5, solid_capstyle="butt", zorder=3)
        ax.plot(statistics.median(fair), y + 0.17, "o", ms=4.5, color="white", mec=BLUE, mew=1.2, zorder=5)
        nb = sum(v > 2 for v in perf)
        if nb:
            ax.text(max(perf) + 0.08, y - 0.17, f"{nb} of {len(cs)} beyond", fontsize=7.5, va="center", ha="left", color="#444")
    ax.axvline(2, color=BLACK, ls=(0, (4, 3)), lw=1.1, zorder=2)
    ax.text(2.06, len(groups) - 0.45, "2 pp line", fontsize=7.5, va="bottom")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups][::-1], fontsize=7.8)
    ax.set_xlim(-0.35, 4.6)
    ax.set_ylim(-0.6, len(groups) - 0.2)
    ax.set_xlabel("Gap to a privileged reference, pp of total power", fontsize=7.8)
    box_ = dict(fc="white", ec="none", alpha=0.85, pad=0.8)
    ax.text(1.32, 2.17, "same noisy GPS (diagnostic only)", fontsize=7.3, color=BLUE, va="center", ha="left", bbox=box_, zorder=6)
    ax.text(1.65, 1.83, "perfect position (our prediction)", fontsize=7.3, color="#444", va="center", ha="left", bbox=box_, zorder=6)
    _title(fig, "Only leader manoeuvres put the controller beyond the line,\nand only against a perfect-position reference")
    fig.subplots_adjust(top=0.76, left=0.21)
    save(fig, "fig2b_wake")


def fig2c(N):
    from matplotlib.lines import Line2D
    n5 = [r for r in N["soar_rows"] if r["N"] == 5 and not r["floor"]]
    n5 = sorted(n5, key=lambda r: r["g_l2l1"][0])
    assert len(n5) == 4
    fig, ax = plt.subplots(figsize=(4.4, 2.25))
    for i, r in enumerate(n5):
        g, lo, hi = r["g_l2l1"]
        ax.plot([lo, hi], [i + 0.17] * 2, color=BLUE, lw=2.0, zorder=3, alpha=0.8)
        ax.plot(g, i + 0.17, "o", ms=6, color=BLUE, zorder=5)
        g2, lo2, hi2 = r["gap"]
        ax.plot([lo2, hi2], [i - 0.17] * 2, color=ORANGE, lw=2.0, zorder=3, alpha=0.85)
        ax.plot(g2, i - 0.17, "o", ms=6, mfc="white", mec=ORANGE, mew=1.6, zorder=5)
    ax.axvline(2, color=BLACK, ls=(0, (4, 3)), lw=1.1, zorder=2)
    ax.text(2.2, len(n5) - 0.45, "2 pp line", fontsize=7.5, va="bottom")
    ax.set_yticks(range(len(n5)))
    ax.set_yticklabels([soar_label(r["cell"]).replace(", ", ",\n", 1) for r in n5], fontsize=7.5)
    ax.set_xlim(-3.2, 25)
    ax.set_ylim(-0.6, len(n5) - 0.3)
    ax.set_xlabel("Lead of the strongest method, pp of altitude ceiling", fontsize=7.8)
    ax.legend(handles=[Line2D([0], [0], color=BLUE, marker="o", lw=2, ms=6, label="vs the simple rule"),
                       Line2D([0], [0], color=ORANGE, marker="o", mfc="white", mew=1.6, lw=2, ms=6, label="vs the simple rule\ngiven the same true data")],
              loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1, frameon=False, fontsize=7.3, handlelength=1.3)
    _title(fig, "Most of the soaring lead is information: give the simple rule\nthe true data and it nearly disappears")
    ax.set_xticks([0, 5, 10, 15])
    fig.subplots_adjust(top=0.78, left=0.3)
    save(fig, "fig2c_soaring")


# ================================================================== Fig 3: L3 slope chart
def fig3(N):
    cells23 = [c for j in load("drone_bucket_l23")["jobs"] for c in j["cells"]]
    ok = [c for c in cells23 if not timed_out(c)]
    bad = [c for c in ok if c["rules"]["rot"]["gap"] > 0.02]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for c in bad:
        gl = 100 * c["best_global_gap"]
        lc = 100 * min(c["rules"]["rot"]["gap"], c["rules"]["local"]["gap"])
        ax.plot([0, 1], [gl, lc], color=BLUE, lw=1.4, alpha=0.55, marker="o", ms=3.6)
    ax.axhline(2, color=BLACK, ls=(0, (4, 3)), lw=1.1)
    ax.text(-0.19, 2.05, "2% line", fontsize=7.5, va="bottom", ha="left")
    ax.set_xlim(-0.2, 1.38)
    ax.set_ylim(0, 4.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Rule that sees the\nwhole team\n{N['L3_global_min']:.2f} to {N['L3_global_max']:.2f}%",
                        f"Best rule that sees\nonly neighbours\n{N['L3_local_min']:.2f} to {N['L3_local_max']:.2f}%"], fontsize=7.5)
    ax.set_ylabel("Gap to the best plan, % of its value", fontsize=7.5)
    ax.text(-0.04, 2.7, f"{N['L3_global_gt2']} of {N['L3_n']}\nabove 2%", fontsize=8, color=BLUE, fontweight="bold", ha="left", va="center")
    ax.text(1.05, 3.3, f"{N['L3_local_gt2']} of {N['L3_n']}\nabove 2%", fontsize=8, color=BLUE, fontweight="bold", ha="left", va="center")
    _title(fig, "Seeing only neighbours leaves a gap\nin 23 settings chosen for it")
    fig.subplots_adjust(top=0.85, bottom=0.2, left=0.15)
    save(fig, "fig3_local_gap")


# ================================================================== Fig 4: sanity gate
CELL_NAMES = {"N5_ar10_s0.3_p0.0_c10_h0.0": "swap cost 10, fast noise 30%", "N5_ar50_s0.05_p0.0_c1_h0.0": "swap cost 1, slow noise 5%",
              "N5_ar50_s0.15_p0.0_c10_h0.0": "swap cost 10, slow noise 15%"}


def fig4(N):
    d = N["ctrl_v2"]
    Us = ["U500", "U1000", "U2000"]
    cells = [c["cell"] for c in d["U500"]["cells"]]
    fig, axs = plt.subplots(1, 3, figsize=(5.0, 2.6), sharey=True)
    rng = np.random.default_rng(1)
    for ax, cell in zip(axs, cells):
        ax.axhspan(0.5, 1.75, color=GREEN, alpha=0.08, lw=0, zorder=0)
        for xi, u in enumerate(Us):
            c = [q for q in d[u]["cells"] if q["cell"] == cell][0]
            col = GREEN if c["passed"] else BLUE
            jit = rng.uniform(-0.16, 0.16, len(c["seeds"]))
            ax.scatter(xi + jit, c["seeds"], s=16, color=col, alpha=0.5, lw=0, zorder=3)
            ax.plot([xi, xi], c["ci"], color=col, lw=2.2, zorder=4)
            ax.plot(xi, c["F"], "o", ms=6, mfc="white", mec=col, mew=1.8, zorder=5)
            if c["passed"]:
                ax.annotate(f"passed\n{c['F']:.2f}", xy=(xi, c["ci"][1]), xytext=(xi - 0.35, 1.2), fontsize=7.4, color=GREEN, fontweight="bold",
                            ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))
            if u == "U2000":
                for sx, sv in zip(jit, c["seeds"]):
                    if sv < -1:
                        ax.annotate("collapsed:\nalmost never\nswaps", xy=(xi + sx, sv), xytext=(xi + sx - 0.12, sv + 0.45), fontsize=6.8, color=VERM,
                                    ha="right", va="center")
        ax.axhline(0.5, color=BLACK, ls=(0, (4, 3)), lw=1.1)
        ax.axhline(0.0, color=GREY, lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["500", "1000", "2000"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-2.1, 1.75)
        ax.set_title(CELL_NAMES[cell].replace(", ", ",\n"), fontsize=7.4, pad=3)
    axs[0].set_ylabel("Share of the known gap recovered\n0 = no better than the local rule\n1 = as good as the whole-team rule", fontsize=6.8)
    axs[0].text(-0.5, 1.6, "pass zone", fontsize=7.2, color=GREEN, fontweight="bold", va="center")
    axs[1].set_xlabel("Training updates", fontsize=7.8)
    fig.suptitle("A better-informed learner passed its sanity gate in only 1 of 3 settings\n(2 needed), so the real run never happened",
                 x=0.01, y=1.13, ha="left", fontsize=8.8, fontweight="bold", linespacing=1.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig4_sanity_gate")


# ================================================================== Fig A1: formation (appendix)
def figA1():
    d = load("horseshoe_formation")["sweep"]["N9_V_optimal_lamb_oseen_rc0.075"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    pos = d["per_position"]
    cm = plt.get_cmap("Blues")
    for p in pos:
        red = 1 - p["ratio_all"]
        ax.scatter(p["x"], 4 * p["y"], s=520, color=cm(0.12 + 0.88 * red), ec="#333", lw=0.8, zorder=3)
        ax.text(p["x"], 4 * p["y"], f"{100 * red:.0f}", ha="center", va="center", fontsize=8.5, color="white" if red > 0.55 else "#111", zorder=4)
    ax.annotate("", xy=(-9.0, 0), xytext=(-3.5, 0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
    ax.text(-6.2, 1.6, "flight direction", fontsize=8, ha="center", color="#444")
    ax.axis("off")
    ax.set_xlim(-10, 35.5)
    ax.set_ylim(-16, 16)
    fig.suptitle("In the model, every follower saves induced drag; the leader saves none", x=0.01, y=0.98, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, 0.02, "Number in circle: model reduction of induced drag (%), 9-drone V, vortex core 0.075 span.\nModel values are higher than measured ones; sideways spacing stretched.",
             fontsize=7.5, color="#444")
    save(fig, "figA1_formation")


if __name__ == "__main__":
    N = compute_numbers()
    json.dump(N, open(os.path.join(FIG, "numbers.json"), "w"), indent=1, default=float)
    fig0()
    fig1(N)
    fig2a(N)
    fig2b(N)
    fig2c(N)
    fig3(N)
    fig4(N)
    figA1()
