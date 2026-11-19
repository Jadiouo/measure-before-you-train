"""Web-page figures (zh + en), redrawn from the paper figures: bigger text, fewer elements, one message each.

Reads ONLY calib/out/*.json, through scripts/make_paper_figs.compute_numbers() (which asserts the audited values);
then cross-checks every scalar used here against paper/figures/numbers.json.  Nothing is typed in by hand except
the schematic panels of the concept figure (marked "schematic" / "illustrative" on the figure itself).
Writes web/portfolio/figs/{zh,en}/*.png.
Run:  .venv/bin/python scripts/make_web_figs.py
"""
import json
import math
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch, Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_paper_figs as P  # noqa: E402  (only its load/compute_numbers/timed_out are used)

ROOT = P.ROOT
OUTDIR = os.path.join(ROOT, "web/portfolio/figs")
BLUE, VERM, ORANGE, SKY, GREEN, GREY, BLACK = P.BLUE, P.VERM, P.ORANGE, P.SKY, P.GREEN, P.GREY, P.BLACK
INK, MUTED = "#1d2127", "#4a525c"
FS, FS_S, FS_T = 12.5, 11.5, 14      # body / small / title font sizes (figures are 6 in wide; shown at about 340-700 px)

N = P.compute_numbers()
NJ = json.load(open(os.path.join(ROOT, "paper/figures/numbers.json")))
for k, v in N.items():                          # figures must agree with the audited paper numbers
    if isinstance(v, (int, float)) and not isinstance(v, bool) and k in NJ and isinstance(NJ[k], (int, float)):
        assert math.isclose(v, NJ[k], rel_tol=1e-9, abs_tol=1e-9), (k, v, NJ[k])

TXT = {
    "zh": dict(
        fam="Noto Sans CJK TC",
        # concept
        c1="1  領頭的最累", c1a="領頭", c1b="跟機", c1c="跟機借前機的尾流，省電；\n領頭的沒有。電量最先見底。",
        c2="2  用規則決定誰來領頭", c2a="定時輪流", c2a2="每隔一段時間換人", c2b="看電量", c2b2="把領頭交給\n電量最多的那架",
        c2c="換人能分攤負擔，但換位要花能量。",
        c3="3  簡單規則離最佳做法有多遠？", c3a="事後看來的最佳做法（上界）", c3b="簡單規則", c3c="進步空間", c3d="2% 線",
        c3e="差距小於 2%，就判斷不值得訓練強化學習。",
        cnote="示意圖，不是資料（第 3 列的橫軸只放大 90% 到 100%）",
        batt="電量（示意）",
        # steps
        s1="先寫下預測", s1a="把「我預期差距多大」\n存進版本紀錄，之後才量。",
        s2="再量差距", s2a="在一整格情境裡，量簡單規則\n離最佳做法有多遠。",
        s3="學習者先過資格考", s3a="在已知有進步空間的情境，\n追回一半以上，成績才採信。",
        # prediction
        p1="我們預測", p1a="約 30% 的情境\n差距超過 2%\n（約 230 個，由 576 格外推）", p2="實際量到", p2a="只有 1 個\n差距超過 2%",
        p3="一個方格 = 一個輪流情境（共 768 個，位置沒有意義）", p4="這一個",
        # 2a
        a1="定時輪流（基本模型）", a2="看電量換領頭（基本模型）", a3="五種規則挑最好（較複雜模型）",
        a_x="離最佳做法的差距（%）　黑色短線 = 中位數", a_line="2% 線", a_cnt="{k} / {n} 超過",
        # 2b
        b1="領機不動或隨機漂移", b2="領機左右蛇行", b3="領機突然移位", bx="離參考的差距（佔總功率的百分點）",
        b_ref1="完美位置參考（我們預測用的）", b_ref2="同樣雜訊 GPS 的參考（診斷用）", b_line="2 pp 線", b_set="{n} 格",
        # 2c
        sx="領先幅度（佔高度上限的百分點）", sl1="對簡單規則", sl2="對「拿到同樣真實資料」的簡單規則",
        s_dens={"sparse": "稀疏", "mid": "中等"}, s_life={"short": "短命", "long": "長壽"}, s_w={"W3.0": "強", "W1.5": "弱"},
        s_lab="熱氣流{d}、{l}、{w}", s_line="2 pp 線",
        # 3
        l1="看得到全隊的規則", l2="只看鄰居的最好規則", ly="離最佳做法的差距（%）", l_cnt="{k} / {n}\n超過 2%", l_line="2% 線",
        # 4
        g_cells=["換位成本高\n雜訊變化快", "換位成本低\n雜訊變化慢", "換位成本高\n雜訊變化慢"],
        gy="追回已知差距的比例\n0 = 不比只看鄰居的規則好\n1 = 和看全隊的規則一樣好",
        g_pass="及格線 0.5", g_zone="及格區", g_ok="通過 {f:.2f}", g_col="幾乎不換位\n而崩潰", gx="訓練 2000 次更新後；點 = 5 次訓練，圓圈與直線 = 平均與 95% 區間",
        # birds
        bd1="定時輪流", bd2="MAPPO 學習者", bd3="看能量的規則", bdx="標準化成績",
        # formation
        f_dir="飛行方向", f_note="圓圈內數字 = 模型算出的誘導阻力降幅（%），9 架 V 形。\n模型值比實飛量到的高；橫向間距被拉寬以便閱讀。",
    ),
    "en": dict(
        fam="Noto Sans CJK TC",
        c1="1  The leader works hardest", c1a="leader", c1b="followers", c1c="Followers ride the leader's wake and save\npower. The leader does not. Its battery empties first.",
        c2="2  A rule picks the next leader", c2a="Time rule", c2a2="swap the lead\nevery so often", c2b="Battery rule", c2b2="give the lead to the\nfullest battery",
        c2c="Swapping spreads the load, but a swap costs energy.",
        c3="3  How far from the best plan?", c3a="best plan in hindsight (upper bound)", c3b="simple rule", c3c="headroom", c3d="2% line",
        c3e="Under 2%: not worth training a learned policy.",
        cnote="Illustration, not data (row 3 axis zoomed to 90-100%)",
        batt="battery (schematic)",
        s1="Write the prediction", s1a="Commit \"how big I expect the gap\"\nto version control, then measure.",
        s2="Measure the gap", s2a="Across a grid of settings: how far is\na simple rule from the best plan?",
        s3="Learner passes a check first", s3a="It must recover at least half of a known\ngap before its results are read.",
        p1="What we predicted", p1a="about 30% of settings\nwith a gap above 2%\n(about 230, scaled up from 576)", p2="What we found", p2a="only 1 setting\nwith a gap above 2%",
        p3="One square = one rotation setting (768 in all; positions are arbitrary)", p4="the one",
        a1="Time rule (basic model)", a2="Battery rule (basic model)", a3="Best of 5 rules (richer model)",
        a_x="Gap to the best plan (%)   black tick = median", a_line="2% line", a_cnt="{k} of {n} beyond",
        b1="Leader straight or drifting", b2="Leader weaves", b3="Leader steps sideways", bx="Gap to a reference (pp of total power)",
        b_ref1="perfect-position reference (our prediction)", b_ref2="same-noisy-GPS reference (diagnostic)", b_line="2 pp line", b_set="{n} settings",
        sx="Lead of the strongest method (pp of altitude ceiling)", sl1="vs the simple rule", sl2="vs the simple rule given the same true data",
        s_dens={"sparse": "sparse", "mid": "medium"}, s_life={"short": "short-lived", "long": "long-lived"}, s_w={"W3.0": "strong", "W1.5": "weak"},
        s_lab="{d} thermals, {l}, {w}", s_line="2 pp line",
        l1="Rule that sees\nthe whole team", l2="Best rule that sees\nonly neighbours", ly="Gap to the best plan (%)", l_cnt="{k} of {n}\nabove 2%", l_line="2% line",
        g_cells=["high swap cost,\nfast noise", "low swap cost,\nslow noise", "high swap cost,\nslow noise"],
        gy="Share of the known gap recovered\n0 = no better than the neighbours-only rule\n1 = as good as the whole-team rule",
        g_pass="pass line 0.5", g_zone="pass zone", g_ok="passed {f:.2f}", g_col="almost never\nswaps: collapsed", gx="After 2000 training updates; dots = 5 runs, ring and bar = mean and 95% interval",
        bd1="Time rule", bd2="MAPPO learner", bd3="Energy-aware rule", bdx="Normalised score",
        f_dir="direction", f_note="Number in circle: model's reduction of induced drag (%), 9-drone V.\nModel values are higher than flight tests; sideways spacing stretched.",
    ),
}


def setup(lang):
    plt.rcParams.update({
        "font.family": [TXT[lang]["fam"], "DejaVu Sans"], "font.size": FS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "figure.dpi": 100, "savefig.dpi": 170, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#888", "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, lang, name):
    d = os.path.join(OUTDIR, lang)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".png"), bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print("wrote", lang, name)


def box(ax, x, y, w, h, fc="#fafafa", ec="#d4d4d4", lw=1.0, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def plane(ax, x, y, s, color, ang=90):
    pts = np.array([[0, 1.0], [0.75, -0.7], [0, -0.25], [-0.75, -0.7]]) * s
    t = np.deg2rad(ang - 90)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ax.add_patch(Polygon(pts @ R.T + [x, y], closed=True, fc=color, ec="white", lw=0.7, zorder=5))


def vline(ax, x, label, ytxt, **kw):
    ax.axvline(x, color=BLACK, ls=(0, (4, 3)), lw=1.3, zorder=2)
    ax.text(x + 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ytxt, label, fontsize=FS_S, va="bottom", ha="left", **kw)


# ----------------------------------------------------------------------------- concept
def fig_concept(lang):
    T = TXT[lang]
    fig, ax = plt.subplots(figsize=(5.5, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 138)
    ax.axis("off")
    H, G = 42, 6
    ys = [138 - H, 138 - 2 * H - G, 138 - 3 * H - 2 * G]
    for y in ys:
        box(ax, 0, y, 100, H)
    for y, t in zip(ys, (T["c1"], T["c2"], T["c3"])):
        ax.text(3, y + H - 5.2, t, fontsize=FS + 1, fontweight="bold", va="center")
    for y in ys[:2]:
        ax.add_patch(FancyArrowPatch((50, y - 0.3), (50, y - G + 0.3), arrowstyle="-|>", mutation_scale=16, color=GREY, lw=1.8, zorder=6))
    # row 1: planes + batteries
    y0 = ys[0]
    cx, cy = 39, y0 + 22
    ax.add_patch(Polygon([[cx - 1.8, cy + 6.0], [cx + 1.8, cy + 6.0], [cx + 11, cy - 8], [cx - 11, cy - 8]], closed=True, fc=SKY, alpha=0.28, ec="none", zorder=3))
    plane(ax, cx, cy + 7.6, 3.5, VERM)
    for dx, dy in ((-8, 0.5), (8, 0.5), (-15, -6.8), (15, -6.8)):
        plane(ax, cx + dx, cy + dy, 3.5, BLUE)
    ax.text(cx + 5.5, cy + 9.4, T["c1a"], fontsize=FS_S, color=VERM, fontweight="bold", va="center")
    ax.text(cx - 20, cy - 6.8, T["c1b"], fontsize=FS_S, color=BLUE, fontweight="bold", va="center", ha="right")
    for i, lv in enumerate([0.40, 0.72, 0.80, 0.86, 0.92]):
        x = 62 + i * 7.0
        ax.add_patch(Rectangle((x, y0 + 15), 5.4, 17, fc="white", ec=GREY, lw=1.2, zorder=4))
        ax.add_patch(Rectangle((x, y0 + 15), 5.4, 17 * lv, fc=VERM if i == 0 else BLUE, ec="none", zorder=5))
    ax.text(62 + 2 * 7.0 + 2.7, y0 + 12.2, T["batt"], fontsize=FS_S - 0.5, color=MUTED, ha="center", va="center")
    ax.text(3, y0 + 5.4, T["c1c"], fontsize=FS_S, va="center", linespacing=1.35)
    # row 2: two rule chips
    y0 = ys[1]
    for j, (h1, h2) in enumerate(((T["c2a"], T["c2a2"]), (T["c2b"], T["c2b2"]))):
        xx = 3 + j * 48.5
        box(ax, xx, y0 + 13, 45.5, 17, fc="white", ec=BLUE, lw=1.8, r=1.5, z=3)
        ax.text(xx + 2.2, y0 + 26, h1, fontsize=FS, fontweight="bold", color=BLUE, va="center", zorder=4)
        ax.text(xx + 2.2, y0 + 18.2, h2, fontsize=FS_S - 0.5, va="center", zorder=4, linespacing=1.25)
    ax.text(3, y0 + 5.2, T["c2c"], fontsize=FS_S, va="center")
    # row 3: bars
    y0 = ys[2]
    qx, L = 4, 92
    xv = lambda v: qx + (v - 0.90) / 0.10 * L
    ax.text(qx, y0 + 29.4, T["c3a"], fontsize=FS_S, va="center")
    ax.add_patch(Rectangle((qx, y0 + 22.2), L, 4.2, fc=GREEN, ec="none", zorder=4))
    ax.text(qx, y0 + 18.4, T["c3b"], fontsize=FS_S, va="center")
    ax.add_patch(Rectangle((qx, y0 + 12.2), xv(0.985) - qx, 4.2, fc=BLUE, ec="none", zorder=4))
    ax.add_patch(Rectangle((xv(0.985), y0 + 12.2), xv(1.0) - xv(0.985), 4.2, fc="none", ec=VERM, lw=1.6, hatch="////", zorder=5))
    ax.plot([xv(0.98)] * 2, [y0 + 10.4, y0 + 27.6], color=BLACK, lw=1.4, ls=(0, (3, 2)), zorder=6)
    ax.text(xv(0.98) - 4.5, y0 + 8.2, T["c3d"], fontsize=FS_S, ha="right", va="center")
    ax.text(xv(1.0), y0 + 8.2, T["c3c"], fontsize=FS_S, color=VERM, fontweight="bold", ha="right", va="center")
    ax.text(3, y0 + 2.6, T["c3e"], fontsize=FS_S, va="center")
    ax.text(50, -2.8, T["cnote"], fontsize=FS_S - 1, color=MUTED, ha="center", va="center")
    save(fig, lang, "fig0_concept")


# ----------------------------------------------------------------------------- steps
def fig_steps(lang):
    T = TXT[lang]
    fig, ax = plt.subplots(figsize=(5.0, 3.83))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    H, gap = 27, 9.5
    for i, (a, b) in enumerate(((T["s1"], T["s1a"]), (T["s2"], T["s2a"]), (T["s3"], T["s3a"]))):
        y = 100 - H - i * (H + gap)
        box(ax, 0, y, 100, H, fc="#fafafa")
        ax.add_patch(Circle((8.5, y + H / 2), 5.2, fc=BLUE, ec="none", zorder=3))
        ax.text(8.5, y + H / 2 - 0.2, str(i + 1), color="white", fontsize=FS + 4, fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(17, y + H - 6.4, a, fontsize=FS + 0.5, fontweight="bold", va="center")
        ax.text(17, y + 9.0, b, fontsize=FS_S - 1, va="center", linespacing=1.4, color=INK)
        if i < 2:
            ax.add_patch(FancyArrowPatch((50, y - 0.4), (50, y - gap + 0.4), arrowstyle="-|>", mutation_scale=16, color=GREY, lw=1.8, zorder=6))
    save(fig, lang, "fig_steps")


# ----------------------------------------------------------------------------- prediction waffle
def fig_pred(lang):
    T = TXT[lang]
    nrow, ncol = 24, 32
    total = nrow * ncol
    assert total == N["L1_cells"] and N["L1_best_gt2"] == 1
    npred = round(N["pred_frac_L1"] * total)
    assert 229 <= npred <= 231
    hit_obs = 17 * ncol + 22
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 3.25))
    for ax, k, title, sub in ((axs[0], npred, T["p1"], T["p1a"]), (axs[1], 1, T["p2"], T["p2a"])):
        ax.set_xlim(0, ncol)
        ax.set_ylim(nrow, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        for i in range(total):
            r, c = divmod(i, ncol)
            hit = i < k if ax is axs[0] else i == hit_obs
            ax.add_patch(Rectangle((c + 0.07, r + 0.07), 0.86, 0.86, fc=VERM if hit else "#e3e3e3", ec="none"))
        nl = 3
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points", fontsize=FS_S, va="bottom", ha="left", linespacing=1.3)
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 12 + nl * FS_S * 1.4), textcoords="offset points",
                    fontsize=FS + 1.5, fontweight="bold", va="bottom", ha="left")
    r, c = divmod(hit_obs, ncol)
    axs[1].annotate(T["p4"], xy=(c + 0.5, r + 1.0), xytext=(c - 5.5, r + 5.3), fontsize=FS_S, color=VERM, fontweight="bold", ha="center", va="top",
                    arrowprops=dict(arrowstyle="-|>", color=VERM, lw=1.5))
    fig.text(0.5, 0.0, T["p3"], fontsize=FS_S - 0.5, color=MUTED, ha="center", va="top")
    fig.subplots_adjust(top=0.64, bottom=0.04, wspace=0.08, left=0.01, right=0.99)
    save(fig, lang, "fig1_prediction")


# ----------------------------------------------------------------------------- strip helpers (rows with a label above each)
def strip_axes(figsize, xlim, ylim, xlabel):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=FS_S)
    ax.tick_params(axis="x", labelsize=FS_S)
    return fig, ax


def fig_rotation(lang):
    T = TXT[lang]
    b0 = P.load("drone_bucket")["cells"]
    cells1 = [c for j in P.load("drone_bucket_l1")["jobs"] for c in j["cells"]]
    rows = [(T["a1"], [100 * c["gap_tail"] for c in b0], ORANGE, N["L0_cells"]),
            (T["a2"], [100 * c["gap_gate"] for c in b0], BLUE, N["L0_cells"]),
            (T["a3"], [100 * c["best_gap"] for c in cells1], BLUE, N["L1_cells"])]
    assert [sum(v > 2 for v in r[1]) for r in rows] == [N["L0_timerot_gt2"], N["L0_battery_gt2"], N["L1_best_gt2"]]
    fig, ax = strip_axes((5.0, 3.33), (-0.3, 13), (-0.55, 3.0), T["a_x"])
    rng = np.random.default_rng(0)
    for i, (lab, vals, col, n) in enumerate(rows):
        y = 2 - i
        vals = np.array(vals)
        ax.scatter(vals, y + rng.uniform(-0.25, 0.22, len(vals)), s=14, color=col, alpha=0.5, lw=0, zorder=3)
        ax.plot([np.median(vals)] * 2, [y - 0.3, y + 0.28], color=BLACK, lw=2.6, zorder=4)
        ax.text(-0.2, y + 0.46, lab, fontsize=FS, fontweight="bold", va="center", bbox=dict(fc="white", ec="none", pad=1.5), zorder=7)
        k = int((vals > 2).sum())
        ax.text(12.9, y + 0.44, T["a_cnt"].format(k=k, n=n), fontsize=FS_S, va="center", ha="right", color=VERM if k else MUTED, fontweight="bold")
    ax.axvline(2, color=BLACK, ls=(0, (4, 3)), lw=1.3, zorder=2)
    ax.text(2.2, -0.42, T["a_line"], fontsize=FS_S, va="center", bbox=dict(fc="white", ec="none", pad=1.5), zorder=7)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    save(fig, lang, "fig2a_rotation")


def fig_wake(lang):
    T = TXT[lang]
    w = P.load("wake_headroom")["cells"]
    groups = [(T["b1"], [c for c in w if c["drift"].startswith("amp")]), (T["b2"], [c for c in w if c["drift"] == "sine"]),
              (T["b3"], [c for c in w if c["drift"] == "step"])]
    assert sum(len(g[1]) for g in groups) == N["wake_cells"]
    pitch = 1.45
    fig, ax = strip_axes((5.0, 3.83), (-0.4, 3.4), (-0.75, 2 * pitch + 0.8), T["bx"])
    for i, (lab, cs) in enumerate(groups):
        y = (2 - i) * pitch
        fair, perf = [c["gap_gpsprivileged"] for c in cs], [c["gap"] for c in cs]
        ax.plot([min(perf), max(perf)], [y + 0.02] * 2, color=GREY, lw=8, solid_capstyle="butt", alpha=0.75, zorder=3)
        ax.plot(statistics.median(perf), y + 0.02, "o", ms=8, color="white", mec="#444", mew=1.5, zorder=5)
        ax.plot([min(fair), max(fair)], [y - 0.3] * 2, color=BLUE, lw=8, solid_capstyle="butt", zorder=3)
        ax.plot(statistics.median(fair), y - 0.3, "o", ms=8, color="white", mec=BLUE, mew=1.8, zorder=5)
        ax.text(-0.35, y + 0.5, lab, fontsize=FS, fontweight="bold", va="center", bbox=dict(fc="white", ec="none", pad=1.5), zorder=7)
        nb = sum(v > 2 for v in perf)
        ax.text(3.35, y + 0.5, T["a_cnt"].format(k=nb, n=len(cs)), fontsize=FS_S, ha="right", va="center", color=VERM if nb else MUTED, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=7)
    assert [sum(c["gap"] > 2 for c in g[1]) for g in groups] == [0, 12, 4]
    ax.axvline(2, color=BLACK, ls=(0, (4, 3)), lw=1.3, zorder=2)
    ax.text(2.04, -0.55, T["b_line"], fontsize=FS_S, va="center")
    ax.set_xticks([0, 1, 2, 3])
    fig.legend(handles=[plt.Line2D([0], [0], color=GREY, lw=8, alpha=0.75, label=T["b_ref1"]),
                        plt.Line2D([0], [0], color=BLUE, lw=8, label=T["b_ref2"])], loc="upper center", ncol=1, frameon=False,
               fontsize=FS_S, bbox_to_anchor=(0.5, 1.13), handlelength=1.6)
    save(fig, lang, "fig2b_wake")


def fig_soaring(lang):
    T = TXT[lang]
    rows = sorted([r for r in N["soar_rows"] if r["N"] == 5 and not r["floor"]], key=lambda r: r["g_l2l1"][0])
    assert len(rows) == 4
    pitch = 1.4
    fig, ax = strip_axes((5.0, 4.33), (-3.5, 25), (-0.75, 3 * pitch + 0.85), T["sx"])
    for i, r in enumerate(rows):
        y = (3 - i) * pitch
        g, lo, hi = r["g_l2l1"]
        ax.plot([lo, hi], [y + 0.02] * 2, color=BLUE, lw=3.2, alpha=0.8, zorder=3)
        ax.plot(g, y + 0.02, "o", ms=9, color=BLUE, zorder=5)
        g2, lo2, hi2 = r["gap"]
        ax.plot([lo2, hi2], [y - 0.3] * 2, color=ORANGE, lw=3.2, alpha=0.9, zorder=3)
        ax.plot(g2, y - 0.3, "o", ms=9, mfc="white", mec=ORANGE, mew=2.2, zorder=5)
        d, l, wk = r["cell"].split("|")
        ax.text(2.6, y + 0.5, T["s_lab"].format(d=T["s_dens"][d], l=T["s_life"][l], w=T["s_w"][wk]), fontsize=FS - 0.5, fontweight="bold", va="center",
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=7)
    ax.axvline(2, color=BLACK, ls=(0, (4, 3)), lw=1.3, zorder=2)
    ax.text(2.3, -0.55, T["s_line"], fontsize=FS_S, va="center")
    ax.set_xticks([0, 5, 10, 15, 20])
    fig.legend(handles=[plt.Line2D([0], [0], color=BLUE, marker="o", ms=9, lw=3.2, label=T["sl1"]),
                        plt.Line2D([0], [0], color=ORANGE, marker="o", mfc="white", mew=2.2, ms=9, lw=3.2, label=T["sl2"])],
               loc="upper center", ncol=1, frameon=False, fontsize=FS_S, bbox_to_anchor=(0.5, 1.11), handlelength=2.0)
    save(fig, lang, "fig2c_soaring")


# ----------------------------------------------------------------------------- local gap slope chart
def fig_local(lang):
    T = TXT[lang]
    cells23 = [c for j in P.load("drone_bucket_l23")["jobs"] for c in j["cells"]]
    ok = [c for c in cells23 if not P.timed_out(c)]
    bad = [c for c in ok if c["rules"]["rot"]["gap"] > 0.02]
    assert len(bad) == N["L3_n"] == 23
    fig, ax = plt.subplots(figsize=(5.0, 3.83))
    gl_all, lc_all = [], []
    for c in bad:
        gl = 100 * c["best_global_gap"]
        lc = 100 * min(c["rules"]["rot"]["gap"], c["rules"]["local"]["gap"])
        gl_all.append(gl)
        lc_all.append(lc)
        ax.plot([0, 1], [gl, lc], color=BLUE, lw=2.0, alpha=0.5, marker="o", ms=5.5, zorder=3)
    kg, kl = sum(v > 2 for v in gl_all), sum(v > 2 for v in lc_all)
    assert (kg, kl) == (N["L3_global_gt2"], N["L3_local_gt2"]) == (0, 21)
    ax.axhline(2, color=BLACK, ls=(0, (4, 3)), lw=1.3, zorder=2)
    ax.text(0.5, 2.06, T["l_line"], fontsize=FS_S, va="bottom", ha="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 4.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([T["l1"], T["l2"]], fontsize=FS)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel(T["ly"], fontsize=FS_S)
    ax.tick_params(axis="y", labelsize=FS_S)
    ax.text(0, 4.35, T["l_cnt"].format(k=kg, n=len(bad)), fontsize=FS, color=BLUE, fontweight="bold", ha="center", va="center")
    ax.text(1, 4.35, T["l_cnt"].format(k=kl, n=len(bad)), fontsize=FS, color=BLUE, fontweight="bold", ha="center", va="center")
    save(fig, lang, "fig3_local_gap")


# ----------------------------------------------------------------------------- sanity check
def fig_gate(lang):
    T = TXT[lang]
    d = N["ctrl_v2"]["U2000"]
    assert d["n_pass"] == 1 and len(d["cells"]) == 3
    fig, ax = plt.subplots(figsize=(5.0, 3.92))
    rng = np.random.default_rng(1)
    ax.axhspan(0.5, 1.6, color=GREEN, alpha=0.09, lw=0, zorder=0)
    for xi, c in enumerate(d["cells"]):
        col = GREEN if c["passed"] else BLUE
        jit = rng.uniform(-0.13, 0.13, len(c["seeds"]))
        ax.scatter(xi + jit, c["seeds"], s=44, color=col, alpha=0.5, lw=0, zorder=3)
        ax.plot([xi, xi], c["ci"], color=col, lw=3.2, zorder=4)
        ax.plot(xi, c["F"], "o", ms=11, mfc="white", mec=col, mew=2.6, zorder=5)
        if c["passed"]:
            ax.text(xi + 0.2, c["F"] + 0.03, T["g_ok"].format(f=c["F"]), fontsize=FS, color=GREEN, fontweight="bold", ha="left", va="center")
        for sx, sv in zip(jit, c["seeds"]):
            if sv < -1:
                ax.annotate("", xy=(xi + sx, sv), xytext=(xi + sx + 0.3, sv), arrowprops=dict(arrowstyle="-", color=VERM, lw=1.0))
        if any(sv < -1 for sv in c["seeds"]):
            ax.text(xi + 0.33, min(c["seeds"]), T["g_col"], fontsize=FS_S - 0.5, color=VERM, ha="left", va="center", linespacing=1.25)
    ax.axhline(0.5, color=BLACK, ls=(0, (4, 3)), lw=1.3)
    ax.axhline(0.0, color=GREY, lw=1.0)
    ax.text(-0.45, 0.56, T["g_pass"], fontsize=FS_S, ha="left", va="bottom")
    ax.text(-0.45, 1.5, T["g_zone"], fontsize=FS_S, color=GREEN, fontweight="bold", va="center")
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(-2.2, 1.65)
    ax.set_xticks(range(3))
    ax.set_xticklabels(T["g_cells"], fontsize=FS_S)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=FS_S)
    ax.set_ylabel(T["gy"], fontsize=FS_S - 1, linespacing=1.35)
    ax.set_xlabel(T["gx"], fontsize=FS_S - 1, labelpad=10)
    save(fig, lang, "fig4_sanity_gate")


# ----------------------------------------------------------------------------- birds (other work)
def fig_birds(lang):
    T = TXT[lang]
    rows = [(T["bd1"], N["bird_rot30"], None), (T["bd2"], N["bird_A2_reward"], N["bird_A2_reward_ci"]), (T["bd3"], N["bird_gate30s"], None)]
    assert [round(r[1], 4) for r in rows] == [0.9875, 1.001, 1.0284] or [round(r[1], 4) for r in rows] == [0.9875, 1.0010, 1.0284]
    fig, ax = strip_axes((5.0, 2.25), (0.96, 1.05), (-0.6, 2.7), T["bdx"])
    for i, (lab, v, ci) in enumerate(rows):
        y = 2 - i
        col = BLUE if ci is not None else GREY
        if ci is not None:
            ax.plot([v - ci, v + ci], [y - 0.1] * 2, color=col, lw=3.2, zorder=3)
        ax.plot(v, y - 0.1, "o", ms=11, color=col if ci is not None else "white", mec=col, mew=2.2, zorder=5)
        ax.text(0.962, y + 0.35, lab, fontsize=FS, fontweight="bold", va="center")
        ax.text(1.048, y + 0.35, f"{v:.3f}" + (f" ± {ci:.3f}" if ci is not None else ""), fontsize=FS_S, va="center", ha="right", color=MUTED)
    ax.set_xticks([0.97, 1.0, 1.03])
    ax.set_xticklabels(["0.97", "1.00", "1.03"])
    save(fig, lang, "fig_birds")


# ----------------------------------------------------------------------------- formation (limits section)
def fig_formation(lang):
    T = TXT[lang]
    d = P.load("horseshoe_formation")["sweep"]["N9_V_optimal_lamb_oseen_rc0.075"]
    fig, ax = plt.subplots(figsize=(5.0, 3.25))
    cm = plt.get_cmap("Blues")
    reds = []
    for p in d["per_position"]:
        red = 1 - p["ratio_all"]
        reds.append(red)
        ax.scatter(p["x"], 4 * p["y"], s=1100, color=cm(0.12 + 0.88 * red), ec="#333", lw=1.0, zorder=3)
        ax.text(p["x"], 4 * p["y"], f"{100 * red:.0f}", ha="center", va="center", fontsize=FS_S, fontweight="bold", color="white" if red > 0.55 else "#111", zorder=4)
    assert abs(reds[0]) < 0.01 and len(reds) == 9     # the leader saves nothing in the model
    ax.annotate("", xy=(-9.5, 0), xytext=(-3.5, 0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))
    ax.text(-6.5, 2.2, T["f_dir"], fontsize=FS_S, ha="center", color=MUTED)
    ax.axis("off")
    ax.set_xlim(-10.5, 36)
    ax.set_ylim(-17, 17)
    fig.text(0.0, -0.01, T["f_note"], fontsize=FS_S - 1, color=MUTED, va="top", linespacing=1.35)
    save(fig, lang, "fig_formation")


if __name__ == "__main__":
    for lg in ("zh", "en"):
        setup(lg)
        for f in (fig_concept, fig_steps, fig_pred, fig_rotation, fig_wake, fig_soaring, fig_local, fig_gate, fig_birds, fig_formation):
            f(lg)
