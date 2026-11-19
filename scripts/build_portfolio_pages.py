"""Build web/portfolio/index.html (zh-Hant) and web/portfolio/en.html from ONE structure, so both pages carry the same
six stations, the same five figures and the same numbers.

Every number comes from paper/figures/numbers.json (written by scripts/make_paper_figs.py from calib/out/*.json) or from
the calib/out result files themselves; the few literals are listed in LITERALS with their source.  Figure geometry is
decided by the numbers (the 960-cell figure has exactly as many coloured cells as the data say, the 23 lines of the
neighbours-only figure are the 23 settings).  Both language versions of every text sit side by side as (zh, en) pairs.

Run:  python3 scripts/build_portfolio_pages.py
"""
import hashlib
import html
import json
import os
import re
import shutil
import statistics
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PF = os.path.join(ROOT, "web/portfolio")
OUT = os.path.join(ROOT, "calib/out")
REPO_URL = "https://github.com/Jadiouo/measure-before-you-train"
N = json.load(open(os.path.join(ROOT, "paper/figures/numbers.json")))


def load(name):
    return json.load(open(os.path.join(OUT, name + ".json")))


def timed_out(c):  # same rule as scripts/make_paper_figs.py
    return c["opt_timeouts"] > 0 or abs(c["opt"] - c["opt_incumbent"]) > 1e-9 * max(1.0, abs(c["opt"]))


# ---------------------------------------------------------------- numbers
LITERALS = dict(
    pred_gate_first=75, pred_gate_second=35,   # paper/main.tex tab:guess, "75% chance on the first try, 35% on the second"
    pred_l1_share=30,                          # paper/main.tex tab:guess, "About 30% of settings"
    pred_l0_within=3, pred_l0_low=2,           # paper/main.tex tab:guess, "stays within 2--3% of the best plan"
)
V = dict(N)
V.update(LITERALS)

c2 = N["ctrl_v2"]["U2000"]["cells"]
V.update(
    n_rot=N["L0_cells"] + N["L1_cells"],
    k_rot=N["L0_battery_gt2"] + N["L1_gate_gt2"],           # battery rule: 4 of 192 basic + 4 of 768 richer
    gate_pass_n=sum(1 for c in c2 if c["passed"]), gate_need=2, gate_n=len(c2),
    gate_first_pass=sum(v["n_pass"] for v in N["ctrl_v1"].values()),
    gate_first_n=len(c2),
    gate_second_pass=N["ctrl_v2"]["U2000"]["n_pass"],
)

# wake surfing (settings x drift); definitions as in paper/main.tex tab:guess
WK = load("wake_headroom")["cells"]
wk_little = [c["gap"] for c in WK if c["drift"] == "amp0" or c["drift"].startswith("amp0.05")]
wk_large = [c for c in WK if c["drift"].startswith("amp0.2")]
wk_lead = [c["gap"] for c in WK if c["drift"] in ("sine", "step")]
V.update(
    wake_little_max=max(wk_little),
    wake_large_max=max(c["gap"] for c in wk_large),
    wake_large_ref_max=max(max(c["priv"], c["priv_gps"]) for c in wk_large),
)

# per-setting data for the two data figures
b0 = load("drone_bucket")["cells"]
cells1 = [c for j in load("drone_bucket_l1")["jobs"] for c in j["cells"]]
over0 = [100 * c["gap_gate"] > 2 for c in b0]                    # battery rule, basic model
over1 = [100 * c["rules"]["gate"]["gap"] > 2 for c in cells1]    # battery rule, richer model
cells23 = [c for j in load("drone_bucket_l23")["jobs"] for c in j["cells"]]
ok23 = [c for c in cells23 if not timed_out(c)]
bad23 = [c for c in ok23 if c["rules"]["rot"]["gap"] > 0.02]
l3_pairs = [(100 * c["best_global_gap"], 100 * min(c["rules"]["rot"]["gap"], c["rules"]["local"]["gap"])) for c in bad23]

# guards: figures and sentences must agree with the paper's numbers
assert (len(over0), sum(over0), len(over1), sum(over1)) == (N["L0_cells"], N["L0_battery_gt2"], N["L1_cells"], N["L1_gate_gt2"])
assert V["n_rot"] == 960 and V["k_rot"] == 8
assert max(c["c"] for c in b0) == max(c["c"] for c, o in zip(b0, over0) if o) == min(c["c"] for c, o in zip(b0, over0) if o)  # all four at the highest swap cost
assert len(l3_pairs) == N["L3_n"] == 23
assert sum(g > 2 for g, _ in l3_pairs) == N["L3_global_gt2"] == 0 and sum(l > 2 for _, l in l3_pairs) == N["L3_local_gt2"] == 21
assert abs(statistics.median(l - g for g, l in l3_pairs) - N["L3_diff_median"]) < 1e-9 and round(N["L3_diff_median"], 2) == 1.41
assert (V["gate_pass_n"], V["gate_need"], V["gate_n"]) == (1, 2, 3) and V["gate_first_pass"] == 0 and V["gate_second_pass"] == 1
assert N["L1_best_gt2"] == 1 and N["L1_cells"] == 768 and N["L1_h15plus_gt2"] == 0 and N["L0_timerot_gt2"] == 132 and N["L0_cells"] == 192
assert round(V["wake_little_max"], 2) == 1.30 and round(V["wake_large_max"], 2) == 1.55 and round(V["wake_large_ref_max"], 1) == 3.5
assert min(wk_lead) > 0 and N["wake_perfect_gt2"] == sum(1 for g in wk_lead if g > 2) == 16   # settings above 2 pp are all lead-manoeuvre ones
assert all(c["gap"] <= 2 for c in WK if c["drift"] not in ("sine", "step"))
assert N["soar_l2l1_cilo_gt2"] >= 1
# the guess table and figure 5 (station 4): every number is one of the above
assert V["pred_l1_share"] == 30 and (V["pred_l0_low"], V["pred_l0_within"]) == (2, 3) and V["pred_gate_first"] == 75
assert V["gate_first_pass"] < V["gate_need"] and V["gate_second_pass"] < V["gate_need"]   # "neither try passed the gate"

# page count of the paper PDF (stated in the links)
try:
    _pg = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", os.path.join(PF, "paper.pdf")], capture_output=True, text=True, check=True).stdout)
    V["paper_pages"] = int(_pg.group(1))
except Exception:
    V["paper_pages"] = 19
assert V["paper_pages"] == 19
shutil.copyfile(os.path.join(ROOT, "paper/supplement_predictions.pdf"), os.path.join(PF, "supplement.pdf"))


V["n_within"] = V["n_rot"] - V["k_rot"]
assert V["n_within"] == 952 and V["n_within"] == len(over0) + len(over1) - sum(over0) - sum(over1)


def F(s):
    return s.format_map(V)


# ---------------------------------------------------------------- palette (the same values as style.css)
INK, BG, BLUE, ORANGE, GREEN = "#0E1A2B", "#F5F7F9", "#0057B8", "#E87722", "#1A936F"
PALE = GRID = "#B8C2CF"   # one pale tint of ink for cells and guide lines
ZH, EN = 0, 1


# ---------------------------------------------------------------- figures (inline SVG, one drawing language)
def esc(s):
    return html.escape(s, quote=True)


def hatch_def(i):
    return (f'<pattern id="{i}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="1.6" height="8" fill="{INK}"/></pattern>')


def svg(variant, w, h, aria, body, defs=""):
    d = f"<defs>{defs}</defs>" if defs else ""
    return (f'<svg class="fg {variant}" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(aria)}" focusable="false">{d}{body}</svg>')


def both(fn, lang):
    return fn(lang, "w") + fn(lang, "n")


def r1(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")


def fig_rotation(lang, v):
    s = [dict(rows=["領頭", "第 2 位", "第 3 位", "第 4 位", "第 5 位"], time="時間",
              aria="輪值圖：五架無人機 A 到 E 輪流領頭。橫軸是時間，縱軸是隊形裡的位置，最上列是領頭；每個時段換一架領頭，領頭的那一段加粗並畫成藍色。"),
         dict(rows=["Lead", "2nd", "3rd", "4th", "5th"], time="Time",
              aria="Rotation chart: five drones A to E take turns leading. Time runs across and position in the formation runs down; the top row is the lead. Each period a different drone leads, and the leading stretch is drawn thick and blue.")][lang]
    if v == "w":
        W, x0, x1, yt, gap, lt, rt = 532, 100, 500, 48, 80, 90, 512
    else:
        W, x0, x1, yt, gap, lt, rt = 328, 90, 296, 44, 56, 80, 308
    step = (x1 - x0) / 5
    ramp = 0.15 * step
    y = lambda r: yt + r * gap
    axis_y = y(4) + 42
    H = axis_y + 38
    names = "ABCDE"
    out = []
    out.append(f'<g stroke="{GRID}" stroke-width="1.5" stroke-dasharray="2 5">' + "".join(f'<path d="M{x0 + step * b:g} {yt - 16}V{y(4) + 10}"/>' for b in range(1, 5)) + "</g>")
    out.append("<g class=\"d\">" + "".join(f'<text x="0" y="{y(r) + 6}">{s["rows"][r]}</text>' for r in range(5)) + "</g>")
    paths = []
    for k in range(5):
        d = f"M{x0:g} {y(k)}"
        for sl in range(1, 5):
            xb = x0 + step * sl
            d += f"H{xb - ramp:g}L{xb + ramp:g} {y((k - sl) % 5)}"
        d += f"H{x1:g}"
        paths.append(f'<path d="{d}"/>')
    out.append(f'<g fill="none" stroke="{INK}" stroke-opacity=".55" stroke-width="2.5" stroke-linejoin="round">' + "".join(paths) + "</g>")
    lead, letters = [], []
    for sl in range(5):
        a = x0 if sl == 0 else x0 + step * sl + ramp
        b = x1 if sl == 4 else x0 + step * (sl + 1) - ramp
        lead.append(f'<path d="M{a:g} {y(0)}H{b:g}"/>')
        letters.append(f'<text x="{(a + b) / 2:g}" y="{y(0) - 14}">{names[sl]}</text>')
    out.append(f'<g fill="none" stroke-width="9" stroke="{BLUE}">' + "".join(lead) + "</g>")
    out.append('<g class="d" text-anchor="middle">' + "".join(letters) + "</g>")
    out.append('<g class="d" text-anchor="end">' + "".join(f'<text x="{lt}" y="{y(r) + 6}">{names[r]}</text>' for r in range(5)) + "</g>")
    out.append('<g class="d">' + "".join(f'<text x="{rt}" y="{y(r) + 6}">{names[(r + 4) % 5]}</text>' for r in range(5)) + "</g>")
    out.append(f'<path d="M{x0} {axis_y}H{x1}" stroke="{INK}" stroke-width="2"/>')
    out.append(f'<path d="' + "".join(f"M{x0 + step * i:g} {axis_y}v8" for i in range(6)) + f'" stroke="{INK}" stroke-width="2"/>')
    out.append(f'<text x="0" y="{axis_y + 30}">{s["time"]}</text>')
    out.append('<g class="t" text-anchor="middle">' + "".join(f'<text x="{x0 + step * i:g}" y="{axis_y + 30}">{i}</text>' for i in range(6)) + "</g>")
    return svg(v, W, H, s["aria"], "".join(out))


def fig_headroom(lang, v):
    s = [dict(best="最好成績", rule="簡單規則", line="2% 線", gap="進步空間", axis="成績（只畫最右邊一段）",
              aria="進步空間示意圖：上面一條橫條是最好成績，下面一條是簡單規則，兩條從同一個起點畫，只畫成績最右邊的一段。簡單規則的右端落在 2% 虛線的右邊，它到最好成績之間的那一小段用斜線網紋標為進步空間，所以差距在 2% 線以內；示意圖，不是資料。"),
         dict(best="Best plan", rule="Simple rule", line="2% line", gap="Headroom", axis="Score (only the right-hand end is drawn)",
              aria="Headroom schematic: the top bar is the best plan and the bar below it is the simple rule, both drawn from the same start and showing only the right-hand end of the score. The simple rule ends to the right of the dashed 2% line, and the short stretch between its end and the best plan is hatched and labelled headroom, so the gap is inside the 2% line. A schematic, not data.")][lang]
    hid = f"hh-{v}"
    if v == "w":
        W, H, bx, ex, tw = 532, 250, 110, 500, 44
        ytop, yrule, lab_best, lab_rule = 70, 130, (0, 94), (0, 154)
        gap_px = 26
        ln_top, ln_bot = 40, 186
        tl_y, br_y, bl_y, ax_y = 30, 198, 224, 244
    else:
        W, H, bx, ex, tw = 328, 262, 24, 320, 40
        ytop, yrule, lab_best, lab_rule = 52, 122, (0, 40), (0, 110)
        gap_px = 22
        ln_top, ln_bot = 30, 186
        tl_y, br_y, bl_y, ax_y = 18, 198, 224, 252
    bh = 34
    xr = ex - gap_px            # right end of the simple rule: to the right of the 2% line
    xl2 = ex - tw               # 2% line
    assert xr > xl2
    o = []
    o.append(f'<text x="{lab_best[0]}" y="{lab_best[1]}" class="d">{s["best"]}</text><text x="{lab_rule[0]}" y="{lab_rule[1]}" class="d">{s["rule"]}</text>')
    o.append(f'<rect x="{bx}" y="{ytop}" width="{ex - bx}" height="{bh}" fill="{INK}"/>')
    o.append(f'<rect x="{bx}" y="{yrule}" width="{xr - bx}" height="{bh}" fill="{BLUE}"/>')
    o.append(f'<rect x="{xr}" y="{yrule}" width="{gap_px}" height="{bh}" fill="url(#{hid})" stroke="{INK}" stroke-width="1.5"/>')
    o.append(f'<path d="M{xl2} {ln_top}V{ln_bot}" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{xl2}" y="{tl_y}" text-anchor="middle" class="d">{s["line"]}</text>')
    o.append(f'<path d="M{xr} {br_y - 6}V{br_y}H{ex}V{br_y - 6}" fill="none" stroke="{INK}" stroke-width="2"/>')
    o.append(f'<text x="{ex}" y="{bl_y}" text-anchor="end" class="d">{s["gap"]}</text>')
    # zigzag axis break at the common start of both bars
    for yy in (ytop, yrule):
        o.append(f'<path d="M{bx - 4} {yy - 4}l8 {bh // 2 + 4}l-8 {bh // 2 + 4}" fill="none" stroke="{BG}" stroke-width="5"/>'
                 f'<path d="M{bx - 4} {yy - 4}l8 {bh // 2 + 4}l-8 {bh // 2 + 4}" fill="none" stroke="{INK}" stroke-width="2"/>')
    o.append(f'<text x="{bx}" y="{ax_y}">{s["axis"]}</text>')
    return svg(v, W, H, s["aria"], "".join(o), hatch_def(hid))


def fig_cells(lang, v):
    k0, k1 = sum(over0), sum(over1)
    n0, n1 = len(over0), len(over1)
    tot, k = n0 + n1, k0 + k1
    s = [dict(b0=f"基本模型　{n0} 個情境", b1=f"較複雜模型　{n1} 個情境", over=f"超過 2% 線：{k} 個", under=f"沒超過：{tot - k} 個",
              aria=f"{tot} 個輪流領頭情境的格狀長帶，每格一個情境，上面一組是基本模型 {n0} 格，下面一組是較複雜模型 {n1} 格。看電量的規則離最好成績超過 2% 的情境上橘色，共 {k} 格（基本 {k0}、較複雜 {k1}），其餘 {tot - k} 格是淡色。"),
         dict(b0=f"Basic model, {n0} settings", b1=f"Richer model, {n1} settings", over=f"Beyond the 2% line: {k}", under=f"Not beyond: {tot - k}",
              aria=f"Band of {tot} leader-rotation settings, one cell each: the upper group is the basic model with {n0} cells and the lower group the richer model with {n1}. Settings where the battery rule is more than 2% from the best plan are orange, {k} in all ({k0} basic, {k1} richer); the other {tot - k} are pale.")][lang]
    # x: one long band, 96 cells per row across the content width (desktop); m: 64 per row (laptop); w: 48 per row at 532
    # (tablet); n: 32 per row at 328 (phone)
    cols, pitch, size, W = {"x": (96, 13.6, 11.6, 1306), "m": (64, 14, 12, 896), "w": (48, 11, 9, 532), "n": (32, 10, 8, 328)}[v]
    o = []

    def block(flags, label, y):
        o.append(f'<text x="0" y="{y + 16}" class="d">{label}</text>')
        y += 28
        assert len(flags) % cols == 0
        cells = []
        for i, f in enumerate(flags):
            cx, cy = (i % cols) * pitch, y + (i // cols) * pitch
            cells.append(f'<rect x="{cx:g}" y="{cy:g}" width="{size}" height="{size}" fill="{ORANGE if f else PALE}"/>')
        o.append("".join(cells))
        return y + (len(flags) // cols) * pitch

    y = block(over0, s["b0"], 0) + 28
    y = block(over1, s["b1"], y) + 30
    if v == "n":
        o.append(f'<rect x="0" y="{y:g}" width="{size}" height="{size}" fill="{ORANGE}"/><text x="{size + 10}" y="{y + 10:g}">{s["over"]}</text>')
        o.append(f'<rect x="0" y="{y + 28:g}" width="{size}" height="{size}" fill="{PALE}"/><text x="{size + 10}" y="{y + 38:g}">{s["under"]}</text>')
        H = y + 52
    else:
        o.append(f'<rect x="0" y="{y:g}" width="{size}" height="{size}" fill="{ORANGE}"/><text x="{size + 10}" y="{y + 10:g}">{s["over"]}</text>')
        o.append(f'<rect x="260" y="{y:g}" width="{size}" height="{size}" fill="{PALE}"/><text x="{260 + size + 10}" y="{y + 10:g}">{s["under"]}</text>')
        H = y + 24
    return svg(v, W, round(H), s["aria"], "".join(o))


def fig_slope(lang, v):
    gmax, gmin_ = max(g for g, _ in l3_pairs), min(g for g, _ in l3_pairs)
    n, ng, nl = len(l3_pairs), sum(g > 2 for g, _ in l3_pairs), sum(l > 2 for _, l in l3_pairs)
    s = [dict(team="看得到全隊", nb="只看鄰居", line="2% 線", cnt=lambda x: f"超過 2%：{x} / {n}", unit="與最好成績的差距 %",
              aria=f"{n} 個情境的斜線圖：每條線一個情境，左端是看得到全隊的規則離最好成績的差距，右端是只看鄰居的最好規則的差距。左端 {n} 個都在 2% 虛線以下（{ng} 個超過），右端有 {nl} 個在虛線以上。"),
         dict(team="Whole-team", nb="Neighbours-only", line="2% line", cnt=lambda x: f"Beyond 2%: {x} of {n}", unit="Gap to the best plan, %",
              aria=f"Slope chart of {n} settings: each line is one setting, from the gap of the whole-team rule to the best plan on the left to the gap of the best neighbours-only rule on the right. All {n} left ends are below the dashed 2% line ({ng} beyond it); {nl} of the right ends are above it.")][lang]
    if v == "w":
        W, y0, sc, xl, xr, ax = 532, 276, 60, 170, 400, 36
        xend, lab_y = 500, 306
    else:
        W, y0, sc, xl, xr, ax = 328, 244, 50, 106, 244, 30
        xend, lab_y = 320, 274
    Y = lambda g: y0 - g * sc
    o = []
    o.append(f'<g stroke="{GRID}" stroke-width="1.5" stroke-dasharray="2 5">' + "".join(f'<path d="M{ax + 8} {Y(g):g}H{xend}"/>' for g in (1, 3, 4)) + "</g>")
    o.append(f'<path d="M{ax + 8} {Y(0):g}H{xend}" stroke="{INK}" stroke-width="2"/>')
    o.append(f'<path d="M{ax + 8} {Y(2):g}H{xend}" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>')
    o.append(f'<text x="{xend}" y="{Y(2) - 8:g}" text-anchor="end" class="d">{s["line"]}</text>')
    o.append('<g class="t" text-anchor="end">' + "".join(f'<text x="{ax}" y="{Y(g) + 5:g}">{g}</text>' for g in range(0, 5)) + "</g>")
    o.append(f'<g fill="none" stroke="{ORANGE}" stroke-width="2" stroke-opacity=".75">' + "".join(f'<path d="M{xl} {Y(g):.1f}L{xr} {Y(l):.1f}"/>' for g, l in l3_pairs) + "</g>")
    o.append(f'<g fill="{ORANGE}">' + "".join(f'<circle cx="{xl}" cy="{Y(g):.1f}" r="4"/><circle cx="{xr}" cy="{Y(l):.1f}" r="4"/>' for g, l in l3_pairs) + "</g>")
    o.append(f'<g class="d" text-anchor="middle"><text x="{xl}" y="{lab_y}">{s["team"]}</text><text x="{xr}" y="{lab_y}">{s["nb"]}</text></g>')
    o.append(f'<g text-anchor="middle"><text x="{xl}" y="{lab_y + 24}">{s["cnt"](ng)}</text><text x="{xr}" y="{lab_y + 24}">{s["cnt"](nl)}</text></g>')
    o.append(f'<text x="0" y="{Y(4) - 14:g}" class="d">{s["unit"]}</text>')
    H = lab_y + 36
    return svg(v, W, H, s["aria"], "".join(o))


def fig_miss(lang, v):
    """Figure 5: the biggest miss. Two equal bars on one 0-100% axis: what was guessed (30%) and what happened (1 of 768)."""
    guess, k, n = V["pred_l1_share"], V["L1_best_gt2"], V["L1_cells"]
    assert (guess, k, n) == (30, 1, 768)
    s = [dict(a="當初猜", b="實際", wide="橘線已加粗才看得見", lead=f"{n} 個裡 {k} 個",
              aria=f"猜錯最大的一項：同一條 0 到 100% 的橫軸上有上下兩條等長的橫條。上面一條是當初猜，藍色從 0 填到 {guess}%；下面一條是實際，只有最左端一條橘色細線，是 {n} 個裡 {k} 個（五條規則取最好的）。橘線已加粗才看得見。"),
         dict(a="Guessed", b="Actual", wide="Orange mark widened to stay visible", lead=f"{k} of {n}",
              aria=f"The biggest miss: two equal bars on one 0 to 100% axis. The top bar, guessed, is filled blue from 0 to {guess}%. The bottom bar, actual, has only a thin orange line at its left end, {k} of {n} settings (the best of five rules). The orange mark is widened to stay visible.")][lang]
    bh, line_w = 34, 3
    if v == "w":
        W, bx, ex = 532, 100, 500
        ya, yb = 52, 126            # top of the two bars
        la, lb = (0, ya + 24), (0, yb + 24)
    else:
        W, bx, ex = 328, 14, 314
        ya, yb = 52, 138
        la, lb = (0, ya - 12), (0, yb - 12)
    L = ex - bx
    xg = bx + L * guess / 100
    axis_y = yb + bh + 56
    o = []
    o.append(f'<text x="{la[0]}" y="{la[1]}" class="d">{s["a"]}</text><text x="{lb[0]}" y="{lb[1]}" class="d">{s["b"]}</text>')
    for y in (ya, yb):
        o.append(f'<rect x="{bx}" y="{y}" width="{L}" height="{bh}" fill="{PALE}"/>')
    o.append(f'<rect x="{bx}" y="{ya}" width="{xg - bx:g}" height="{bh}" fill="{BLUE}"/>')
    o.append(f'<rect x="{bx}" y="{yb}" width="{line_w}" height="{bh}" fill="{ORANGE}"/>')
    # dashed guide at the guessed value, through both bars (it stops below them so that it never crosses the leader's label)
    o.append(f'<path d="M{xg:g} {ya - 10}V{yb + bh + 8}" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5" fill="none"/>')
    # leader from the orange line to its label
    lx = bx + line_w / 2
    o.append(f'<path d="M{lx:g} {yb + bh + 2}V{yb + bh + 22}H{lx + 30:g}" stroke="{INK}" stroke-width="2" fill="none"/>')
    o.append(f'<text x="{lx + 38:g}" y="{yb + bh + 28}" class="d">{s["lead"]}</text>')
    # axis: 0, 30%, 100%
    o.append(f'<path d="M{bx} {axis_y}H{ex}" stroke="{INK}" stroke-width="2"/>')
    ticks = ((bx, "0", "middle"), (xg, f"{guess}%", "middle"), (ex, "100%", "end"))
    o.append(f'<path d="' + "".join(f"M{x:g} {axis_y}v8" for x, _, _ in ticks) + f'" stroke="{INK}" stroke-width="2"/>')
    o.append('<g class="t">' + "".join(f'<text x="{x:g}" y="{axis_y + 30}" text-anchor="{a}">{t}</text>' for x, t, a in ticks) + "</g>")
    o.append(f'<text x="0" y="{axis_y + 62}">{s["wide"]}</text>')
    return svg(v, W, axis_y + 70, s["aria"], "".join(o))


def fig_gate(lang, v):
    p, need, n = V["gate_pass_n"], V["gate_need"], V["gate_n"]
    s = [dict(a="三個對照情境", d1="通過至少", d2=f"{need} 個？", yes="是", no="否", y1="去測", y2="正式題目", n1="結果", n2="不採計", took="實際走到", skip="沒走到", pas=f"過 {p} 個",
              aria=f"資格考流程圖：三個對照情境中，通過的只有 {p} 個（一個實心圓、{n - p} 個空心圓）。判斷「通過至少 {need} 個？」答案是否，走到實心橘色的「結果不採計」；「是」那一支通往「去測正式題目」，沒有走到，用墨色網紋與虛線框表示。"),
         dict(a="3 control settings", d1="Passed at", d2=f"least {need}?", yes="Yes", no="No", y1="Test the", y2="real problem", n1="Results", n2="not counted", took="Taken", skip="Not reached", pas=f"Passed {p}",
              aria=f"Sanity gate flow: of three control settings, {p} was passed (one filled dot, {n - p} hollow dots). The test 'passed at least {need}?' is answered no, which leads to the solid orange box 'results not counted'. The yes branch leads to 'test the real problem', was not reached, and is drawn with an ink hatch and a dashed frame.")][lang]
    hid = f"hg-{v}"
    o = []

    def band(x, y, w, h):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="url(#{hid})" stroke="{INK}" stroke-width="1.5"/>'

    def box(x, y, w, h, dashed=False, fill="none", stroke=INK):
        da = ' stroke-dasharray="7 5"' if dashed else ""
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="2"{da}/>'

    def dots(x, y):
        return (f'<circle cx="{x}" cy="{y}" r="12" fill="{INK}"/>'
                + "".join(f'<circle cx="{x + 40 * i}" cy="{y}" r="10" fill="{BG}" stroke="{INK}" stroke-width="3.5"/>' for i in range(1, n)))

    if v == "w":
        W, H = 532, 340
        o.append(box(1, 110, 176, 110, fill=BG))
        o.append(f'<text x="14" y="138" class="d">{s["a"]}</text>')
        o.append(dots(34, 172))
        o.append(f'<text x="14" y="208">{s["pas"]}</text>')
        o.append(f'<path d="M177 165H214" stroke="{ORANGE}" stroke-width="6" fill="none"/><path d="M212 156l14 9l-14 9z" fill="{ORANGE}"/>')
        o.append(box(227, 125, 132, 80, fill=BG))
        o.append(f'<text x="293" y="160" text-anchor="middle" class="d">{s["d1"]}</text><text x="293" y="182" text-anchor="middle" class="d">{s["d2"]}</text>')
        o.append(f'<path d="M359 165H384V262H400" stroke="{ORANGE}" stroke-width="6" fill="none"/><path d="M398 253l14 9l-14 9z" fill="{ORANGE}"/>')
        o.append(band(379, 55, 10, 110) + band(379, 55, 34, 10))
        o.append(f'<text x="370" y="232" class="d" text-anchor="end">{s["no"]}</text><text x="370" y="112" class="d" text-anchor="end">{s["yes"]}</text>')
        o.append(f'<rect x="413" y="230" width="118" height="64" fill="{ORANGE}"/>')
        o.append(f'<text x="472" y="258" text-anchor="middle" class="d">{s["n1"]}</text><text x="472" y="280" text-anchor="middle" class="d">{s["n2"]}</text>')
        o.append(box(413, 28, 118, 64, dashed=True, fill=BG))
        o.append(f'<text x="472" y="56" text-anchor="middle" class="d">{s["y1"]}</text><text x="472" y="78" text-anchor="middle" class="d">{s["y2"]}</text>')
        ly = 326
        o.append(f'<path d="M0 {ly - 5}H34" stroke="{ORANGE}" stroke-width="6"/><text x="44" y="{ly}">{s["took"]}</text>')
        o.append(band(230, ly - 11, 34, 12) + f'<text x="274" y="{ly}">{s["skip"]}</text>')
    else:
        W, H = 328, 440
        o.append(box(41, 8, 246, 110, fill=BG))
        o.append(f'<text x="164" y="36" text-anchor="middle" class="d">{s["a"]}</text>')
        o.append(dots(110, 70))
        o.append(f'<text x="164" y="106" text-anchor="middle">{s["pas"]}</text>')
        o.append(f'<path d="M164 118V150" stroke="{ORANGE}" stroke-width="6" fill="none"/><path d="M155 148l9 14l9 -14z" fill="{ORANGE}"/>')
        o.append(box(84, 162, 160, 64, fill=BG))
        o.append(f'<text x="164" y="200" text-anchor="middle" class="d">{s["d1"]} {s["d2"]}</text>')
        o.append(f'<path d="M164 226V262H70V288" stroke="{ORANGE}" stroke-width="6" fill="none"/><path d="M61 286l9 14l9 -14z" fill="{ORANGE}"/>')
        o.append(band(164, 257, 94, 10) + band(253, 257, 10, 45))
        o.append(f'<text x="112" y="250" class="d" text-anchor="middle">{s["no"]}</text><text x="212" y="250" class="d" text-anchor="middle">{s["yes"]}</text>')
        o.append(f'<rect x="0" y="302" width="150" height="64" fill="{ORANGE}"/>')
        o.append(f'<text x="75" y="330" text-anchor="middle" class="d">{s["n1"]}</text><text x="75" y="352" text-anchor="middle" class="d">{s["n2"]}</text>')
        o.append(box(178, 302, 150, 64, dashed=True, fill=BG))
        o.append(f'<text x="253" y="330" text-anchor="middle" class="d">{s["y1"]}</text><text x="253" y="352" text-anchor="middle" class="d">{s["y2"]}</text>')
        ly = 396
        o.append(f'<path d="M0 {ly - 5}H34" stroke="{ORANGE}" stroke-width="6"/><text x="44" y="{ly}">{s["took"]}</text>')
        o.append(band(0, ly + 14, 34, 12) + f'<text x="44" y="{ly + 26}">{s["skip"]}</text>')
        H = 436
    return svg(v, W, H, s["aria"], "".join(o), hatch_def(hid))


# ---------------------------------------------------------------- text (zh, en) pairs
LANG = {
    ZH: dict(code="zh-Hant", file="index.html", other="en.html", other_code="en"),
    EN: dict(code="en", file="en.html", other="index.html", other_code="zh-Hant"),
}

TITLE = ("先量再訓練：無人機編隊的進步空間測試", "Measure before you train: a headroom test for drone formations")
DESC = ("一個無人機編隊的模擬，與一組實驗：在訓練之前，先量簡單規則離最好成績還差多少，並要求學習方法先通過資格考。",
        "A drone-formation simulation and a set of experiments: before training, measure how far a simple rule falls short of the best plan, and make a learner pass a qualifying check first.")
MAST = ("Hao-Ting Ho　研究作品", "Hao-Ting Ho　Research portfolio")
MAST_LINKS = (("論文 PDF", "程式", "English"), ("Paper PDF", "Code", "中文"))
ROUTE_LABEL = ("目錄", "Contents")
HERE = ("你在這裡", "You are here")
EXT_LABEL = ("外部連結", "Links")
NAMES = (("做了什麼", "方法三步", "三個發現", "猜想與實際", "還沒解決的", "論文與程式"),
         ("What I did", "The method", "Three findings", "Guess vs. result", "Still open", "Paper and code"))
SENT = (("先量一量，再決定要不要訓練", "先猜、再量、學習方法要先過資格考", "一個幾乎沒有空間，一個結論較窄，一個還有差距",
         "當初猜的，有對有錯", "學習方法補得上「只看鄰居」的差距嗎？我不知道", "細節都在這裡"),
        ("Measure first, then decide whether to train", "Guess first, measure, and make the learner pass a qualifying check",
         "One leaves almost no room, one supports only a narrow conclusion, one still has a gap",
         "Some of my guesses were right, some were wrong", "Can a learner close the neighbours-only gap? I do not know", "The details are here"))
FIG = ("圖", "Fig")
# station colour: 1 and 2 blue (what I did, how), 3 and 4 orange (what came out), 5 hatched (unresolved), 6 green (where to read on)
COLOR = ("blue", "blue", "orange", "orange", "hatch", "green")

S1_H1 = ("我做了一套無人機編隊的模擬，<br class=\"dk\">用它回答一個問題：<br class=\"dk\">訓練出來的策略，還能比簡單的規則好多少？",
         "I built a drone-formation simulation <br class=\"dk\">to answer one question: <br class=\"dk\">how much better than a simple rule can a trained policy do?")
S1_P = ("編隊飛行時，跟在後面的無人機借前機的氣流省電，領頭的沒有。第一顆電池見底，任務就結束。所以誰來領頭、什麼時候換人，決定整隊能飛多遠。",
        "In formation flight, each drone behind the leader saves energy by riding the wake of the drone ahead; the leader gets no such help. The first empty battery ends the mission, so who leads, and when the lead changes hands, decides how far the team flies.")
S1_RES = (f"<b>先講結果：</b>在基本與較複雜兩種模型的 {V['n_rot']} 個情境裡，看電量的簡單規則有 {V['n_within']} 個離最好成績不到 2%。只看鄰居時，在挑出來的 {V['L3_n']} 個情境還有一段差距；我試的學習設定還沒取得測試它的資格。",
          f"<b>The result first:</b> across the {V['n_rot']} settings of the basic and richer models, a simple rule that reads battery levels is within 2% of the best plan in {V['n_within']}. With neighbours only, a gap remains in {V['L3_n']} selected settings; the learner set-up I tried has not yet qualified to be tested on it.")
S1_NOTE = ("全部是電腦模擬，沒有實機。多數程式與實驗由 AI coding agents 在我的指示下完成。",
           "All simulation, no hardware. Most code and experiments were done by AI coding agents under my direction.")

FIGS = {
    1: dict(name=("輪值圖", "Rotation chart"), sent=("換人領頭，把耗電分攤給每一架（示意，不是資料）", "Leaders take turns, spreading the drain across all the drones (schematic, not data)"), fn=fig_rotation),
    2: dict(name=("進步空間", "Headroom"), sent=("差距落在 2% 線以內，就判斷不值得訓練（示意，不是資料）", "A gap inside the 2% line means training is judged not worth trying (schematic, not data)"), fn=fig_headroom),
    3: dict(name=(f"{V['n_rot']} 個情境", f"{V['n_rot']} settings"), sent=(f"看電量的規則，只有 {V['k_rot']} 個超過 2% 線", f"The battery rule: only {V['k_rot']} are beyond the 2% line"), fn=fig_cells),
    4: dict(name=("只看鄰居 對 看得到全隊", "Neighbours-only vs. whole-team"), sent=(f"{V['L3_n']} 個情境的差距", f"The gap in {V['L3_n']} settings"), fn=fig_slope),
    5: dict(name=("猜錯最大的一項", "The biggest miss"), sent=(f"當初猜約 {V['pred_l1_share']}% 的情境留有超過 2% 的空間，實際是 {V['L1_cells']} 個裡 {V['L1_best_gt2']} 個（五條規則取最好的）", f"I guessed about {V['pred_l1_share']}% of settings would leave more than 2% of room; the actual count is {V['L1_best_gt2']} of {V['L1_cells']} (the best of five rules)"), fn=fig_miss),
    6: dict(name=("資格考", "Sanity gate"), sent=(f"{V['gate_n']} 個對照情境過了 {V['gate_pass_n']} 個，需要 {V['gate_need']} 個", f"{V['gate_pass_n']} of {V['gate_n']} control settings passed; {V['gate_need']} were required"), fn=fig_gate),
}

S2_P = ("訓練失敗的時候，分不出是「本來就沒有進步空間」，還是「只是沒訓練好」。所以我先量空間。",
        "When a training run fails, you cannot tell whether there was no room to improve or the training was simply poor. So I measure the room first.")
S2_ROWS = [
    (("先寫下猜想", "Write the guess down"), ("量之前，先把猜想寫下來存檔。猜錯的也照報。先後順序靠的是我自己的版本紀錄，不是第三方證明。",
                                       "Before measuring, write my guess down and save it. Wrong guesses are reported too. The order rests on my own version-control record, not on third-party proof.")),
    (("量差距", "Measure the gap"), ("算出簡單規則的成績，再算出「事後什麼都知道」時能做到的最好成績（論文裡叫上界）。兩者的差距就是進步空間。差距小於 2%，就判斷不值得訓練。",
                                "Score the simple rule, then score the best plan achievable with hindsight (the paper's upper bound). The difference is the headroom. If it is under 2%, I judge training not worth trying.")),
    (("資格考", "Sanity gate"), ("測學習方法之前，先讓資訊較多的版本在三個已知有差距可補的對照情境上試，至少要有兩個補回一半以上。沒過，學習方法的結果就不採計。",
                              "Before a learner is tested, a better-informed version of it is tried on three control settings with a known gap. It must recover at least half of that gap in at least two of them. If it does not, the learner's results are not counted.")),
]
S2_HEAD = (("步驟", "名稱", "說明"), ("Step", "Name", "What it is"))
S2_NOTE = ("2% 這條線是我量之前自己選的，不是算出來的。", "The 2% line is my own choice, made before measuring, not a derived quantity.")
S2_FIGNOTE = None

S3_INTRO = ("模擬裡有三種任務：輪流領頭；貼尾流（跟機貼著前機的尾流飛）；滑翔（滑翔機互相通報上升氣流）。",
            "The simulation has three tasks: leader rotation; wake surfing (a follower flies in the wake of the drone ahead); and soaring (gliders announce rising air to one another).")
S3 = [
    dict(h=("看電量的簡單規則，幾乎沒留下空間", "A simple rule that reads battery levels leaves almost no room"),
         p=(f"「把領頭交給電量最多的那架」這條規則，在基本與較複雜兩種模型共 {V['n_rot']} 個輪流領頭情境裡，只有 {V['k_rot']} 個離最好成績超過 2%。「每隔一段時間換人」的規則在基本模型裡常常差得遠。",
            f"The rule “give the lead to the drone with the most charge” is more than 2% from the best plan in only {V['k_rot']} of {V['n_rot']} leader-rotation settings, counting the basic and richer models. In the basic model, a time rule that hands the lead on at fixed intervals is often far off."),
         note=("這些規則是在評分用的同一批情境上調整的，這對規則有利。", "These rules were tuned on the same instances that score them, which favours the rules."), fig=3),
    dict(h=("對上更強的參考對象時有差距，但能下的結論比較窄", "Against stronger references there is a gap, but the conclusion is narrower"),
         p=("貼尾流和滑翔這兩種任務算不出最好成績，只能和拿得到額外資訊的更強方法（參考對象）比。參考對象在部分貼尾流情境、多數滑翔情境領先超過 2 個百分點。貼尾流那幾個，只出現在和「看得到真實位置」的參考對象比的時候。參考對象的成績不是最好成績，所以這不能證明學習方法有進步空間，也不能證明沒有。",
            "The wake-surfing and soaring tasks have no best plan to compare with, only stronger reference controllers that are given extra information. Their lead exceeds 2 percentage points in some wake-surfing settings (against one of the two references only) and in most soaring settings. But a reference is not an upper bound, so this does not show that a learner has room, nor that it has none."),
         note=None, fig=None),
    dict(h=("只看得到鄰居時，還有一段差距", "Seeing only neighbours still leaves a gap"),
         p=(f"只看得到相鄰無人機的電量時（在有雜訊的情境裡），最好的規則比看得到全隊的規則差一截。{V['L3_n']} 個情境的中位數差距是最好成績的 {V['L3_diff_median']:.2f}%。",
            f"When each drone sees only its neighbours' batteries (in noisy settings), the best rule trails the whole-team rule by a median of {V['L3_diff_median']:.2f}% of the best plan, across {V['L3_n']} settings."),
         note=(f"這 {V['L3_n']} 個情境是因為出現差距才被挑出來的，所以會高估它有多常見。", f"These {V['L3_n']} settings were picked because they showed a gap, so they overstate how common it is."), fig=4),
]

S4_P = ("每一項都先寫下猜想再量。下面是主要的五項。",
        "Every item was guessed first and then measured. Here are the five main ones.")
S4_HEAD = (("項目", "當初猜的", "實際", "結果"), ("Item", "What I guessed", "What happened", "Result"))
VERDICT = {"right": ("blue", ("猜對", "Right")), "wrong": ("orange", ("猜錯", "Wrong")), "half": ("ink", ("對一半", "Half right"))}
S4_ROWS = [   # one short line per cell, with the paper's qualifiers kept; every number is read from V
    dict(name=("輪流領頭（基本）", "Basic rotation"), v="right",
         g=(f"換位便宜時，看電量的規則離最好成績 {V['pred_l0_low']}% 到 {V['pred_l0_within']}% 以內", f"When swapping is cheap, a rule that reads battery levels stays within {V['pred_l0_low']} to {V['pred_l0_within']}% of the best plan"),
         a=(f"{V['L0_cells']} 個裡只有 {V['L0_battery_gt2']} 個超過 2%", f"Only {V['L0_battery_gt2']} of {V['L0_cells']} settings beyond 2%")),
    dict(name=("輪流領頭（較複雜）", "Richer rotation"), v="wrong",
         g=(f"約 {V['pred_l1_share']}% 的情境留有超過 2% 的空間", f"About {V['pred_l1_share']}% of settings leave more than 2% of room"),
         a=(f"{V['L1_cells']} 個裡只有 {V['L1_best_gt2']} 個（五條規則取最好的）", f"Only {V['L1_best_gt2']} of {V['L1_cells']} (the best of five rules)")),
    dict(name=("貼尾流", "Wake surfing"), v="half",
         g=("尾流飄得厲害時差距才超過 2 個百分點，飄得小時不會", "Gaps above 2 percentage points only when the wake drifts a lot, not when it drifts little"),
         a=("飄得小時確實不會；超過的卻都在領機做機動時", "As guessed, none at small drift; yet every gap above 2\u00a0pp came when the lead aircraft manoeuvred")),
    dict(name=("滑翔", "Soaring"), v="right",
         g=("簡單規則拿到真實資料，就追得回大部分差距", "A simple rule given the real data recovers most of the gap"),
         a=("追回了大部分；剩下的領先沒有把握超過 2 個百分點", "Most was recovered; the remaining lead was never confidently above 2\u00a0pp")),
    dict(name=("學習方法的資格考", "Learner sanity gate"), v="wrong",
         g=(f"第一次有 {V['pred_gate_first']}% 的機會通過", f"A {V['pred_gate_first']}% chance of passing on the first try"),
         a=(f"第一次沒通過；第二次過 {V['gate_second_pass']} 個，需要 {V['gate_need']} 個",
            f"The first try failed; the second passed {['none', 'one', 'two', 'three'][V['gate_second_pass']]} control setting, where {['none', 'one', 'two', 'three'][V['gate_need']]} were required")),
]
assert [r["v"] for r in S4_ROWS] == ["right", "wrong", "half", "right", "wrong"]
S4_AFTER = (("猜錯最大的一項是較複雜的輪流領頭。我以為無人機彼此差異大時會留下空間。結果差異大的情境沒有一個超過 2%，唯一超過的那個不在其中。",
             "我的解讀（沒有驗證過）：差異大時，把最弱的那架放在最省電的位置就接近最好，簡單規則很容易做到。"),
            ("The biggest miss was the richer leader rotation. I expected room where the drones differ a lot. None of those settings went beyond 2%, and the one setting that did is not among them.",
             "My reading (untested): when drones differ a lot, putting the weakest one where it saves most gets close to the best, and a simple rule does that easily."))
assert V["L1_h15plus_gt2"] == 0 and V["L1_best_gt2"] == 1   # "none" where the drones differ a lot; exactly one elsewhere
S4_NOTE = ("每一項學到什麼，在論文第 4 節的表。全部猜想的統計在論文附錄，其中一部分不是盲測，是在看過較早或部分結果之後寫的。",
           "What each one taught is in the table in Section 4 of the paper. The statistics for all guesses are in the paper's appendix; some of them were not blind but written after earlier or partial results.")

S5_P = (("學習方法得先通過資格考。我讓資訊較多的版本在三個對照情境上試，只過一個，需要兩個。照事先寫好的規則，正式的訓練就沒有進行。",
         "少了資格考，失敗的訓練容易被讀成「學習方法補不上這段差距」。實際上，這個學習設定連規則補得回來的差距都還學不會，所以它對原本的問題什麼都沒說。"),
        ("A learner must first pass the sanity gate. I tried a better-informed version of it on three control settings. It passed one, and two were required. Following the rules written in advance, the real training was not run.",
         "Without the gate, a failed training run could be read as “learning cannot close this gap”. The correct reading is that this learner setting cannot yet recover a gap that a rule is known to recover, so it says nothing about the original question."))

S6_LINKS = ((f"論文（PDF，{V['paper_pages']} 頁）", "補充資料：全部猜想的逐條對照（PDF）", "程式與結果檔（GitHub）"),
            (f"Paper (PDF, {V['paper_pages']} pages)", "Supplement: every guess set against its result (PDF)", "Code and result files (GitHub)"))
S6_HEAD = (("去哪裡讀", "主要限制"), ("Where to read more", "Main limits"))
S6_LIMITS = (("全部是模擬，裡面的物理是假設的，不是實測。", "2% 線是自己選的。換成 1%，有些結論會變。", "這個檢查只能說「學習方法加不了多少」，不能說學習沒有其他價值。"),
             ("Everything is simulated, and the physics is assumed, not measured.", "The 2% line is my own choice. With 1%, some conclusions change.", "This check can say a learner adds little over a rule. It cannot say learning has no other value."))
FOOT = (("作者 Hao-Ting Ho 提出研究問題、做每一次方向決定，並提出每份猜想的方向。AI coding agents 把這些方向擴寫成書面的猜想檔。多數程式、實驗的執行、分析與論文的起草，是 AI coding agents（Anthropic 的 Claude 模型，透過 Claude Code 使用）在作者指示下完成的。論文的內部審查也用 AI agents 模擬。作者對內容負全責。",
         "授權：程式 MIT；論文與圖 CC BY 4.0。"),
        ("Author Hao-Ting Ho set the research question, made every change of direction and proposed the direction of each prediction. AI coding agents expanded these into the written prediction files. Most of the code, the running of the experiments, the analysis and the drafting of the paper were carried out by AI coding agents (Anthropic's Claude models, used through Claude Code) under the author's direction. Internal reviews of the paper were also simulated with AI agents. The author takes full responsibility for the content.",
         "Licence: code MIT; paper and figures CC BY 4.0."))


# ---------------------------------------------------------------- cache-proof static files
# The browser may keep an old style.css or route.js for ever (the local server sends no cache headers), which once paired a new
# page with an old style sheet.  Every static file the pages load therefore carries a content hash in its URL: ?v=<sha1[:10]>.
ASSET = {}          # published name -> versioned URL; filled in by main() once the files are final
VER_LEN = 10
STATIC_EXT = (".css", ".js", ".pdf", ".woff", ".woff2")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()[:VER_LEN]


def U(name):
    return ASSET.get(name, name)


# ---------------------------------------------------------------- rendering

def tag(num, name, sent, kind="st", color=None, heading=True):
    if kind == "st":
        inner = f"<span>{num}</span>" if color == "hatch" else num
        h = f"<h2>{name}</h2>" if heading else f'<strong class="nm" id="s{num}-name">{name}</strong>'
        return f'<div class="tag c-{color}"><b>{inner}</b>{h}<p>{sent}</p></div>'
    return f'<figcaption class="tag fig"><b>{num}</b><strong class="nm">{name}</strong><p>{sent}</p></figcaption>'


def figure(lang, n, note=None, full=False):
    f = FIGS[n]
    nm = f"{FIG[lang]} {n}"
    nt = f'<p class="note">{note[lang]}</p>' if note else ""
    if n == 3:   # long band across the content width on desktop (x); 64 per row on laptops (m); 48 per row on tablets (w); 32 per row on phones (n)
        svgs = "".join(f["fn"](lang, v) for v in "xmwn")
    else:
        svgs = both(f["fn"], lang)
    return (f'<figure class="plate{" full" if full else ""}">{tag(nm, f["name"][lang], f["sent"][lang], "fig")}{svgs}{nt}</figure>')


def keep_together(name):
    """Wrap a full-width parenthesis group in a no-break span so that a narrow column breaks before it, not inside it."""
    m = re.search(r"（| \(", name)
    return name if not m else f'{name[:m.start()]}{m.group(0)[:-1] if m.group(0)[0] == " " else ""}<span class="nb">{m.group(0).lstrip(" ")}{name[m.end():]}</span>'


def render(lang):
    L = LANG[lang]
    nm = NAMES[lang]
    mast = (f'<header class="mast"><span>{MAST[lang]}</span><nav aria-label="{EXT_LABEL[lang]}">'
            f'<a href="{U("paper.pdf")}">{MAST_LINKS[lang][0]}</a><a href="{REPO_URL}">{MAST_LINKS[lang][1]}</a>'
            f'<a href="{L["other"]}" hreflang="{L["other_code"]}" lang="{L["other_code"]}">{MAST_LINKS[lang][2]}</a></nav></header>')
    lis = "".join(f'<li><a href="#s{i + 1}"><span class="dot c-{COLOR[i]}"></span><span class="lab"><b class="n">{i + 1}</b> <span class="nm">{nm[i]}</span></span></a></li>' for i in range(6))
    route = (f'<nav class="route" aria-label="{ROUTE_LABEL[lang]}" data-here="{HERE[lang]}"><div class="in">'
             '<div class="line" aria-hidden="true"><i class="b"></i><i class="o"></i><i class="h"></i><i class="g"></i></div>'
             f'<ol>{lis}</ol>'
             f'<span class="you" aria-hidden="true"><span class="you-t">{HERE[lang]}</span><span class="you-d"></span></span>'
             '<p class="readout" aria-hidden="true"></p></div></nav>')

    def plate(i):
        # station 1 opens with the page title (h1), so its plate name is a plain element, labelled for assistive technology
        return tag(i + 1, nm[i], SENT[lang][i], "st", COLOR[i], heading=i > 0)

    # 1
    s1 = (f'<section class="st s1" id="s1" aria-labelledby="s1-name"><div class="read">{plate(0)}<h1>{S1_H1[lang]}</h1><p class="b body">{S1_P[lang]}</p><p class="b body res">{S1_RES[lang]}</p>'
          f'<p class="note foot">{S1_NOTE[lang]}</p></div>{figure(lang, 1)}</section>')
    # 2
    head = "".join(f"<th>{h}</th>" for h in S2_HEAD[lang])
    rows = "".join(f'<tr><td class="n">{i + 1}</td><th scope="row">{r[0][lang]}</th><td class="d">{r[1][lang]}</td></tr>' for i, r in enumerate(S2_ROWS))
    s2 = (f'<section class="st s2" id="s2"><div class="read">{plate(1)}<p class="b body">{S2_P[lang]}</p>'
          f'<table class="tt steps"><thead><tr><th>{S2_HEAD[lang][0]}</th><th>{S2_HEAD[lang][1]}</th><th>{S2_HEAD[lang][2]}</th></tr></thead><tbody>{rows}</tbody></table>'
          f'<p class="note">{S2_NOTE[lang]}</p></div>{figure(lang, 2)}</section>')
    # 3
    parts = []
    for i, f in enumerate(S3):
        fig = figure(lang, f["fig"], f["note"], full=f["fig"] == 3) if f["fig"] else ""
        note = "" if f["fig"] or not f["note"] else f'<p class="note">{f["note"][lang]}</p>'
        parts.append(f'<div class="row"><div class="read"><h3><span class="n">3.{i + 1}</span> {f["h"][lang]}</h3><p class="b body">{f["p"][lang]}</p>{note}</div>{fig}</div>')
    s3 = f'<section class="st s3" id="s3"><div class="read">{plate(2)}<p class="b body">{S3_INTRO[lang]}</p></div>{"".join(parts)}</section>'
    # 4: a short table, then the two sentences and the figure side by side
    H4 = S4_HEAD[lang]
    rows4 = "".join(
        f'<tr><th scope="row">{keep_together(r["name"][lang])}</th><td data-l="{H4[1]}">{keep_together(r["g"][lang])}</td><td data-l="{H4[2]}">{keep_together(r["a"][lang])}</td>'
        f'<td><span class="mk c-{VERDICT[r["v"]][0]}"><i></i>{VERDICT[r["v"]][1][lang]}</span></td></tr>' for r in S4_ROWS)
    s4 = (f'<section class="st s4" id="s4"><div class="read">{plate(3)}<p class="b body">{S4_P[lang]}</p></div>'
          f'<table class="tt guess"><thead><tr>{"".join(f"<th>{h}</th>" for h in H4)}</tr></thead><tbody>{rows4}</tbody></table>'
          f'<div class="row"><div class="read"><p class="b body">{S4_AFTER[lang][0]}</p><p class="b body">{S4_AFTER[lang][1]}</p></div>{figure(lang, 5)}</div>'
          f'<p class="note wide">{S4_NOTE[lang]}</p></section>')
    # 5
    s5 = (f'<section class="st s5" id="s5"><div class="read">{plate(4)}<p class="b body">{S5_P[lang][0]}</p><p class="b body">{S5_P[lang][1]}</p></div>{figure(lang, 6)}</section>')
    # 6
    hrefs = (U("paper.pdf"), U("supplement.pdf"), REPO_URL)
    links = "".join(f'<p class="go"><a href="{h}">{t}</a></p>' for h, t in zip(hrefs, S6_LINKS[lang]))
    lim = "".join(f'<li class="b">{t}</li>' for t in S6_LIMITS[lang])
    s6 = f'<section class="st s6" id="s6"><div class="read">{plate(5)}</div><div class="row"><div class="read"><h3>{S6_HEAD[lang][0]}</h3><div class="b links">{links}</div></div><div class="lim"><h3>{S6_HEAD[lang][1]}</h3><ul>{lim}</ul></div></div></section>'
    foot = f'<footer><div class="wrap"><p>{FOOT[lang][0]}</p><p>{FOOT[lang][1]}</p></div></footer>'
    page = f'''<!doctype html>
<html lang="{L["code"]}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(TITLE[lang])}</title>
<meta name="description" content="{esc(DESC[lang])}">
<link rel="alternate" hreflang="{L["other_code"]}" href="{L["other"]}">
<link rel="icon" href="data:,">
<meta name="color-scheme" content="light">
<link rel="stylesheet" href="{U("style.css")}">
<script src="{U("js/route.js")}"></script>
</head>
<body>
{mast}
{route}
<main class="wrap">
{s1}
{s2}
{s3}
{s4}
{s5}
{s6}
</main>
{foot}
</body>
</html>
'''
    return page


# ---------------------------------------------------------------- style sheet (written to web/portfolio/style.css by main())
CSS = r'''@font-face { font-family: "Noto Sans TC"; font-style: normal; font-weight: 400; font-display: swap; src: url(fonts/NotoSansTC-400.woff) format("woff"); unicode-range: U+2E80-2FDF, U+3000-303F, U+3400-4DBF, U+4E00-9FFF, U+F900-FAFF, U+FE30-FE4F, U+FF00-FFEF; }
@font-face { font-family: "Noto Sans TC"; font-style: normal; font-weight: 700; font-display: swap; src: url(fonts/NotoSansTC-700.woff) format("woff"); unicode-range: U+2E80-2FDF, U+3000-303F, U+3400-4DBF, U+4E00-9FFF, U+F900-FAFF, U+FE30-FE4F, U+FF00-FFEF; }
@font-face { font-family: "Barlow"; font-style: normal; font-weight: 400; font-display: swap; src: url(fonts/Barlow-400.woff2) format("woff2"); }
@font-face { font-family: "Barlow"; font-style: normal; font-weight: 500; font-display: swap; src: url(fonts/Barlow-500.woff2) format("woff2"); }
@font-face { font-family: "Barlow"; font-style: normal; font-weight: 700; font-display: swap; src: url(fonts/Barlow-700.woff2) format("woff2"); }
@font-face { font-family: "Barlow Semi Condensed"; font-style: normal; font-weight: 500; font-display: swap; src: url(fonts/BarlowSemiCondensed-500.woff2) format("woff2"); }
@font-face { font-family: "Barlow Semi Condensed"; font-style: normal; font-weight: 700; font-display: swap; src: url(fonts/BarlowSemiCondensed-700.woff2) format("woff2"); }

/* Breakpoints: below 720 phone; 720-1023 tablet, one column; 1024 and up laptop and desktop, text left and figure right. */
:root {
  --bg: #F5F7F9; --ink: #0E1A2B; --blue: #0057B8; --orange: #E87722; --green: #1A936F;
  --text: "Noto Sans TC", "Barlow", "PingFang TC", "Microsoft JhengHei", sans-serif;
  --num: "Barlow Semi Condensed", "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif;
  --s: 16px; --m: 18px; --l: 28px;
  --pad: 16px;
  --hatch: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11.31' height='11.31'%3E%3Cpath d='M-3 3l6-6M0 11.31L11.31 0M8.31 14.31l6-6' stroke='%230E1A2B' stroke-width='1.6'/%3E%3C/svg%3E");
  --route-h: 84px;
}
@media (min-width: 720px) { :root { --m: 20px; --l: 32px; --pad: 48px; } }
/* the headline stays the largest size; it scales smoothly from 28px at 1024 to 36px at 1440 */
@media (min-width: 1024px) { :root { --m: 18px; --l: clamp(28px, 8.3px + 1.923vw, 36px); --pad: 40px; } }
@media (min-width: 1280px) { :root { --m: 20px; } }
@media (min-width: 1360px) { :root { --pad: 64px; } }

* { box-sizing: border-box; margin: 0; }
html { background: var(--bg); color: var(--ink); color-scheme: light; scrollbar-color: var(--ink) var(--bg); scroll-padding-top: calc(var(--route-h) + 16px); }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
body { background: var(--bg); color: var(--ink); font: 400 var(--m)/1.8 var(--text); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; text-spacing-trim: normal; }
::selection { background: var(--blue); color: var(--bg); }
a { color: inherit; text-underline-offset: 5px; text-decoration-thickness: 1.5px; }
a:hover { text-decoration-thickness: 3px; }
a:focus-visible { outline: 3px solid var(--orange); outline-offset: 3px; }

/* masthead */
.mast { min-height: 56px; padding: 8px var(--pad); background: var(--ink); color: var(--bg); display: flex; justify-content: space-between; align-items: center; gap: 12px 24px; flex-wrap: wrap; font: 500 var(--s)/1.4 var(--num); }
.mast nav { display: flex; gap: 12px 36px; flex-wrap: wrap; }
.mast a:focus-visible { outline-color: var(--orange); }

/* route map, which is also the table of contents */
.route { background: var(--bg); padding: 12px var(--pad) 8px; }
@media (min-width: 1024px) { .route { padding-top: 26px; padding-bottom: 12px; } .route.stuck { padding-top: 10px; padding-bottom: 6px; } }
.route .in { position: relative; max-width: 1312px; margin: 0 auto; padding-top: 28px; }
.route ol { list-style: none; padding: 0; display: grid; grid-template-columns: repeat(6, 1fr); }
.route li a { display: block; position: relative; text-decoration: none; padding-top: 36px; line-height: 28px; font: 500 var(--s)/28px var(--text); white-space: nowrap; }
.route li a:hover .nm { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 5px; }
.route .dot { position: absolute; left: 2px; top: 2px; width: 24px; height: 24px; border-radius: 50%; border: 4px solid var(--ink); background: var(--bg); }
.route .dot.c-hatch { border-style: dashed; }
.route .n { font: 700 var(--s)/28px var(--num); margin-right: 6px; }
.route a[aria-current] .nm { font-weight: 700; }
.route .line { position: absolute; left: 0; right: 0; top: 36px; height: 12px; }
.route .line i { position: absolute; top: 0; height: 12px; }
.route .line .b { left: 14px; width: 33.333%; background: var(--blue); }
.route .line .o { left: calc(14px + 33.333%); width: 33.333%; background: var(--orange); }
.route .line .h { left: calc(14px + 66.667%); width: 16.667%; background: var(--hatch); border: 1.5px solid var(--ink); }
.route .line .g { left: calc(14px + 83.333%); right: 0; background: var(--green); }
.route .you { position: absolute; top: 0; left: calc(var(--pos, 0) * 100% / 6); width: 0; pointer-events: none; transition: left .55s cubic-bezier(.16, 1, .3, 1); }
.route .you-t { position: absolute; left: 0; top: 0; font: 500 var(--s)/24px var(--text); white-space: nowrap; }
.route .you-d { position: absolute; left: 0; top: 28px; width: 28px; height: 28px; border-radius: 50%; background: var(--ink); }
.route .readout { display: none; }
.js .route { position: sticky; top: 0; z-index: 10; }
.js .route.stuck { box-shadow: 0 1px 0 var(--ink); }
@media (prefers-reduced-motion: reduce) { .route .you { transition: none; } }
@media (min-width: 1360px) { .route li a { font: 500 var(--m)/28px var(--text); } .route .n { font: 700 var(--m)/28px var(--num); } }

/* narrow route (phone and tablet): dots and a one-line readout */
@media (max-width: 1023px) {
  .route .in { padding-top: 0; }
  .route .lab { display: none; }
  .route li a { padding-top: 0; height: 28px; }
  .route .line { top: 8px; }
  .route .you-t { display: none; }
  .route .you-d { top: 0; }
  .js .route .readout { display: block; font: 500 var(--s)/28px var(--text); padding-top: 6px; }
  .js .route .readout b { font: 700 var(--s)/28px var(--num); }
  :root { --route-h: 92px; }
  /* without scripting the map is a plain list of links */
  html:not(.js) .route .line, html:not(.js) .route .you { display: none; }
  html:not(.js) .route ol { grid-template-columns: 1fr 1fr; }
  html:not(.js) .route .lab { display: inline; }
  html:not(.js) .route li a { height: auto; padding: 2px 0; }
  html:not(.js) .route .dot { display: none; }
}
@media (min-width: 1024px) { :root { --route-h: 108px; } }

/* layout */
.wrap { max-width: calc(1312px + 2 * var(--pad)); margin: 0 auto; padding: 0 var(--pad); }
main.wrap { padding-top: 40px; padding-bottom: 96px; }
.st { display: block; }
.st + .st { margin-top: 96px; padding-top: 8px; }
.st > .read + .plate, .row > .read + .plate { margin-top: 40px; }
.st > .read + .row { margin-top: 32px; }
.row + .row { margin-top: 48px; }
.st { scroll-margin-top: calc(var(--route-h) + 16px); }
@media (min-width: 1024px) {
  .st, .row { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); column-gap: 24px; align-items: start; }
  .st > .read, .row > .read { grid-column: 1 / span 7; }
  .st > .plate, .row > .plate, .row > .lim { grid-column: 8 / span 5; }
  .st > .read + .plate, .row > .read + .plate { margin-top: 0; }
  .st > .tt, .st > .note.wide { grid-column: 1 / -1; }
  .row { grid-column: 1 / -1; margin-top: 56px; }
  .row > .plate.full { grid-column: 1 / -1; margin-top: 32px; }
  .st > .read + .row { margin-top: 40px; }
  .s1 > .read, .s1 > .plate { align-self: stretch; }
  .s1 > .read { display: flex; flex-direction: column; }
  .s1 { min-height: calc(100svh - 280px); }   /* the first screen holds station 1 only; station 2 starts below the fold */
}

/* station plate and figure label: number, name, one sentence */
.tag { display: grid; grid-template-columns: 64px 1fr; }
.tag b { background: var(--blue); color: var(--bg); font: 700 var(--m)/44px var(--num); text-align: center; }
.tag h2, .tag .nm { display: block; font: 700 var(--m)/44px var(--text); padding-left: 16px; border-bottom: 2px solid var(--ink); }
.tag p { grid-column: 2; padding: 8px 0 0 16px; font: 400 var(--s)/1.6 var(--text); }
.tag.c-orange b { background: var(--orange); color: var(--ink); }
.tag.c-green b { background: var(--green); color: var(--ink); }
.tag.c-hatch b { background: var(--hatch), var(--bg); box-shadow: inset 0 0 0 1.5px var(--ink); display: flex; align-items: center; justify-content: center; }
.tag.c-hatch b span { background: var(--bg); color: var(--ink); line-height: 28px; padding: 0 8px; }
.tag.fig b { background: var(--ink); color: var(--bg); font-family: var(--text); }
figcaption.tag { margin: 0; }

/* reading text */
h1 { font: 700 var(--l)/1.4 var(--text); margin: 40px 0 28px; text-wrap: balance; }
h1 .dk { display: none; }
@media (min-width: 720px) { h1 .dk { display: inline; } }
@media (min-width: 1024px) {
  h1 { margin-top: 24px; margin-bottom: 16px; }
  html[lang="en"] .s1 h1 { margin: 20px 0 14px; }
  /* the first screen holds the whole of station 1: tighter rhythm so the small print stays on screen */
  .s1 .read > p.body { line-height: 1.65; }
  .s1 .read > p.body.res, html[lang="en"] .s1 .read > p.body.res { margin-top: 12px; margin-bottom: 14px; }
  .s1 .note.foot { padding-top: 10px; }
  main.wrap { padding-top: 24px; }
  .route { padding-top: 20px; }
}
/* a short laptop screen (1024x768): tighter still, nothing removed */
@media (min-width: 1024px) and (max-height: 800px) {
  main.wrap { padding-top: 12px; }
  .route { padding-top: 12px; padding-bottom: 6px; }
  h1, html[lang="en"] .s1 h1 { margin-top: 16px; margin-bottom: 10px; }
  .s1 .read > p.body, html[lang="en"] .s1 .read > p.body { line-height: 1.5; }
  .s1 .read > p.body.res, html[lang="en"] .s1 .read > p.body.res { margin-top: 8px; margin-bottom: 10px; }
  .read > .tag + .b.body { margin-top: 16px; }
  .tag p { padding-top: 4px; }
}
/* on a narrow laptop the forced line breaks would leave a lone character on a line; let the balanced wrapping decide */
@media (min-width: 1024px) and (max-width: 1279px) { h1 .dk { display: none; } }
h3 { font: 700 var(--m)/1.5 var(--text); margin: 0 0 12px; padding-top: 14px; border-top: 1px solid var(--ink); }
h3 .n { font-family: var(--num); margin-right: 8px; }
.row > .read > h3 { margin-top: 0; }
.read > .tag + .b, .read > .tag + h3 { margin-top: 28px; }
.b.body, p.b { max-width: 720px; }
.read > p.b + p.b { margin-top: 16px; }
.read > .tag + .b.body { margin-top: 32px; }
.s1 .read > p.body { margin: 0; }
.s1 .read > p.body.res { margin-top: 16px; margin-bottom: 28px; }
html[lang="en"] .s1 h1 { margin: 28px 0 20px; }
html[lang="en"] .s1 .read > p.body { line-height: 1.6; }
html[lang="en"] .s1 .read > p.body.res { margin-bottom: 16px; }
.note { margin-top: 16px; font: 400 var(--s)/1.6 var(--text); max-width: 720px; }
.note.foot { margin-top: auto; padding-top: 14px; border-top: 1px solid var(--ink); max-width: none; }
.s1 .note.foot { margin-top: auto; }
.s1 .read { min-height: 0; }
@media (max-width: 1023px) { .s1 .note.foot { margin-top: 28px; } }
figure.plate { display: block; }
.plate .note { margin-top: 12px; }

/* figures: one drawing language, ink hairlines, three route colours, hatching for what is unresolved.
   n = phone, w = tablet and wide laptop, m = narrow laptop (figure 3 only), x = laptop and desktop (figure 3 only).
   In the figure column of a 1024-1199 laptop the narrow drawing is used again, so the text keeps its size. */
.fg { display: block; width: 100%; height: auto; margin-top: 24px; font-variant-numeric: tabular-nums; overflow: visible; }
.fg.n { max-width: 400px; }
.fg.w, .fg.m, .fg.x { display: none; }
.fg.w { max-width: 532px; }
.fg.m { max-width: 960px; }
.fg.x { max-width: 1312px; }
@media (min-width: 720px) { .fg.n { display: none; } .fg.w { display: block; } }
@media (min-width: 1024px) and (max-width: 1199px) { .plate:not(.full) .fg.w { display: none; } .plate:not(.full) .fg.n { display: block; max-width: 440px; } }
@media (min-width: 1024px) { .plate.full .fg.w { display: none; } .plate.full .fg.m { display: block; } }
@media (min-width: 1200px) { .plate.full .fg.m { display: none; } .plate.full .fg.x { display: block; } }
.fg text { font: 400 var(--s) var(--text); fill: var(--ink); }
.fg .t, .fg .t text { font: 500 var(--s) var(--num); }
.fg .d, .fg .d text { font-weight: 700; }

/* timetable-style tables */
.tt { width: 100%; border-collapse: collapse; font: 400 var(--s)/1.6 var(--text); margin-top: 28px; }
.tt thead th { font: 700 var(--s)/1.6 var(--text); text-align: left; padding: 6px 24px 6px 0; border-bottom: 2px solid var(--ink); }
.tt td, .tt tbody th { padding: 14px 24px 14px 0; border-bottom: 1px solid var(--ink); vertical-align: top; text-align: left; font-weight: 400; }
.tt tbody th { font-weight: 700; }
.tt td.n { font: 700 var(--s)/1.6 var(--num); }
.tt.steps { max-width: none; }
.tt.steps thead th:first-child, .tt.steps td.n { width: 64px; padding-right: 8px; }
.tt.steps tbody th { width: 140px; }
/* the guess-and-result table: four short columns, same size as the reading text */
.tt.guess { margin-top: 36px; font-size: var(--m); line-height: 1.6; }
.tt.guess thead th { font-size: var(--s); }
.tt.guess td, .tt.guess tbody th { padding-right: 20px; }
.tt.guess thead th:nth-child(1) { width: 28%; }
.tt.guess thead th:nth-child(2) { width: 30%; }
.tt.guess thead th:nth-child(3) { width: 24%; }
.tt.guess .nb { display: inline-block; }   /* a parenthesis group wraps as a whole, never in the middle */
@media (min-width: 720px) and (max-width: 1023px) { .tt.guess td, .tt.guess tbody th { padding-right: 14px; } }
@media (min-width: 1024px) {
  .tt.guess thead th:nth-child(1) { width: 22%; }
  .tt.guess thead th:nth-child(2) { width: 33%; }
  .tt.guess thead th:nth-child(3) { width: 27%; }
}
.mk { display: inline-block; font: 500 var(--m)/1.6 var(--text); white-space: nowrap; }
.mk i { display: inline-block; width: 12px; height: 12px; margin-right: 8px; vertical-align: -1px; box-sizing: border-box; }
.mk.c-blue i { background: var(--blue); }
.mk.c-orange i { background: var(--orange); }
.mk.c-ink i { border: 2px solid var(--ink); background: transparent; }
.s4 .note.wide { max-width: 720px; }
.s4 .tt + .row { margin-top: 40px; }
@media (min-width: 1024px) { .s4 .tt + .row { margin-top: 48px; } }
@media (max-width: 719px) {
  .tt thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
  .tt, .tt tbody { display: block; }
  .tt.steps tr { display: grid; grid-template-columns: 40px 1fr; border-bottom: 1px solid var(--ink); padding: 12px 0; }
  .tt.steps td, .tt.steps th { border: 0; padding: 0; }
  .tt.steps td.d { grid-column: 1 / -1; margin-top: 4px; }
  .tt.steps tbody th { width: auto; }
  /* each guess becomes a short stack; the verdict stays on the first line, next to the item */
  .tt.guess tr { display: grid; grid-template-columns: 1fr auto; column-gap: 16px; border-bottom: 2px solid var(--ink); padding: 14px 0 6px; }
  .tt.guess tbody > tr > * { display: block; border: 0; padding: 0 0 12px; }
  .tt.guess tbody > tr > :nth-child(2), .tt.guess tbody > tr > :nth-child(3) { grid-column: 1 / -1; }
  .tt.guess tbody > tr > :nth-child(2)::before, .tt.guess tbody > tr > :nth-child(3)::before { content: attr(data-l); display: block; font: 700 var(--s)/1.6 var(--text); }
  .tt.guess tbody > tr > :nth-child(4) { grid-column: 2; grid-row: 1; text-align: right; }
}

/* last station: links and limits */
.links { margin-top: 0; }
.go { padding: 12px 0; border-top: 1px solid var(--ink); font-weight: 700; }
.go:last-child { border-bottom: 1px solid var(--ink); }
.lim { margin-top: 32px; max-width: 720px; }
.lim ul { list-style: none; padding: 0; }
.lim li { padding: 12px 0; border-top: 1px solid var(--ink); }
.lim li:last-child { border-bottom: 1px solid var(--ink); }
@media (min-width: 1024px) { .lim { margin-top: 0; } }
.read > h3 + .links .go:first-child, .lim h3 + ul li:first-child { border-top: 0; }

/* colophon */
footer { background: var(--ink); color: var(--bg); padding: 40px 0 48px; font: 400 var(--s)/1.7 var(--text); }
footer p { max-width: 900px; }
footer p + p { margin-top: 12px; }
'''


CJK_RANGES = [(0x2E80, 0x2FDF), (0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF)]
CJK_UNICODE_RANGE = "U+2E80-2FDF, U+3000-303F, U+3400-4DBF, U+4E00-9FFF, U+F900-FAFF, U+FE30-FE4F, U+FF00-FFEF"
CJK_PUNCT = "，。、；：？！「」『』（）《》〈〉【】～　．％"


def build_cjk_fonts(pages):
    """Subset Noto Sans CJK TC (Regular, Bold) to the CJK characters that appear in the generated pages, as woff.
    Latin text and Latin-range punctuation stay in Barlow (the face is limited by unicode-range in style.css)."""
    from fontTools.ttLib import TTCollection
    from fontTools import subset
    chars = set(CJK_PUNCT)
    for html_text in pages:
        chars.update(ch for ch in html_text if any(lo <= ord(ch) <= hi for lo, hi in CJK_RANGES))
    chars = sorted(chars)
    sizes = {}
    for weight, src in ((400, "Regular"), (700, "Bold")):
        coll = TTCollection(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{src}.ttc")
        font = next(f for f in coll.fonts if f["name"].getDebugName(1) == "Noto Sans CJK TC")
        font.recalcTimestamp = False     # same input, same bytes, so the version hash only changes when the content does
        opt = subset.Options()
        opt.flavor = "woff"
        opt.layout_features = ["kern", "halt", "palt", "vhal", "vpal", "chws", "vchw", "vert", "vrt2", "locl"]
        opt.notdef_outline = True
        opt.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]
        sub = subset.Subsetter(opt)
        sub.populate(text="".join(chars))
        sub.subset(font)
        out = os.path.join(PF, "fonts", f"NotoSansTC-{weight}.woff")
        font.flavor = "woff"
        font.save(out)
        sizes[weight] = os.path.getsize(out)
    print("cjk glyphs", len(chars), "font bytes", sizes)


def versioned_css():
    """style.css with every font file URL carrying the hash of that font file (fonts first, then the CSS, then the pages)."""
    def one(m):
        rel = m.group(1)
        return f"url({rel}?v={sha(os.path.join(PF, rel))})"
    return re.sub(r"url\((fonts/[^)?]+)\)", one, CSS)


def check_versions():
    """Regression check: read the written pages and the style sheet back and make sure every local static URL carries
    ?v=<hash of the file as it is now>.  HTML-to-HTML links (the language switch) are navigation, not cached assets."""
    problems, n = [], 0
    refs = []
    for lang in (ZH, EN):
        name = LANG[lang]["file"]
        text = open(os.path.join(PF, name), encoding="utf-8").read()
        refs += [(name, u, PF) for u in re.findall(r'\b(?:href|src)="([^"]+)"', text)]
    css = open(os.path.join(PF, "style.css"), encoding="utf-8").read()
    refs += [("style.css", u, PF) for u in re.findall(r"url\(([^)]+)\)", css) if not u.startswith("data:") and not u.startswith("'") and not u.startswith('"')]
    for where, u, base in refs:
        if u.startswith(("#", "http://", "https://", "mailto:", "data:")):
            continue
        path, _, q = u.partition("?")
        if path.endswith(".html"):
            continue
        n += 1
        if not path.endswith(STATIC_EXT):
            problems.append(f"{where}: unexpected local URL {u}")
        elif q != "v=" + sha(os.path.join(base, path)):
            problems.append(f"{where}: {u} lacks the current content hash (expected ?v={sha(os.path.join(base, path))})")
    if problems:
        raise SystemExit("version check FAILED:\n  " + "\n  ".join(problems))
    print("version check ok:", n, "local static URLs, all carry the current ?v= hash")


if __name__ == "__main__":
    ASSET.clear()
    pages_text = [render(lang) for lang in (ZH, EN)]      # first pass: only to learn which CJK characters the pages use
    for lang, out in zip((ZH, EN), pages_text):
        assert not re.search(r"草稿|\bdraft\b|TODO|undefined|NaN|\{|\}", re.sub(r"<[^>]+>", " ", out), re.I), LANG[lang]["file"]
    build_cjk_fonts(pages_text)
    open(os.path.join(PF, "style.css"), "w", encoding="utf-8").write(versioned_css())
    for name in ("style.css", "js/route.js", "paper.pdf", "supplement.pdf"):
        ASSET[name] = f"{name}?v={sha(os.path.join(PF, name))}"
    for lang in (ZH, EN):
        out = render(lang)                                  # second pass: with the content hashes
        open(os.path.join(PF, LANG[lang]["file"]), "w", encoding="utf-8").write(out)
        print("wrote", LANG[lang]["file"], len(out))
    check_versions()
