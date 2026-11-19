# -*- coding: utf-8 -*-
import re, sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
from scorecard_rows import *

import argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser()
_ap.add_argument("--outdir", default=".scratch/scorecard_check",
                 help="where to write the generated files (default .scratch/scorecard_check/; "
                      "'paper' writes the official locations: paper/*.tex and docs/predictions/scorecard_sources.md)")
_args = _ap.parse_args()
_outdir = _args.outdir if os.path.isabs(_args.outdir) else os.path.join(ROOT, _args.outdir)
PREDDIR = ROOT + "/docs/predictions"
TAIL = os.path.join(ROOT, "scripts", "scorecard_tail.md")
# The result files quoted in column "Outcome" (notes/impl/*.md) live in the author's private repository.
# Without them the line numbers of scorecard_sources.md cannot be computed, so only the two .tex files are written.
HAVE_RESULTS = all(os.path.exists(os.path.join(ROOT, f)) for f in RESULT_FILES.values())
OUT_TEX = _outdir + "/appendix_predictions.tex"
OUT_SUPP = _outdir + "/supplement_predictions.tex"
OUT_MD = (ROOT + "/docs/predictions" if os.path.normpath(_outdir) == os.path.join(ROOT, "paper") else _outdir) + "/scorecard_sources.md"
os.makedirs(_outdir, exist_ok=True)
os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)

def load(path):
    with open(path, encoding="utf-8") as f:
        return f.read().split("\n")
_cache = {}
MISSES = []
def lines_of(kind, key):
    if kind == "pred":
        p = PREDDIR + "/" + PRED[key][2]
    else:
        p = ROOT + "/" + RESULT_FILES[key]
    if p not in _cache:
        _cache[p] = load(p)
    return _cache[p]

def find(kind, key, quote):
    if kind != "pred" and not HAVE_RESULTS:
        return 0
    ls = lines_of(kind, key)
    # the private result notes use the author's internal wording for the masked role term (written as escapes here)
    q = quote if kind == "pred" else quote.replace("\u3014\u4f5c\u8005\u3015", "\u4f7f\u7528\u8005")
    hits = [i + 1 for i, l in enumerate(ls) if q in l]
    if not hits:
        MISSES.append((kind, key, quote))
        return 0
    return hits[0]

# ------------------------------------------------------------ LaTeX helpers
def esc(s):
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    rep = [("±", r"$\pm$"), ("≥", r"$\ge$"), ("≤", r"$\le$"), ("×", r"$\times$"),
           ("α", r"$\alpha$"), ("σ", r"$\sigma$"), ("τ", r"$\tau$"), ("≈", r"$\approx$"),
           ("–", "--"), ("−", r"$-$"), ("“", "``"), ("”", "''"),
           ("<", r"$<$"), (">", r"$>$")]
    for a, b in rep:
        s = s.replace(a, b)
    return s

VLABEL = {"held": "held", "failed": "failed", "partly": "partly", "na": "not assessed"}

def vcell(r):
    d = r"$^{\dagger}$" if r["b"] == "T" and r["v"] != "na" else ""
    return VLABEL[r["v"]] + d

# ------------------------------------------------------------ counts
cnt = collections.Counter(r["v"] for r in ROWS)
N = len(ROWS)
assert sum(cnt.values()) == N

# ------------------------------------------------------------ build tables
COLS = r"@{}>{\raggedright\arraybackslash}p{2.2in}>{\raggedright\arraybackslash}p{2.95in}>{\raggedright\arraybackslash}p{0.8in}@{}"
MAXROWS = 9

def table_chunks(gkey, label):
    rs = [r for r in ROWS if r["g"] == gkey]
    chunks = []
    if gkey == "l3":
        chunks = [rs]  # one table, two sub-headers
    else:
        k = (len(rs) + MAXROWS - 1) // MAXROWS
        per = (len(rs) + k - 1) // k
        chunks = [rs[i:i + per] for i in range(0, len(rs), per)]
    out = []
    for ci, ch in enumerate(chunks):
        cap = label
        if len(chunks) > 1:
            cap += " (%d/%d)" % (ci + 1, len(chunks))
        t = []
        t.append(r"\begin{center}\begin{minipage}{\linewidth}\footnotesize")
        t.append(r"\captionof{table}{%s}" % esc(cap))
        t.append(r"\begin{tabular}{%s}" % COLS)
        t.append(r"\toprule")
        t.append(r"Prediction & Outcome & Verdict \\")
        t.append(r"\midrule")
        lastsub = None
        for r in ch:
            if r["sub"] and r["sub"] != lastsub:
                if lastsub is not None:
                    t.append(r"\midrule")
                t.append(r"\multicolumn{3}{@{}l}{\textit{%s}} \\" % esc(r["sub"]))
                t.append(r"\addlinespace[2pt]")
                lastsub = r["sub"]
            t.append("%s & %s & %s \\\\" % (esc(r["p"]), esc(r["o"]), vcell(r)))
            t.append(r"\addlinespace[3pt]")
        t[-1] = r"\bottomrule"
        t.append(r"\end{tabular}")
        t.append(r"\end{minipage}\end{center}")
        out.append("\n".join(t))
    return out

cond_tex = []
cond_tex.append(r"\begin{center}\begin{minipage}{\linewidth}\footnotesize")
cond_tex.append(r"\captionof{table}{Conditions the prediction files named for being wrong, and whether they were triggered. These rows are not counted in the verdicts.}")
cond_tex.append(r"\begin{tabular}{%s}" % COLS)
cond_tex.append(r"\toprule")
cond_tex.append(r"Condition (family) & Outcome & Status \\")
cond_tex.append(r"\midrule")
for c in CONDS:
    status_short = "triggered" if c[3].startswith("Triggered") else ("not triggered" if c[3].startswith("Not triggered") else "not evaluable")
    rest = c[3].split(":", 1)[1].strip() if ":" in c[3] else c[3]
    cond_tex.append("%s (%s) & %s & %s \\\\" % (esc(c[2]), esc(c[1]), esc(rest), status_short))
    cond_tex.append(r"\addlinespace[3pt]")
cond_tex[-1] = r"\bottomrule"
cond_tex.append(r"\end{tabular}")
cond_tex.append(r"\end{minipage}\end{center}")


supp_intro = (
 "Every prediction in the eleven prediction files is listed below with what was measured. "
 "Verdicts are the project's own where its records give one; a dagger marks verdicts we assigned from recorded numbers, "
 "and ``not assessed'' means the records allow none. A prediction stated as a probability is scored by its more likely side. "
 "The predictions were written before each full measurement, but not before all earlier results: "
 "the round-2 soaring file followed two small trial runs whose rough numbers had been seen, "
 "round 3 followed the round-2 results, and the richer rotation file covered 576 settings while 768 were measured.")
assert 80 <= len(supp_intro.replace("``","").replace("''","").split()) <= 120

closing = ("Of the %d predictions in these tables, %d held, %d failed, %d held partly, and %d were not assessed."
           % (N, cnt["held"], cnt["failed"], cnt["partly"], cnt["na"]))

# ---- full table body (supplement)
body_parts = [supp_intro, ""]
for gkey, label in GROUPS:
    for t in table_chunks(gkey, label):
        body_parts.append(t); body_parts.append("")
body_parts.extend(cond_tex); body_parts.append(""); body_parts.append(closing); body_parts.append("")
full_body = "\n".join(body_parts)

# mechanical recount from tex rows of the full body
blocks = full_body.split(r"\begin{center}\begin{minipage}{\linewidth}\footnotesize")
recount = collections.Counter()
for blk in blocks[1:]:
    if "Conditions the prediction files named" in blk:
        continue
    for ln in blk.split("\n"):
        m = re.search(r" & (held|failed|partly|not assessed)(?:\$\^\{\\dagger\}\$)? \\\\$", ln)
        if m:
            recount[m.group(1)] += 1
assert recount["held"] == cnt["held"] and recount["failed"] == cnt["failed"] \
    and recount["partly"] == cnt["partly"] and recount["not assessed"] == cnt["na"], (recount, cnt)
assert sum(recount.values()) == N

pre = [l for l in open(ROOT + "/paper/main.tex", encoding="utf-8").read().split("\n")[:17]]
supp = "\n".join(pre) + r"""
\title{Supplement: all predictions and how they fared}
\author{}
\date{}
\begin{document}
\maketitle
% Auto-generated by scripts/gen_scorecard.py (rows in scripts/scorecard_rows.py); do not edit by hand.
% Who wrote the predictions (author's confirmation, 2026-11-19): the author proposed the direction of each prediction and AI coding agents expanded these into the written files; the text below uses the passive voice on purpose.

""" + full_body + "\n\\end{document}\n"
open(OUT_SUPP, "w", encoding="utf-8").write(supp)

# ---- summary table (appendix)
FAMS = [("BV", "Birds v2"), ("RB", "Rotation basic"), ("RR", "Rotation richer"), ("RN", "Rotation noisy"),
        ("WK", "Wake surfing"), ("SA", "Soaring round 1"), ("SB", "Soaring round 2"), ("SC", "Soaring round 3"),
        ("LM", "Language-model commander"), ("LA", "Neighbours-only learner first"), ("LB", "Neighbours-only learner second")]
def fam_of(r): return re.match(r"[A-Z]+", r["id"]).group(0)
SUMROWS = []
for k, name in FAMS:
    rs = [r for r in ROWS if fam_of(r) == k]
    c = collections.Counter(r["v"] for r in rs)
    assigned = sum(1 for r in rs if r["b"] == "T" and r["v"] != "na")
    SUMROWS.append((name, len(rs), c["held"], c["partly"], c["failed"], c["na"], assigned))
assert sum(x[1] for x in SUMROWS) == N
tot = tuple(sum(x[i] for x in SUMROWS) for i in range(1, 7))
assert tot == (N, cnt["held"], cnt["partly"], cnt["failed"], cnt["na"], sum(1 for r in ROWS if r["b"] == "T" and r["v"] != "na"))
for x in SUMROWS:
    assert x[1] == x[2] + x[3] + x[4] + x[5]

def lnum(kind, key, q): return find(kind, key, q)
l_s2 = lnum("pred", "s2", "寫於正式評估之前。誠實聲明")
l_s3 = lnum("pred", "s3", "我已看過第二輪結果")
l_n = lnum("pred", "rot2", "只讀過 L1 說明與結果")
l_llm = lnum("pred", "llm", "我只看過 oracle 的整體量級")
l_l3b = lnum("pred", "l3b", "看過第一份失敗結果之後")
l_r1 = lnum("pred", "rot1", "共 576 格")
# notes/impl/drone_bucket_l1.md is private; line 22 is the value found there (used when that file is absent)
l_d1 = lnum("res", "D1", "個體差異多加了 ±5% 一級") if HAVE_RESULTS else 22

sum_intro = (
 "This table summarises how the predictions written for this project fared. Verdicts are those we recorded at the time or, "
 "where our records give numbers but no verdict, those we assigned when compiling this table. "
 "A prediction stated as a probability is scored by its more likely side. "
 "The row-by-row table is in the supplement (\\path{supplement_predictions.pdf}) and in "
 "\\path{docs/predictions/scorecard_sources.md}.")
assert len(sum_intro.split()) <= 90, len(sum_intro.split())

disc = (
 "Some of our predictions were written after earlier results or small trial runs: soaring round 2 after two trial runs "
 "(\\path{soaring_headroom2/PREDICTION.md}, line %d), round 3 after the round-2 results (\\path{soaring_headroom3/PREDICTION.md}, line %d), "
 "noisy rotation after the richer-rotation results (\\path{drone_bucket/PREDICTION_L23.md}, line %d), "
 "the language-model commander after a look at the oracle (\\path{llm_commander/PREDICTION.md}, line %d), "
 "and the second learner after the first one failed (\\path{l3_mappo_prereg_v2.md}, section 0). "
 "The richer-rotation prediction covered 576 settings, but 768 were measured because we added a $\\pm$5\\%% level "
 "(\\path{drone_bucket/PREDICTION_L1.md}, line %d; \\path{drone_bucket_l1.md}, line %d). "
 "Verdicts such as ``too high'', ``too low'' or ``right direction, wrong size'' were all recorded as held partly; "
 "we chose this mapping when compiling this table (\\path{scorecard_sources.md}, section 1)."
 % (l_s2, l_s3, l_n, l_llm, l_r1, l_d1))

st = []
st.append(r"\begin{center}\small")
st.append(r"\begin{tabular}{@{}l>{\raggedleft\arraybackslash}p{0.8in}>{\raggedleft\arraybackslash}p{0.38in}>{\raggedleft\arraybackslash}p{0.5in}>{\raggedleft\arraybackslash}p{0.45in}>{\raggedleft\arraybackslash}p{0.62in}>{\raggedleft\arraybackslash}p{0.75in}@{}}")
st.append(r"\toprule")
st.append(r"Family & Predictions & Held & Held partly & Failed & Not assessed & Verdicts assigned for this table \\")
st.append(r"\midrule")
for x in SUMROWS:
    st.append("%s & %d & %d & %d & %d & %d & %d \\\\" % (esc(x[0]), *x[1:]))
st.append(r"\midrule")
st.append("Total & %d & %d & %d & %d & %d & %d \\\\" % tot)
st.append(r"\bottomrule")
st.append(r"\end{tabular}")
st.append(r"\end{center}")
app = "\n".join([
 "% Auto-generated by scripts/gen_scorecard.py (rows in scripts/scorecard_rows.py); do not edit by hand.",
 r"\section{All predictions and how they fared}\label{app:scorecard}", "",
 sum_intro, "", "\n".join(st), "", disc, ""])
open(OUT_TEX, "w", encoding="utf-8").write(app)

# ------------------------------------------------------------ Markdown (zh-TW)
PRIVATE_NOTE_LINES = [
    "> **Note on paths and commit hashes.** Paths that start with `notes/`, `.scratch/` or `docs/writing/`, and commit",
    "> hashes used with `git show`, refer to the author's private development repository. They are not",
    "> part of this public export. The prediction files are copied verbatim into `docs/predictions/`,",
    "> and the commit record is in `docs/provenance.md`.",
    "",
]
ZV = {"held": "held（成立）", "failed": "failed（不成立）", "partly": "partly（部分成立）", "na": "not assessed（無判定）"}
ZB = {"P": "專案紀錄的判定", "T": "本次判定"}

def predsrc(r):
    f, c, _ = PRED[r["pq"][0]]
    ln = find("pred", r["pq"][0], r["pq"][1])
    return "`%s@%s` 第 %d 行：「%s」" % (f, c, ln, r["pq"][1])

def ressrc(item):
    k, q = item
    if k == "raw":
        return "（計算或 JSON 取值）%s" % q
    ln = find("res", k, q)
    if k == "PG":
        return "私人開發紀錄（PROGRESS，第 %d 行）：「%s」" % (ln, q)
    return "`%s` 第 %d 行：「%s」" % (RESULT_FILES[k], ln, q)

md = list(PRIVATE_NOTE_LINES)
md.append("# 預測對照表（附錄 `app:scorecard`）的來源與判定理由")
md.append("")
md.append("日期 2026-11-18。對應檔案：`paper/appendix_predictions.tex`。本檔與 tex 由同一份資料產生（`scripts/scorecard_rows.py`、`scripts/gen_scorecard.py`），所以列數與判定條數一致。")
md.append("")
md.append("## 輸出檔（兩個）")
md.append("")
md.append("1. `paper/appendix_predictions.tex`：論文附錄用的**摘要片段**（可被 `\\input`；一段說明、一張 11 列加 Total 的摘要表、三句揭露）。由 `scripts/gen_scorecard.py` 讀 `scripts/scorecard_rows.py` 產生。")
md.append("2. `paper/supplement_predictions.tex`（編譯成 `paper/supplement_predictions.pdf`）：**補充檔**，含完整逐條表與打臉條件表。同一支 `scripts/gen_scorecard.py` 產生；補充檔 PDF 另用 latexmk 編譯。")
md.append("3. 本檔：逐列來源與判定理由。同一支腳本產生（第 6、7 節來自 `scripts/scorecard_tail.md`）。")
md.append("")
md.append("摘要表各列條數（Predictions／Held／Held partly／Failed／Not assessed／Verdicts assigned for this table）：")
md.append("")
for x in SUMROWS:
    md.append("- %s：%d／%d／%d／%d／%d／%d" % x)
md.append("- Total：%d／%d／%d／%d／%d／%d" % tot)
md.append("")
md.append("「Verdicts assigned for this table」＝本次判定且給了 held／failed／partly 的條數（tex 內有 † 者）；另有 SA17 一條本次處理後判 not assessed，不計入此欄，所以本欄 34、本檔「本次判定」總數 35。")
md.append("")
md.append("## 0. 總數")
md.append("")
md.append("共 %d 條預測：held %d、failed %d、partly %d、not assessed %d（條數由表格列機械數出，tex 內的統計句即此數）。另有 %d 條「預測檔自己寫下的打臉／判讀條件」列在獨立的條件表，不計入上述條數。" % (N, cnt["held"], cnt["failed"], cnt["partly"], cnt["na"], len(CONDS)))
md.append("")
md.append("按家族：")
md.append("")
fam = collections.OrderedDict()
for gkey, label in GROUPS:
    rs = [r for r in ROWS if r["g"] == gkey]
    c = collections.Counter(r["v"] for r in rs)
    md.append("- %s：%d 條（held %d／failed %d／partly %d／not assessed %d）" % (label, len(rs), c["held"], c["failed"], c["partly"], c["na"]))
md.append("")
nP = sum(1 for r in ROWS if r["b"] == "P"); nT = sum(1 for r in ROWS if r["b"] == "T")
md.append("判定依據：專案紀錄的判定 %d 條；本次判定 %d 條（tex 內以 † 標示，not assessed 不加 †）。" % (nP, nT))
md.append("")
md.append("## 1. 判定規則與對映（先看這裡）")
md.append("")
md.append("- 四種判定：held（成立）、failed（不成立）、partly（部分成立）、not assessed（專案紀錄沒有判定，或無法判定）。")
md.append("- 「專案紀錄的判定」＝照 `notes/impl/*.md` 的「與預測對照」節或私人開發紀錄（PROGRESS）寫下的字眼；本次**沒有重新裁決**，只把字眼對映到四類：")
md.append("  - 對／大致對／成立／一致／落在預期內／在範圍內 → held")
md.append("  - 錯／不成立／多於預測／打臉條件成立 → failed")
md.append("  - 部分對／部分錯／方向對（量級錯或高估）／偏高／偏低／數字對原因錯／結論對數字被污染 → partly（預測的方向或定性主張成立，但數字落在預測範圍外）")
md.append("  - 專案寫的是「排他性斷言被反例推翻」（例：「只有……」「低估了 <2% 的範圍」「不成立為通則」）→ failed")
md.append("  - 專案寫「沒量／無結論」→ not assessed")
md.append("- 「本次判定」＝專案紀錄有明確數字、可與預測直接比對，但沒有對這一條下判定；每條都寫了理由。拿不準的寫 not assessed。")
md.append("- 機率型預測（例：「通過 35%」「機率約 20%」）：專案有判定的照專案；沒有判定的，以預測給較高機率的一側是否發生來判（held＝較可能的一側發生）。這是單次結果，不能驗證機率本身；讀者看 Outcome 欄即可。")
md.append("- 有明確數字區間的預測（例：「+2 到 +10」「5–7 格」）：格數在內、點估計有格子超出 → partly；整批都在外 → failed。")
md.append("- 預測檔裡「先驗」「背景」「基準值」「設計說明」等不是預測的段落沒有列入（例：生存 v2 第 2 節基準表、第 5 節解讀規則本身、L3 第 7 節算力預算、旋轉 L1 的「先驗」節、LLM 的「一個先決觀察」）。預測檔的「解讀規則／打臉條件」另列條件表。")
md.append("")
md.append("## 2. 會影響預測可信度的事實（逐家族）")
md.append("")
facts = [
 ("鳥 v2（`2838e68`）", "預測檔自述「不是盲測」：v2 的模型修改是〔作者〕看過 v1 結果後決定，閾值是看過腳本基準後才定（`survival_v2_prereg.md` 第 3 行、第 63 行）。預測 commit 只含文字檔，不含訓練腳本。條件與規模（12 run）與實際執行相同。"),
 ("輪流，基本（`a0619ab`）", "預測檔自述「沒有跑過任何程式」（第 3 行）。預測針對 192 格，量測也是 192 格（`drone_bucket.md` 第 7 行）。"),
 ("輪流，richer（`79b4dbc`）", "預測針對 576 格（預測檔第 5 行：「共 576 格」），最後量測是 768 格：實作時多加了 ±5% 個體差異一級（`drone_bucket_l1.md` 第 22 行：「個體差異多加了 ±5% 一級（規格只要 ±0、±15、±30）」）。預測檔自述「寫這份之前只讀過舊模型結果與馬蹄渦說明檔」。論文圖 2 的「約 230/768」是從 576 格外推（`docs/canonical_numbers.md` 第 86 行）。"),
 ("輪流，noisy（`e8d6059`）", "預測 commit 在 L1 結果 commit（13:26:38）之後 2 分鐘；預測檔自述「只讀過 L1 說明與結果」。預測與量測網格同為 160 格。"),
 ("貼尾流（`f49e6e8`）", "預測檔標題寫「written before any code was run」。同 GPS 參考（`priv_gps`）是診斷用的新增項目，不在預測內；它是在看到完美位置結果之前或之後加入，紀錄找不到（`docs/canonical_numbers.md` 第 113 行）。預測沒有寫網格，結果檔為 108 格 + 5 個錨點。"),
 ("滑翔 第 1 輪（`071574f`）", "預測檔寫「模型尚未跑」。24 格與預測相同。結果檔第 4 節自述調參過程不是一次到位：第一版定心位移太小、第一版共享規則在 dense/short/W3 輸給不共享、離開門檻網格下界被選中，三次都在看到結果之後改規則（`soaring_headroom.md` 第 47 行：「三輪結論方向相同」）。預測 commit 不含程式（只有預測檔）。"),
 ("滑翔 第 2 輪（`4eacb94`）", "預測檔自述：程式寫完後做過兩次小規模 smoke（網格縮成 1/4），「我有看到輸出的大致數字，但下面的預測是依設計與先驗寫的，沒有據此調整」（第 3 行）。結果檔限制節也記了這點（`soaring_headroom2.md` 第 39 行）。事實查核另查到 `smoke.json` mtime 比預測 commit 早 30 秒（`.scratch/audit_20261118/03_paper_factcheck.md` 問題 7）。"),
 ("滑翔 第 3 輪（`02dc53f`）", "預測檔自述：「我已看過第二輪結果（L2-L1 = +3.2 到 +13.9 pp）」（第 3 行）；拆解設計（L1.5a／L1.5b）是看過第 2 輪資料後才定（事實查核問題 7）。"),
 ("LLM 指揮官（`5c53a88`）", "預測檔自述：只看過 oracle 的整體量級（「環境裡好規則與壞規則差約 0-20%」），沒看過 LLM 輸出（第 5 行）；`llm_commander.md` 第 46 行補充：預測是在看過「gate 幾乎總是最好」之後寫的。作答的兩個 AI agent 被要求不用工具，但各自有 1 次工具呼叫，內容未查（私人開發紀錄，PROGRESS 第 388 行）。14 條指令與預測相同。"),
 ("學習者 第 1 份（`38ce89e`）", "預測檔自述已看過 ≤20 次更新的接線除錯跑（第 9 行），沒有任何訓練過的 L3 結果。預測 commit 同時含訓練程式。"),
 ("學習者 第 2 份（`0e16639`）", "預測檔自述這是「看過第一份失敗結果之後」的第二次嘗試（第 0 節），看過 3 格×3 種子×3 個 U 的逐種子 F_G 與訓練 log；〔作者〕決定只再試這一次。"),
 ("全部", "預測由誰撰寫：作者已於 2026-11-19 確認，作者提出每份預測的方向，AI agent 擴寫成預測檔（`docs/canonical_numbers.md` 第 114 行）；tex 說明段落仍用被動語態。時間戳是 git author time，可被改動（論文附錄 A 已揭露）。"),
]
for t, s in facts:
    md.append("- **%s**：%s" % (t, s))
md.append("")
md.append("## 3. 逐列來源")
md.append("")
md.append("格式：列編號｜預測（英文轉述）｜預測檔@commit 行號與原文｜結果出處與原文｜判定與依據。行號指檔案內的行；預測檔取自預測 commit 當時的版本（`git show <hash>:<path>`），結果檔取自目前工作目錄。")
md.append("")
cur = None
for gkey, label in GROUPS:
    md.append("### %s" % label)
    md.append("")
    for r in [x for x in ROWS if x["g"] == gkey]:
        md.append("#### %s" % r["id"])
        md.append("- 預測（表內）：%s" % r["p"])
        md.append("- 預測原文：%s" % predsrc(r))
        for it in r["rs"]:
            md.append("- 結果出處：%s" % ressrc(it))
        md.append("- 結果（表內）：%s" % r["o"])
        md.append("- 判定：**%s**｜依據：%s｜%s" % (ZV[r["v"]], ZB[r["b"]], r["why"]))
        md.append("")
md.append("## 4. 預測檔自己寫下的打臉／判讀條件（條件表，不計入條數）")
md.append("")
for c in CONDS:
    f, cc, _ = PRED[c[4][0]]
    ln = find("pred", c[4][0], c[4][1])
    md.append("#### %s %s" % (c[0], c[1]))
    md.append("- 條件（表內）：%s" % c[2])
    md.append("- 原文：`%s@%s` 第 %d 行：「%s」" % (f, cc, ln, c[4][1]))
    for it in c[5]:
        md.append("- 結果出處：%s" % ressrc(it))
    md.append("- 狀態：%s" % c[3])
    md.append("")
md.append("## 5. 找不到結果或判定的預測")
md.append("")
na = [r for r in ROWS if r["v"] == "na"]
for r in na:
    md.append("- **%s**（%s）：%s　→ %s" % (r["id"], [l for k, l in GROUPS if k == r["g"]][0], r["p"], r["why"]))
md.append("")
md.append("另外，下列預測專案紀錄有結果文字、但沒有判定字眼，由本次判定（共 %d 條 held/failed/partly，tex 內以 † 標示）：" % sum(1 for r in ROWS if r["b"] == "T" and r["v"] != "na"))
md.append("")
md.append("、".join("%s" % r["id"] for r in ROWS if r["b"] == "T" and r["v"] != "na"))
md.append("")
md.append("以下**沒有**對專案已寫下的判定做任何改動；對映（例：「偏低」→ partly）的規則見第 1 節，如果作者希望更嚴或更寬，只需改 `scripts/scorecard_rows.py` 的 `v` 欄後重新產生。")
md.append("")
tail = open(TAIL, encoding="utf-8").read()
dag = sum(1 for r in ROWS if r["b"] == "T" and r["v"] != "na")
tail = tail.replace("%PCOUNT%", str(nP)).replace("%TCOUNT%", str(nT)).replace("%DAG%", str(dag)).replace("%HELD%", str(cnt["held"])).replace("%FP%", str(cnt["failed"] + cnt["partly"]))
if HAVE_RESULTS:
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md) + "\n" + tail)
else:
    print("note: notes/impl result files not found (public export); wrote the two .tex files only, not scorecard_sources.md")
if MISSES:
    for m in MISSES: print("MISS", m)
    raise SystemExit(1)
print("rows", N, dict(cnt), "P", nP, "T", nT)
