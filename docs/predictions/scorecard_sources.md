> **Note on paths and commit hashes.** Paths that start with `notes/`, `.scratch/` or `docs/writing/`, and commit
> hashes used with `git show`, refer to the author's private development repository. They are not
> part of this public export. The prediction files are copied verbatim into `docs/predictions/`,
> and the commit record is in `docs/provenance.md`.

# 預測對照表（附錄 `app:scorecard`）的來源與判定理由

日期 2026-11-18。對應檔案：`paper/appendix_predictions.tex`。本檔與 tex 由同一份資料產生（`scripts/scorecard_rows.py`、`scripts/gen_scorecard.py`），所以列數與判定條數一致。

## 輸出檔（兩個）

1. `paper/appendix_predictions.tex`：論文附錄用的**摘要片段**（可被 `\input`；一段說明、一張 11 列加 Total 的摘要表、三句揭露）。由 `scripts/gen_scorecard.py` 讀 `scripts/scorecard_rows.py` 產生。
2. `paper/supplement_predictions.tex`（編譯成 `paper/supplement_predictions.pdf`）：**補充檔**，含完整逐條表與打臉條件表。同一支 `scripts/gen_scorecard.py` 產生；補充檔 PDF 另用 latexmk 編譯。
3. 本檔：逐列來源與判定理由。同一支腳本產生（第 6、7 節來自 `scripts/scorecard_tail.md`）。

摘要表各列條數（Predictions／Held／Held partly／Failed／Not assessed／Verdicts assigned for this table）：

- Birds v2：5／4／0／1／0／0
- Rotation basic：12／5／3／4／0／2
- Rotation richer：14／4／4／5／1／1
- Rotation noisy：11／3／5／3／0／1
- Wake surfing：12／5／1／5／1／2
- Soaring round 1：18／6／4／4／4／2
- Soaring round 2：14／5／7／2／0／8
- Soaring round 3：9／6／2／1／0／6
- Language-model commander：12／6／1／5／0／10
- Neighbours-only learner first：2／0／0／1／1／0
- Neighbours-only learner second：4／1／0／3／0／2
- Total：113／45／27／34／7／34

「Verdicts assigned for this table」＝本次判定且給了 held／failed／partly 的條數（tex 內有 † 者）；另有 SA17 一條本次處理後判 not assessed，不計入此欄，所以本欄 34、本檔「本次判定」總數 35。

## 0. 總數

共 113 條預測：held 45、failed 34、partly 27、not assessed 7（條數由表格列機械數出，tex 內的統計句即此數）。另有 14 條「預測檔自己寫下的打臉／判讀條件」列在獨立的條件表，不計入上述條數。

按家族：

- Birds v2 (survival game, learned policies)：5 條（held 4／failed 1／partly 0／not assessed 0）
- Rotation, basic：12 條（held 5／failed 4／partly 3／not assessed 0）
- Rotation, richer：14 條（held 4／failed 5／partly 4／not assessed 1）
- Rotation, noisy：11 條（held 3／failed 3／partly 5／not assessed 0）
- Wake surfing：12 條（held 5／failed 5／partly 1／not assessed 1）
- Soaring, round 1：18 條（held 6／failed 4／partly 4／not assessed 4）
- Soaring, round 2：14 條（held 5／failed 2／partly 7／not assessed 0）
- Soaring, round 3：9 條（held 6／failed 1／partly 2／not assessed 0）
- Language-model commander：12 條（held 6／failed 5／partly 1／not assessed 0）
- Neighbours-only learner：6 條（held 1／failed 4／partly 0／not assessed 1）

判定依據：專案紀錄的判定 78 條；本次判定 35 條（tex 內以 † 標示，not assessed 不加 †）。

## 1. 判定規則與對映（先看這裡）

- 四種判定：held（成立）、failed（不成立）、partly（部分成立）、not assessed（專案紀錄沒有判定，或無法判定）。
- 「專案紀錄的判定」＝照 `notes/impl/*.md` 的「與預測對照」節或私人開發紀錄（PROGRESS）寫下的字眼；本次**沒有重新裁決**，只把字眼對映到四類：
  - 對／大致對／成立／一致／落在預期內／在範圍內 → held
  - 錯／不成立／多於預測／打臉條件成立 → failed
  - 部分對／部分錯／方向對（量級錯或高估）／偏高／偏低／數字對原因錯／結論對數字被污染 → partly（預測的方向或定性主張成立，但數字落在預測範圍外）
  - 專案寫的是「排他性斷言被反例推翻」（例：「只有……」「低估了 <2% 的範圍」「不成立為通則」）→ failed
  - 專案寫「沒量／無結論」→ not assessed
- 「本次判定」＝專案紀錄有明確數字、可與預測直接比對，但沒有對這一條下判定；每條都寫了理由。拿不準的寫 not assessed。
- 機率型預測（例：「通過 35%」「機率約 20%」）：專案有判定的照專案；沒有判定的，以預測給較高機率的一側是否發生來判（held＝較可能的一側發生）。這是單次結果，不能驗證機率本身；讀者看 Outcome 欄即可。
- 有明確數字區間的預測（例：「+2 到 +10」「5–7 格」）：格數在內、點估計有格子超出 → partly；整批都在外 → failed。
- 預測檔裡「先驗」「背景」「基準值」「設計說明」等不是預測的段落沒有列入（例：生存 v2 第 2 節基準表、第 5 節解讀規則本身、L3 第 7 節算力預算、旋轉 L1 的「先驗」節、LLM 的「一個先決觀察」）。預測檔的「解讀規則／打臉條件」另列條件表。

## 2. 會影響預測可信度的事實（逐家族）

- **鳥 v2（`2838e68`）**：預測檔自述「不是盲測」：v2 的模型修改是〔作者〕看過 v1 結果後決定，閾值是看過腳本基準後才定（`survival_v2_prereg.md` 第 3 行、第 63 行）。預測 commit 只含文字檔，不含訓練腳本。條件與規模（12 run）與實際執行相同。
- **輪流，基本（`a0619ab`）**：預測檔自述「沒有跑過任何程式」（第 3 行）。預測針對 192 格，量測也是 192 格（`drone_bucket.md` 第 7 行）。
- **輪流，richer（`79b4dbc`）**：預測針對 576 格（預測檔第 5 行：「共 576 格」），最後量測是 768 格：實作時多加了 ±5% 個體差異一級（`drone_bucket_l1.md` 第 22 行：「個體差異多加了 ±5% 一級（規格只要 ±0、±15、±30）」）。預測檔自述「寫這份之前只讀過舊模型結果與馬蹄渦說明檔」。論文圖 2 的「約 230/768」是從 576 格外推（`docs/canonical_numbers.md` 第 86 行）。
- **輪流，noisy（`e8d6059`）**：預測 commit 在 L1 結果 commit（13:26:38）之後 2 分鐘；預測檔自述「只讀過 L1 說明與結果」。預測與量測網格同為 160 格。
- **貼尾流（`f49e6e8`）**：預測檔標題寫「written before any code was run」。同 GPS 參考（`priv_gps`）是診斷用的新增項目，不在預測內；它是在看到完美位置結果之前或之後加入，紀錄找不到（`docs/canonical_numbers.md` 第 113 行）。預測沒有寫網格，結果檔為 108 格 + 5 個錨點。
- **滑翔 第 1 輪（`071574f`）**：預測檔寫「模型尚未跑」。24 格與預測相同。結果檔第 4 節自述調參過程不是一次到位：第一版定心位移太小、第一版共享規則在 dense/short/W3 輸給不共享、離開門檻網格下界被選中，三次都在看到結果之後改規則（`soaring_headroom.md` 第 47 行：「三輪結論方向相同」）。預測 commit 不含程式（只有預測檔）。
- **滑翔 第 2 輪（`4eacb94`）**：預測檔自述：程式寫完後做過兩次小規模 smoke（網格縮成 1/4），「我有看到輸出的大致數字，但下面的預測是依設計與先驗寫的，沒有據此調整」（第 3 行）。結果檔限制節也記了這點（`soaring_headroom2.md` 第 39 行）。事實查核另查到 `smoke.json` mtime 比預測 commit 早 30 秒（`.scratch/audit_20261118/03_paper_factcheck.md` 問題 7）。
- **滑翔 第 3 輪（`02dc53f`）**：預測檔自述：「我已看過第二輪結果（L2-L1 = +3.2 到 +13.9 pp）」（第 3 行）；拆解設計（L1.5a／L1.5b）是看過第 2 輪資料後才定（事實查核問題 7）。
- **LLM 指揮官（`5c53a88`）**：預測檔自述：只看過 oracle 的整體量級（「環境裡好規則與壞規則差約 0-20%」），沒看過 LLM 輸出（第 5 行）；`llm_commander.md` 第 46 行補充：預測是在看過「gate 幾乎總是最好」之後寫的。作答的兩個 AI agent 被要求不用工具，但各自有 1 次工具呼叫，內容未查（私人開發紀錄，PROGRESS 第 388 行）。14 條指令與預測相同。
- **學習者 第 1 份（`38ce89e`）**：預測檔自述已看過 ≤20 次更新的接線除錯跑（第 9 行），沒有任何訓練過的 L3 結果。預測 commit 同時含訓練程式。
- **學習者 第 2 份（`0e16639`）**：預測檔自述這是「看過第一份失敗結果之後」的第二次嘗試（第 0 節），看過 3 格×3 種子×3 個 U 的逐種子 F_G 與訓練 log；〔作者〕決定只再試這一次。
- **全部**：預測由誰撰寫：作者已於 2026-11-19 確認，作者提出每份預測的方向，AI agent 擴寫成預測檔（`docs/canonical_numbers.md` 第 114 行）；tex 說明段落仍用被動語態。時間戳是 git author time，可被改動（論文附錄 A 已揭露）。

## 3. 逐列來源

格式：列編號｜預測（英文轉述）｜預測檔@commit 行號與原文｜結果出處與原文｜判定與依據。行號指檔案內的行；預測檔取自預測 commit 當時的版本（`git show <hash>:<path>`），結果檔取自目前工作目錄。

### Birds v2 (survival game, learned policies)

#### BV1
- 預測（表內）：A2 (MAPPO, α=1) sustains rotation in at least 2 of 3 seeds (confidence: medium).
- 預測原文：`notes/impl/survival_v2_prereg.md@2838e68` 第 46 行：「A2（MAPPO α=1）維持，≥2/3（多數）」
- 結果出處：`notes/impl/survival_v2_results.md` 第 27 行：「| A2 | 維持 ≥2/3（多數） | 3/3 維持 → **成立** | 1.0010±0.0171 |」
- 結果（表內）：3 of 3 seeds sustained; seed-level reward 1.0010 ± 0.0171.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案結果檔條件層級表直接寫「成立」。

#### BV2
- 預測（表內）：A2 shows significant energy-aware hand-over: at least 2/3 of the sustaining runs energy-aware (confidence about 40%).
- 預測原文：`notes/impl/survival_v2_prereg.md@2838e68` 第 47 行：「A2 能量感知顯著**（≥2/3 的維持 run 為能量感知）」
- 結果出處：`notes/impl/survival_v2_results.md` 第 32 行：「3 個維持 run 全為「能量盲」（0.343/0.337/0.338 ≤0.40）」
- 結果出處：`notes/impl/survival_v2_results.md` 第 32 行：「「能量感知顯著」**不成立**」
- 結果（表內）：All 3 sustaining runs energy-blind: incoming leader had the highest energy in 0.343, 0.337, 0.338 of hand-overs (threshold 0.40); seed mean 0.3394 ± 0.0074.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案結果檔明寫「不成立」。預測附帶的信心「約 40%」不改變判定（主張本身未成立）。

#### BV3
- 預測（表內）：B2 (IPPO, α=1) collapses in at least 2 of 3 seeds (confidence: medium).
- 預測原文：`notes/impl/survival_v2_prereg.md@2838e68` 第 48 行：「B2（IPPO α=1）塌縮，≥2/3**」
- 結果出處：`notes/impl/survival_v2_results.md` 第 28 行：「| B2 | 塌縮 ≥2/3 | 2 塌縮 + 1 中間 → **多數（2/3）** | 0.9470±0.0431 |」
- 結果出處：`notes/impl/survival_v2_results.md` 第 45 行：「若判塌縮則 B2 為 3/3」
- 結果（表內）：2 collapsed and 1 “middle” (last-decile swap rate 0.0050, exactly on the collapse boundary); condition-level result “majority (2/3)”.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案紀錄判為「多數（2/3）」，符合預測「≥2/3」。注意 B2 s21 在邊界上被判「中間」，結果檔 L45 已自述。

#### BV4
- 預測（表內）：C2 (IPPO, α=0) collapses in at least 2 of 3 seeds (confidence: medium-high).
- 預測原文：`notes/impl/survival_v2_prereg.md@2838e68` 第 49 行：「C2（IPPO α=0）塌縮，≥2/3**」
- 結果出處：`notes/impl/survival_v2_results.md` 第 29 行：「| C2 | 塌縮 ≥2/3 | 3/3 → **成立** | 0.8253±0.0009 |」
- 結果（表內）：3 of 3 collapsed to a fixed leader (reward 0.8251, 0.8258, 0.8251).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案結果檔寫「成立」。

#### BV5
- 預測（表內）：D2 (MAPPO, α=0) does not sustain: at most 1 of 3 seeds sustains (confidence: medium).
- 預測原文：`notes/impl/survival_v2_prereg.md@2838e68` 第 50 行：「D2（MAPPO α=0）不維持（維持 ≤1/3）**」
- 結果出處：`notes/impl/survival_v2_results.md` 第 30 行：「| D2 | 維持 ≤1/3 | 0/3 維持 → **成立** | 0.8624±0.1603 |」
- 結果（表內）：0 of 3 sustained (two collapsed, one “middle”); seed-level reward 0.8624 ± 0.1603.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案結果檔寫「成立」。

### Rotation, basic

#### RB1
- 預測（表內）：Homogeneous fleet, c_swap 0 or 1: time-rotation gap below 2% (about 0.5–1.5%).
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 8 行：「c_swap = 0、1：差距 < 2%（約 0.5–1.5%）」
- 結果出處：`notes/impl/drone_bucket.md` 第 151 行：「| 同質、c≤1：差距 < 2% | 24 格中 22 格 < 2%；N=5、N=7 的 s=0.3, c=1 為 2.04%、2.08%，略破 | 大致對 |」
- 結果（表內）：22 of 24 cells below 2%; two cells (N=5 and N=7, s=0.3, c=1) at 2.04% and 2.08%.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「大致對」，對映為 held。

#### RB2
- 預測（表內）：Homogeneous: the only cells that might exceed 2% are s=0.3 with N=10.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 9 行：「唯一可能破 2% 的是 s=0.3 且 N=10 的格子」
- 結果出處：`notes/impl/drone_bucket.md` 第 152 行：「| 同質、s=0.3, N=10 可能破 2% | N=10,s=0.3 同質 c≤3 為 0.9–1.5%，未破（c=10 才 2.9%）；N=5,7 的 s=0.3 先破 | 錯：以為 N 大越糟 |」
- 結果（表內）：N=10, s=0.3, c≤3: 0.9–1.5%, not above 2% (2.9% only at c=10); N=5 and N=7 with s=0.3 crossed 2% first.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### RB3
- 預測（表內）：Homogeneous, c_swap=3: borderline, about 1–3%; above 2% more likely for N≥7 and s≥0.2.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 10 行：「c_swap = 3：臨界，約 1–3%；N≥7 與 s≥0.2 較可能 > 2%」
- 結果出處：`notes/impl/drone_bucket.md` 第 153 行：「| 同質、c=3 臨界 1–3% | 0.2–3.3%，s=0.3 破、s=0.1 遠低於 | 對 |」
- 結果（表內）：0.2–3.3%; above 2% at s=0.3, far below at s=0.1.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### RB4
- 預測（表內）：Homogeneous, c_swap=10: most cells above 2% (about 3–8%); N=3 an exception.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 11 行：「c_swap = 10：多數格子 > 2%（約 3–8%）」
- 結果出處：`notes/impl/drone_bucket.md` 第 154 行：「| 同質、c=10：多數 > 2%，N=3 例外 | 12 格中 7 格 > 2%（s≥0.2 幾乎都破，s=0.1 全 < 2%）；N=3 不是例外，s=0.3 達 4.2% | 部分對：漏了 s=0.1 全不破；N=3 錯 |」
- 結果（表內）：7 of 12 cells above 2% (nearly all with s≥0.2, none with s=0.1); N=3 not an exception, 4.2% at s=0.3.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「部分對」。

#### RB5
- 預測（表內）：±5% individual difference: all cells with N≥5 above 2% (about 3–9%); N=3 with s=0.1 may be below 2%.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 12 行：「個體差異 ±5%：N≥5 時全部 > 2%（約 3–9%，N 越大越大）」
- 結果出處：`notes/impl/drone_bucket.md` 第 155 行：「| ±5%、N≥5 全部 > 2% | N=5 全破；N=7,10 在 s=0.1 時 1.2–2.3%，只有 N=7,c=10 一格破 | 部分錯：漏了 s=0.1 的補償上限效應（我只在 N=3 提到） |」
- 結果出處：`notes/impl/drone_bucket.md` 第 47 行：「| 3 | 0.1 | **2.7 (2.6–2.9)** | **2.8 (2.6–2.9)** | **2.8 (2.7–3.0)** | **3.0 (2.8–3.1)** |」
- 結果（表內）：N=5 all above 2%; N=7 and N=10 at s=0.1 gave 1.2–2.3%, only one cell (N=7, c=10) above 2%; N=3, s=0.1 was 2.7–3.0%.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「部分錯」，對映為 partly（N=5 全破成立，N 越大越大不成立）。N=3,s=0.1 的 2.7–3.0% 取自差距表。

#### RB6
- 預測（表內）：±15% and ±30%: nearly all cells above 2%, and above 5%.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 14 行：「個體差異 ±15%、±30%：幾乎所有格子 > 2%，且 > 5%」
- 結果出處：`notes/impl/drone_bucket.md` 第 156 行：「| ±15%、±30% 幾乎全 > 2% 且 > 5% | 83% 格 > 2%（80/96，除 s=0.1 且 N≥7 外全破），但 > 5% 只有 40/96（中位 4.3–4.4%，最大 12%） | 方向對，量級高估 |」
- 結果（表內）：80 of 96 cells (83%) above 2%, but only 40 of 96 above 5% (median 4.3–4.4%, maximum 12%).
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「方向對，量級高估」，對映為 partly（見判定規則）。

#### RB7
- 預測（表內）：The gap grows monotonically with N (and with individual difference).
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 15 行：「差距隨 N 與個體差異單調上升」
- 結果出處：`notes/impl/drone_bucket.md` 第 157 行：「| 差距隨 N 單調上升 | 相反：h>0 時平均差距 N=3: 6.0%、N=5: 5.0%、N=7: 4.0%、N=10: 3.3%，隨 N 下降 |」
- 結果（表內）：Opposite for N: with h>0 the mean gap was 6.0% (N=3), 5.0% (N=5), 4.0% (N=7), 3.3% (N=10).
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」；專案只判了 N 的部分，個體差異部分沒有單獨判定。

#### RB8a
- 預測（表內）：“Most settings have a gap below 2%” will not hold.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 16 行：「「多數設定下差距 < 2%」不成立」
- 結果出處：`notes/impl/drone_bucket.md` 第 7 行：「最優與「時間輪退尾（每格取最佳 τ）」的差距 > 2% 的有 **132 格（69%）**」
- 結果出處：`notes/impl/drone_bucket.md` 第 7 行：「「多數設定差距 < 2%」的停止條件**沒有成立**」
- 結果（表內）：132 of 192 cells (69%) above 2%; the stop condition “most settings below 2%” was not met.
- 判定：**held（成立）**｜依據：本次判定｜專案結論第一點寫出 132/192 與停止條件未成立，但它的「與預測對照」表沒有單列這一條，故標本次判定。預測與結果的網格同為 192 格。

#### RB8b
- 預測（表內）：The only cells below 2% are homogeneous with c_swap≤1: about 1/8 or fewer of 192 cells.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 16 行：「真正 < 2% 的只有同質且 c_swap ≤ 1 的格子，約占全部 192 格中 1/8 以下」
- 結果出處：`notes/impl/drone_bucket.md` 第 158 行：「| 結論：真正 < 2% 的只有同質 c≤1，約 1/8 以下 | < 2% 的有 60 格（31%），其中 23 格是有個體差異但 s=0.1, N≥7 | 低估了 < 2% 的範圍 |」
- 結果（表內）：60 cells (31%) below 2%, including 23 heterogeneous cells with s=0.1 and N≥7.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「低估了 < 2% 的範圍」。預測是「只有……」的排他性斷言且被反例推翻，對映為 failed（對映規則見開頭）。

#### RB9
- 預測（表內）：The battery gate (3) is within 2–3% of optimal for c_swap≤3.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 17 行：「電量閘門 (3) 預期在 c_swap ≤ 3 時與最優差 < 2–3%」
- 結果出處：`notes/impl/drone_bucket.md` 第 159 行：「| 閘門 (3) 在 c≤3 與最優差 < 2–3% | 全部格子差 ≤ 2.2%，c=10 才少數破 2% | 對，比預期更好 |」
- 結果（表內）：Gate gap 0.0%–2.2% everywhere; only 4 cells above 2% (all s=0.3, c_swap=10).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對，比預期更好」。

#### RB10
- 預測（表內）：RL room lies mainly in the hidden-coefficient version (k_i not visible).
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 18 行：「RL 的空間主要在看不到 k_i 的隱藏係數版本」
- 結果出處：`notes/impl/drone_bucket.md` 第 10 行：「用耗電估 k_i 的閘門也幾乎不退化」
- 結果出處：`notes/impl/drone_bucket.md` 第 143 行：「| 0.1 | 中位 3.1% | 1.2% | 中位 -0.01% | -0.75% |」
- 結果（表內）：A gate using estimated k_i still beat time rotation (median gain 3.1%) and matched the true-k gate (median −0.01%, worst −0.75% at 10% noise).
- 判定：**failed（不成立）**｜依據：本次判定｜專案的「與預測對照」表沒有這一條；但結果檔結論與第 5 節直接寫出隱藏版也幾乎不退化（即該版本沒有留給 RL 的空間），與預測相反，故本次判為 failed。

#### RB11
- 預測（表內）：Hidden version: with noiseless drain readings the gate barely degrades; with 10% noise degradation is small (below 1–2%) and still beats time rotation.
- 預測原文：`.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 20 行：「有 10% 雜訊時退化小（< 1–2%），仍優於時間輪」
- 結果出處：`notes/impl/drone_bucket.md` 第 160 行：「| 隱藏版：σ=10% 退化 < 1–2% | 對閘門(真值)的退化中位 ≈ 0，最差 -0.75% | 對 |」
- 結果（表內）：Degradation versus true-k gate: median ≈ 0, worst −0.75%; still better than time rotation for h>0.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

### Rotation, richer

#### RR1
- 預測（表內）：±0% individual difference: best rule within 2% of the bound in all 144 cells, at most 3 exceptions.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 14 行：「個體差異 +-0%：最佳規則（預期是時間輪，τ 約 D/N）對最優差距 < 2%，全部 144 格中至多 3 格破」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 94 行：「| ±0：最佳規則 < 2%，至多 3 格破 | 1 格（2.18%，撞時間上限），其餘 < 1.6% | 對 |」
- 結果（表內）：1 cell above 2% (2.18%, N=10, echelon, c=10, solver hit its time limit; 1.89% on the primal solution); all others below 1.6%.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。預測針對 576 格，量測為 768 格（多加 ±5% 一級）；±0% 子集的格數兩者相同。

#### RR2
- 預測（表內）：Rule (d) alone may reach 2–3% at c_swap≥3 (homogeneous).
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 15 行：「(d) 單獨在 c>=3 可能 2-3%」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 95 行：「| ±0 時 (d) c≥3 可能 2–3% | (d) 破 2% 的 10 格全是 c=10（8 格同質、2 格 ±5%）；c=3 最大 1.9% | 部分對（只有 c=10 破） |」
- 結果（表內）：Rule (d) above 2% in 10 cells, all at c_swap=10 (8 homogeneous, 2 at ±5%); at c_swap=3 the maximum is 1.9%.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「部分對」。

#### RR3
- 預測（表內）：±15%: best-rule gap 1–5%, about 30% of cells above 2%.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 16 行：「+-15%：最佳規則差距落在 1-5%。> 2% 集中在 f=0.5 或 0.4、最佳設定、N>=5、c>=3 的格子，約 30% 的格子破 2%」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 96 行：「| ±15%：差距 1–5%，約 30% 的格子破 2% | 0 格破；最差格 0.59% | **錯** |」
- 結果（表內）：0 cells above 2%; worst cell 0.59%.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### RR4
- 預測（表內）：±30%: about 45% of cells above 2%, maximum 5–8%.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 18 行：「+-30%：差距更大，約 45% 的格子破 2%，最大值 5-8%」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 97 行：「| ±30%：約 45% 破 2%，最大 5–8% | 0 格破；最大 0.23% | **錯** |」
- 結果（表內）：0 cells above 2%; maximum 0.23%.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### RR5
- 預測（表內）：All 576 cells: best rule above 2% in about 30% of cells (range 15–45%).
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 19 行：「全 576 格合計：最佳規則 > 2% 的格數約 30%（區間 15-45%）」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 98 行：「| 全表最佳規則 > 2% 約 30%（15–45%） | 1/768 = 0.13% | **錯，差兩個數量級** |」
- 結果（表內）：1 of 768 cells (0.13%); the grid measured had 768 cells, not 576.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯，差兩個數量級」。網格 576 → 768（見 drone_bucket_l1.md L22）。

#### RR6
- 預測（表內）：Rule (d) alone exceeds 2% in 5–10 percentage points more cells than the best rule.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 19 行：「(d) 單獨 > 2% 的格數比最佳規則多約 5-10 個百分點」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 99 行：「| (d) 單獨比最佳規則多 5–10 個百分點 | 10 格 vs 1 格，約 1.2 個百分點 | 方向對，量級錯 |」
- 結果（表內）：10 cells versus 1 cell, about 1.2 percentage points.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「方向對，量級錯」，對映為 partly。

#### RR7
- 預測（表內）：N=3 V-formation (leader plus two equivalent followers): best rule below 2% in all cells with c_swap≤3.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 20 行：「N=3 V 形只有「領頭 / 兩個等價跟機」兩級，等同舊模型，預期最佳規則（電量閘門）< 2%（所有 c<=3 的格子）」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 100 行：「| N=3 V 形等同舊模型，< 2% | 最大 1.42% | 對 |」
- 結果（表內）：Maximum 1.42%.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### RR8
- 預測（表內）：N=3 echelon (three levels) has a gap slightly larger than the V.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 21 行：「梯隊 N=3 有三級，略大於 V。」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 62 行：「| 3 | V | 實際 / 最佳 | 0.96 / 0.94 | 1.20 / 1.09 |」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 63 行：「| 3 | 梯隊 | 實際 / 最佳 | 0.60 / 0.89 | 0.84 / 1.44 |」
- 結果（表內）：Not compared in the record. Worst best-rule gap at ±0% (actual/best setting): V 0.96/0.94, echelon 0.60/0.89.
- 判定：**not assessed（無判定）**｜依據：專案紀錄的判定｜專案「與預測對照」沒有這一條。差距表在 ±0% 與 ±5% 各層的 V、梯隊大小互有高低，不足以判定，故 not assessed。

#### RR9
- 預測（表內）：The gap rises monotonically with individual difference, f_ind, position setting (actual to best) and c_swap.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 22 行：「維度趨勢：差距隨個體差異、f_ind、位置設定從實際到最佳、c 單調上升」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 101 行：「| 差距隨個體差異、f_ind、設定、c 單調上升 | f_ind、設定、c 單調上升；個體差異**相反**：同質最大，異質反而最小 | 個體差異方向**錯** |」
- 結果（表內）：Rose with f_ind, setting and c_swap; fell with individual difference (largest for homogeneous cells).
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案寫「個體差異方向錯」，其餘三維成立；四維合成一條，對映為 partly。

#### RR10
- 預測（表內）：The trend with N is not monotone.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 22 行：「我預期不單調」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 102 行：「| 隨 N 不單調 | 不單調 | 對 |」
- 結果（表內）：Not monotone (mean gap 0.24, 0.19, 0.16, 0.18% for N=3, 5, 7, 10).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### RR11
- 預測（表內）：Best rule is (d) or (e) in heterogeneous cells and time rotation (b) in homogeneous cells.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 23 行：「最佳規則預期是 (d) 或 (e)（成本感知）在個體差異格子」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 104 行：「| 最佳規則：異質格 (d)/(e)，同質格 (b) | 閘門最多；同質格最佳是閘門／時間輪／單對調混合 | 錯 |」
- 結果（表內）：Gate (c) was best in 374 of 768 cells; in homogeneous cells the best was a mix of gate, time rotation and single swap.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### RR12
- 預測（表內）：For N≥5 the gate (c) loses to (d)/(e) by 0.5–2 percentage points.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 23 行：「閘門 (c) 在 N>=5 因為只能換領頭，用不到深位置，預期輸給 (d)/(e) 0.5-2 個百分點」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 103 行：「| 閘門 (c) 輸給 (d)/(e) 0.5–2 個百分點 | 閘門是最佳規則的占 374 格（最多），差距幾乎都在 0.1% 內 | **錯** |」
- 結果（表內）：Gate was the best rule most often (374 cells); gaps almost all within 0.1%.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### RR13
- 預測（表內）：Most cells below 2%, so RL is not worth trying; it is worth it only in the ±15/30%, large f_ind, best-setting, c_swap≥3 blocks.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 24 行：「結論預測：多數格子 < 2%，不值得上 RL；值得的是 +-15/30%、f_ind 大、最佳設定、c>=3 的區塊。」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 8 行：「**最佳規則對流體上界的差距 > 2% 的只有 1 格**」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 80 行：「個體差異 ±15%：所有 (N,隊形,設定) 最差格 ≤ 0.59%」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 81 行：「個體差異 ±30%：最差格 ≤ 0.23%」
- 結果（表內）：767 of 768 cells below 2% (first half held); but the ±15% and ±30% blocks had no cell above 2% (worst 0.59% and 0.23%).
- 判定：**partly（部分成立）**｜依據：本次判定｜專案「與預測對照」沒有單列這一條。前半（多數 <2%）成立；後半點名的區塊（±15%、±30%）在專案結果中沒有任何一格超過 2%，故本次判為 partly。

#### RR14
- 預測（表內）：Anchors: new simulator equals the old one for fixed, time rotation and gate; new LP equals old MILP at c=0 and may differ by under 0.5% at c>0.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L1.md@79b4dbc` 第 25 行：「錨點（所有跟飛位置節省相同）：新模擬器與舊 drone_bucket.simulate 在固定、時間輪、閘門三條上航程比相同到浮點誤差」
- 結果出處：`notes/impl/drone_bucket_l1.md` 第 105 行：「| 錨點：模擬器相同、LP 在 c=0 相同 | 見第 6 節，連 c>0 都相同到 5e-4 | 對（c>0 比預期更好） |」
- 結果（表內）：Simulator bit-identical; LP difference at most 4.8e-4 even for c>0.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

### Rotation, noisy

#### RN1
- 預測（表內）：Independent noise (all σ): gap below 2%, except possibly about 2% for homogeneous c=10 cells (maximum about 2.2%).
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 14 行：「獨立雜訊（σ 全部）：差距 < 2%，除了 h=0 的 c=10 格可能 ~2%（沿用 L1 同質 c=10 弱點）。最大 ~2.2%。」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 72 行：「| 1 獨立雜訊 <2%，除 h=0 c=10 邊緣 | 報表上 N=10 h=0 的 12 格 >2%（最大 3.73），但全是上界逾時；排除後全部 <2%（最大 1.13%） | 結論對、數字被上界鬆度污染 |」
- 結果（表內）：Headline table: 12 cells of N=10, h=0 above 2% (maximum 3.73), all with a timed-out upper bound; excluding them, all below 2% (maximum 1.13%).
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「結論對、數字被上界鬆度污染」，對映為 partly（見判定規則）。

#### RN2
- 預測（表內）：Correlated gusts, L=10: below 2%; σ=30% may be 1.5–2.5%.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 15 行：「相關陣風 L=10：與獨立相近，< 2%；σ=30% 可能 1.5–2.5%」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 73 行：「| 2 AR L=10 σ=30% 可能 1.5–2.5% | 1.65%（N=5 h=0） | 對 |」
- 結果（表內）：1.65% (N=5, h=0).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### RN3
- 預測（表內）：Correlated gusts, L=50, σ=30%, c=10: mostly above 2% (2–5%).
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 16 行：「σ=30% 在 c=10 多半 > 2%（2–5%）」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 74 行：「| 3 AR L=50 σ=30% c=10 多半 >2%（2–5%） | 最大 1.47%（N=5 h=0）、0.50%（N=5 h=15%），無 >2% | **錯** |」
- 結果（表內）：Maximum 1.47% (N=5, h=0) and 0.50% (N=5, h=15%); none above 2%.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### RN4
- 預測（表內）：Events (p=0.2) add to the noise; above 2% concentrated at c=10 with σ≥15%.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 17 行：「事件 p=0.2：與雜訊疊加，> 2% 集中在 c=10 且 σ≥15%」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 75 行：「| 4 事件 + σ≥15% + c=10 >2% | 事件使差距增加 ≤0.3 個百分點，無 >2%（可信格） | **錯** |」
- 結果（表內）：Events raised the gap by at most 0.3 percentage points; none above 2% on trusted cells.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### RN5
- 預測（表內）：About 15–25% of the 160 cells (24–40) above 2%, concentrated at AR L=50, σ≥15%, c=10, N=10.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 18 行：「全 160 格：約 15–25% 的格子（24–40 格）最佳規則 > 2%，集中在 AR L=50 x σ≥15% x c=10 x N=10。N=10 比 N=5 多。」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 76 行：「| 5 約 15–25% 的格子 >2% | 報表 23/160（14%，全是逾時的 N=10 h=0）；可信格 0/160 | 數字對、原因錯 |」
- 結果（表內）：Headline table: 23/160 (14%), all timed-out N=10, h=0 cells; on trusted cells 0/160.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「數字對、原因錯」，對映為 partly。可信格（上界未逾時）0/160，讀者應以這個數字理解。預測與量測的網格同為 160 格。

#### RN6
- 預測（表內）：The estimated gate (EMA) beats the plain gate by 0.3–1 percentage points under correlated gusts and events.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 19 行：「預估閘門 (EMA) 比沿用閘門在相關陣風與事件格好 0.3–1 個百分點」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 77 行：「| 6 gate_est 在相關陣風與事件優於 gate 0.3–1 點 | 平均差 0.03–0.13%，且 gate_est 較差 | **錯** |」
- 結果（表內）：On average worse by 0.03–0.13%.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### RN7
- 預測（表內）：With σ=0, p=0 and global information the result returns to L1 exactly.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 20 行：「σ=0、p=0、全域資訊：回到 L1（閘門範圍與 L1 的 gate 完全相同，上界與 L1 的 LP 相同）」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 78 行：「| 7 錨點回到 L1 | 逐位元相同 | 對 |」
- 結果（表內）：Bit-identical to L1.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### RN8
- 預測（表內）：Local-rule loss versus the global gate: 1–6% at h=0.15; at h=0 about 1–3% for N=5 and 2–8% for N=10.
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 23 行：「h=0 時 N=5 約 1–3%、N=10 約 2–8%」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 79 行：「| 8 局部規則損失：h=0.15 為 1–6%、h=0 N=5 為 1–3%、N=10 為 2–8% | h=0.15 為 0.9–1.2%（偏低端）；h=0 N=5 為 2.7–3.9%、N=10 為 3.6–5.9% | 部分對（h=0.15 在區間下緣；h=0 略高於 N=5 預測） |」
- 結果（表內）：h=0.15: 0.9–1.2% (low end); h=0: N=5 2.7–3.9%, N=10 3.6–5.9%.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「部分對」。

#### RN9
- 預測（表內）：Local rule above 2% from the bound in more than 60% of the L3 cells (N=10 and c=10 nearly all).
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 24 行：「預測 > 60% 的格子 > 2%，N=10 與 c=10 幾乎全部 > 2%」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 80 行：「| 9 L3 >60% 的格子 >2% | 50%（16 格中 8 格），都在 h=0 | 偏高 |」
- 結果（表內）：50% (8 of 16 cells), all at h=0.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「偏高」（預測偏高）；方向（多數格子超過 2%）對、數字不對，對映為 partly。

#### RN10
- 預測（表內）：With a sorted starting arrangement the local rule's loss drops below 1.5% (most loss is initial sorting).
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 25 行：「把局部規則的初始排列換成已排序（消融）後，損失降到 < 1.5%：大部分損失來自初始排序成本，不是動態規則。」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 81 行：「| 10 局部規則換成已排序初始排列後損失 <1.5% | h=0.15：成立（降到 ~0）；h=0：**不成立**，損失不變 | 部分對 |」
- 結果（表內）：h=0.15: holds (drops to about 0); h=0: does not hold, loss unchanged.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「部分對」。

#### RN11
- 預測（表內）：No RL room for independent noise; if any, at correlated gusts or events with c=10; L3 is the layer most likely to have room (loss above 2%).
- 預測原文：`.scratch/drone_bucket/PREDICTION_L23.md@e8d6059` 第 26 行：「推論：L2 的獨立雜訊沒有 RL 空間；若有空間在相關陣風 / 事件 x c=10；L3 是最有可能有空間的一層（資訊受限造成的損失 > 2%）。」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 8 行：「**L2：沒有確定的 >2% 格子。**」
- 結果出處：`notes/impl/drone_bucket_l23.md` 第 9 行：「**L3：有。**」
- 結果（表內）：L2: no confirmed cell above 2%. L3: local rule loses 2.7–3.9% (N=5) and 3.6–5.9% (N=10) at h=0; the correlated-gust/event question was untested because L2 had no cell above 2%.
- 判定：**held（成立）**｜依據：本次判定｜專案「與預測對照」沒有單列這條推論。專案結論（L2 沒有、L3 有）與兩個子句的方向一致，故本次判為 held；「若有空間則在……」為條件句，未被檢驗。

### Wake surfing

#### WK1
- 預測（表內）：Gap above 2 pp at A=0.2 for both τ (all sensing levels).
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 12 行：「1. Gap > 2 pp: A=0.2 for both tau (all sensing levels)」
- 結果出處：`notes/impl/wake_headroom.md` 第 33 行：「| A=0.2（兩種 τ）差距 >2 pp | **錯。** 差距 ≤1.55。原因：A=0.2 時所有方法的絕對節省都掉到 2–3.5 pp，可贏的絕對量本來就小。 |」
- 結果（表內）：All below 2 pp; maximum 1.55 pp (A=0.2, τ=30, GPS 0.08).
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### WK2
- 預測（表內）：A=0.1 with τ=30 and high roll noise: borderline, 1.5–2.5 pp.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 12 行：「A=0.1 only with tau=30 and high roll noise (borderline, 1.5-2.5 pp)」
- 結果出處：`notes/impl/wake_headroom.md` 第 34 行：「| A=0.1 τ=30 高滾轉雜訊邊緣 1.5–2.5 pp | **錯（偏高）。** 0.89–1.41。 |」
- 結果出處：（計算或 JSON 取值）calib/out/wake_headroom.json 的 cells（drift=amp0.1_tau30，roll 非 low 的 6 格）gap = 1.2、1.2、1.08、1.08、1.39、1.39（本次自 JSON 取值）
- 結果（表內）：0.89–1.41 pp over the 12 cells (high roll noise: 1.08–1.39).
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯（偏高）」。

#### WK3
- 預測（表內）：Slow sine maneuver (120 s period) below 2 pp.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 14 行：「sine maneuver (slow, 120 s period)」
- 結果出處：`notes/impl/wake_headroom.md` 第 35 行：「| 慢 sine、step <2 pp | **sine 錯**（2.04–2.79），step 對一半（GPS 0.08 才 >2）。特權法能把慢而平滑的位移追到幾乎零誤差，傳統法的估計有延遲。 |」
- 結果出處：`notes/impl/wake_headroom.md` 第 24 行：「| sine | 4.55 | 4.90 | 4.18 | 7.27 | 6.37 | 2.04/2.29/2.79 | 1.18/1.46/1.77 | 10/12 |」
- 結果（表內）：Sine: 2.04–2.79 pp in all 12 cells (10 with lower bound above 2); the privileged method has perfect own position.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「sine 錯」。

#### WK4
- 預測（表內）：Step maneuver: may exceed 2 pp only if the estimator is slow; expected about 1 pp, so not above 2.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 13 行：「the step maneuver may exceed 2 pp averaged over 600 s only if the estimator is slow (I expect ~1 pp, so NOT > 2)」
- 結果出處：`notes/impl/wake_headroom.md` 第 35 行：「step 對一半（GPS 0.08 才 >2）」
- 結果出處：`notes/impl/wake_headroom.md` 第 25 行：「| step | 4.12 | 5.30 | 4.88 | 7.23 | 6.37 | 1.47/1.74/2.58 | 0.89/1.07/1.25 | 4/12 |」
- 結果（表內）：Step above 2 pp only in the 4 cells with GPS σ=0.08 (2.53–2.58 pp); cell-wise minimum/median/maximum 1.47/1.74/2.58 pp.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「step 對一半」。

#### WK5
- 預測（表內）：A in {0, 0.05}: gap below 2 pp everywhere.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 14 行：「A in {0, 0.05} everywhere」
- 結果出處：`notes/impl/wake_headroom.md` 第 36 行：「| A≤0.05 全部 <2 pp | 對。 |」
- 結果出處：`notes/impl/wake_headroom.md` 第 17 行：「| amp0 | 6.41 | 6.51 | 5.99 | 7.29 | 6.41 | 0.25/0.76/1.30 |」
- 結果（表內）：Yes: largest gaps 1.30 pp (A=0) and 1.29 pp (A=0.05, τ=30).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### WK6
- 預測（表內）：A=0.1 with τ=5: below 2 pp (the oracle itself cannot track).
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 14 行：「A=0.1 with tau=5 (the oracle itself cannot track)」
- 結果出處：`notes/impl/wake_headroom.md` 第 20 行：「| amp0.1 τ5 | 4.34 | 4.32 | 4.05 | 4.70 | 4.34 | 0.14/0.32/0.59 |」
- 結果（表內）：Gap 0.14/0.32/0.59 pp (minimum/median/maximum over 12 cells).
- 判定：**held（成立）**｜依據：本次判定｜專案「與預測對照」沒有單列這一條；差距表中 amp0.1 τ5 最大 0.59 pp，明確 <2，故本次判 held。

#### WK7
- 預測（表內）：Best traditional method: the roll-moment Kalman setpoint when roll noise is low and τ=30; a fixed setpoint ties or wins when drift is small or roll noise high.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 15 行：「the roll-moment Kalman setpoint (controller 2) when roll noise is low and tau=30」
- 結果出處：`notes/impl/wake_headroom.md` 第 37 行：「| 最佳傳統法：kf 為主，小漂移或高滾轉雜訊時固定設定點打平 | 大致對；快漂移（τ=5, A≥0.1）固定設定點勝。 |」
- 結果出處：`notes/impl/wake_headroom.md` 第 12 行：「kf）108 格中 90 格最好，固定設定點 18 格」
- 結果（表內）：Kalman best in 90 of 108 cells; fixed setpoint best in 18 (A=0.2 τ=5, 12 cells; A=0.1 τ=5, 6 cells).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「大致對」，對映為 held；專案註明快漂移時是固定設定點勝（預測沒提這點）。

#### WK8
- 預測（表內）：ESC (extremum seeking) is never best.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 16 行：「ESC (3) never best」
- 結果出處：`notes/impl/wake_headroom.md` 第 38 行：「| ESC 不會最好 | 對（0/108）。 |」
- 結果（表內）：Best in 0 of 108 cells.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### WK9
- 預測（表內）：z-axis drift is weakly observable from roll moment, so the Kalman filter closes only about half the gap at A=0.2.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 18 行：「z-axis drift is weakly observable from roll moment (table is even in dz), so KF leaves residual z error: expect it to close only ~half the gap at A=0.2.」
- 結果出處：`notes/impl/wake_headroom.md` 第 39 行：「| z 向漂移滾轉力矩弱可觀 | 我沒有單獨量這一點，無結論。 |」
- 結果（表內）：Not measured separately; the record draws no conclusion.
- 判定：**not assessed（無判定）**｜依據：專案紀錄的判定｜專案原文「我沒有單獨量這一點，無結論」。

#### WK10
- 預測（表內）：Anchors: with zero drift and zero noise all four methods within 0.1 pp; privileged ≥ others in every condition.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 19 行：「Anchors: with zero drift and zero noise, all four within 0.1 pp. Privileged >= others in every condition (within bootstrap noise).」
- 結果出處：`notes/impl/wake_headroom.md` 第 43 行：「**四者在 0.1 pp 內，通過**」
- 結果出處：`notes/impl/wake_headroom.md` 第 44 行：「`priv − 各傳統法` 最小為 +0.07 pp（正，**通過**）」
- 結果（表內）：Fixed 8.000, privileged 8.000, Kalman 8.000, ESC 7.916 pp; smallest privileged minus traditional gap +0.07 pp.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案錨點檢查寫「通過」。

#### WK11
- 預測（表內）：Headline: headroom exists only at large drift (A≥0.2, maybe 0.1 slow), and even there the Kalman filter closes a large part of it.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 20 行：「the headline will be "headroom exists only at large drift (A>=0.2, maybe 0.1 slow) and even there the KF closes a large part of it"」
- 結果出處：`notes/impl/wake_headroom.md` 第 9 行：「**隨機漂移（AR(1)）的 84 格，差距全部 <2 pp（最大 1.55，amp 0.2 τ=30、GPS 0.08）。**」
- 結果出處：`notes/impl/wake_headroom.md` 第 10 行：「**差距 >2 pp 只出現在領航機機動**」
- 結果出處：`notes/impl/wake_headroom.md` 第 12 行：「kf 勝出幅度小：比固定設定點多 0.1–0.3 pp（amp0.1–0.2 τ=30）」
- 結果（表內）：All 84 random-drift cells below 2 pp (maximum 1.55); gaps above 2 pp appeared only under leader maneuvers; Kalman beat the fixed setpoint by only 0.1–0.3 pp (A=0.1–0.2, τ=30).
- 判定：**failed（不成立）**｜依據：本次判定｜專案「與預測對照」沒有單列這一條。預測的標題句說餘裕只在大漂移出現；專案結果是大漂移（隨機漂移 84 格）全部 <2 pp，>2 pp 只出現在領機機動，與標題句相反（同 WK1、WK2），故本次判為 failed。

#### WK12
- 預測（表內）：Maximum gap about 3–5 pp, at A=0.2, τ=30, high roll noise, high GPS noise.
- 預測原文：`.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 21 行：「the max gap I expect is ~3-5 pp at A=0.2, tau=30, high roll noise, high GPS noise」
- 結果出處：`notes/impl/wake_headroom.md` 第 40 行：「| 最大差距 3–5 pp | 錯，最大 2.79 pp（sine），且那是特權有完美自身位置的版本。 |」
- 結果（表內）：Maximum 2.79 pp (sine), and that is the privileged method with perfect own position.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

### Soaring, round 1

#### SA1
- 預測（表內）：(a) Single glider, reference minus tuned rule: above 2 pp in nearly all sparse and mid-density conditions (about 16 or more of 24).
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 7 行：「預測 >2 pp：稀疏與中密度的幾乎全部條件（約 16 組以上）」
- 結果出處：`notes/impl/soaring_headroom.md` 第 9 行：「24 格中 20 格點估計 >2 pp（18 格區間下界 >2）」
- 結果出處：`notes/impl/soaring_headroom.md` 第 19 行：「| 對。20/24 點估計 >2；dense|W3 的 4 格 <2。」
- 結果（表內）：20 of 24 cells with point estimate above 2 pp (18 with lower bound above 2).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。預測的 24 格與量測的 24 格相同。

#### SA2
- 預測（表內）：(a) Densest, long-life, W/ws=3: below 2 pp (about 4 cells, both noise levels).
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 8 行：「預測 <2 pp：密度最高、壽命長、W/ws=3 的條件（4 組左右，雜訊高低都是）」
- 結果出處：`notes/impl/soaring_headroom.md` 第 9 行：「只有 4 個 dense + W/ws=3 的格子沒有空間（0.0–0.8 pp）」
- 結果（表內）：Only the 4 dense, W/ws=3 cells had no room (0.0–0.8 pp).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」（同列：dense|W3 的 4 格 <2）。

#### SA3
- 預測（表內）：(a) Dense with W/ws=1.5: above 2 pp.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 9 行：「密 + W/ws=1.5：預測 >2 pp」
- 結果出處：`notes/impl/soaring_headroom.md` 第 19 行：「dense + W1.5 預測 >2：dense|short 對（21–32 pp），dense|long|W1.5|low 只有 3.3 [1.0,6.3]（比預測小）」
- 結果（表內）：dense/short: 21–32 pp; dense/long/W1.5/low only 3.3 [1.0, 6.3], smaller than predicted.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案把 (a) 三小點合併判「對」，並註明 dense/long/W1.5/low 較小；該格點估計仍 >2（區間下界 1.0）。

#### SA4
- 預測（表內）：(b) Sharing versus not, unlimited comms, N=5 and 10: sparse and mid density with long life above 2 pp (about 8 or more cells per N).
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 12 行：「N=5、10、通訊無限：預測稀疏與中密度、壽命長 >2 pp（約 8 組以上，各 N）」
- 結果出處：`notes/impl/soaring_headroom.md` 第 20 行：「| 部分對。sparse|long|W3 與 mid|long 全對；sparse|long|W1.5 只有 N=10 的低雜訊成立（規則本身在 W1.5 稀疏找不到熱氣流，Amin 都在 0.33 的滑降地板）。 |」
- 結果出處：`notes/impl/soaring_headroom.md` 第 10 行：「N=5：無限通訊 8/24 格、中等通訊 7/24 格區間下界 >2；N=10：10/24、10/24」
- 結果（表內）：Lower bound above 2 pp in 8/24 cells (N=5) and 10/24 (N=10); sparse/long/W1.5 held only for N=10 low noise.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「部分對」。

#### SA5
- 預測（表內）：(b) Short life: shared information goes stale, gain shrinks; sparse/short may be below 2 pp.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 13 行：「壽命短（約 5 分鐘）：共享的資訊很快過期，預測增益縮小，稀疏短壽命可能 <2 pp」
- 結果出處：`notes/impl/soaring_headroom.md` 第 21 行：「| 對。短壽命幾乎全是 0；例外是 N=10 的 mid|short|W3（8–11 pp）。 |」
- 結果（表內）：Short life almost all about 0; exception N=10 mid/short/W3 (8–11 pp).
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### SA6
- 預測（表內）：(b) Medium comms (2 km of a 5 km world): gain smaller than unlimited, but still mostly above 2 pp for sparse/long.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 14 行：「預測增益比無限小，但仍多數 >2 pp 於稀疏/長壽命」
- 結果出處：`notes/impl/soaring_headroom.md` 第 10 行：「通訊中等（2 km）與無限幾乎一樣（兩者 Amin 最大差 0.10）」
- 結果出處：`notes/impl/soaring_headroom.md` 第 10 行：「N=5：無限通訊 8/24 格、中等通訊 7/24 格區間下界 >2；N=10：10/24、10/24」
- 結果（表內）：Medium and unlimited nearly equal (largest Amin difference 0.10); cells with lower bound above 2: N=5 7/24 versus 8/24, N=10 10/24 versus 10/24.
- 判定：**partly（部分成立）**｜依據：本次判定｜專案「與預測對照」沒有單列這一條。後半（稀疏/長壽命仍多數 >2）成立；前半「比無限小」只在 N=5 差 1 格，N=10 相同、整體「幾乎一樣」，故本次判 partly。

#### SA7
- 預測（表內）：(b) Densest condition: below 2 pp.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 15 行：「密度最高：預測 <2 pp」
- 結果出處：`notes/impl/soaring_headroom.md` 第 22 行：「| (b) 密度最高 <2 pp | **錯。** dense|long|W1.5：N=5 高雜訊 12.9、N=10 兩種雜訊 8–21 pp；dense|long|W3 N=10 約 2 pp。 |」
- 結果（表內）：dense/long/W1.5: N=5 high noise 12.9, N=10 both noise levels 8–21 pp; dense/long/W3 at N=10 about 2 pp.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### SA8
- 預測（表內）：(b) W/ws=1.5: sharing gain small, mostly below 2 pp.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 16 行：「W/ws=1.5：共享增益小（熱氣流本來就難用），預測 <2 pp 居多」
- 結果出處：`notes/impl/soaring_headroom.md` 第 23 行：「| (b) W/ws=1.5 <2 pp 居多 | **錯。** mid|long|W1.5 11–26 pp，dense|long|W1.5 如上。短壽命的 W1.5 才是 ≈0。 |」
- 結果（表內）：mid/long/W1.5 11–26 pp; dense/long/W1.5 as above; only short-life W1.5 was about 0.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### SA9
- 預測（表內）：(c) Reference minus shared rule: about 55–65% of the 48 cells above 2 pp.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 19 行：「粗估 24 x 2（N）= 48 格中約 55–65% 差距 >2 pp」
- 結果出處：`notes/impl/soaring_headroom.md` 第 24 行：「| (c) 約 55–65% 的格 >2 pp | **偏低。** 實際約 81%（區間下界）到 92%（點估計）。 |」
- 結果（表內）：About 81% (lower bound) to 92% (point estimate).
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案原判「偏低」（預測偏低）；多數格子 >2 pp 的方向對、百分比區間未涵蓋實際值，對映為 partly。

#### SA10
- 預測（表內）：(c) Dense with W/ws=3: below 2 pp.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 20 行：「預測 <2 pp：密度最高 + W/ws=3 的條件。」
- 結果出處：`notes/impl/soaring_headroom.md` 第 25 行：「| (c) dense + W3 <2 pp | 長壽命對（0.0–0.8）；**短壽命錯**：dense|short|W3 在 N=5 是 3.8–5.9、N=10 是 5.9–20.1 pp（規則在短壽命熱氣流下追過期的熱氣流或搜尋不夠快）。 |」
- 結果（表內）：Long life correct (0.0–0.8); short life wrong (N=5 3.8–5.9, N=10 5.9–20.1 pp).
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案結果「長壽命對、短壽命錯」，合成一條為 partly。

#### SA11
- 預測（表內）：(c) The N=5 gap is larger than the N=10 gap.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 19 行：「N=5 的差距 > N=10（人多覆蓋多）」
- 結果出處：`notes/impl/soaring_headroom.md` 第 26 行：「| (c) N=5 差距 > N=10 | 看條件：sparse|long|W3 對（N=5 32.0–32.5 vs N=10 24.4–30.8）、其他多半相近或 N=10 更大（dense|short|W3|high：5.9 vs 20.1）。不成立為通則。 |」
- 結果（表內）：Depends on condition: sparse/long/W3 yes (32.0–32.5 versus 24.4–30.8); otherwise similar or N=10 larger (dense/short/W3/high 5.9 versus 20.1).
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案結果「不成立為通則」，對映為 failed。

#### SA12
- 預測（表內）：(c) Inference: the true RL room is smaller than (c), because the reference has information RL cannot get; the gap should be smaller after removing information.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 21 行：「所以真正 RL 空間會比 (c) 小；我預期扣掉「資訊」之後，差距更小。」
- 結果出處：`notes/impl/soaring_headroom.md` 第 11 行：「推論：真正受同樣感測限制的 RL 空間會小得多，但這個測試沒辦法量出多小。」
- 結果（表內）：This round could not measure how much smaller.
- 判定：**not assessed（無判定）**｜依據：專案紀錄的判定｜專案寫「這個測試沒辦法量出多小」。第 2、3 輪才開始做資訊對等的比較，但未回頭判這一條。

#### SA13
- 預測（表內）：Best single-glider rule: centering helps (shift steps above 0 beat switching it off).
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 24 行：「單機：定心有用（偏移步數 >0 勝過關閉）」
- 結果出處：`notes/impl/soaring_headroom.md` 第 27 行：「| 最佳規則：定心有用 | 對。shgain=0 在 24 格都沒被選。 |」
- 結果（表內）：Centering shift (shgain) was never 0 across the 24 conditions.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」。

#### SA14
- 預測（表內）：Best single-glider rule: a medium leave threshold.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 24 行：「離開門檻中等」
- 結果出處：`notes/impl/soaring_headroom.md` 第 68 行：「th_leave：−0.8×13、−0.5×6、−0.3×5」
- 結果出處：`notes/impl/soaring_headroom.md` 第 47 行：「−0.8 仍是最常選（13/24）」
- 結果（表內）：Most chosen threshold −0.8 m/s (13/24 conditions), the lowest value of the final grid; a later 60-seed check judged it a plateau, not a boundary.
- 判定：**not assessed（無判定）**｜依據：專案紀錄的判定｜專案沒有對「離開門檻中等」下判定；−0.8 是最終網格的最低值，但專案自行判為平台，故 not assessed。

#### SA15
- 預測（表內）：Best rule: the two-wing difference signal gains less than 1 pp at high noise.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 24 行：「高雜訊時用兩翼差訊號的優勢很小（<1 pp）」
- 結果出處：`notes/impl/soaring_headroom.md` 第 29 行：「| 兩翼差訊號增益 <1 pp | 沒量。使用兩翼差的設定在 13/24 格被選，但沒有做開關消融，無結論。 |」
- 結果（表內）：Not measured; no on/off ablation.
- 判定：**not assessed（無判定）**｜依據：專案紀錄的判定｜專案原文「沒量……無結論」。

#### SA16
- 預測（表內）：Best shared rule: adding a MacCready-type “leave for a better known thermal” helps.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 25 行：「共享：加上「更好的已知熱氣流存在就離開現在這個」（MacCready 型）有幫助」
- 結果出處：`notes/impl/soaring_headroom.md` 第 28 行：「| 最佳規則：共享加 MacCready 型離開有幫助 | **錯。** mc=0.6 只在 8/80 個設定被選。 |」
- 結果（表內）：mc=0.6 was chosen in only 8 of 80 settings.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### SA17
- 預測（表內）：Best shared rule: the crowding penalty has almost no effect.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 25 行：「擁擠懲罰對結果幾乎無影響」
- 結果出處：`notes/impl/soaring_headroom.md` 第 12 行：「擁擠懲罰 22/80 個設定選到」
- 結果（表內）：Crowding penalty was chosen in 22 of 80 settings; no on/off ablation.
- 判定：**not assessed（無判定）**｜依據：本次判定｜專案沒有對這一條下判定，也沒有做開關消融；被選中次數不能直接推出「幾乎無影響」，故 not assessed。

#### SA18
- 預測（表內）：Anchor (no noise, dense, long, W/ws=3): all methods within 2 pp (on the main metric Amin), reference ≥ rule.
- 預測原文：`.scratch/soaring_headroom/PREDICTION.md@071574f` 第 28 行：「無雜訊 + 密 + 長壽命 + W/ws=3：各法差距 <2 pp，參考解 ≥ 規則」
- 結果出處：`notes/impl/soaring_headroom.md` 第 30 行：「| 錨點：各法差距 <2 pp、參考 ≥ 規則 | Amin 對；能量指標不對（4.4–7.5 pp，N=5 的 (c) 能量可達 10.2 pp）。參考 ≥ 規則對。 |」
- 結果（表內）：Amin gaps 0.0 (N=1 and N=5) and reference ≥ rule in every cell; the energy metric Emin was not close (4.4–7.5 pp).
- 判定：**held（成立）**｜依據：本次判定｜專案原文是「Amin 對；能量指標不對」。預測檔把差距單位定為主指標 Amin，故本次判 held；能量指標的落差已寫在 Outcome 欄。

### Soaring, round 2

#### SB1
- 預測（表內）：L1 minus L0: long life +2 to +10 pp, short life about −2 to +2; 3–4 of 8 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 29 行：「| L1 − L0 | 長壽命 +2 到 +10；短壽命 −2 到 +2 | 3–4 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 24 行：「| L1−L0：長壽命 +2 到 +10，短壽命約 0 | 方向對，長壽命更大（mid|long +7.3/+12.9、sparse|long +5.6/+24.2，N=10 mid|long|W3 +19.7）；短壽命約 0（對）。 |」
- 結果（表內）：Long life larger (N=5: +5.6, +7.3, +12.9, +24.2; N=10 mid/long/W3 +19.7); short life about 0; 4 cells with lower bound above 2.
- 判定：**partly（部分成立）**｜依據：專案紀錄的判定｜專案寫「方向對，長壽命更大」，對映為 partly。

#### SB2
- 預測（表內）：L2 minus L1: +2 to +10 pp (larger for long life); 5–7 of 8 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 30 行：「| L2 − L1 | +2 到 +10（長壽命較大） | 5–7 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 25 行：「| L2−L1 +2 到 +10，5–7 格 Y | 7/8 格 Y，範圍 +0.3 到 +13.9（mid|short|W3 超出上界）。 |」
- 結果（表內）：7 of 8 cells; range +0.3 to +13.9 (mid/short/W3 above the range).
- 判定：**partly（部分成立）**｜依據：本次判定｜專案有結果文字但沒有判定字眼。格數（7）在預測內；點估計範圍有 2 格在預測外（+13.9、+0.3），故本次判 partly。

#### SB3
- 預測（表內）：L3 minus L2: +1 to +6 pp.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 31 行：「| L3 − L2 | +1 到 +6 | 4–6 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 26 行：「| L3−L2 +1 到 +6 | 大致對（+0.3 到 +5.1）。 |」
- 結果（表內）：+0.3 to +5.1.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「大致對」，對映為 held。專案沒有核對「4–6 格」那一欄。

#### SB4
- 預測（表內）：L3 minus L1: +4 to +15 pp; 6–8 of 8 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 32 行：「| L3 − L1 | +4 到 +15 | 6–8 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 53 行：「| sparse|short|W1.5|low | 0.169/0.682/1.00 | +0.0 [-0.1,+0.2]- | +0.3 [+0.1,+0.5]- | +0.6 [+0.3,+0.9]- |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 50 行：「| mid|short|W3.0|low | 0.291/0.975/0.17 | +0.0 [+0.0,+0.0]- | +13.9 [+11.2,+16.7]Y | +19.0 [+15.9,+22.1]Y |」
- 結果（表內）：N=5: +0.6 to +19.0 pp; 7 of 8 cells with lower bound above 2.
- 判定：**partly（部分成立）**｜依據：本次判定｜專案「與預測對照」沒有這一列。從第二輪表算出：格數 7（在 6–8 內），但 +19.0 與 +0.6 在 +4 到 +15 之外，故 partly。

#### SB5
- 預測（表內）：zone minus L1: −3 to +3 pp; 0–2 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 33 行：「| zone − L1 | −3 到 +3 | 0–2 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 27 行：「| zone/prior−L1 約 0，Y 格 0–2/0–3 | 對，0 格 Y。 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 16 行：「zone −3.7 到 +0.8，prior −3.5 到 +1.3」
- 結果（表內）：−3.7 to +0.8; 0 cells.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」（zone 與 prior 同列）。zone 最低 −3.7 略低於預測的 −3，專案仍判對。

#### SB6
- 預測（表內）：prior minus L1: −2 to +4 pp; 0–3 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 34 行：「| prior − L1 | −2 到 +4 | 0–3 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 27 行：「| zone/prior−L1 約 0，Y 格 0–2/0–3 | 對，0 格 Y。 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 16 行：「zone −3.7 到 +0.8，prior −3.5 到 +1.3」
- 結果（表內）：−3.5 to +1.3; 0 cells.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「對」（與 zone 同列）。prior 最低 −3.5 低於預測的 −2，專案仍判對。

#### SB7
- 預測（表內）：plan minus L1: −4 to +4 pp; 0–2 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 35 行：「| plan − L1 | −4 到 +4 | 0–2 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 28 行：「| plan−L1 −4 到 +4，Y 格 0–2 | Y 格 0；但 sparse|long|W3 為 −6.2，超出下界；規劃器在長壽命格明顯輸給 L1。 |」
- 結果（表內）：−6.2 to +1.1; 0 cells; clearly worse than L1 in long-life cells (sparse/long/W3 −6.2).
- 判定：**partly（部分成立）**｜依據：本次判定｜專案有結果文字但沒有判定字眼；格數成立、下界 −6.2 超出預測，故本次判 partly。

#### SB8
- 預測（表內）：ref minus L1: +10 to +35 pp (mainly an information gap); all 8 cells above 2.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 36 行：「| ref − L1 | +10 到 +35（主要是資訊差） | 8 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 29 行：「| ref−L1 +10 到 +35 | +18.5 到 +50.3，sparse|short|W3、mid|short|W3 超出上界。 |」
- 結果（表內）：+18.5 to +50.3, all 8 cells with lower bound above 2; two cells (sparse/short/W3, mid/short/W3) above the range.
- 判定：**partly（部分成立）**｜依據：本次判定｜專案有結果文字但沒有判定字眼；8/8 格成立，但 2 格超出預測上界，故本次判 partly。

#### SB9
- 預測（表內）：Floor cells (L1 landing rate ≥ 0.5): expected 0–2 cells (sparse/short most likely).
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 38 行：「地板格：預期 0–2 格（sparse|short 最可能）」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 30 行：「| 地板格 0–2 | 4 格（落地率 >=0.5），多於預測。 |」
- 結果（表內）：4 cells.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案寫「多於預測」。預測給的是 0–2 的明確數字，實際 4，對映為 failed。

#### SB10
- 預測（表內）：Amin above 0.9 in most cells with low landing rate, so Amin gaps are usually below 2 pp and only Emin separates methods.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 38 行：「Amin 在多數格 >0.9，落地率低，所以 Amin 差距通常 <2 pp，Emin 才分得開」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 49 行：「| mid|short|W1.5|low | 0.194/0.793/0.97 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 51 行：「| sparse|long|W1.5|low | 0.230/0.879/0.58 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 53 行：「| sparse|short|W1.5|low | 0.169/0.682/1.00 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 54 行：「| sparse|short|W3.0|low | 0.185/0.725/0.98 |」
- 結果（表內）：L1 Amin above 0.9 in 4 of 8 cells (0.994, 0.998, 0.975, 0.987), 4 floor cells; L2 minus L1 in Amin below 2 pp in 4 of 8 cells (+0.4, +0.2, −0.5, +1.3).
- 判定：**failed（不成立）**｜依據：本次判定｜專案沒有這一條的判定。由第二輪表數出：8 格中 Amin>0.9 的僅 4 格、落地率高的 4 格，Amin 差距 <2 pp 的 4 格；「多數」「通常」都只有一半，故本次判 failed。

#### SB11
- 預測（表內）：N=10: same direction as N=5; L2 minus L1 +2 to +8, plan minus L1 −3 to +3.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 39 行：「N=10：方向同 N=5，L2 − L1 +2 到 +8，plan − L1 −3 到 +3。」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 15 行：「N=10 三格同向（+4.9 到 +11.9）」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 90 行：「| mid|long|W3.0|low | 0.601/1.000/0.00 | -19.7 [-23.0,-16.3]- | +11.9 [+8.7,+15.0]Y | +16.4 [+13.5,+19.4]Y | +1.4 [-1.6,+4.6]? | +1.3 [-1.8,+4.6]? | -2.3 [-4.8,+0.4]- |」
- 結果（表內）：L2 minus L1: +11.9, +4.9, +5.0 (one above the range); plan minus L1: −2.3, −0.2, −1.7.
- 判定：**partly（部分成立）**｜依據：本次判定｜專案沒有單列 N=10 的判定。3 格中 L2−L1 有 1 格（+11.9）超出預測、plan−L1 三格都在範圍內，故本次判 partly。

#### SB12
- 預測（表內）：Planner: criterion C1 (lower bound above 2 pp) in at least 3 cells has probability 15%; the most likely conclusion is “undecided”.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 40 行：「規劃器：C1 在 >=3 格成立的機率我給 15%；C3 因此多半不會被檢驗。最可能的結論是「未決」（規劃器沒贏，但它是簡版）。」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 31 行：「| 結論：規劃器未決 | 一致。 |」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 17 行：「C1（下界 >2）0/8 格成立（N=10 0/3）」
- 結果（表內）：C1 held in 0/8 cells (0/3 at N=10); recorded as “undecided”, not “no room”.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原判「一致」。

#### SB13
- 預測（表內）：Information-matched methods (zone, prior, plan) gain below 2 pp over L1, mostly.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 41 行：「(i) 資訊對等且可達成的方法（zone、prior、plan）相對 L1 的增益 <2 pp 居多」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 16 行：「沒有哪個只靠全隊觀測＋先驗的方法贏過 L1」
- 結果（表內）：None has a lower bound above 2 pp: zone −3.7 to +0.8, prior −3.5 to +1.3, plan −6.2 to +1.1.
- 判定：**held（成立）**｜依據：本次判定｜專案結論寫出數字；與預測的對照是本次做的。

#### SB14
- 預測（表內）：Any L2/L3 gap comes from measurement noise and centering error (information), not from decision structure.
- 預測原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 41 行：「(ii) L2/L3 的差距若存在，來源是量測噪聲與定心誤差（資訊量），不是決策結構。」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 36 行：「差距幾乎全是資訊差：L1.5b（只加真值資訊、決策結構不變）追回 L2−L1 的 76%–164%」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 37 行：「L1.5a（L2 的決策結構但只用觀測估計）**比 L1 差**」
- 結果（表內）：Round 3: giving L1 the true information (L1.5b) recovered 76%–164% of L2 minus L1; observation-only L2 structure (L1.5a) was worse than L1. Noise and centering error were not separated.
- 判定：**partly（部分成立）**｜依據：本次判定｜專案第三輪結論支持「資訊而非決策結構」，但它加入的是全部真值（中心、強度、漂移），沒有把量測噪聲與定心誤差分開；故本次判 partly。

### Soaring, round 3

#### SC1
- 預測（表內）：L1.5a minus L1: −2 to +5 pp; 1–3 of 8 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 19 行：「| L1.5a - L1 | -2 到 +5 | 1-3 |」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 41 行：「預測 L1.5a−L1 為 −2 到 +5；實際 −0.2 到 −24.5，在長壽命格輸得比預測多很多（sparse|long|W3 −24.5）。」
- 結果（表內）：−0.2 to −24.5 pp; no cell better than L1.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案把它列在「和預測不同」；沒有任何一格 >2、且有格遠低於預測下界，對映為 failed。

#### SC2
- 預測（表內）：L1.5b minus L1: +1 to +9 pp; 3–5 of 8 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 20 行：「| L1.5b - L1 | +1 到 +9 | 3-5 |」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 31 行：「| sparse|short|W1.5 | 5 | 地板 1.00 | 16.9 / 16.8 / 17.4 / 17.2 | 68.2 / 67.1 / 70.6 / 69.5 | -0.2 [-0.3, -0.1] | +0.5 [+0.3, +0.7] |」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 27 行：「| mid|short|W3.0 | 5 | 非地板 0.17 | 29.1 / 22.4 / 40.5 / 43.1 | 97.5 / 84.2 / 99.3 / 100.0 | -6.8 [-8.7, -4.8] | +11.4 [+8.8, +14.1] |」
- 結果（表內）：N=5: +0.5 to +11.4 pp; 7 of 8 cells with lower bound above 2.
- 判定：**partly（部分成立）**｜依據：本次判定｜專案沒有這一列的判定（只在「和預測不同」列了「追回一半以上」）。方向（正）成立，但格數 7 超出 3–5、最大 +11.4 超出 +9，故本次判 partly。

#### SC3
- 預測（表內）：L2 minus L1 (re-run, should reproduce round 2): +3 to +14 pp; 7 cells.
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 21 行：「| L2 - L1（重跑，應重現第二輪） | +3 到 +14 | 7 |」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 7 行：「L1、L2 的逐種子 Emin 與第二輪 JSON 完全一致（差 0.0）」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 31 行：「| sparse|short|W1.5 | 5 | 地板 1.00 | 16.9 / 16.8 / 17.4 / 17.2 | 68.2 / 67.1 / 70.6 / 69.5 | -0.2 [-0.3, -0.1] | +0.5 [+0.3, +0.7] | +0.3 [+0.1, +0.5] |」
- 結果（表內）：+0.3 to +13.9 pp; 7 of 8 cells with lower bound above 2; per-seed values identical to round 2 (difference 0.0).
- 判定：**held（成立）**｜依據：本次判定｜專案沒有單列；重現（差 0.0）與格數 7 成立，+0.3 一格低於預測範圍下緣，以預測的主旨（重現第二輪）判 held。

#### SC4
- 預測（表內）：L2 minus max(L1.5a, L1.5b): −1 to +4 pp; 0–2 cells with lower bound above 2.
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 22 行：「| L2 - max(L1.5a, L1.5b) | -1 到 +4 | 0-2 |」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 43 行：「預測 L2−max 為 −1 到 +4，實際 −0.8 到 +2.5，在範圍內；非地板格下界 >2 的格數預測 0–2，實際 0。」
- 結果（表內）：−0.8 to +2.5 pp; 0 cells.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案寫「在範圍內」。

#### SC5
- 預測（表內）：L1.5b recovers about half or more of L2 minus L1 (probability 60%).
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 26 行：「L1.5b 追回大部分差距機率約 60%」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 42 行：「預測 L1.5b 追回「一半以上」（機率 60%）；實際追回約 76%–164%，多數格 ≥89%（L2−L1.5b 的點估計 −0.8 到 +2.5）。」
- 結果（表內）：It recovered 76%–164%, most cells ≥ 89%.
- 判定：**held（成立）**｜依據：本次判定｜專案把它列在「和預測不同」（追回量比預測大），沒有下判定；預測的事件（追回一半以上）發生，故本次判 held。

#### SC6
- 預測（表內）：L1.5a recovers most of the gap and becomes the new best hand-written rule (probability about 20%).
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 26 行：「L1.5a 追回大部分差距（成為新最佳手寫規則）機率約 20%」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 37 行：「L1.5a（L2 的決策結構但只用觀測估計）**比 L1 差**：−0.2 到 −24.5 pp，沒有一格贏。不是「更好的手寫規則」。」
- 結果（表內）：Did not happen: L1.5a was worse than L1 in all cells; the best achievable rule remains L1.
- 判定：**held（成立）**｜依據：本次判定｜機率型預測（20%）依判定規則以較可能的一側計：該事件沒有發生，故 held。

#### SC7
- 預測（表內）：L1.5a's leave decision rests on noisy filtered values and will be worse, possibly below L1 in long-life cells because of jittery leaving.
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 24 行：「L1.5a 的離開判斷靠有雜訊的濾波值，會較差，可能因抖動離開而低於 L1 在長壽命格」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 46 行：「它輸的原因（離開誤判、量測噪聲）我沒有做診斷，屬推論。」
- 結果（表內）：L1.5a was below L1 in every row (−0.2 to −24.5; sparse/long/W3 −24.5); the cause (leaving errors or measurement noise) was not diagnosed.
- 判定：**partly（部分成立）**｜依據：本次判定｜結果（低於 L1）成立；預測的機制（抖動離開）專案明說沒有診斷，故 partly。

#### SC8
- 預測（表內）：Floor cells: the same 4 as in round 2; non-floor cells are mid/long ×2, mid/short/W3, sparse/long/W3.
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 25 行：「地板格（L1 落地率 >=0.5）：預期仍是第二輪那 4 格；非地板 4 格是 mid|long x2、mid|short|W3、sparse|long|W3。」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 35 行：「（N=5 的非地板格：mid|long|W1.5、mid|long|W3、mid|short|W3、sparse|long|W3；」
- 結果（表內）：Same sets (N=5 non-floor: mid/long/W1.5, mid/long/W3, mid/short/W3, sparse/long/W3).
- 判定：**held（成立）**｜依據：本次判定｜結果表的「L1 落地率」欄與非地板格名單與預測相同。

#### SC9
- 預測（表內）：Conclusion “no evidence of room for MARL in this setting” (probability about 85%); at least 3 non-floor cells with lower bound above 2 (probability about 12%).
- 預測原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 26 行：「結論預測：「此設定下 MARL 無證據空間」機率約 85%」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 43 行：「結論方向（無證據空間）與預測一致」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 35 行：「L2−max 的區間下界 >2 的非地板格 = **0 / 4**」
- 結果（表內）：0 of 4 non-floor cells; verdict “no evidence of room for MARL in this setting”.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案原文「結論方向（無證據空間）與預測一致」。

### Language-model commander

#### LM1
- 預測（表內）：Valid output: Haiku 93% (80–100%), Sonnet 100% (93–100%).
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 13 行：「| 輸出有效（valid）| 93%（80-100%）| 100%（93-100%）|」
- 結果出處：私人開發紀錄（PROGRESS，第 385 行）：「兩者 14/14 有效、14/14 遵守硬限制」
- 結果（表內）：14/14 valid for both (100%).
- 判定：**held（成立）**｜依據：本次判定｜專案只寫「錯」於遵守率與 regret 中位數，沒有單判有效率；Haiku 點估計 93% 低於實際 100%，但實際值在 80–100% 區間內，故本次判 held。

#### LM2
- 預測（表內）：Hard-limit compliance (constraint_ok, 14 items): Haiku 71% (50–86%), Sonnet 86% (71–100%).
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 14 行：「| 硬限制遵守率（14 條中 constraint_ok；矛盾條以滿足任一解讀計）| 71%（50-86%）| 86%（71-100%）|」
- 結果出處：私人開發紀錄（PROGRESS，第 386 行）：「預測 Haiku 遵守 71%、Sonnet 86%，regret 中位數 4%／1% → 錯」
- 結果（表內）：14/14 for both (100%).
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### LM3
- 預測（表內）：Items with explicit fields (c2, c3, t4, c6, t9): compliance Haiku 90%, Sonnet 100%.
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 15 行：「| 其中「寫在 no_lead / max_swaps 這類明講欄位」的條目（c2 c3 t4 c6 t9）| 90% | 100% |」
- 結果出處：（計算或 JSON 取值）results/haiku.json 與 sonnet.json（commit f1c439d）：c2、c3、t4、c6、t9 的 constraint_ok 皆為 true；c6 的 no_lead=[1,6]、max_swaps=6
- 結果（表內）：5/5 compliant for both; both filled no_lead [1, 6] for c6 and max_swaps for c3, t4, c6, t9.
- 判定：**held（成立）**｜依據：本次判定｜專案沒有單判這一條。5 條不足以區分 90% 與 100%，結果（5/5）與預測相容，故本次判 held。

#### LM4
- 預測（表內）：Regret median over all items: Haiku 4% (0–12%), Sonnet 1% (0–5%).
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 16 行：「| 全部條目 regret 中位數 | 4%（0-12%）| 1%（0-5%）|」
- 結果出處：私人開發紀錄（PROGRESS，第 386 行）：「預測 Haiku 遵守 71%、Sonnet 86%，regret 中位數 4%／1% → 錯」
- 結果出處：私人開發紀錄（PROGRESS，第 385 行）：「中位數都 ≈0」
- 結果（表內）：Medians about 0.01% (Haiku) and 0.002% (Sonnet), computed from the results files; the record says “about 0”.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。預測範圍的下界是 0，所以結果落在範圍邊緣；點估計 4%／1% 與實際相差大，仍照專案判 failed。

#### LM5
- 預測（表內）：Regret mean over all items: Haiku 6% (2–12%), Sonnet 2.5% (0.5–6%).
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 17 行：「| 全部條目 regret 平均 | 6%（2-12%）| 2.5%（0.5-6%）|」
- 結果出處：（計算或 JSON 取值）results/haiku.json、sonnet.json 的 regret 欄：平均 0.001425 與 0.001132（本次自 JSON 計算）
- 結果出處：私人開發紀錄（PROGRESS，第 385 行）：「regret 最大 Haiku 1.15%、Sonnet 1.31%」
- 結果（表內）：Means 0.14% (Haiku) and 0.11% (Sonnet), computed from the results files; maxima 1.15% and 1.31% (both on t10).
- 判定：**failed（不成立）**｜依據：本次判定｜專案沒有單判平均。兩者實際平均都低於預測區間下界（2% 與 0.5%），故本次判 failed。

#### LM6
- 預測（表內）：Max-swaps items (c3, t4, c6, t9): models fill max_swaps but often pair it with time or rotation, so regret is larger; Haiku may skip max_swaps on t4/t9 (about 25% violation), Sonnet about 8%.
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 21 行：「會填 max_swaps，但常搭配 time 或 rot（「少換位」被理解成「定期輪替但少一點」），所以 constraint_ok 大致成立、regret 會偏大」
- 結果出處：（計算或 JSON 取值）results/haiku.json、sonnet.json：t4、t9、c3、c6 的 params.rotation 皆為 gate 或 fixed；regret 最大 0.0025（haiku t4）
- 結果（表內）：Both filled max_swaps in all four, always with gate or fixed (never time or rotation); no violation; regret at most 0.25% (Haiku, t4).
- 判定：**failed（不成立）**｜依據：本次判定｜專案沒有判定。預測的行為模式（搭配 time/rot、regret 偏大）沒有出現，故本次判 failed。

#### LM7
- 預測（表內）：Protect items (p5, p8, v12): the models may not adopt protect and fill no_lead instead; violation Haiku 20%, Sonnet 10%.
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 22 行：「schema 的 protect 語意（鎖在最省位置、不換位）模型可能不採用，改填 no_lead」
- 結果出處：（計算或 JSON 取值）results/haiku.json、sonnet.json：p5、p8、v12 的 params.protect 皆非空
- 結果（表內）：Both used protect in p5, p8 and v12 (p8: Haiku also no_lead [2], Sonnet protect only); no violation.
- 判定：**failed（不成立）**｜依據：本次判定｜專案沒有判定。預測模型可能不採用 protect，實際兩者都採用，故本次判 failed。

#### LM8
- 預測（表內）：s7 (machine 4 leads at most 10%): models mostly fill no_lead=[4]; violation Haiku 30%, Sonnet 15%.
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 23 行：「模型多半直接 no_lead=[4]（這樣正確）」
- 結果出處：私人開發紀錄（PROGRESS，第 385 行）：「s7（4 號領頭 ≤10%）都用 no_lead=[4] 保守解」
- 結果（表內）：Both used no_lead=[4]; no violation.
- 判定：**held（成立）**｜依據：本次判定｜專案紀錄寫明兩者都用 no_lead=[4]。違規機率型預測依判定規則以較可能的一側計，故 held。

#### LM9
- 預測（表內）：t10 (every machine leads, at most 8 swaps): tenure too large is the common error; violation Haiku 40%, Sonnet 20%.
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 24 行：「預期 Haiku t10 違規 40%，Sonnet 20%」
- 結果出處：私人開發紀錄（PROGRESS，第 385 行）：「regret 最大 Haiku 1.15%、Sonnet 1.31%（都在 t10「全員輪流領頭且換位 ≤8」）」
- 結果（表內）：Both satisfied the limit (rotation, max_swaps 8; tenure 10 and 40); regret 1.15% and 1.31%, the largest of the 14.
- 判定：**held（成立）**｜依據：本次判定｜兩者都遵守；機率型預測依判定規則以較可能的一側計（Haiku 60%、Sonnet 80% 不違規），故 held。

#### LM10
- 預測（表內）：Contradictory items (x13, x14): Sonnet notes the conflict about 85%, Haiku about 55%; Haiku may combine inconsistent settings (25%).
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 26 行：「Sonnet 在 notes 點出衝突的機率約 85%，Haiku 約 55%」
- 結果出處：私人開發紀錄（PROGRESS，第 385 行）：「兩者在 x13、x14 都在 notes 點出矛盾並選一個解讀」
- 結果（表內）：Both noted the conflict in x13 and x14 and chose one reading; no inconsistent combination.
- 判定：**held（成立）**｜依據：本次判定｜專案紀錄寫明結果；「點出衝突」以關鍵字比對判定（llm_commander.md L44 自述為粗略）。

#### LM11
- 預測（表內）：Ambiguous items (v11, v12): compliance near 100%; both models choose the “fewer swaps” reading.
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 25 行：「只要選任一合理解讀就算遵守，遵守率接近 100%；regret 取決於它選哪個。預期兩者都選「少換位」解讀。」
- 結果出處：（計算或 JSON 取值）results/haiku.json、sonnet.json：v11 matched_readings=['穩=少換位']，v12 constraint_ok=true
- 結果（表內）：Both compliant on v11 and v12; v11 matched the “fewer swaps” reading for both.
- 判定：**held（成立）**｜依據：本次判定｜專案沒有單判。

#### LM12
- 預測（表內）：Rule choice on unconstrained items (g1, c2, s7 …): Sonnet picks gate about 70%, Haiku about 45%; a time rule with small tenure would cost 10–20% regret.
- 預測原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 27 行：「Sonnet 在無特別要求的條目（g1、c2、s7 等）選 gate 的機率約 70%（Haiku 約 45%，其餘多為 rot 或 time）」
- 結果出處：（計算或 JSON 取值）results/haiku.json、sonnet.json：params.rotation 計數 gate 11、fixed 2、rot 1（兩者相同）
- 結果（表內）：g1, c2, s7: gate for both on 3/3; gate in 11 of 14 items for both; no time rule used.
- 判定：**partly（部分成立）**｜依據：本次判定｜專案沒有判定。Sonnet 多選 gate 的方向對、Haiku 45% 與實際 100% 差很多、time 規則一次都沒出現，故本次判 partly。

### Neighbours-only learner

#### LA1
- 預測（表內）：Positive control passes: at U=500 35%; by U≤1000 65%; by U≤2000 75%.
- 預測原文：`notes/impl/l3_mappo_prereg.md@38ce89e` 第 62 行：「正向對照：U=500 通過 35%；U≤1000 通過 65%；U≤2000 通過 75%。」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 6 行：「正向對照（MAPPO-G，3 格 × 3 種子）在 U=500、1000、2000 都是 0/3 格通過」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 14 行：「事前預測「U≤2000 通過 75%」錯。」
- 結果（表內）：0 of 3 cells passed at U=500, 1000 and 2000.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案原判「錯」。

#### LA2
- 預測（表內）：L3 main run: F_A point estimate 0.15 (80% interval −0.15 to 0.45), F_B about 0.05; verdict probabilities not recovered 40%, partial 25%, recovered 10%, uninterpretable 25%; at least 15 of 21 N=5 cells keep gap_M above 2%.
- 預測原文：`notes/impl/l3_mappo_prereg.md@38ce89e` 第 63 行：「F_A 點估計 **0.15**（80% 區間 −0.15 到 0.45），F_B 約 0.05。判讀機率：沒補回 40%、部分補回 25%、補回 10%、無法判讀 25%。」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 6 行：「**L3 結果「無法判讀」，L3 正式不跑，停下問〔作者〕**」
- 結果（表內）：Not executed (positive control failed); verdict “uninterpretable”.
- 判定：**not assessed（無判定）**｜依據：專案紀錄的判定｜L3 主跑依預先登記未執行，這些預測沒有可比的結果。

#### LB1
- 預測（表內）：Positive control passes: at U=500 10%; by U≤1000 25%; by U≤2000 35%.
- 預測原文：`notes/impl/l3_mappo_prereg_v2.md@0e16639` 第 38 行：「正向對照通過：U=500 通過 10%；U≤1000 通過 25%；U≤2000 通過 35%。」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 44 行：「預測「U≤2000 通過 35%」：沒通過，落在預期內。」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 32 行：「U=500：0/3 格；U=1000：0/3 格；U=2000：1/3 格」
- 結果（表內）：0/3 cells at U=500, 0/3 at 1000, 1/3 at 2000; did not pass.
- 判定：**held（成立）**｜依據：專案紀錄的判定｜專案寫「落在預期內」。

#### LB2
- 預測（表內）：F_G median over the three cells at U=2000: 0.35 (80% interval 0.0–0.7).
- 預測原文：`notes/impl/l3_mappo_prereg_v2.md@0e16639` 第 39 行：「通過與否的 F_G 點估計：U=2000 時三格 F_G 的中位數 **0.35**（80% 區間 0.0–0.7）」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 44 行：「F_G 中位數預測 0.35，實際 U=2000 三格中位數 −0.00。」
- 結果（表內）：−0.00 (cell values −0.42, 0.74, −0.00).
- 判定：**failed（不成立）**｜依據：本次判定｜專案只並列預測與實際，沒有判定字眼。實際中位數 −0.0026（calib/out/l3_mappo_control_v2.json），略低於 80% 區間下緣 0.0，故本次判 failed。論文第 3 節已寫出此落差。

#### LB3
- 預測（表內）：At least one cell shows a collapsed seed (entropy below 0.01 and F_G below −1): probability 30%.
- 預測原文：`notes/impl/l3_mappo_prereg_v2.md@0e16639` 第 39 行：「至少一格出現「塌縮」種子（熵 <0.01 且該種子 F_G <−1）的機率 30%」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 45 行：「U=2000 的 15 個 run 中有 2 個塌縮」
- 結果（表內）：It happened: 2 of 15 runs at U=2000 collapsed (c10_ar10 seed 4, F −1.74; c10_ar50 seed 4, F −1.87).
- 判定：**failed（不成立）**｜依據：本次判定｜機率型預測依判定規則以較可能的一側計：預測給 30%，事件發生了，故 failed。

#### LB4
- 預測（表內）：Diagnosis: collapse comes from too-low initial “no-move” entropy; lowering the bias from +5 to +2 should remove it.
- 預測原文：`notes/impl/l3_mappo_prereg_v2.md@0e16639` 第 22 行：「推論：初始「不動」機率 ≈0.98，早期更新中換位動作很少被取樣到，換位的優勢估計雜訊大，個別種子容易把換位壓成 0。」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 45 行：「**打臉條件 (b) 成立**」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 45 行：「所以「塌縮來自初始熵太低」的診斷（推論）不成立或至少不充分。」
- 結果（表內）：Two of 15 runs at U=2000 still collapsed; the record: the diagnosis “is not established, or at least insufficient”.
- 判定：**failed（不成立）**｜依據：專案紀錄的判定｜專案寫「打臉條件 (b) 成立；診斷不成立或至少不充分」，對映為 failed。

## 4. 預測檔自己寫下的打臉／判讀條件（條件表，不計入條數）

#### CD1 Birds v2
- 條件（表內）：A2 sustains rotation but is energy-blind: “visible energy gaps are still not learned”.
- 原文：`notes/impl/survival_v2_prereg.md@2838e68` 第 55 行：「A2 維持但**能量盲**（維持 run 皆 ≤0.40）→ 「看得見體力差仍學不到依能量分工」成立」
- 結果出處：`notes/impl/survival_v2_results.md` 第 36 行：「A2 維持但能量盲 → 照規則：**看得見體力差仍學不到依能量分工**」
- 狀態：Triggered: 3 of 3 sustaining runs energy-blind.

#### CD2 Soaring, round 2
- 條件（表內）：The planner loses or ties L1: “undecided”, not “no room”.
- 原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 24 行：「規劃器輸或打平 = 「未決」，不是「沒空間」。」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 17 行：「規劃器輸或打平 → 記為**未決**，不是沒空間。」
- 狀態：Triggered: plan minus L1 −6.2 to +1.1.

#### CD3 Soaring, round 2
- 條件（表內）：The gap appears only in floor cells: “no evidence of room for MARL”.
- 原文：`.scratch/soaring_headroom2/PREDICTION.md@4eacb94` 第 24 行：「止損：差距只出現在地板格 → 「此設定下 MARL 無證據空間」。」
- 結果出處：`notes/impl/soaring_headroom2.md` 第 18 行：「規劃器的差距沒有出現在任何格（地板或非地板），所以不觸發「只在地板格有差距」」
- 狀態：Not triggered: the planner's gap appeared in no cell.

#### CD4 Soaring, round 3
- 條件（表內）：L1.5a recovers most of the gap: “a better hand-written rule”.
- 原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 14 行：「L1.5a 追回大部分差距 -> 「更好的手寫規則」」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 37 行：「不是「更好的手寫規則」」
- 狀態：Not triggered: L1.5a was worse than L1 in every cell.

#### CD5 Soaring, round 3
- 條件（表內）：L1.5b recovers most of the gap: “information gap”.
- 原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 14 行：「L1.5b 追回大部分差距 -> 資訊差」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 36 行：「差距幾乎全是資訊差」
- 狀態：Triggered: recovered 76%–164%.

#### CD6 Soaring, round 3
- 條件（表內）：“Evidence of room for MARL” needs a lower bound above 2 pp for L2 minus max(L1.5a, L1.5b) in at least 3 non-floor cells; otherwise “no evidence of room”.
- 原文：`.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 14 行：「否則「此設定下 MARL 無證據空間」」
- 結果出處：`notes/impl/soaring_headroom3.md` 第 35 行：「L2−max 的區間下界 >2 的非地板格 = **0 / 4**」
- 狀態：Triggered (the “otherwise” branch): 0 of 4 non-floor cells.

#### CD7 Language-model commander
- 條件（表內）：The two models' compliance differs by less than 5 percentage points: “model size affects compliance” does not hold.
- 原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 30 行：「若兩者遵守率差距 < 5 個百分點，則「模型大小影響遵守」不成立。」
- 結果出處：私人開發紀錄（PROGRESS，第 386 行）：「「模型大小影響遵守」不成立」
- 狀態：Triggered: both 14/14, a 0 point difference.

#### CD8 Language-model commander
- 條件（表內）：Median regret below 1% for both models: the environment is too lenient.
- 原文：`.scratch/llm_commander/PREDICTION.md@5c53a88` 第 31 行：「若 regret 中位數兩者都 < 1%，則環境太寬鬆，結果主要說明 gate 是顯而易見的選擇。」
- 結果出處：私人開發紀錄（PROGRESS，第 386 行）：「環境太寬鬆，gate 是顯而易見的選擇」
- 狀態：Triggered: both medians about 0.

#### CD9 Learner, first
- 條件（表內）：(c) The positive control passes already at U=500 with F_G ≥ 0.9.
- 原文：`notes/impl/l3_mappo_prereg.md@38ce89e` 第 64 行：「(c) 正向對照在 U=500 就通過且 F_G ≥0.9（我高估了難度）」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 6 行：「正向對照（MAPPO-G，3 格 × 3 種子）在 U=500、1000、2000 都是 0/3 格通過」
- 狀態：Not triggered: 0/3 cells passed at U=500.

#### CD10 Learner, first
- 條件（表內）：(a) F_A lower bound ≥ 0.5; (b) F_A ≤ −0.1 with upper bound below 0; (d) F_B lower bound above 0.3.
- 原文：`notes/impl/l3_mappo_prereg.md@38ce89e` 第 64 行：「**打臉條件**：(a) F_A 的 CI 下界 ≥0.5（我低估了局部學習）」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 6 行：「**L3 結果「無法判讀」，L3 正式不跑，停下問〔作者〕**」
- 狀態：Not evaluable: the L3 main run was not executed.

#### CD11 Learner, second
- 條件（表內）：(a) The positive control passes already at U=500.
- 原文：`notes/impl/l3_mappo_prereg_v2.md@0e16639` 第 41 行：「(a) U=500 就通過（我高估了難度）」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 32 行：「U=500：0/3 格；U=1000：0/3 格；U=2000：1/3 格」
- 狀態：Not triggered: 0/3 at U=500.

#### CD12 Learner, second
- 條件（表內）：(b) At least 2 of 15 runs at U=2000 still collapse: the “low initial entropy” diagnosis is refuted.
- 原文：`notes/impl/l3_mappo_prereg_v2.md@0e16639` 第 41 行：「(b) 15 個 run（3 格×5 種子）在 U=2000 中仍有 ≥2 個塌縮」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 45 行：「**打臉條件 (b) 成立**」
- 狀態：Triggered: 2 of 15 runs collapsed.

#### CD13 Learner, second
- 條件（表內）：(c) All 15 runs have F_G above 0 but fewer than 2 of 3 cells pass.
- 原文：`notes/impl/l3_mappo_prereg_v2.md@0e16639` 第 41 行：「(c) 15 個 run 全部 F_G>0 但 3 格中通過 <2 格」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 46 行：「條件 (c) 未成立（仍有 F_G<0 的種子）」
- 狀態：Not triggered: several runs had F_G below 0.

#### CD14 Learner, second
- 條件（表內）：(d) The control passes with a point estimate of F_G ≥ 0.9.
- 原文：`notes/impl/l3_mappo_prereg_v2.md@0e16639` 第 41 行：「(d) 通過但 F_G 點估計 ≥0.9」
- 結果出處：`notes/impl/l3_mappo_results.md` 第 46 行：「(a)、(d) 未成立」
- 狀態：Not triggered: the control did not pass.

## 5. 找不到結果或判定的預測

- **RR8**（Rotation, richer）：N=3 echelon (three levels) has a gap slightly larger than the V.　→ 專案「與預測對照」沒有這一條。差距表在 ±0% 與 ±5% 各層的 V、梯隊大小互有高低，不足以判定，故 not assessed。
- **WK9**（Wake surfing）：z-axis drift is weakly observable from roll moment, so the Kalman filter closes only about half the gap at A=0.2.　→ 專案原文「我沒有單獨量這一點，無結論」。
- **SA12**（Soaring, round 1）：(c) Inference: the true RL room is smaller than (c), because the reference has information RL cannot get; the gap should be smaller after removing information.　→ 專案寫「這個測試沒辦法量出多小」。第 2、3 輪才開始做資訊對等的比較，但未回頭判這一條。
- **SA14**（Soaring, round 1）：Best single-glider rule: a medium leave threshold.　→ 專案沒有對「離開門檻中等」下判定；−0.8 是最終網格的最低值，但專案自行判為平台，故 not assessed。
- **SA15**（Soaring, round 1）：Best rule: the two-wing difference signal gains less than 1 pp at high noise.　→ 專案原文「沒量……無結論」。
- **SA17**（Soaring, round 1）：Best shared rule: the crowding penalty has almost no effect.　→ 專案沒有對這一條下判定，也沒有做開關消融；被選中次數不能直接推出「幾乎無影響」，故 not assessed。
- **LA2**（Neighbours-only learner）：L3 main run: F_A point estimate 0.15 (80% interval −0.15 to 0.45), F_B about 0.05; verdict probabilities not recovered 40%, partial 25%, recovered 10%, uninterpretable 25%; at least 15 of 21 N=5 cells keep gap_M above 2%.　→ L3 主跑依預先登記未執行，這些預測沒有可比的結果。

另外，下列預測專案紀錄有結果文字、但沒有判定字眼，由本次判定（共 34 條 held/failed/partly，tex 內以 † 標示）：

RB8a、RB10、RR13、RN11、WK6、WK11、SA6、SA18、SB2、SB4、SB7、SB8、SB10、SB11、SB13、SB14、SC2、SC3、SC5、SC6、SC7、SC8、LM1、LM3、LM5、LM6、LM7、LM8、LM9、LM10、LM11、LM12、LB2、LB3

以下**沒有**對專案已寫下的判定做任何改動；對映（例：「偏低」→ partly）的規則見第 1 節，如果作者希望更嚴或更寬，只需改 `scripts/scorecard_rows.py` 的 `v` 欄後重新產生。


## 6. 抽查（表內數字回原始檔核對）

方法：以固定種子（`random.seed(20261118)`）從 113 列中隨機抽 8 列，把該列 Outcome 欄的數字回到原始 JSON 或 git 版本重新計算（腳本輸出存於 `.scratch/audit_20261118/tmp/sample_check_out2.txt`，LLM 兩列另跑 `git show f1c439d:.scratch/llm_commander/results/<model>.json`）。另外在抽樣前已先用 JSON 核對過 wake、drone_bucket、l1、l23、survival_v2、soaring3、L3 v2 的主要數字（`sample_check_out.txt`），也全部相符。

| 抽到的列 | 表內數字 | 原始檔重算結果 | 相符 |
|---|---|---|---|
| SB12 | planner 對 L1 的區間下界 >2：N=5 為 0/8、N=10 為 0/3 | `calib/out/soaring_headroom2.json` 的 `gap_vs_L1_Emin.plan`：N=5 八格 0 格、N=10 三格 0 格（點估計 −1.8…+1.1、−2.3…−0.2） | 是 |
| LM12 | g1、c2、s7 兩模型皆 gate；14 條中 gate 11 | f1c439d 的 haiku.json、sonnet.json：三條皆 gate；計數 gate 11、fixed 2、rot 1（兩者相同） | 是 |
| RR10 | 平均差距 0.24、0.19、0.16、0.18%（N=3、5、7、10） | `calib/out/drone_bucket_l1.json` 各 N 的 `best_gap` 平均：0.24、0.19、0.16、0.18 | 是 |
| WK2 | 0.89–1.41 pp（高滾轉雜訊 1.08–1.39） | `calib/out/wake_headroom.json`，`drift=amp0.1_tau30` 12 格 gap 最小 0.89、最大 1.41；roll 非 low 六格 1.2、1.2、1.08、1.08、1.39、1.39 | 是 |
| RB5 | N=5 全 >2%；N=7、10 在 s=0.1 為 1.2–2.3% 且只有 N=7,c=10 一格破；N=3,s=0.1 為 2.7–3.0% | `calib/out/drone_bucket.json`（h=0.05）：N=5 全 >2% 為真；N=7/10、s=0.1 範圍 1.2–2.3，超過 2% 的只有 (7, c=10, 2.27%)；N=3、s=0.1 為 2.7、2.8、2.8、3.0 | 是 |
| RB1 | 同質、c≤1 的 24 格中 22 格 <2%；兩格為 2.04%、2.08% | `calib/out/drone_bucket.json`：24 格，<2% 的 22 格；≥2% 的為 (N=5, s=0.3, c=1, 2.04%)、(N=7, s=0.3, c=1, 2.08%) | 是 |
| LM2 | 兩模型皆 14/14 遵守 | f1c439d 的 haiku.json、sonnet.json：`valid` 與 `constraint_ok` 皆 14/14 | 是 |
| BV3 | B2 兩個塌縮、一個中間；中間那個最後十分位換位率 0.0050 | `calib/out/survival_v2_seeds.json` 的 `B2.runs`：seed 21「中间」0.0050、seed 22「塌缩」0.0031、seed 23「塌缩」0.0042 | 是 |

8/8 相符。順帶核對（非隨機抽到，但在製表過程中用 JSON 算過）：wake sine 12 格 gap 2.04–2.79、84 格隨機漂移最大 1.55 且 0 格 >2；drone_bucket 時間輪 >2% 為 132/192、閘門最大 2.22%；l1 為 768 格、最佳規則 >2% 僅 1 格（最大 2.18%）、±15% 最大 0.59%、±30% 最大 0.23%、閘門最佳 374 格；l23 為 160 格、最佳全域規則 >2% 為 23 格且全是 N=10 h=0、N=5 最大 1.71%；survival_v2 A2 三個 run 的 `incoming_is_max` 為 0.3429、0.3374、0.3380；soaring3 mid/short/W3 N=5 的 L1.5b−L1 = +11.4 [8.8, 14.1]、L2−max = +2.5 [0.1, 5.1]；l3_mappo_control_v2 U=2000 三格 F_G = −0.4208、0.7352、−0.0026（中位 −0.0026）；LLM 結果 JSON 的 regret 平均 Haiku 0.001425、Sonnet 0.001132，中位數 1.1e-4 與 2.1e-5。

## 7. 給作者的提醒（不影響表，但值得看）

1. 這張表的 113 條裡，專案紀錄原本就有判定（或明寫沒量、無結論）的是 78 條；其餘 35 條由本次處理，其中 34 條給了 held／failed／partly（tex 內有 †，摘要表「Verdicts assigned for this table」欄即此數），1 條（SA17）判 not assessed。如果你想讓表更保守，把本次判定的 partly／failed 改成 not assessed 只需動 `scripts/scorecard_rows.py`。
2. 事實查核提到的「只報最戲劇的兩個預測」：本表揭露後，held 條數（45）其實不少，但 failed 與 partly 加起來 61 條，高於 held；論文主文目前只講 wake surfing 與 L3 第二份兩件事。
3. 輪流 richer 家族的預測針對 576 格、量測 768 格；摘要附錄只用一句話帶過，論文第 3 節（`main.tex` 第 88–92 行）與圖 2 的 caption 仍建議明寫。
4. 「預測由誰撰寫」：作者已於 2026-11-19 確認（`docs/canonical_numbers.md` 第 114 行）。tex 用被動語態，檔頭註解記下了這項確認。
5. 專案紀錄的「與預測對照」只涵蓋 L1、L2、L3、貼尾流、滑翔第 1、2 輪、LLM（在私人開發紀錄 PROGRESS）、學習者；滑翔第 3 輪只有「和預測不同」三條，鳥 v2 只有條件層級表，沒有逐條對照（第 3 節各列的「本次判定」多半落在這幾個家族）。
