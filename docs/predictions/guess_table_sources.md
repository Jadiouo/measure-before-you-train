> **Note on paths and commit hashes.** Paths that start with `notes/`, `.scratch/` or `docs/writing/`, and commit
> hashes used with `git show`, refer to the author's private development repository. They are not
> part of this public export. The prediction files are copied verbatim into `docs/predictions/`,
> and the commit record is in `docs/provenance.md`.

# 正文表 `tab:guess`（猜想／實際／學到什麼）的來源對照

日期 2026-11-19。對應檔案：`paper/main.tex` 第 4 節（Results）開頭的表。作者要求把預測統計句改寫成「猜想、實際、學到什麼」的對照。

## 規則
- 「猜想」欄只轉述預測檔原文；預測檔原文取自預測 commit 當時的版本（`git show <hash>:<path>`，`.scratch/` 下的預測檔已在工作目錄外，故以 commit 為準；逐條行號與引句另見 `scorecard_sources.md` 第 3 節）。
- 「實際」欄只用現有稿子或 `scorecard_sources.md` 已記載的結果。
- 「學到什麼」欄只有兩種：(a) 量測直接顯示的事實；(b) 專案當時紀錄寫下的解讀，句中標明 `Our reading, untested` 或 `conjecture, untested`。
- 表中選了 5 列（成立 2 列：基本層、貼尾流的小漂移半句；不成立或部分成立 3 列）。表說明已寫「列是挑選的，完整統計見附錄」。
- 「學到什麼」欄沒有找到出處因而放棄的寫法：見檔尾「放棄或改寫」。

## 第 1 列：基本層（Basic rotation）

| 格 | 表中文字 | 出處 | 類別 |
|---|---|---|---|
| 猜想 | A rule that reads battery levels stays within 2–3% of the best plan when swapping is cheap. | `scorecard_sources.md` 第 191–197 行（RB9）；預測檔 `.scratch/drone_bucket/PREDICTION.md@a0619ab` 第 17 行：「電量閘門 (3) 預期在 c_swap ≤ 3 時與最優差 < 2–3%」。判定 held | 預測轉述 |
| 實際 | Right, and better: within 2% in all but 4 of 192 settings, all at the highest swap cost. | `notes/impl/drone_bucket.md` 第 9 行：「電量閘門 (3)…與最優只差 0.0%–2.2%，只有 4 格 > 2%（皆為 s=0.3、c_swap=10…）」；第 159 行：「對，比預期更好」；c 的格值 {0,1,3,10} 見第 15 行；稿內 `4 of 192` 見 main.tex Results 4.1 | 結果 |
| 學到什麼 | Measured: the time rule did far worse, more than 2% from the best plan in 132 of 192 settings. | `notes/impl/drone_bucket.md` 第 7 行：「差距 > 2% 的有 **132 格（69%）**」；稿內同句見 Results 4.1（"132 of 192 settings leave the time rule more than 2% from the bound"） | (a) 量測事實 |

## 第 2 列：richer 層（Richer rotation）

| 格 | 表中文字 | 出處 | 類別 |
|---|---|---|---|
| 猜想 | About 30% of settings would leave more than 2% of room, mostly where the drones differ most. | `scorecard_sources.md` RR3、RR4、RR5、RR13（第 229–250、300–308 行）；預測檔 `PREDICTION_L1.md@79b4dbc` 第 16 行：「約 30% 的格子破 2%」、第 18 行：「+-30%：差距更大，約 45% 的格子破 2%」、第 19 行：「全 576 格合計：最佳規則 > 2% 的格數約 30%」、第 24 行：「值得的是 +-15/30%、f_ind 大、最佳設定、c>=3 的區塊」；稿內 Results 4.1：「The prediction pointed to settings with large drone differences」 | 預測轉述 |
| 實際 | Only 1 of 768 did, and none of the settings with large differences. | `notes/impl/drone_bucket_l1.md` 第 98 行：「1/768 = 0.13%」；第 96、97 行：±15% 最差格 0.59%、±30% 最大 0.23%（0 格破）；稿內 Results 4.1：「none of those exceeds 2%」 | 結果 |
| 學到什麼 | Our reading, untested: when drones differ a lot, the best plan is simply to put the weakest drone where it saves most, which a rule does easily. | `notes/impl/drone_bucket_l1.md` 第 107 行：「預測最大的錯誤：…異質度 ±15% 以上時，B_i/k_i 的差距遠大於節省量，最優近似「把最弱的一架放最省位置」，規則式輕易達到。…（推論）」 | (b) 專案紀錄的解讀（標「推論」），表中標 untested |

## 第 3 列：貼尾流（Wake surfing）

| 格 | 表中文字 | 出處 | 類別 |
|---|---|---|---|
| 猜想 | Gaps above 2 pp would appear only at large drift; with little or no drift they would stay below 2 pp. | 預測檔 `.scratch/wake_headroom/PREDICTION.md@f49e6e8` 第 12 行：「Gap > 2 pp: A=0.2 for both tau」；第 14 行：「Gap < 2 pp (no RL room): A in {0, 0.05} everywhere」；第 20 行：「the headline will be "headroom exists only at large drift (A>=0.2, maybe 0.1 slow)…"」；`scorecard_sources.md` WK1（第 398 行起）、WK5（第 429 行起）、WK11（第 474 行起） | 預測轉述（小漂移半句判 held，大漂移半句判 failed） |
| 實際 | Half right. Little or no drift: at most 1.30 pp. Large random drift: also below 2 pp (at most 1.55). Gaps above 2 pp appeared only when the lead aircraft manoeuvred. | `notes/impl/wake_headroom.md` 第 36 行：「A≤0.05 全部 <2 pp｜對。」；第 9 行：「隨機漂移（AR(1)）的 84 格，差距全部 <2 pp（最大 1.55…）」；第 10 行：「差距 >2 pp 只出現在領航機機動」；稿內附錄 C 表 `tab:wake`：No drift 最大 1.30、Large, slow 最大 1.55；Results 4.2：「all of them when the lead aircraft manoeuvres」。「Half right」是本表對 WK5 held ＋ WK1／WK11 failed 的概括 | 結果 |
| 學到什麼 | Our reading, untested: at large drift even the references save at most 3.5 pp of power, so there is little to win. | `notes/impl/wake_headroom.md` 第 33 行：「A=0.2（兩種 τ）差距 >2 pp｜錯。差距 ≤1.55。原因：A=0.2 時所有方法的絕對節省都掉到 2–3.5 pp，可贏的絕對量本來就小。」（該「原因」未另行檢驗） | (b) 專案紀錄的解讀；`2–3.5 pp` 是新數字（見 new_numbers.tsv）；calib/out/wake_headroom.json（drift 以 amp0.2 開頭的 24 格：priv 最大 3.47、priv_gps 最大 3.41） |

## 第 4 列：滑翔（Soaring）

| 格 | 表中文字 | 出處 | 類別 |
|---|---|---|---|
| 猜想 | Giving the simple rule the true thermal data would recover most of its gap to the strongest method. | 預測檔 `.scratch/soaring_headroom3/PREDICTION.md@02dc53f` 第 26 行：「L1.5b 追回大部分差距機率約 60%」（L1.5b＝簡單規則的決策結構加真值資訊）；`scorecard_sources.md` SC5（第 763 行起）。判定 held（本次判定，見該節） | 預測轉述 |
| 實際 | It did, and the strongest method's remaining lead was never confidently above 2 pp. | `notes/impl/soaring_headroom3.md` 第 36 行：L1.5b 追回 L2−L1 的 76%–164%；第 42 行：「預測 L1.5b 追回「一半以上」（機率 60%）；實際追回約 76%–164%」；稿內 Results 4.2：「Against the simple rule's decision structure given the same data, its lead is never confidently above 2 pp」 | 結果 |
| 學到什麼 | Measured: the gap is mostly information, not decision structure. The reference is the best we tried, not an optimum. | `soaring_headroom3.md` 第 36 行：「差距幾乎全是資訊差」；稿內 Results 4.2：「Here the reference is "the best we tried" rather than a true optimum」「the large differences we found come from information rather than decision structure」 | (a) 量測事實（後一句是稿內既有但書） |

## 第 5 列：學習器資格考（Learner sanity gate）

| 格 | 表中文字 | 出處 | 類別 |
|---|---|---|---|
| 猜想 | The better-informed control learner would pass the sanity gate: 75% chance on the first try, 35% on the second. | `notes/impl/l3_mappo_prereg.md@38ce89e` 第 62 行：「正向對照：U=500 通過 35%；U≤1000 通過 65%；U≤2000 通過 75%。」；`l3_mappo_prereg_v2.md@0e16639` 第 38 行：「U≤2000 通過 35%」；`scorecard_sources.md` LA1（第 889 行起）、LB1（第 904 行起）；稿內附錄 E（Preregistration details）：「The first preregistration gave a 75% chance of passing by 2000 updates…The second gave 35%」 | 預測轉述 |
| 實際 | The first try passed in none of the three control settings; the second in one, where two were required. | `notes/impl/l3_mappo_results.md` 第 6 行：「0/3 格通過」；第 32 行：「U=2000：1/3 格（要 ≥2 格…）」；稿內 Results 4.3：「passed in none of the three control settings; a second…passed in one, where two were required」 | 結果 |
| 學到什麼 | Measured: some runs still collapsed, so our diagnosis of the first failure, too little initial exploration, was at best incomplete. Our conjecture, untested: costly early swaps push the policy to swap too rarely. | 前半：`l3_mappo_results.md` 第 45 行：「U=2000 的 15 個 run 中有 2 個塌縮…所以「塌縮來自初始熵太低」的診斷（推論）不成立或至少不充分」；稿內附錄 D：「so the explanation that low initial entropy caused the collapses was at best incomplete」。後半：第 47 行：「（推論：c10 格換位成本高，早期亂換位的懲罰使策略往「幾乎不換」塌縮…；未測試。）」；稿內附錄 D：「we conjecture, without testing, that early swaps are punished by their cost…」 | 前半 (a) 量測事實，後半 (b) 解讀（untested）；本格是 (a)＋(b) |

## 「學到什麼」欄類別一覽
| 列 | 類別 |
|---|---|
| 1 基本層 | (a) |
| 2 richer 層 | (b) |
| 3 貼尾流 | (b) |
| 4 滑翔 | (a) |
| 5 資格考 | (a)＋(b) |

## 放棄或改寫（找不到出處）
- 原想為貼尾流小漂移那半句另寫一格「學到什麼」（例如 Kalman 控制器勝過固定設定點的幅度），但稿內沒有這些數字且為避免新增數字，改為把貼尾流併成一列，不另列。
- 想在滑翔列寫「只用觀測估計的決策結構更差的原因」，專案紀錄寫的是「它輸的原因（離開誤判、量測噪聲）我沒有做診斷，屬推論」（`soaring_headroom3.md` 第 46 行「限制」節），沒有可寫的已驗證說法，故不寫。
- 想在 richer 列補 ±5% 一格的原因，出處只有同一段「（推論）」，已併入第 2 列的 (b) 句。
- 雜訊層（noisy）沒有列入：其「與預測對照」寫「數字對、原因錯」，原因說明牽涉逾時的界，稿內描述複雜，不適合放進白話表。
