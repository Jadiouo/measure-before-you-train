> **Note on paths and commit hashes.** Paths that start with `notes/`, `.scratch/` or `docs/writing/`, and commit
> hashes used with `git show`, refer to the author's private development repository. They are not
> part of this public export. The prediction files are copied verbatim into `docs/predictions/`,
> and the commit record is in `docs/provenance.md`.

# canonical_numbers：論文 paper/main.tex 的每個數字

規則：只用下列來源。「key」是 `paper/figures/numbers.json` 的欄位；該檔由 `.venv/bin/python scripts/make_paper_figs.py` 從 `calib/out/*.json` 產生（腳本內 assert 會對 notes/impl/portfolio_numbers.md 已核對過的數字）。找不到的標 [TODO]。圖也由同一腳本產生。
單位：L0–L3 為相對差距 % ＝(上界−規則)/上界；貼尾流為總功率的百分點（pp）；滑翔為最低那架時間平均高度占上限的百分點（pp）。

## A. 輪換排程（calib/out/drone_bucket*.json）
| 論文中的數字 | key | 來源與欄位 |
|---|---|---|
| 192 格；時間輪 132 格 >2%、最大 12.14、(中位 3.52) | L0_cells, L0_timerot_gt2, L0_timerot_max | drone_bucket.json `cells[].gap_tail`（×100） |
| 192 格：看電量規則 4 格 >2%、最大 2.22、中位 0.095（768 格的看電量規則見 H 區） | L0_battery_gt2, L0_battery_max, L0_battery_median | drone_bucket.json `cells[].gap_gate` |
| 1% 線：11/192（看電量）、22/768、28/108（同 GPS）、52/108（完美位置） | L0_battery_gt1, L1_best_gt1, wake_fair_gt1, wake_perfect_gt1 | 同上與 drone_bucket_l1.json、wake_headroom.json |
| 768 格；最佳規則 1 格 >2%（2.18）、中位 0.045 | L1_cells, L1_best_gt2, L1_best_max, L1_best_median | drone_bucket_l1.json `jobs[].cells[].best_gap` |
| 該格以對偶界 2.18、以最佳可行解 1.89 | L1_over_cell_gap_dual, L1_over_cell_gap_incumbent | 同上 `best_gap`、`opt`、`opt_incumbent`：1−(1−best_gap)·opt/opt_incumbent |
| 152 格求解器未證最優 | L1_timeout_cells | `opt_timeouts>0` 或 `opt≠opt_incumbent` |
| 160 格；62 格逾時；98 格可信 | L23_cells, L23_timeout_cells, L23_trusted_cells | drone_bucket_l23.json 同上判準 |
| 全域規則 23 格 >2%（全為逾時格）；可信格最大 1.71 | L23_global_gt2, L23_global_gt2_trusted, L23_global_max_trusted | `best_global_gap` |
| L3：23 格（98 個可信格中 rot>2%）；20 格是 N=5、AR、h=0；其餘 3 格 | L3_n, L3_n_n5ar_h0, L3_n_other | `rules.rot.gap>0.02`；`N==5, noise=="ar", h==0` |
| 局部 1.73–3.82；全域 0.69–1.71；逐格差 0.52–2.17（中位 1.41）；局部>2 有 21 格、全域>2 有 0 格 | L3_local_*, L3_global_*, L3_diff_*, L3_local_gt2, L3_global_gt2 | 局部＝min(`rules.rot.gap`,`rules.local.gap`)；全域＝`best_global_gap`；差＝局部−全域（占上界 %） |
| 預測「約 30% 格子會破 2%」；預測時網格 576 格 | pred_frac_L1, pred_grid_L1 | **不在 calib/out**：`git show 79b4dbc:.scratch/drone_bucket/PREDICTION_L1.md`（第 4 點、標題行的「共 576 格」） |
| 附錄 A：網格參數（N∈{3,5,7,10}、f_ind∈{0.3,0.4,0.5}、c∈{0,1,3,10}、h 的 4 級、200 實例、60 個 LP 實例、σ/AR/事件機率） | — | notes/impl/drone_bucket_l1.md、drone_bucket_l23.md 第 2 節；PREDICTION_L1.md 網格行；h 的 4 級＝`best_gap` 依 h 分組（L1_by_h 的鍵） |
| 本檔之 f_ind 掃描 0.3–0.5 是假設 | — | portfolio_numbers.md C 區 |

## B. 貼尾流（calib/out/wake_headroom.json）
| 數字 | key | 來源 |
|---|---|---|
| 108 格；公平參考最大 1.77、95% 區間上限最大 1.83、0 格 >2；最差 −0.24 | wake_cells, wake_fair_max, wake_fair_ci_hi_max, wake_fair_gt2, wake_fair_min | `cells[].gap_gpsprivileged`、`gap_gpsprivileged_ci[1]` |
| 完美位置參考最大 2.79；16 格點估計 >2（sine 12、step 4）；14 格區間下界>2（預測檔定義的參考；同 GPS 參考是診斷用，預測檔 `priv_gps` 0 次） | wake_perfect_max, wake_perfect_gt2, wake_perfect_cilo_gt2 | `cells[].gap`、`gap_ci[0]` |
| 最佳傳統法 kf 90／固定 18／ESC 0 | wake_best_counts | `cells[].best` |
| 附錄 B 表（9 情境 min/median/max） | — | 同 JSON，依 `drift` 分組；指令見 .scratch/writing/（本次以 python 即時算出；可重算：對 cells 依 drift 取 gap_gpsprivileged 與 gap） |
| 預測「最大差距 3–5 pp」 | — | notes/impl/wake_headroom.md「與預測對照」末列 |
| 64 評估種子；GPS σ∈{0.02,0.04,0.08}b；12 格/情境 | — | notes/impl/wake_headroom.md 設定段 |

## C. 滑翔（calib/out/soaring_headroom2.json、soaring_headroom3.json）
| 數字 | key | 來源 |
|---|---|---|
| 11 格（N=5 8 格、非地板 4 格） | soar_cells, soar_n5_cells, soar_n5_nonfloor | soaring_headroom3.json `results[].cells[N]`、`floor_cell` |
| 最強法減簡單規則決策＋真值（L2−max(L1.5a,L1.5b)＝L2−L1.5b）：最高 +2.5 [0.1, 5.1]（mid\|short\|W3.0）；區間下界>2 為 0 格。注意 L1.5b 拿真值，不是 simple method；對簡單規則 L1 的差距見 H 區 | soar_top_*, soar_nonfloor_cilo_gt2, soar_all_cilo_gt2 | `cells[N].gaps.Emin["L2-max(L15a,L15b)"]` |
| L1.5b 追回 76–164%（9/11 ≥89%）；L1.5a 比 L1 差 0.2–24.5（11/11） | soar_recov_b_*, soar_l15a_* | `gaps.Emin["L15b-L1"]/["L2-L1"]`；`["L15a-L1"]` |
| 通報比各飛各的多 5.6–24.2（4 個長壽命 N=5 格） | soar_announce_gain_* | soaring_headroom2.json `cells["5"].gap_vs_L1_Emin.L0[0]` 取負，名稱含 `|long|` |
| 起始高度 1600、上限 3200、T=2400、沉率 1、空速 12 | — | notes/impl/soaring_headroom2.md 設定段；portfolio_numbers.md B 區（soaring_headroom3.json `consts`） |
| 60 評估種子、24 調參種子 | — | notes/impl/soaring_headroom2.md |

## D. L3 MAPPO 正向對照（calib/out/l3_mappo_control.json, l3_mappo_control_v2.json）
| 數字 | key | 來源 |
|---|---|---|
| v1：U=500/1000/2000 全 0/3 通過；最佳均值 0.29 | ctrl_v1 | l3_mappo_control.json `{500,1000,2000}.n_pass`、`cells[].F_G` |
| v2：0/3、0/3、1/3；通過格 0.74 [0.43, 0.97]；其餘 −0.42、−0.00 | ctrl_v2 | l3_mappo_control_v2.json `U{500,1000,2000}` |
| 附錄 F 表（各格各 U 的 F_G 與 CI） | ctrl_v1, ctrl_v2 | 同上 `F_G`、`ci95` |
| U=2000 的 15 個 run 中 2 個 F_G<−1（−1.74、−1.87） | ctrl_v2_U2000_neg_runs | `per_seed` |
| 該 2 run 熵 0.004、取樣換位 0.8／0.9 | — | **不在 JSON**：notes/impl/l3_mappo_results.md「對照預測與打臉條件」（熵由訓練 log，.scratch/l3/，不進 git） |
| 預測：v1 U≤2000 通過 75%；v2 35%；v2 F_G 中位預測 0.35 | — | notes/impl/l3_mappo_prereg.md 第 6 節；l3_mappo_prereg_v2.md 第 3 節 |
| 通過條件：≥2/3 格 F_G≥0.5 且 CI 下界>0；none_bias +5→+2；種子 3→5 | — | l3_mappo_prereg.md 第 4 節；l3_mappo_prereg_v2.md 第 1–2 節 |
| 網路 2×64 tanh、5 步一決策、256 回合/更新、200 評估實例 | — | l3_mappo_prereg.md 第 1、8 節 |

## E. 鳥類與語言模型指揮官（附錄；主文只引一句）
| 數字 | key／來源 |
|---|---|
| 報酬 1.001±0.017；減 rot30 +0.013±0.017；rot30 0.9875；gate30s 1.0284；接手者最高能量 0.339（機率 1/3） | bird_A2_*, bird_rot30, bird_gate30s；survival_v2_seeds.json `A2.reward_seedlevel`、`minus_rot30_seedlevel`、`incoming_max_seedlevel`、`_refs` |
| 3 種子、5e7 步、512 評估回合、3 隻鳥 | notes/impl/survival_v2_results.md 第一段（portfolio_numbers.md B 區「2」） |
| 14 條指令、兩模型 14/14 有效且遵守、regret 最大 1.15%／1.31%、預測 71%／86% | portfolio_numbers.md A2、B 區「8」；私人開發紀錄（PROGRESS）「LLM 指揮官錄製」；.scratch/llm_commander/results/*.json |
| 模型別名、2026-11-02、版本字串未記錄、各 1 次工具呼叫未查 | portfolio_numbers.md A3 末列；私人開發紀錄（PROGRESS）同節 |

## F. 時間線（附錄 G 表）
`git show -s --format='%h|%aI|%s' <hash>`：2838e68/98937d9、a0619ab/a7e76aa、79b4dbc/3c774e7、e8d6059/ae62fc8、f49e6e8/3437930、071574f/a477e3b、4eacb94/3219473、02dc53f/7c215c2、5c53a88/f1c439d、38ce89e/74245e9、0e16639/040f391。間隔為兩個 commit 的 author 時間差；author 時間可自設，無第三方證明。「6 分鐘到約 2.5 小時」指輪換、貼尾流、滑翔各列（不含鳥類、LLM、L3）；此句已從主文刪除（P3），只留附錄表。表的 L3 列在文中改稱 neighbours-only learner。兩個學習者的間隔含 GPU 排隊與暫停（b74aa7e），不是訓練時間。

## G. 其他文字數字
| 數字 | 來源 |
|---|---|
| Maskew：領頭 1、跟機 0.2（約降 80%）；F/A-18 一個條件阻力降 >20% | notes/research/F1_drone_formation_energy.md B1、A1（Maskew 為數值；Ray 為實飛，Mach 0.56） |
| 圖 A1 各位置降幅 72/82/85/86 | horseshoe_formation.json `sweep.N9_V_optimal_lamb_oseen_rc0.075.per_position[].ratio_all`（降幅＝1−ratio） |
| 重疊 15%、校準目標 0.21（三翼 V） | notes/impl/horseshoe_formation.md「模型」；portfolio_numbers.md B 區「3」 |
| V 與梯隊總量差 0.000（設定造成） | portfolio_numbers.md A 區「3」 |
| 該 commit 訊息稱 L3 差距為規則假象，後更正 | git ae62fc8 訊息；私人開發紀錄（PROGRESS）「內部審查第一輪＋更正 L3」 |
| 滑翔 5.6–24 pp（舊網頁的綁定 JSON 外） | 已由 soaring_headroom2.json 反推（見 C 區），不再是 [TODO] |

## H. 內部審查第一輪後新增或改口徑的數字（notes/review/closed_door_paper_r1.md；已用 make_paper_figs.py 內 assert 核對）
| 論文中的數字 | key | 來源與欄位 |
|---|---|---|
| 768 格：看電量規則單獨 4 格 >2%、最大 2.33（摘要、4.1、Table 1） | L1_gate_gt2, L1_gate_max | drone_bucket_l1.json `rules.gate.gap`（×100）。注意：1/768（2.18）是五規則事後最佳 `best_gap`，不是看電量規則 |
| 個體差異 ±15%、±30% 共 384 格，0 格 >2%，最大 0.59 | L1_h15plus_cells, L1_h15plus_gt2, L1_h15plus_max | drone_bucket_l1.json `best_gap`，依 `abs(h)>=0.15`（分組最大：±0 2.18、±5% 1.45、±15% 0.59、±30% 0.23） |
| 預測區間 15–45%（圖 2） | pred_range_L1 | **不在 calib/out**：`git show 79b4dbc:.scratch/drone_bucket/PREDICTION_L1.md` 第 4 點。「約 230/768」＝0.30×768，是外推（預測針對 576 格），圖上標 extrapolated |
| 貼尾流：傳統法勝過同 GPS 參考 34/108 格（全是 kf）、其中 22 格 95% 區間上限 <0；最差 −0.24 | wake_fair_neg_cells, wake_fair_neg_ci_cells, wake_fair_neg_all_kf | wake_headroom.json `gap_gpsprivileged`、`gap_gpsprivileged_ci[1]`、`best` |
| 貼尾流：隨機漂移 84 格，完美位置參考最大 1.55（圖 4 第一列） | wake_random_drift_cells, wake_random_drift_perfect_max | wake_headroom.json `drift` 以 `amp` 開頭者 |
| 滑翔：最強法 − 簡單規則（L2−L1）11 格中 10 格 95% 區間下界 >2 | soar_l2l1_cilo_gt2 | soaring_headroom3.json `gaps.Emin["L2-L1"][1]` |
| 滑翔：4 個有效 5 機格 L2−L1 為 5.9／6.8／10.0／13.9，區間下界 4.0–11.3 | soar_l2l1_informative, soar_l2l1_informative_lo_min/max | 同上，非地板格（`floor_cell==False`） |
| 滑翔：最強法 − 簡單規則決策＋真值（L2−L1.5b，等於 L2−max(L1.5a,L1.5b)，11/11 格相同）在有效格 −0.8 到 +2.5 | soar_gap_vs_sameinfo_informative_min/max | `gaps.Emin["L2-max(L15a,L15b)"]`；「11/11 相同」是審查檔的比對，腳本只取後者 |
| 160 格中 16 格沒有逐步雜訊（σ=0；8 格無事件、8 格只有事件）；f_ind=0.4、c∈{1,10}（附錄 A） | L23_sigma0_cells | drone_bucket_l23.json `sigma`、`p_event`；`config.f_ind` |
| **成本**：L1＋L2/3 共 928 格，量測約 9.8 CPU 小時、每格約 38 CPU 秒（4 個程序的工作時間合計，不是牆鐘；牆鐘 7508 s＋3950 s） | cpu_hours_L1_L23, cpu_s_per_setting_L1_L23 | drone_bucket_l1.json `jobs[].time_sim_s+time_opt_s`（5.68 h，其中 92% 在求解器：L1_solver_share）＋drone_bucket_l23.json `jobs[].time_s`（4.14 h） |
| **成本**：訓練 5 種子每格約 200 GPU 秒（U=2000，vmap） | gpu_s_per_setting_train5 | **不在 calib/out**：notes/impl/l3_mappo_results.md「實測成本」（198–210 s）；訓練 log 不進 git。兩者裝置不同（CPU vs GPU），文中只寫各自數字，不寫倍數 |
| Table 1：整數規劃是流體模型上界、離散化誤差約 0.1% | — | notes/impl/drone_bucket.md 第 167 行 |
| Table 1：區塊 20 步的 hindsight 上界嚴格說不保證，區塊檢查差 ≤0.15% | — | notes/impl/drone_bucket_l23.md 第 2 節（N=5、6 實例、σ=30%） |
| 附錄 A：richer 層五條規則（fixed、rot、gate、sort、pair）；noisy 層（fixed、rot、gate、gate_est、local） | — | drone_bucket_l1.json `rules` 鍵、drone_bucket_l23.json `rules` 鍵；規則定義見 notes/impl/drone_bucket_l1.md、drone_bucket_l23.md |
| 附錄 C：第一輪滑翔起始高度與上限為第二輪的一半、指標 Amin（在空時間），第二輪改 Emin | — | notes/impl/soaring_headroom.md、soaring_headroom2.md 設定段 |
| 附錄 F：接線除錯 ≤20 次更新、一個格子 | — | notes/impl/l3_mappo_prereg.md 第 0 節 |
| 附錄 F：v1 預測 75% 未通過；v2 預測 35%，未通過屬預期內；v2 預測中位數 0.35、觀察 −0.00 | — | notes/impl/l3_mappo_results.md 對照預測節 |

審查建議但**沒採用**進主文的數字（因為會增加篇幅，且附錄／圖已有）：貼尾流相對參考節省 2.0–7.0 pp、差距最高達 39%（P6）；預測區間只放在圖 2，不放正文（P4）。若要用，來源是 wake_headroom.json `gap_gpsprivileged/priv_gps`。

## 圖與表的對應（paper/figures/，由 scripts/make_paper_figs.py 產生）
fig0_concept（概念，不含資料）；fig1_prediction（L1_best_gt2、pred_*）；fig2a_rotation（L0/L1 逐格）；fig2b_wake（wake_headroom.json，三列：隨機漂移 84／領機蛇行 12／領機階躍 12）；fig2c_soaring（4 個有效 5 機格，兩個差距）；fig3_local_gap；fig4_sanity_gate；figA1_formation。舊的 fig2_distance 三面板已拆成 fig2a/b/c 並刪除。

## 未找到／[TODO]
- Reddy 2016、2018 論文標題（notes/research 只有作者、期刊、頁碼）：refs.bib 標 [TODO verify title]。
- 預先登記方法論文獻、MAPPO／PPO／JAX 引用：main.tex 標 [TODO cite]。
- AI 代理的模型名稱與版本、投稿場地的揭露文字：[TODO]（貢獻聲明已寫明專案由作者與 AI 代理合作，細節待補）。
- 2026-11-19 更新（已解決）：作者署名已定（Hao-Ting Ho）；controller-ladder 的 sanity gate 出處已引用（refs.bib 的 ho2026ladder）。公開 repo 連結已定：https://github.com/Jadiouo/measure-before-you-train（公開時生效；`\repourl` 已是此值）。
- L0（192 格）的實際計算時間：找不到紀錄（drone_bucket.md 無）；成本數字不含 L0。
- 同 GPS 參考是在看到完美位置結果之前還是之後加的：紀錄找不到（預測檔沒有；結果 commit 只隔 6 分鐘）。文中只寫「不在預測內」。
- 預測檔是誰寫的：作者提出每份預測的方向，AI agent 擴寫成預測檔（作者 2026-11-19 確認）。
- Ray 2002（只讀摘要與關鍵字檢索）、Binetti 2003（只引書目）的閱讀程度：main.tex 只揭露 2018、2023 兩篇，是否一併寫待作者決定。
- 小型無人機誘導阻力占總功率比例：找不到實測，掃描 0.3–0.5。


## I. v4 新增與衍生的數字（2026-11-18）

逐列取自 docs/writing/revision-20261118_data/new_numbers.tsv。

| 數字 | 出現在 v4 的哪裡 | 出處 |
|---|---|---|
| 8 of 960 | Abstract ("all but 8 of 960 settings"), Section 4.1, Conclusion; 8 = 4 of 192 + 4 of 768 | v3 L31 (4 of 192, 4 of 768), L147-L148; docs/canonical_numbers.md A; 03 problem 6 |
| 27 (conditions) | Section 4.2: "4 of 27 distinct conditions"; Appendix C: 108 settings = 27 conditions x 4 sensing-noise combinations | R4; calib/out/wake_headroom.json (verified by the author: 27 groups of 4 cells) |
| 4 of 27 (conditions) | Section 4.2 and Appendix C: the 16 settings beyond 2 pp lie in 4 of the 27 conditions, each a complete group of four | R4; calib/out/wake_headroom.json |
| 15 (of 27 conditions) | Appendix C: the 52 settings beyond 1 pp lie in 15 of the 27 conditions | R4; calib/out/wake_headroom.json |
| 4 (sensing-noise combinations per condition) | Appendix C: each condition paired with 4 sensing-noise combinations | R4; v3 Appendix B states 12 settings per drift scenario = 3 GPS levels x 4 |
| 14 (of 16) | Appendix A (wake surfing): in 14 of the 16 settings the lower end of the 95% interval also exceeds 2 pp | F17; docs/canonical_numbers.md B (wake_perfect_cilo_gt2 = 14) |
| 5 seconds (solver time limit) | Appendix B: solver time limit of 5 seconds of wall-clock time per solve | R1; scripts/drone_bucket_l1.py line 229 and drone_bucket_l23.py line 210 (options time_limit = tl, default --tl 5.0); calib/out/drone_bucket_l1.json and drone_bucket_l23.json config.tl = 5.0 |
| 60 seconds (block-size check) | Appendix B: "(60 seconds in the block-size check)" | R1; scripts/drone_bucket_l23.py line 344 (tl=60 in block_check) |
| four worker processes | Appendix B: "four worker processes ran in parallel" | R1; scripts/drone_bucket_l1.py line 396 and drone_bucket_l23.py lines 373, 381 (mp Pool(4)) |
| 3 seeds (95% t interval) | Appendix H and Appendix K: the +- values are 95% t intervals over 3 seeds | F14; notes/impl/survival_v2_results.md (seed-level t interval, k=3) |
| single instance (deterministic settings) | Appendix B: richer-level h=0 settings and noisy-level settings with no noise, no event and h=0 use one instance | F15; calib/out/drone_bucket_l1.json (n_inst=1 for h=0), drone_bucket_l23.json (n_inst=1, n_lp=1 for noise none, sigma 0, p_event 0) |
| 1 of 768 in the abstract/finding sentence "richer" | Abstract, Section 4.1 | F18; v3 L30, L133 |
| no new number this round | Round 2 (abstract rewrite, five-figure body, Related Work citations, Conclusion sentence) | guard G2 PASS: every number in v4 occurs in v3 or in this file |
| 113 | Appendix J scorecard table (paper/appendix_predictions.tex, generated by the colleague, not by this revision); 45 of 113 is the Total row. The main text no longer quotes this count (the Section 3 sentence was removed on 2026-11-19) | paper/appendix_predictions.tex; its own sources in docs/predictions/scorecard_sources.md; docs/predictions/scorecard_sources.md section 2 |
| 20261118 | Appendix J: path docs/predictions/scorecard_sources.md | paper/appendix_predictions.tex |
| 3.5（`at most 3.5 pp`） | v4 第 4 節正文表 `tab:guess` 貼尾流列「學到什麼」格：大漂移時連兩個參考的絕對節省也不超過 3.5 pp（標為 untested 的解讀） | notes/impl/wake_headroom.md 第 33 行（與預測對照表的原因欄）；第 20–22 行（各情境節省表）；calib/out/wake_headroom.json（drift 以 amp0.2 開頭的 24 格：`priv` 最大 3.47、`priv_gps` 最大 3.41） |
| 其餘數字（30%、2%、1 of 768、4 of 192、132 of 192、1.30、1.55、75%、35%） | `tab:guess` 各格 | v3 正文與附錄皆已有；逐格出處見 docs/predictions/guess_table_sources.md |

附錄預測摘要表（paper/appendix_predictions.tex）的 Total 列：預測 113 條／held 45／held partly 27／failed 34／not assessed 7；其中「這次才判定」（Verdicts assigned for this table）34 條。逐條來源與判定理由見 docs/predictions/scorecard_sources.md。

2026-11-19 作者確認：預測檔由作者本人撰寫
