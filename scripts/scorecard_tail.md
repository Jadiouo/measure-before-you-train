
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

1. 這張表的 113 條裡，專案紀錄原本就有判定（或明寫沒量、無結論）的是 %PCOUNT% 條；其餘 %TCOUNT% 條由本次處理，其中 %DAG% 條給了 held／failed／partly（tex 內有 †，摘要表「Verdicts assigned for this table」欄即此數），1 條（SA17）判 not assessed。如果你想讓表更保守，把本次判定的 partly／failed 改成 not assessed 只需動 `scripts/scorecard_rows.py`。
2. 事實查核提到的「只報最戲劇的兩個預測」：本表揭露後，held 條數（%HELD%）其實不少，但 failed 與 partly 加起來 %FP% 條，高於 held；論文主文目前只講 wake surfing 與 L3 第二份兩件事。
3. 輪流 richer 家族的預測針對 576 格、量測 768 格；摘要附錄只用一句話帶過，論文第 3 節（`main.tex` 第 88–92 行）與圖 2 的 caption 仍建議明寫。
4. 「預測由誰撰寫」：作者已於 2026-11-19 確認（`docs/canonical_numbers.md` 第 114 行）。tex 用被動語態，檔頭註解記下了這項確認。
5. 專案紀錄的「與預測對照」只涵蓋 L1、L2、L3、貼尾流、滑翔第 1、2 輪、LLM（在私人開發紀錄 PROGRESS）、學習者；滑翔第 3 輪只有「和預測不同」三條，鳥 v2 只有條件層級表，沒有逐條對照（第 3 節各列的「本次判定」多半落在這幾個家族）。
