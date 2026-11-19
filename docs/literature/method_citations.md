# G：方法類引用核對與前例搜尋紀錄

- 查核日期：2026-11-18
- 每筆皆附本次開啟的來源網址（WebFetch 實際開啟；「只有搜尋結果」者會明說，且不放進 BibTeX）。
- 重要限制：WebFetch 回傳的是小模型對網頁的整理，不是逐字全文。所以下列「層級」最高只標到「摘要頁」，沒有任何一筆我敢標「全文」。要引用具體數字或原句前，請自己開 PDF 核對。
- 搜尋關鍵字只用文獻標題、作者、年份與一般主題詞，沒有送出專案名稱、使用者資訊或論文句子。
- 本檔只作為引用與搜尋紀錄，沒有修改專案其他檔案。

---

## A. 候選文獻逐筆核對

### A1. PPO（核對成功，與候選一致）
- 書目：John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347 [cs.LG], v1 2017-07-20，v2 2017-08-28。arXiv 頁面沒有列正式發表處。
- 來源：https://arxiv.org/abs/1707.06347
- 層級：摘要頁。
- 說了什麼：提出 PPO，交替採樣與用截斷式代理目標做多輪小批次更新，比 TRPO 簡單，在機器人運動與 Atari 上表現良好。
- 與論文的關係：是 MAPPO 的底層演算法，放在「學習方法用 MAPPO」處一併引用。

### A2. MAPPO（核對成功；有兩點要注意）
- 書目：Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu. *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*. Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track。arXiv:2103.01955（v1 2021-03-02，v4 2022-11-04）。
- 來源：https://arxiv.org/abs/2103.01955 ；https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html
- 層級：兩處都是摘要頁。
- 與候選的出入：(a) 候選寫「約 2022」，正確是 arXiv 2021 首版、NeurIPS 2022 正式發表，引用時建議用 2022 並附 arXiv 編號。(b) 我開的兩個頁面摘要都沒有出現「MAPPO」這個縮寫，是否在正文用這名稱我沒有核對到。如論文裡稱 MAPPO，請自行開全文確認該縮寫。頁數：NeurIPS 頁面未列，查不到。arXiv 頁面的作者名沒有中間名；Bayen 的 DBLP 連結寫 Alexandre M. Bayen，未在此使用。
- 說了什麼：在 particle-world、SMAC、Google Research Football、Hanabi 四個合作式多智能體測試床上，PPO 類方法只需少量調參就能匹敵或超過離策略方法；附消融與實作建議。
- 與論文的關係：直接支持「用 MAPPO 當學習方法」，也支持「以 PPO 為強基準」的選擇。

### A3. 預先登記：Nosek 等人（核對成功；候選的「PNAS 約 2018」正確）
- 書目：Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, David T. Mellor. *The preregistration revolution*. Proceedings of the National Academy of Sciences, 115(11), 2600–2606, 2018. DOI: 10.1073/pnas.1708274114。
- 來源：https://api.crossref.org/works/10.1073/pnas.1708274114 （書目、摘要）；https://pmc.ncbi.nlm.nih.gov/articles/PMC5856500/ （內容摘要）。PNAS 官網 403 打不開，PubMed 要 cookie 打不開。
- 層級：Crossref 書目加 PMC 頁面摘要。
- 說了什麼：區分「預測（prediction）」與「事後解釋（postdiction）」；把研究問題與分析計畫在看到結果前登記，才能讓確認性檢定的結論可信；摘要原句："An effective solution is to define the research questions and analysis plan before observing the research outcomes."
- 與論文的關係：「訓練前先寫下預測」的方法論依據。注意這是心理學／社會科學脈絡，不是 RL；引用時只能說「借用」這個做法。

### A4. Registered Reports：Chambers（核對成功，但層級低）
- 書目：Christopher D. Chambers. *Registered Reports: A new publishing initiative at Cortex*. Cortex, 49(3), 609–610, 2013. DOI: 10.1016/j.cortex.2012.12.016。
- 來源：https://api.crossref.org/works/10.1016/j.cortex.2012.12.016 （書目）。內容描述來自搜尋結果的二手整理；Cardiff 的 PDF（https://orca.cardiff.ac.uk/45177/1/Chambers_Cortex_2013b_GreenOA.pdf）我試開但回 403，所以沒讀到內文。FORRT 頁（https://forrt.org/curated_resources/registered-reports-a-new-publishing-init/）只有一行描述，且頁面日期標 2020，與 2013 無關，只是策展登錄日期。
- 層級：只有書目。
- 說了什麼（二手）：Cortex 開設新文章類型，方法與分析計畫在收集資料前先同儕審查，通過即「原則接受」（in-principle acceptance）。這一點我沒在原文核對。
- 與論文的關係：「預先登記」的期刊版制度前例；本論文若沒有走期刊 Stage 1 審查，只能說「借用其精神」。

### A5. Henderson 等人（核對成功）
- 書目：Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger. *Deep Reinforcement Learning That Matters*. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 2018（AAAI-18）。DOI: 10.1609/aaai.v32i1.11694。arXiv:1709.06560。頁碼：兩個來源都沒列，查不到。
- 來源：https://arxiv.org/abs/1709.06560 ；https://aaai.org/papers/11694-deep-reinforcement-learning-that-matters/ ；https://api.crossref.org/works/10.1609/aaai.v32i1.11694
- 層級：摘要頁。
- 說了什麼：深度 RL 的結果因環境非確定性、方法本身變異與報告不一致而難以重現，建議統一的實驗與報告做法。
- 與論文的關係：支持「RL 評估需要更嚴格流程」的動機，放在引言或方法動機處。

### A6. Agarwal 等人（核對成功；出現小差異）
- 書目：Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, Marc G. Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. Advances in Neural Information Processing Systems 34 (NeurIPS 2021)。arXiv:2108.13264。NeurIPS 頁面標示獲 outstanding paper award。頁碼查不到。
- 來源：https://arxiv.org/abs/2108.13264 ；https://papers.nips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
- 層級：摘要頁。
- 與候選的出入：候選只寫「約 2021」，確認為 NeurIPS 2021。arXiv 頁面本身沒寫發表處，發表處靠 NeurIPS 頁面確認。
- 說了什麼：只報點估計會誤導；建議區間估計、performance profile 與 interquartile mean；釋出 rliable。
- 與論文的關係：統計報告方式（種子數、區間、IQM）的依據。

### A7a. Mania、Guy、Recht（核對成功；標題有兩個版本）
- 書目（正式版）：Horia Mania, Aurelia Guy, Benjamin Recht. *Simple random search of static linear policies is competitive for reinforcement learning*. Advances in Neural Information Processing Systems 31 (NeurIPS 2018)。頁碼 1800–1809 只出現在搜尋結果的整理，我開的 NeurIPS 頁面沒列，所以標「待核對」。
- 預印本：同作者，*Simple random search provides a competitive approach to reinforcement learning*. arXiv:1803.07055, 2018-03-19（v1）。
- 來源：https://papers.nips.cc/paper/2018/hash/7634ea65a4e6d9041cfd3f7de18e334a-Abstract.html ；https://arxiv.org/abs/1803.07055
- 與候選的出入：候選的標題「Simple random search…」對應 arXiv 標題；正式發表的標題不同。引用請決定用哪個版本。
- 層級：摘要頁。
- 說了什麼：對線性策略做隨機搜尋，在 MuJoCo 運動基準上達到最先進的樣本效率，且比最快的對手快至少 15 倍；跨數百個種子顯示變異很大，主張需要新基準與評估方法。
- 與論文的關係：「簡單方法追上深度 RL」的前例，支持先量手寫簡單規則的合理性。

### A7b. Rajeswaran 等人（核對成功）
- 書目：Aravind Rajeswaran, Kendall Lowrey, Emanuel V. Todorov, Sham M. Kakade. *Towards Generalization and Simplicity in Continuous Control*. Advances in Neural Information Processing Systems 30 (NIPS 2017)。arXiv:1703.02660（v1 2017-03-08，v2 2018-03-20）。
- 來源：https://arxiv.org/abs/1703.02660 ；https://papers.nips.cc/paper_files/paper/2017/hash/9ddb9dd5d8aee9a76bf217a2a3c54833-Abstract.html
- 層級：摘要頁。
- 說了什麼：線性與 RBF 策略在多個連續控制任務上與神經網路策略相當；主張不該預設用多層網路，應與簡單替代方案並排比較。
- 與論文的關係：同 A7a。

### A8. Brown、Smith、Sun（核對成功）
- 書目：David B. Brown, James E. Smith, Peng Sun. *Information Relaxations and Duality in Stochastic Dynamic Programs*. Operations Research, 58(4, Part 1), 785–801, 2010. DOI: 10.1287/opre.1090.0796。三位作者皆在 Duke 大學 Fuqua 商學院。
- 來源：https://api.crossref.org/works/10.1287/opre.1090.0796 ；https://econpapers.repec.org/RePEc:inm:oropre:v:58:y:2010:i:4-part-1:p:785-801 。Dartmouth 的 PDF（https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxations_and_Duality_in_Stochastic_Dynamic_Programs.pdf）我開了，但內容是壓縮的二進位流，讀不出文字。
- 層級：摘要頁（EconPapers）。
- 說了什麼：放寬「決策只能用當下資訊」的限制並加上懲罰項，得到隨機動態規劃的上界（成本問題則為下界）；證明弱對偶、強對偶與互補鬆弛；這些對偶界可與「模擬啟發式策略得到的下界」互相搭配，用來衡量啟發式離最佳差多遠。
- 與論文的關係：「離線上界」的理論來源。重要：該文的上界帶懲罰項；無懲罰的完全資訊（hindsight）上界通常較鬆，這一點是搜尋結果的二手說法（Smith 的綜述），我沒在原文核對。若論文用的是無懲罰的事後最佳值，應寫成「一個有效但可能偏鬆的上界」，不要寫成緊的對偶界。

---

## B. 前例搜尋紀錄（查詢日期皆為 2026-11-18；工具：WebSearch 通用網頁搜尋，再用 WebFetch 開頁核對）

範圍聲明：這一輪只用通用網頁搜尋，沒有系統性檢索 Google Scholar、Semantic Scholar、ACM、IEEE、dblp 的全文或引用網路，也沒有逐篇讀全文。結論只代表「這一輪沒找到」，不代表不存在。

| # | 查詢字串 | 看了哪幾筆 | 結論 |
|---|---|---|---|
| 1 | preregistration of reinforcement learning experiments hypotheses registered before running | 搜尋結果十筆（心理學預先登記、一篇 2026 年 AI agent 預先登記、一個 ML 預測研究模板）；開了 Vaccaro 那篇 | 找到 AI agent 實驗的預先登記立場文（見 B-1）；沒找到 RL 專屬做法 |
| 2 | gap between heuristic and optimal upper bound before training reinforcement learning scheduling headroom | 十筆；開了 arXiv:2609.37125 | 找到一個「訓練前量 headroom」的近似做法（見 B-2），但領域是 LLM agent 的觀測預算，不是排程 RL，且該術語在我看到的頁面沒出現 |
| 3 | information relaxation dual bound to evaluate reinforcement learning policy performance gap | 十筆；開了 arXiv:2506.00801 | 找到用資訊鬆弛對偶做 RL 的論文（見 B-3），但目的是「訓練時用對偶當目標」，不是「訓練前決定值不值得」 |
| 4 | perfect-information hindsight optimal benchmark online scheduling policy evaluation learned vs heuristic | 十筆；開了 arXiv:2207.06272 | hindsight 當基準在線上排程文獻很常見；Hindsight Learning 是用事後資訊「訓練」RL（見 B-4）；其餘只看到搜尋整理，沒開 |
| 5 | Pre-registration in machine learning NeurIPS 2020 workshop Bertinetto | 十筆；開了 PMLR v148 | NeurIPS 2020 有 ML 預先登記工作坊（見 B-5），但是機器學習整體，不是 RL 專屬 |
| 6 | positive control sanity check learning algorithm must solve known solvable task before interpreting results reinforcement learning | 十筆（DeepMind Control Suite、AgileRL debugging 文件、部落格、Reddit） | 找到零散的「先在已知可解任務上驗證」實務說法；沒找到把它當作「必須通過才解讀主結果」的資格考規範。這些來源我沒有逐一開啟，只有搜尋結果整理，不列為已核對文獻 |
| 7 | simple heuristic baseline outperforms deep reinforcement learning scheduling combinatorial optimization strong baselines | 十筆；開了 arXiv:2207.14595 | 排程領域兩邊證據都有；SoC 那篇是在比較神經與啟發式排程，結論偏向「RL 在 SoC 上先天不利」（見 B-6），不是先量差距的流程 |
| 8 | registered report machine learning reinforcement learning study plan accepted in principle before results | 十筆（軟體工程 registered reports 等） | 沒找到 ML／RL 的 registered report 實例 |
| 9 | Chambers 論文標題核對（Registered Reports Cortex） | 十筆 | 僅用於書目核對，不屬前例搜尋 |
| 10 | should we use reinforcement learning? estimate optimality gap of heuristic against oracle bound to decide whether learning is worthwhile | 十筆；開了 arXiv:2508.01329 | 找到 Berseth（見 B-7）：量「學到的策略離自己產生過的最佳經驗差多少」，不是量簡單規則離上界 |
| 11 | multi-agent reinforcement learning benchmark oracle upper bound heuristic baselines gap UAV task allocation MAPPO negative result | 十筆；開了 arXiv:2503.07662 | 找到 HIPPO-MAT（見 B-8）：IPPO 對集中式匈牙利法的差距；沒找到 MAPPO 加上界加簡單規則的完整流程 |

（另有 Mania NeurIPS 標題核對、Brown 書目核對各一次查詢，不屬前例搜尋，未列入上表編號；總計前例搜尋 9 次，書目核對 3 次。）

### 前例搜尋的結論，依目標 (i)–(iv)
- (i) 訓練前先量簡單規則到上界的差距，以決定值不值得訓練：沒有找到做同樣事情的研究。最接近的是 B-2（headroom 診斷，但領域不同）。
- (ii) 對 RL 實驗預先登記預測：沒找到 RL 專屬的。找到 ML 層級（B-5）與 AI agent 層級（B-1）的倡議。
- (iii) 學習方法先在已知可學的對照問題通過檢查才解讀主結果：沒找到當成規範的前例；只有零散的實務說法（搜尋 6，未開啟來源）。
- (iv) 用 perfect-information／hindsight／information-relaxation 上界評估 RL 或啟發式的剩餘空間：理論有成熟前例（A8），線上排程也常用 hindsight 基準（搜尋 4，多數只有搜尋整理）；把它用在「評估 RL 前的剩餘空間」有 B-2、B-7、B-8 這幾個近似做法。

### 每筆找到的相關文獻

**B-1. Vaccaro（AI agent 實驗的預先登記）**
- 書目：Michelle Vaccaro. *Preregistration for Experiments with AI Agents*. arXiv:2606.11217，v1 提交日 2026-05-03（頁面標分類 cs.CY）。備註標示 ICML 2026 Spotlight 立場論文。
- 差異註記：arXiv 編號前綴 2606 對應 2026 年 6 月，但頁面提交日是 5 月 3 日，兩者不一致，我無法解釋；引用前請自行重開確認。
- 來源：https://arxiv.org/abs/2606.11217 ；https://arxiv.org/abs/2606.11217v1
- 層級：摘要頁。
- 說了什麼：主張把預先登記延伸到對 LLM 與 AI agent 的實驗，列出模型選擇、提示詞等研究者自由度，提出登記範本。頁面沒說是否涵蓋 RL。
- 關係：同一方向的相近倡議，但對象是 LLM agent，不是 RL；不構成「做過同樣流程」。

**B-2. Di 等人（structural headroom 的近似做法）**
- 書目：Zhengkun Di, Bin Shi, Kai Sun, Yiming Xu, Bo Dong. *When Should Agents Check External State? Budgeting Observations for Stored Intentions*. arXiv:2609.37125，v1 2026-09-29（v3 2026-10-16），cs.AI，預印本。
- 來源：https://arxiv.org/abs/2609.37125
- 層級：摘要頁。我打開的頁面沒有出現 "structural headroom" 這個詞（那個詞來自搜尋結果的整理，我沒有在頁面上核對到），所以不要引用這個術語。
- 說了什麼：把外部查核的預算分配當資源分配問題，提出 BudgetPM；用事後（hindsight）排程做監督；設計準則是「容量足夠時局部門檻即可，資源互相競爭時未來感知的監督才有幫助」。
- 關係：類似「先看需不需要更複雜方法」的想法，但沒有確認它是否在訓練前量差距。領域不同。

**B-3. Chen 等人（資訊鬆弛加 RL）**
- 書目：Nan Chen, Mengzhou Liu, Xiaoyan Wang, Nanyi Zhang. *Adversarial Reinforcement Learning: A Duality-Based Approach To Solving Optimal Control Problems*. arXiv:2506.00801，v1 2025-06-01（v2 2025-07-02）；頁面標示 2025 Winter Simulation Conference 錄取。
- 來源：https://arxiv.org/abs/2506.00801
- 層級：摘要頁。
- 說了什麼：用資訊鬆弛對偶把控制問題寫成策略與對抗懲罰的 min-max，取得緊的對偶間隙。
- 關係：是把資訊鬆弛和 RL 連起來的前例；但用途是產生更好的策略與界，不是決定該不該訓練。

**B-4. Sinclair 等人（Hindsight Learning）**
- 書目：Sean R. Sinclair, Felipe Frujeri, Ching-An Cheng, Luke Marshall, Hugo Barbalho, Jingling Li, Jennifer Neville, Ishai Menache, Adith Swaminathan. *Hindsight Learning for MDPs with Exogenous Inputs*. arXiv:2207.06272，v1 2022-07-13（v3 2023-10-23），預印本，頁面沒列發表處。
- 來源：https://arxiv.org/abs/2207.06272
- 層級：摘要頁。
- 說了什麼：用外生變數的事後樣本做反事實評估來加速策略改進，在雲端 VM 配置等問題上報告勝過啟發式與 RL 方法。
- 關係：把 hindsight 資訊用在訓練；本論文是把它用在評估剩餘空間，兩者用途不同。

**B-5. Bertinetto 等人（ML 預先登記工作坊）**
- 書目：Luca Bertinetto, João F. Henriques, Samuel Albanie, Michela Paganini, Gül Varol. *Preface*. Proceedings of Machine Learning Research, vol. 148（NeurIPS 2020 Workshop on Pre-registration in Machine Learning），2021，p. i。
- 來源：https://proceedings.mlr.press/v148/bertinetto21a.html （只有書目頁，沒有摘要）。工作坊描述來自搜尋結果的整理（NeurIPS 的工作坊網頁沒有逐一開啟）。
- 層級：只有書目。
- 說了什麼（二手）：2020 年 12 月 11 日舉辦的虛擬工作坊，試行「先投方法與實驗協定、錄取後才跑結果」的出版模式。
- 關係：ML 領域已有預先登記的制度化嘗試，可當作 A3 的 ML 側佐證。引用時建議引這個前言作為工作坊出處，別當成方法論文。

**B-6. Sung 與 Ryu（SoC 排程：神經對啟發式）**
- 書目：Tegg Taekyong Sung, Bo Ryu. *Deep Reinforcement Learning for System-on-Chip: Myths and Realities*. arXiv:2207.14595，2022-07-29，頁面註明已投稿 IEEE。
- 來源：https://arxiv.org/abs/2207.14595
- 層級：摘要頁。
- 說了什麼：叢集排程的神經排程器搬到 SoC 上表現差；提出 EIM 方法改善。
- 關係：弱相關，只說明「RL 排程器不一定勝過啟發式」；不是先量差距的流程。是否與搜尋整理所說的「簡單基準可能勝出」一致，我沒有在摘要上核對到，不要用它支持那個說法。

**B-7. Berseth（最佳經驗對學習策略的差距）**
- 書目：Glen Berseth. *Is Exploration or Optimization the Problem for Deep Reinforcement Learning?* arXiv:2508.01329，v1 2025-08-02，cs.LG。
- 來源：https://arxiv.org/abs/2508.01329
- 層級：摘要頁。
- 說了什麼：提出次最佳性估計器；報告最佳經驗比學到的策略好 2–3 倍，結論是深度 RL「只利用了一半它產生的好經驗」。
- 關係：同為「用一個參考值診斷 RL 剩餘空間」，但參考值是自己的最佳經驗，不是離線上界；不是訓練前的決策。

**B-8. Ratnabala 等人（HIPPO-MAT）**
- 書目：Lavanya Ratnabala, Robinroy Peter, Aleksey Fedoseev, Dzmitry Tsetserukou. *HIPPO-MAT: Decentralized Task Allocation Using GraphSAGE and Multi-Agent Deep Reinforcement Learning*. arXiv:2503.07662，2025-03-08（v1），預印本。
- 來源：https://arxiv.org/abs/2503.07662
- 層級：摘要頁。
- 說了什麼：UAV／UGV 去中心化任務分配，使用 IPPO；報告相對集中式匈牙利法有 16.49% 差距、勝過貪婪式去中心化基準。
- 關係：多機任務分配的 MARL 文獻裡，相對集中式參考解報告差距的例子；但只是事後比較，不是預先登記，也沒有資格考。

### 未開啟、僅在搜尋結果出現的線索（不放進 BibTeX，需要時再核對）
- DeepMind Control Suite（arXiv:1801.00690）：搜尋整理說它的任務要至少被一個學習 agent 解出才收入基準，可能可作為「資格考」的弱前例。我沒開頁面。
- Balseiro、Brown、Chen（2017）關於排程的 information relaxation 下界：搜尋整理提到，我沒開頁面。
- Balseiro 與 Brown 的 Operations Research 論文（啟發式的資訊鬆弛界）：同上。
- 搜尋結果裡的 Sinclair 之外幾篇線上排程 hindsight 基準論文（EV 充電、LLM 推論排程等）：同上。

---

## C. 建議（以下是建議，不是事實）

### C1. 「As far as we know, no earlier study…」該怎麼寫
依這一輪有限的搜尋，我建議把句子縮成「組合」而不是「首創」，並寫明範圍。例如：

> To our knowledge, and within a limited literature search (general web search, November 2026; no systematic database review), we did not find an earlier study that combines all three of: (i) registering predictions before training, (ii) measuring the gap between a hand-written heuristic and an offline (hindsight) upper bound to decide whether learning is worthwhile, and (iii) requiring the learner to pass a positive-control check on a problem known to be improvable before its main results are interpreted. Each ingredient has precedent separately: preregistration [nosek2018prereg; chambers2013registered; bertinetto2021prereg], information-relaxation and hindsight bounds [brown2010information], and simple baselines matching deep RL [mania2018random; rajeswaran2017simplicity].

重點：
1. 不要寫「no earlier study has done any of these」，因為 B-1、B-5 已有預先登記的倡議，B-2、B-7 有近似的差距診斷。
2. 不要寫成「first」。
3. 句子裡要有搜尋範圍與日期，搜尋紀錄放附錄或補充材料。
4. 若論文的「資格考」和我找到的前例（B 搜尋 6）不是同一件事，要自己說明差別；那一項我只有零散線索，沒有可引用的論文。

### C2. 最該引用的文獻與位置
| 論點 | 引用鍵 |
|---|---|
| 學習方法使用 MAPPO | schulman2017ppo、yu2022mappo |
| 預先登記：先寫預測再看結果（借用其做法） | nosek2018prereg；可加 chambers2013registered（制度版；我只核對到書目） |
| ML 領域已有預先登記嘗試 | bertinetto2021prereg（只是前言，描述工作坊；請在內文標明是 workshop preface） |
| RL 評估重現性與統計報告 | henderson2018matters、agarwal2021precipice |
| 簡單方法追上深度 RL，所以要先量簡單規則 | mania2018random、rajeswaran2017simplicity |
| 事後資訊上界的理論 | brown2010information（並註明：論文用的若是無懲罰的完全資訊上界，需寫成「可能偏鬆」） |
| 近似前例（相關工作中提及即可） | vaccaro2026prereg（先確認 arXiv 編號與日期）、berseth2025exploration |

### C3. BibTeX（只填我核對過的欄位；頁碼查不到的就沒填）

```bibtex
@misc{schulman2017ppo,
  author        = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  title         = {Proximal Policy Optimization Algorithms},
  year          = {2017},
  eprint        = {1707.06347},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@inproceedings{yu2022mappo,
  author    = {Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  title     = {The Surprising Effectiveness of {PPO} in Cooperative Multi-Agent Games},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  year      = {2022},
  note      = {Datasets and Benchmarks Track. arXiv:2103.01955}
}

@article{nosek2018prereg,
  author  = {Nosek, Brian A. and Ebersole, Charles R. and DeHaven, Alexander C. and Mellor, David T.},
  title   = {The preregistration revolution},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {115},
  number  = {11},
  pages   = {2600--2606},
  year    = {2018},
  doi     = {10.1073/pnas.1708274114}
}

@article{chambers2013registered,
  author  = {Chambers, Christopher D.},
  title   = {Registered Reports: A new publishing initiative at {Cortex}},
  journal = {Cortex},
  volume  = {49},
  number  = {3},
  pages   = {609--610},
  year    = {2013},
  doi     = {10.1016/j.cortex.2012.12.016}
}

@inproceedings{henderson2018matters,
  author    = {Henderson, Peter and Islam, Riashat and Bachman, Philip and Pineau, Joelle and Precup, Doina and Meger, David},
  title     = {Deep Reinforcement Learning That Matters},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {32},
  number    = {1},
  year      = {2018},
  doi       = {10.1609/aaai.v32i1.11694}
}

@inproceedings{agarwal2021precipice,
  author    = {Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and Courville, Aaron C. and Bellemare, Marc G.},
  title     = {Deep Reinforcement Learning at the Edge of the Statistical Precipice},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  year      = {2021},
  note      = {arXiv:2108.13264}
}

@inproceedings{mania2018random,
  author    = {Mania, Horia and Guy, Aurelia and Recht, Benjamin},
  title     = {Simple random search of static linear policies is competitive for reinforcement learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018},
  note      = {arXiv:1803.07055 (earlier title: Simple random search provides a competitive approach to reinforcement learning)}
}

@inproceedings{rajeswaran2017simplicity,
  author    = {Rajeswaran, Aravind and Lowrey, Kendall and Todorov, Emanuel V. and Kakade, Sham M.},
  title     = {Towards Generalization and Simplicity in Continuous Control},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017},
  note      = {arXiv:1703.02660}
}

@article{brown2010information,
  author  = {Brown, David B. and Smith, James E. and Sun, Peng},
  title   = {Information Relaxations and Duality in Stochastic Dynamic Programs},
  journal = {Operations Research},
  volume  = {58},
  number  = {4, part 1},
  pages   = {785--801},
  year    = {2010},
  doi     = {10.1287/opre.1090.0796}
}

@proceedings{bertinetto2021prereg,
  editor    = {Bertinetto, Luca and Henriques, Jo{\~a}o F. and Albanie, Samuel and Paganini, Michela and Varol, G{\"u}l},
  title     = {Proceedings of the NeurIPS 2020 Workshop on Pre-registration in Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {148},
  year      = {2021}
}

@misc{vaccaro2026prereg,
  author        = {Vaccaro, Michelle},
  title         = {Preregistration for Experiments with {AI} Agents},
  year          = {2026},
  eprint        = {2606.11217},
  archivePrefix = {arXiv},
  note          = {Check arXiv id and submission date before use: id prefix and the listed 2026-05-03 date disagree. ICML 2026 position paper per arXiv comments.}
}

@misc{berseth2025exploration,
  author        = {Berseth, Glen},
  title         = {Is Exploration or Optimization the Problem for Deep Reinforcement Learning?},
  year          = {2025},
  eprint        = {2508.01329},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

BibTeX 備註：
- `bertinetto2021prereg` 我只核對到 PMLR 前言頁的書目（頁面標題為 *Preface*，p. i）；上面寫成 proceedings 形式，是否符合你的引用格式請自行決定。
- `yu2022mappo` 與 `agarwal2021precipice` 頁碼查不到，所以沒填；`henderson2018matters` 同。
- 預印本版年份：`vaccaro2026prereg` 的年份採 2026（提交日期 2026-05-03），編號疑點見上。
