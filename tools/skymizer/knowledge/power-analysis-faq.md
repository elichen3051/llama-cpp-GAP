# Power Analysis FAQ — 事前 vs 事後、iid 假設、80% vs 100%

> [!WARNING]
> 本文保留作歷史背景；其中把 `--num-eval-tokens` 的收益寫成
> σ_b² + σ_w²/T、以及以 observed pilot effect 規劃 N 的段落，不適用於
> ordered、dependent、position-nonstationary token prefixes。現行方法與 CLI
> 見 [seq-power-analysis.md](seq-power-analysis.md)。

版本：2026-07-03
定位：回答四個關於 power analysis 的實務問題，脈絡是 `tools/skymizer` 的
paired quantization fidelity pipeline（`random_subsample_power.py`、
`variance_decomposition.py`、`saved_metrics_paired_compare.py`）。
數學背景見 [notes-power-variance-decomposition.md](notes-power-variance-decomposition.md)（變異數分解與 $N_{80}(T)$ 推導）、
配對推斷框架見 [llm_quantization_paired_eval_teaching.md](llm_quantization_paired_eval_teaching.md)。
標準作業流程見 [quantization-eval-sop.md](quantization-eval-sop.md)。

---

## 0. 核心結論

1. **是的，power analysis 的正統位置在事前**：它是「設計」工具，決定要收多少
   item（$N$）、每題評多少 token（$T$）才值得開跑。事後才算 power 的多數用法
   是統計謬誤——唯一的例外見第 2 點。
2. **我們的「事後」power analysis 其實是合法的**，因為它回答的不是「這次實驗
   的 power 是多少」，而是「**下一次**（下一個 candidate、例行回歸測試）該收多少」
   ——即把既有 collection 當 pilot data 做 **design analysis**。要避免的是
   「observed power」：用觀察到的效應量回頭算本次實驗的 power，那是 p 值的
   單調函數，零資訊且循環論證。
3. **iid 假設要放對層級**：我們的推斷單位是 item（不是 token）。假設是
   「item 是母體的可交換抽樣」+「token 在 item 內允許相關」（兩層模型），
   外推時再加上「效應 $\mu$ 與變異結構 $(\sigma_b, \sigma_w)$ 在下次實驗仍然成立」
   （平穩性／可傳輸性）。teacher-forced 量測本身是確定性的——所有不確定性
   都來自「題目是抽樣的」，不是量測噪音。
4. **100% power 不是目標而是警訊**：設計上 power → 100% 需要 $N \to \infty$；
   量到 100% 通常代表 (a) 實驗過度配置（顯著性失去過濾意義，瑣碎偏差也顯著）、
   (b) 經驗估計撞到天花板（reps 有限，100% 只是解析度不足）、或 (c) 你在用
   觀察到的大效應回推，帶著 winner's curse。80% 是成本—錯誤率的折衷慣例。

---

## 1. 正確的（事前）power analysis 流程

七個步驟，右欄是本 repo 的對應工具：

| 步驟 | 內容 | 工具對應 |
|---|---|---|
| 1. 定義決策與 estimand | 配對平均差 $\Delta = \mathbb{E}[d_i]$、單/雙尾、$\alpha$（預設雙尾 0.05） | report 的 verdict 定義 |
| 2. 定義 **SESOI**（smallest effect size of interest，或稱 MDE 目標） | 「多大的 ΔKLD / ΔNLL 才會改變部署決策」——領域判斷，**不是**觀察值 | 人工決定，寫進 pre-registration |
| 3. 估 nuisance 參數 | $\hat\sigma_b^2$、$\hat\sigma_w^2$、$\bar T$、$\rho_1$ 診斷 | `variance_decomposition.py`（pilot dirs，CPU 秒級） |
| 4. 解出設計 | $N_{80}(T)$ 公式（下）＋ token-vs-item 決策規則 | 同上；經驗曲線用 `random_subsample_power.py` |
| 5. 加安全邊際 | failed rows、重尾、參數估計不確定性 → 對 $N$ 加 10–25% | 人工 |
| 6. Pre-register | 固定 primary metric、weighting、$\alpha$、seed、資料集與 knobs（tf_chunk 等） | `collect_meta.json` 記錄大半，其餘寫 config 註記 |
| 7. 收集並跑**一次**預定的檢定 | 報告點估計＋CI＋效應量，不只 verdict | `saved_metrics_paired_compare.py` |

核心公式（推導見 notes-power 筆記 §2–3）：

$$
N_{1-\beta}(T) \;\approx\; \Big(z_{1-\alpha/2} + z_{1-\beta}\Big)^2 \,
\frac{\sigma_b^2 + \sigma_w^2 / T}{\mu_*^2}
$$

其中 $\mu_*$ 是 **SESOI**（不是 pilot 的 $\hat\mu$）。$T$ 的選擇由
token 噪音占比決定：$\sigma_w^2/T \gg \sigma_b^2$ 時拉長 `--num-eval-tokens`
有效；反之只有加題數有用（notes-power §7 的決策規則）。

最常被跳過、也最重要的是**步驟 2**。沒有 SESOI 的 power analysis 沒有定義：
power 是 $P(\text{顯著} \mid \mu = \mu_*, \sigma, N)$，$\mu_*$ 空著就無從計算。
拿 pilot 觀察值填 $\mu_*$ 是常見替代，但要知道它把問題從「偵測我在乎的效應」
偷換成「複製我上次看到的效應」，且帶 winner's curse（見 §4）。

---

## 2. 事後 power analysis 還有用嗎？怎麼解釋？

把「事後」拆成三種完全不同的東西：

### 2.1 Observed（retrospective）power——謬誤，不要做

定義：用**同一批資料**的 $\hat\mu$、$\hat\sigma$ 回頭算「本次實驗的 power」。

為什麼無效：在常態近似下 observed power 是 p 值的**單調函數**
（Hoenig & Heisey 2001, *The Abuse of Power*）——p 剛好 0.05 時 observed power
恆約 50%，p 越小它越高。它不含 p 值以外的任何新資訊，卻常被誤用成兩種循環論證：

- 「不顯著但 observed power 低 → 效應其實存在」（低 power 正是不顯著的重述）；
- 「不顯著且 observed power 高 → 證明了虛無假設」（observed power 高只因 $\hat\mu$ 恰好大，與 $H_0$ 真假無關）。

**正確替代品是 CI**：對本次結果，CI 的位置與寬度完整表達了「效應多大、
估得多準」，observed power 想說的一切它都說得更好。搭配 SESOI 還能直接做
non-inferiority／equivalence 判讀（teaching 文件 §10）。

### 2.2 Sensitivity analysis（固定 $N$ 反解 MDE）——合法

不代入 $\hat\mu$，只用變異數估計反問：「$N=199$、$T=1024$ 下，我們有 80% power
偵測的最小效應 $\mu_{80}$ 是多少？」

$$
\mu_{80} \;=\; \big(z_{1-\alpha/2}+z_{0.8}\big)\sqrt{\tfrac{\sigma_b^2+\sigma_w^2/T}{N}}
$$

解讀不顯著結果時這很有用：若 $\mu_{80}$ 遠小於任何實務在乎的量，
「不顯著」是有資訊的（效應若存在也小到不重要——正式版走 equivalence test）；
若 $\mu_{80}$ 很大，這次實驗本來就無力回答問題，不顯著什麼都不代表。

### 2.3 Design analysis for the next run——我們在做的，合法

`random_subsample_power.py` 與 `variance_decomposition.py` 把**已收好的母體**
當 pilot，回答「下一次收 $N$ 題會怎樣」。這在邏輯上就是**事前** power analysis
——只是服務對象是下一次收集（新 candidate、例行回歸、擴充資料集），
不是回頭替本次結果背書。解釋時注意三件事：

1. **條件性**：曲線條件在「這個 (dataset, model pair) 的效應／噪音結構可代表
   下一次」。換一對量化（$\mu$ 不同）曲線不可直接搬用；不變的部分通常是
   $\sigma_b, \sigma_w$（同資料集、同 reference 時），可用 §2.2 的 MDE 形式轉述。
2. **有限母體退化**：subsample 是從 199/348 題**不放回**抽樣。$N$ 接近母體時
   子集彼此高度重疊、且與定義「真 verdict」的全母體共用資料——$N=$ 母體時只剩
   一個子集，power 退化為 0%/100%。**曲線上端不是「fresh sample 的 power」**；
   實務上只解讀 $N \lesssim$ 母體一半的區段，上端當診斷看。
3. **它量的是隨機抽樣**：若實際收集用的是排序前綴（VLM 的
   `--sort-by num_images` + `DATASET_LIMIT`），那是**非隨機樣本**，
   與曲線假設不符——這正是這支工具當初存在的理由（量化「若隨機抽 $N$ 題」
   與「前綴 $N$ 題」的差異）。見 SOP §6 的抽樣守則。

---

## 3. iid 假設放在哪一層？

你的理解方向對，但要精確化成三層：

### 3.1 量測層：確定性，沒有 iid 問題

teacher-forced scoring 沒有 sampling：給定 tokens、模型、knobs，logits 與
metrics 是確定的（bit 級可重現）。所以**不存在「重跑幾次取平均」的噪音**，
repeats $r$ 無意義——這與 task-level stochastic decoding 的世界（teaching 文件
§7）根本不同。所有統計不確定性來自下面兩層。

### 3.2 Item 層：這裡才是 iid（可交換）假設

推斷單位是 item。假設 $d_1,\dots,d_N$ 是 item 母體的獨立可交換抽樣
——paired bootstrap 重抽的是 item，$N_{80}$ 公式裡的 $N$ 也是 item 數。
這個假設會被破壞的方式：同源題目群（同一張圖多題）、資料集排序後取前綴
（selection bias）、failed rows 與難度相關（missing not at random）。

### 3.3 Token 層：**明確不假設 iid**

同一題內相鄰 token 的 $\delta_{i,t}$ 正自相關（$\rho_1$ 診斷），且共享 item
效應 $b_i$。兩層模型 $\delta_{i,t} = \mu + b_i + \varepsilon_{i,t}$ 就是為了
不把 token 當獨立樣本。若誤把 token 當 iid 單位，等效樣本數會被高估數十倍
（$N \cdot \bar T$ 對 $N$），power 與 CI 全面過度樂觀。自相關的一階修正是
等效 token 數 $T_{\mathrm{eff}} = T / (1 + 2\sum_k \rho_k) < T$；
我們的動差估計不修 $\rho$，代價是 $\hat\sigma_b^2$ **偏高（保守）**
——方向見 notes-power §6.4，那段是對的。

### 3.4 外推時追加的假設——「$\mu$、$\sigma$ 相等」

你說的「假設 $\mu$、$\delta$ 相等然後外推」對應的正式說法是**可傳輸性**：
把 pilot 估出的 $(\mu, \sigma_b, \sigma_w)$ 帶到下一次實驗，隱含

- 下次抽的 item 來自同一母體（同 dataset/subset、同抽樣方式）；
- 下次的 candidate pair 效應結構類似（換 pair 時 $\mu$ 幾乎必變，
  $\sigma$ 常近似可沿用）；
- $T$ 外推時前後段訊號一致（notes-power §6.3：per-position $\mu$ 前段集中時
  拉長 $T$ 會稀釋效應，$\mathrm{SNR}(\infty)$ 是不可達上界）。

另注意分布形狀：KLD 差的分布**重尾**（少數災難題主導），常態近似的 $N_{80}$
在小 $N$ 偏樂觀，這是我們同時保留經驗式 `random_subsample_power.py`
（不吃常態假設，直接重放實際 bootstrap verdict）的原因。兩者互為 sanity check。

---

## 4. 為什麼 100% power 未必是好事？理論 vs 實務 power 的落差

### 4.1 80% 是成本—錯誤率的折衷，不是懶惰

$N \propto (z_{1-\alpha/2} + z_{1-\beta})^2$。以雙尾 $\alpha=0.05$ 為基準
（80% 時該係數 $= 2.80^2 \approx 7.85$）：

| 目標 power | 係數 | 相對 80% 的樣本成本 |
|---|---|---|
| 80% | 7.85 | 1.00× |
| 90% | 10.51 | 1.34× |
| 95% | 13.00 | 1.66× |
| 99% | 18.37 | 2.34× |
| 100% | $\infty$ | $\infty$ |

power 對 $N$ 邊際報酬遞減，而 $\beta$（漏報）的成本通常低於把 GPU 預算翻倍
——漏掉的效應下次收集還會再來，預算花掉就沒了。80/20 就是這個折衷的慣例值；
若漏報很貴（例如出貨前最後驗收），提到 90% 是合理的，但要自覺是在買保險。

### 4.2 設計出 ≈100% power ⇒ 實驗過度配置（overpowered）

在 SESOI 處 power ≈ 100% 意味著 $N$ 遠超需求，後果不只是浪費：

- **顯著性失去過濾功能**：遠小於 SESOI 的差異——包括管線瑕疵造成的系統性
  小偏差（tf-chunk 不一致、reference 漂移、資料前處理差異）——也會顯著。
  此時決策必須改看「效應量 vs SESOI」與 equivalence 判準（teaching §10），
  verdict 欄位只剩參考價值。
- **正確的反應是縮規模或改問題**：同樣預算改測更多 candidate、更多資料集，
  或把問題升級為精確估計（把 CI 寬度壓到 SESOI 的幾分之一）。

### 4.3 經驗式 power 顯示 100%：多半是解析度天花板

`random_subsample_power.py` 用 $R$ 次重抽估 power，解析度只有 $1/R$：

- $R=100$ 全顯著：Clopper–Pearson 單尾 95% 下界 $0.05^{1/100} \approx 0.970$
  ——只能宣稱 power ≳ 97%（rule of three 近似 $1 - 3/R$）。
- $R=5$ 全顯著（我們冒煙測試的規模）：下界 $0.05^{1/5} \approx 0.55$
  ——「100%」幾乎沒有資訊量。

所以報 power 曲線時應附二項式（Wilson/CP）區間，且不要用 100% 的格子
去比較設計優劣——它們只是「都超出解析度」。

### 4.4 用觀察效應回推的 100%：winner's curse

若這個 pair 是因為「看起來差很多」才被挑來分析，$\hat\mu$ 條件在被選中這件事
上就是**向上偏**的。拿它當 $\mu_*$ 算出的 power 高估、算出的 $N$ 低估，
複製實驗常「意外」失敗。修正：$\mu_*$ 用 SESOI 或對 $\hat\mu$ 收縮
（例如取其 CI 下界）。

### 4.5 理論 power vs 實務 power 的落差清單

| # | 落差來源 | 方向 | 對策（本 repo） |
|---|---|---|---|
| 1 | $\mu_*$ 填了 pilot 觀察值（winner's curse） | power 高估 | 用 SESOI／$\hat\mu$ 的 CI 下界 |
| 2 | token 自相關、item 群聚沒建模 | 把 token 當 iid → 嚴重高估；我們的兩層模型＋動差法 → $\hat\sigma_b^2$ 偏高、略保守 | 看 `variance_decomposition.py` 的 $\rho_1$ 診斷 |
| 3 | 重尾分布，常態近似失真 | 小 $N$ 高估 | 以 `random_subsample_power.py` 經驗曲線互驗 |
| 4 | 多重比較：6 metrics × 2 weightings ≈ 12 格 | 有效 $\alpha$ 膨脹 → 「隨便哪格顯著」的假陽性遠超 5% | pre-register 單一 primary（SOP §3）；其餘當探索 |
| 5 | Optional stopping：收一點看一次，顯著就停 | $\alpha$ 膨脹 | 事前定 $N$；resume 只補收到預定 $N$，不看結果加碼 |
| 6 | Failed rows 非隨機缺失（長題/OOM 先陣亡） | $\hat\mu$、$\hat\sigma$ 偏差 | manifest 的 FAIL 列要檢查與長度/難度的相關 |
| 7 | 母體漂移：換 subset、換模型家族、pilot 過期 | 不定 | 換件即重跑分解（CPU 秒級，別省） |
| 8 | 配對假設實務破壞：ref 非 bit-identical、tf_chunk 不一致 | 多為 power 低估＋偏差 | pipeline 已 fail-loud（guard fields、ref 位元檢查）；`--allow-ref-drift` 要當例外審批 |
| 9 | $T$ 外推假設訊號不稀釋 | 高估 $T\to\infty$ 收益 | 先做 per-position bucket 檢查（notes-power §6.3） |
| 10 | bootstrap iters/seed 在臨界處抖動（僅 `--ci-method studentized|bca|percentile`；預設 `t` 無此問題） | verdict 翻面 | 臨界結果提高 `--bootstrap-iters` 重跑確認 |

---

## 5. 對既有 knowledge/ 文件的核對記錄

依指示帶著懷疑讀過與本題相關的段落，結論：

- [notes-power-variance-decomposition.md](notes-power-variance-decomposition.md)：
  §2–4 推導與動差估計正確；§6.4 自相關使 $\hat\sigma_b^2$ 高估（保守）的
  **方向正確**；§6.3 對 $\mathrm{SNR}(\infty)$ 是「不可達上界」的提醒重要且正確。
  無需更正。
- [llm_quantization_paired_eval_teaching.md](llm_quantization_paired_eval_teaching.md)
  §10（non-inferiority／equivalence 的 CI 判準，含 loss-like 方向處理）與
  §13（task-level protocol）正確。注意其脈絡是 task-level stochastic decoding；
  本 FAQ §3.1 說明了與 logit-level 確定性量測的差異，兩者不衝突。
- 未逐字審完三份 1400–2100 行的教學文件全文；後續引用其他段落時應個別核對。
