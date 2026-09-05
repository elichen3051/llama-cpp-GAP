# 題內變異隨 K 非 1/K 收斂：問題、理論定位與修正方法（文獻佐證）

日期：2026-08-28。來源：deep-research workflow（106 agents；24 個來源、
114 條抽取 claims、25 條經 3 票對抗式驗證 → 23 確認/2 否決 → 8 條
findings；引文逐字對過原始 PDF/摘要），合併本地 balanced-panel 檢驗
（qwen3.5-4b ins-std10 與 wikitext pair）與
`~/backtest-2026-08-26-power-allocation` 的實測。完整逐條報告（含每條
finding 的票數、被否決 claims 全文、open questions、24 來源清單）另存
`~/deep-research-variance-decomposition.md`（非 repo 檔案）。

---

## 1. 問題描述

### 1.1 設定與記號

一次量化比較實驗：~N 個 items（題目），每題在 teacher forcing 下對
候選 A、B 各得逐 token 的 KLD（對共同 reference），配對差
`δ_i,t = kld_A(i,t) − kld_B(i,t)`，t = 1..T_i。裁決統計量是截斷平均

    d_i(K) = (1/min(K,T_i)) · Σ_{t≤min(K,T_i)} δ_i,t

再對 {d_i(K)} 做跨題 paired t-test。設計問題（「加 token 還是加題？」）
需要知道 Var(d_i(K)) 隨 cap K 的行為。

Miller（arXiv:2411.00640）§3.1 對「每題 K 次**獨立重採樣**」推導出

    Var(d_i(K)) ≈ σ²_b + σ²_w/K        （二成分律，下稱 1/K 律）

我們過去把它映射到「答案 token 位置」上做 power 規劃。問題：token
位置不是獨立重採樣——同一條軌跡的不同位置，均值與變異都隨位置變。

### 1.2 實測失效（backtest 2026-08-26，F1/F7/F9）

- **F1（697 個 pair×cap 格）**：K ≤ 256 時 1/K 律誤估實測
  Var(d_i(K)) 至 ~2×，且**方向依 pair 而異**——vision pairs 低估
  （log2(pred/emp) 中位數 −0.41…−0.97；token 變異前段大）、wikitext
  pair 高估（後段大）。K ≥ 1024 之後律大致回收。
- **F7/F9**：μ(K) 本身 cap 相依——44/75 pairs 在內部 K 達峰，
  4/75 pairs 隨 K **變號**。cap 不只是精度旋鈕，是 estimand 選擇。
- 序列相關很弱：逐位置 lag-1 自相關 ≈ +0.03，batch-means design
  effect ≈ 1.2 —— 一開始被誤讀為「自相關無關」（見 §2.3 修正）。

### 1.3 本地歸因分解（2026-08-27 balanced-panel）

把每題軌跡對齊成 balanced panel、擬合「獨立雜訊 + 位置變異剖面 w_t」
模型：

- 位置剖面模型把 vision K=16 的 log2(pred/emp) 從 **−0.67 修到
  +0.01** ⇒ 一階原因是**位置非平穩**（前段 token 變異可達 ~10×）。
- 但 K=64–256 仍殘留 ~1.4× 低估，與 Bartlett 加權積分自相關
  τ(K) ≈ 1.2–1.3 吻合——ACF 個別很小（lag-1≈0.04）卻緩衰到 lag~32，
  積分後不可忽略。「lag-1 小 ⇒ 自相關無關」是 IACT 文獻明白警告的
  謬誤（τ = 1 + 2Σρ_k）。
- ⇒ 兩個機制都在場：**位置非平穩一階、弱長程相關二階**。且依 §3.3，
  非平穩下實測 ACF 本身含 contamination，兩者無法乾淨分離。

### 1.4 後果：power 規劃錯多少

變異被低估 ~2× ⇒ 所需 N 被低估 ~2×；照律規劃的「80% power」樣本在
真實變異下實際 power 掉到約 50%（noncentrality 縮 √2）。反方向
（wikitext 型）則是浪費一倍的收集預算。變號 pairs 更糟：單一固定 cap
可能給出方向性錯誤的裁決——這不是精度問題，是 estimand 問題。

---

## 2. 理論定位 — 修正不是加一個係數，是換掉估計量

### 2.1 實測 var(K) ladder 有正式名字（finding 1，high，15-0 票）

它正是 **cluster-robust / sandwich 變異估計量**（White 1984；
Liang & Zeger 1986）在「item = cluster、token 位置 = cluster 內觀測」
映射下的形式。代數事實：fixed K、intercept-only、item 為 cluster 時
CR0 sandwich 恰化簡為 d_i(K) 的跨題樣本變異（差 n/(n−1)）。其一致性
**完全不限制 cluster 內共變異**——不要求平穩、不要求位置同變異、不
要求任何相關模型；只要求 item 間獨立與 item 數夠多（Cameron & Miller
2015 原文："heteroskedastic- and cluster-robust ... do not require
specification of a model for within-cluster error correlation"；
Liang-Zeger Thm 2）。few-cluster 下偏等已知限制在 ~54k items 不適用。

### 2.2 1/K 律是什麼（finding 2，high，3-0）

Compound-symmetry（exchangeable／等相關）工作模型的**特例**——Kish &
Frankel 1974、Kloek 1981、Moulton 的 design-effect 系譜。CS 下
K-mean 變異 = [(σ²_b+σ²_w)/K][1+ρ(K−1)] = σ²_b + σ²_w/K **恰好
成立**；前提是「順序無關」。位置非平穩直接違反此前提——失效不是
bug，是 working model 假設破產。Kish 的 deff 公式在「變異隨層而異」
時原文即稱 "not always relevant"。

### 2.3 精確的變異恆等式（finding 6，high，3-0）

FDA/longitudinal 形式（Yao-Müller-Wang 2005）：把每題 delta 軌跡視為
隨機曲線 X_i(t)，mean μ(t)、共變異面 G(s,t)=Cov(X(s),X(t)) 皆不受限：

    Var(d_i(K)) = (1/K²)·Σ_{s,t≤K} G(s,t) + σ²_meas/K

1/K 律 = 「G(s,t) 為常數面」的退化情形。實測 ladder 直接估計左邊，
天生正確；擬合 G 面是唯一有原則的外插途徑（model-based）。

---

## 3. 三個工作結論的判定

### (a) 1/K 律在位置非平穩下失效；實測 var(K) 曲線是正確的設計面 — 證實（high）

§2.1–2.3 全部；另外 longitudinal 文獻明文准許非參數共變異估計
（Diggle & Verbyla 1998："either as a guide to the formulation of a
parametric model or as the basis for formal inference without imposing
parametric assumptions"；平方殘差 kernel smoothing "without assuming
stationarity"）。

### (b) 主要修正是位置相依變異、不是弱自相關 — 證實，**表述要精確**

文獻面（findings 3–5）：

- Batch means / IAT/ESS / Geyer initial-sequence / 譜估計的理論對象是
  **平穩鏈的 long-run variance** σ² = Σ_t γ_t（Geyer 1992 Thm 2.1；
  Flegal & Jones 2010 的一致性定理皆明文假設平穩）。它們修的是「序列
  相關使 σ²_g ≠ Var_π(g)」，對持續的位置變異剖面**沒有任何條款**。
- 量級：HAR 首階 size-distortion 隨譜曲率 ω^(q) 縮放（Lazarus-Lewis-
  Stock-Watson 2018；Sun 2014a 展開式）。教科書警示在 ρ≈0.5–0.7
  （ω^(2)=4–15.6，size 11.4%–18%）；lag-1≈0.03 給 ω^(2)=0.064——小
  **60–250×**。（此腿單一來源，finding 標 medium；方向由平穩性論證
  與實測 deff 1.2 獨立佐證。）
- 非平穩下平穩式 HAC/IACT 機器**不只無用、還有偏**：時間變異的二階矩
  在樣本自共變異/periodogram 造成 "low frequency contamination"
  （假長記憶），平穩式 LRV 估計膨脹、檢定 undersized、power 大損，
  fixed-b/長頻寬更糟（Casini 2023；Casini-Deng-Perron；Casini-Perron）。
  非平穩下正確對象是位置積分的 local long-run variance
  J = ∫₀¹c(u,0)du + Σ_{k≥1}∫₀¹(c(u,k)+c(u,k)')du；理論證明的補救是
  **對時間做非參數平滑**——正是「直接估計逐位置變異剖面」。

**表述紀律（來自對抗式驗證，重要）**：

- 籠統版「Newey-West 以平穩性為前提，所以不是正確工具」被否決
  **0-3**。正確表述：「**平穩式、只修自相關的修正不是主要槓桿**」，
  不是「相依修正不必要」。Casini 文獻的補救仍是局部地平滑相依。
- Casini 結果的場景是單一時間序列 HAR，套到多短序列是**類比**；兩條
  最尖銳的方向性陳述 2-1 通過（不是 3-0）。
- 一條用 Moulton 因子量化「pooled per-token 分析在 ρ≈0.03、K≈256 下
  變異膨脹 ~13×」的 claim 被否決 1-2 —— pooled 分析的
  anticonservativeness **沒有已驗證的量化**；我們的 per-item 聚合設計
  本來就迴避它，但不要引用那個 13×。
- 本地歸因（§1.3）維持：位置非平穩一階、弱長程相關二階
  （τ≈1.2–1.3）；兩者皆由實測曲線自動涵蓋。

### (c) Power 規劃在預註冊 cap 上用該 cap 的實測變異；外插＝假設 — 證實（high）

- GEE 樣本數/power 方法學（Liu & Liang 1997 score 版；Shih 1997、
  Pan 2001 Wald 版；Wang 2014 綜述、歸屬已溯源）就是把 **robust
  sandwich 變異**代入 noncentral 檢定統計量——「在實際設計下用估計量
  的實測變異規劃 N」正是 ladder + iterated-t 的做法；54k 語料扮演
  供應 plug-in 值的 pilot data。不等題長的古典修正（Kish 1987；
  Eldridge et al. 2006 加權平均 cluster size / CV 修正）建模的仍是
  相關而非位置剖面——實測路徑一併涵蓋。
- μ(t) 隨位置變動甚至變號 ⇒ cap 是 estimand 選擇：GLS/位置加權會
  **改變 estimand**（Liang-Zeger robust 只保護變異誤設，不保護均值
  模型改變）⇒「預註冊 cap、變號 pair 分段報告」協議必要。
  **誠實註記**：驗證中沒有找到把 truncation cap 視為 estimand 一部分
  的明文協議文獻（ICH E9(R1) 是最近的框架類比）；這一腿目前是第一性
  推理 + FDA 模型的變號 μ(t)，不是可引用的協議文獻。
- 外插到未觀測長度：有原則性機器（FPCA/PACE 共變異面、local-in-time
  smoothing）但本質 model-based——照 tool 現行標示「extrapolation =
  assumption」處理正確。

---

## 4. 解決方法總表

### 4.1 主處方（已落地）

| 作法 | 理論身分 | 狀態 |
|---|---|---|
| 逐 cap 實測 μ(K)/Var/N ladder | cluster-robust sandwich（White；Liang-Zeger） | `cli/variance_decomposition.py` 主輸出 |
| 1/K 律只作 model-check | CS 特例診斷（log2_pred_over_emp，F1 指標） | 同上，附 UNDER/OVER 方向 |
| sign-flip 診斷 + 預註冊 cap | estimand 紀律（μ(K) 非單調、4/75 變號） | 同上 + SOP |
| N 用 iterated-t（非 z 閉式） | 保守方向（backtest F4：noncentral-t 預測偏保守是校準過的方向） | 同上 |
| sqrt-rule K* seed + 成本模型 | 預算分配起點（wall_s ≈ a+b·n_eval+c·n_prefill） | 同上（K*=863 wikitext 重現） |

### 4.2 禁用清單

- **不要**引入 Newey-West / ESS / IACT / batch-means 式**平穩**修正：
  在位置非平穩資料上有偏（low-frequency contamination，§3(b)）。
- **不要**由 1/K 律規劃 N 而不附實測檢查欄。
- **不要**在 μ(t) 變號的 pair 上用位置加權平均（estimand 改變）。

### 4.3 文獻建議、尚未實作的增強（按實用性排序）

1. **Diggle-Verbyla (1998) 平方殘差 kernel smoothing**：對位置變異
   剖面 w_t 非參數平滑（"without assuming stationarity"）——強化
   ladder 的小樣本穩定性。成本低。
2. **FPCA / PACE（Yao-Müller-Wang 2005）**：估 G(s,t) 面 + empirical-
   Bayes 曲線重建；唯一有理論根據的「外插到更長序列」途徑（標明
   model-based；informative T_i 的偏置是 open question）。中成本。
3. **DK-HAC（Casini 2023）**：lag 與位置雙 kernel 的非平穩 LRV 估計
   （公開實作 github.com/alessandro-casini/DK-HAC）——ladder 的平滑化
   類比；需從單序列改編到多短序列。
4. **位置加權 FGLS + sandwich 保護**：1/w_t 加權提效率、sandwich 保
   有效性（Cameron-Miller eq. 15 範式）。**重大警告**：μ(t) 非常數時
   加權改變 estimand——僅 μ(t) 近常數的 pair 可用，且需預註冊權重並
   與等權估計並列報告。
5. 未在任何確認 claim 出現、仍屬 open：variance-stabilizing
   transforms、stratified-by-position 估計量、per-item 變異 shrinkage。

### 4.4 對 skymizer 的落地含意

- `cli/variance_decomposition.py` 現行設計（實測 ladder 為主、1/K 律
  只作 model-check、外插標為假設、sign-flip 警告、iterated-t N）＝
  文獻認可的做法，不需結構性更改。
- 可選增強（未實作）：w_t 剖面 kernel 平滑輸出；G(s,t) 面估計供外插。
- 措辭紀律：「自相關不是主因」應寫成「弱長程相關是二階項
  （τ≈1.2–1.3），位置非平穩是一階項；兩者皆由實測曲線自動涵蓋，且
  平穩式、只修自相關的修正在此類資料上不適用」。

---

## 5. 驗證狀態速覽

- 23/25 claims 確認（多數 3-0）；findings 1,2,5,6,7,8 為 high、
  finding 4（量級腿）為 medium（單一來源）。
- 被否決：Moulton 13× pooled 警告（1-2）；籠統版「Newey-West 不適用」
  （0-3）——兩者**皆不可引用**。
- Casini 移轉是類比（單序列 → 多短序列）；最尖銳兩條 2-1。
- RQ4（2024–2026 LLM-eval 統計文獻）實質未答：Miller 2411.00640、
  Bowyer et al. 2503.01747（CLT 批評）、Madaan et al. 2406.10229、
  Lotfi et al. 2606.00206（逐位置 KL 非均勻）已定位、claims 已抽取但
  未經驗證——閱讀清單見 `~/deep-research-variance-decomposition.md` §7。

---

## 6. 參考文獻（除註明外皆經逐字對抗式驗證）

**Cluster-robust / GEE：**
- Liang, K.-Y. & Zeger, S. (1986). Biometrika 73:13–22 — GEE sandwich
  一致性（Thm 2：任意 working correlation）。
- White, H. (1984) — cluster-robust 變異的計量經濟起源。
- Cameron, A.C. & Miller, D. (2015). J. Human Resources —
  實務指南；Moulton/Kloek 精確性條件；FGLS+sandwich（eq. 15）。
- Gallis, Li & Turner (2020). Stata Journal (PMC8942127) — 小樣本修正
  （Kauermann-Carroll、Mancl-DeRouen）。

**Design effects：**
- Kish & Frankel (1974)；Kloek (1981), Econometrica；Kish (1987/1992)。
- Eldridge, Ashby & Kerry (2006). IJE 35:1292 — 不等 cluster size：
  DE = 1 + {(CV²+1)·n̄ − 1}·ρ。
- Zipf & Valliant, PracTools vignette — deff 公式的限制條款。

**平穩序列相關工具箱（對照組）：**
- Geyer (1992). Statistical Science 7:473–483 — initial-sequence／
  平穩鏈 CLT。
- Flegal & Jones (2010). Annals of Statistics 38:1034–1070 —
  batch means/譜估計一致性（平穩假設明文）。
- Lazarus, Lewis, Stock & Watson (2018). JBES — HAR 實務建議；
  size-distortion 展開（Sun 2014a, Econometrica）。

**非平穩 long-run variance：**
- Casini (2023). J. Econometrics 235:372–392 (arXiv:2103.02981) —
  low-frequency contamination；DK-HAC。
- Casini, Deng & Perron. Econometric Theory 42:294–335
  (arXiv:2103.01604)；Casini & Perron (arXiv:2111.14590)。

**非平穩變異剖面 / FDA：**
- Diggle & Verbyla (1998). Biometrics 54:401–415 — variogram/平方殘差
  的非參數（非平穩）平滑。
- Yao, Müller & Wang (2005). JASA 100:577–590 — PACE/FPCA；G(s,t) 面。
- Diggle, Heagerty, Liang & Zeger. Analysis of Longitudinal Data (OUP)。

**GEE power：**
- Liu & Liang (1997). Biometrics 53:937–947；Shih (1997)；Pan (2001).
  Controlled Clinical Trials 22:211–227；Wang (2014) 綜述（歸屬已
  溯源）；Rutterford et al. (PMC4521133)。

**Estimand 框架：**
- ICH E9(R1) addendum (EMA) — estimands 概念類比（未驗證出 cap 協議）。

**LLM-eval 統計（未經驗證的線索）：**
- Miller (2024). arXiv:2411.00640 — §3.1 K-resample 律 = CS 特例。
- Bowyer, Ivanova & Aitchison (ICML 2025). arXiv:2503.01747。
- Madaan et al. arXiv:2406.10229；Lotfi et al. arXiv:2606.00206。

**本地證據：**
- `~/backtest-2026-08-26-power-allocation`（F1/F7/F9/F4；
  `varfit_curves.csv` 與本 tool 逐格交叉驗證）。
- 2026-08-27 balanced-panel 位置剖面/τ 分解（本 repo 對話紀錄）。
- 完整 deep-research 報告：`~/deep-research-variance-decomposition.md`。
