# 加 eval tokens 還是加題數？— paired KLD 檢定力的變異數分解

> [!WARNING]
> 本文的二層 1/T 模型現在只作 historical/model-check diagnostic，不能用來
> 外推 sequential-prefix power。token cap 會改變 estimand，variance 也不保證
> 單調收斂；現行方法見 [seq-power-analysis.md](seq-power-analysis.md)。

> 2026-06-13。回答的問題：「從統計上看，`--num-eval-tokens -1` 對『測得到效應』幫助可能有限——
> Thinking 的瓶頸是 item 間變異，不是每題的 token 數」這句話的依據是什麼？
> 結論先講：這句話**只說對了一半**——做完分解後發現，cap 1024 下 Thinking 的主導噪音
> 其實仍是 token 級的（53–70%），拉長 eval tokens 有實質幫助，但天花板由 item 間異質性決定。
> 本文含推導、估計方法與四個母體的實測數字。

## 1. 設定

對每個 item（題目）$i = 1, \dots, N$，teacher-forcing 其答案的 $T_i$ 個 token，
在每個位置 $t$ 取得 paired 差：

$$
\delta_{i,t} \;=\; \mathrm{KLD}_B(i,t) - \mathrm{KLD}_A(i,t)
$$

（A、B 是對同一個 bit-identical reference 量測的兩個 candidate，故 $\delta_{i,t}$ 是逐
token 配對差。）item 級分數是 token 平均：

$$
d_i \;=\; \frac{1}{T_i} \sum_{t=1}^{T_i} \delta_{i,t}
$$

item-weighted 檢定統計量為 $\bar d = \frac{1}{N}\sum_i d_i$，
paired bootstrap 對 $\bar d$ 建 95% CI，CI 不含 0 即為顯著 verdict。

## 2. 變異數分解

把 $\delta_{i,t}$ 寫成二層模型：

$$
\delta_{i,t} \;=\; \mu \;+\; b_i \;+\; \varepsilon_{i,t},
\qquad
b_i \sim (0, \sigma_b^2), \quad \varepsilon_{i,t} \sim (0, \sigma_w^2)
$$

- $\mu$：全體平均效應（mmproj 量化造成的真實 ΔKLD）
- $b_i$：item 級偏離——**題目間的真實異質性**（不同題對 mmproj 擾動的敏感度不同）
- $\varepsilon_{i,t}$：token 級噪音

對 token 平均後（暫設 token 間獨立，見 §6）：

$$
\boxed{\;
\mathrm{Var}(d_i) \;=\; \sigma_b^2 \;+\; \frac{\sigma_w^2}{T_i}
\;}
$$

**這就是整件事的核心**：$d_i$ 的變異有兩個來源，
加 eval tokens（增大 $T_i$）只能壓縮第二項，且報酬率是 $1/T$；
$\sigma_b^2$ 是加多少 token 都動不了的地板。

## 3. 檢定力與所需題數

定義 item 級訊噪比：

$$
\mathrm{SNR}(T) \;=\; \frac{\mu}{\sqrt{\sigma_b^2 + \sigma_w^2 / T}}
$$

雙尾 $\alpha=0.05$、目標 power $1-\beta=0.8$ 時（$z_{0.975}=1.96$, $z_{0.8}=0.84$），
所需題數的常態近似：

$$
N_{80}(T) \;\approx\; \left( \frac{z_{0.975} + z_{0.8}}{\mathrm{SNR}(T)} \right)^{\!2}
\;=\; \left( \frac{2.8}{\mathrm{SNR}(T)} \right)^{\!2}
$$

兩個極限：

$$
\mathrm{SNR}(\infty) = \frac{\mu}{\sigma_b}
\quad\Longrightarrow\quad
N_{80}^{\min} = \left( \frac{2.8\,\sigma_b}{\mu} \right)^{\!2}
$$

**無論評多少 token，$N_{80}$ 都不可能低於 $N_{80}^{\min}$。**
「該加 token 還是加題數」取決於目前位置：
若 $\sigma_w^2/T \gg \sigma_b^2$（token 噪音主導）→ 加 token 有效；
若 $\sigma_w^2/T \ll \sigma_b^2$（item 異質性主導）→ 只有加題數有用。

## 4. 從既有 metrics 估計（動差法）

所有量都能直接從已收集的 per-token npz 算出，零 GPU 成本：

$$
\widehat{\mathrm{Var}}(d_i) = \mathrm{Var}_{\text{items}}(d_1,\dots,d_N), \qquad
\widehat{\frac{\sigma_w^2}{T}} = \frac{1}{N}\sum_i \frac{s_i^2}{T_i}, \qquad
\hat\sigma_b^2 = \widehat{\mathrm{Var}}(d_i) - \widehat{\frac{\sigma_w^2}{T}}
$$

其中 $s_i^2$ 是 item $i$ 內 $\delta_{i,t}$ 的樣本變異數（ddof=1）。

## 5. 四個母體的實測結果（kld，item-weighted）

| 母體 | $\bar T$ | $\mathrm{Var}(d_i)$ | token 噪音占比 | $\sigma_b^2$ 占比 | SNR 現況 | SNR$(\infty)$ | $N_{80}$ 現況 | $N_{80}^{\min}$ | 可用題數 |
|---|---|---|---|---|---|---|---|---|---|
| vision × Instruct | 599 | 5.91e-6 | 65.8% | 34.2% | 0.249 | 0.425 | 127 | 44 | 300 ✅ |
| vision × Thinking | 1005 | 2.21e-6 | 69.9% | 30.1% | 0.073 | 0.133 | 1481 | **446** | 454 ⚠️ |
| std10 × Instruct | 447 | 1.43e-5 | 73.8% | 26.2% | 0.061 | 0.120 | 2080 | 545 | 332 ❌ |
| std10 × Thinking | 1704 | 1.73e-6 | 53.4% | 46.6% | 0.051 | 0.074 | 3053 | **1422** | 467 ❌ |

（cap：vision 收集於 1024、std10 收集於 2048；$\bar T$ 是實際平均 eval tokens，
短答案題未吃滿 cap。表中數字由 `variance_decomposition.py` 產生，使用精確的
$z_{0.975}+z_{0.8}=2.8016$ 並對 $N$ 取上整——正文示意用的 $2.8$ 會差 ±1–3 題。
另注意 std10 × Instruct 僅 ~4% 的題吃滿 cap 2048：它 73.8% 的 token 噪音
**無法靠加大 cap 改善**（答案已全數收錄），工具的 verdict 行會自動偵測並標註。）

## 6. 解讀與修正

1. **原始說法的修正**：「Thinking 的瓶頸是 item 間變異」在 cap 1024/2048 下並不成立——
   四個母體的 token 噪音占比都過半（53–74%）。直覺錯在從「SNR 低」直接推「item 變異主導」,
   跳過了分解。正確流程：先算 $\hat\sigma_b^2$ 與 $\sigma_w^2/T$ 的占比，再決定加哪一邊。

2. **但天花板論證仍然成立，而且決定了最終可行性**：
   - vision × Thinking：$N_{80}^{\min} = 446$，貼著手上的 454 題——但注意
     $N_{80}^{\min}$ 是 $T\to\infty$ 的不可達極限。實際 `--num-eval-tokens -1`
     只把 $\bar T$ 拉到 ~8200：$\mathrm{Var} = 6.66\text{e-}7 + 1.55\text{e-}3/8200
     = 8.55\text{e-}7$，SNR ≈ 0.117，$N_{80}(\bar T{=}8200) \approx 573 > 454$，
     以點估計算 454 題的期望 power 約 **70%**（非 80%）。
     ——**仍值得跑**（§6.4 的保守性給了上行空間），但預期是貼線偵測。
   - std10 兩格：$N_{80}^{\min}$（545 / 1422）都超過可用題數（332 / 467），
     **加滿 token 也救不回來**，只有擴充資料集一途。
   - vision × Instruct：$N_{80}^{\min}=44$——若先把 token 噪音壓掉
     （它的訊號集中在前 32 token，等效於用很少的「高訊號 token」就達到低噪音），
     44 題就夠。這解釋了為什麼它是唯一輕鬆顯著的格子。

3. **$T\to\infty$ 預測的隱含假設——訊號不稀釋、異質性不變**：$\mathrm{SNR}(\infty)$
   假設位置 1024/2048 之後的 token (a) 帶有同樣的 $\mu$，且 (b) item 效應 $b_i$
   與前段一致（$\sigma_b^2$ 不變）。vision × Instruct 的實測顯示訊號集中在
   前段（後段 $\mu$ 衰減 10 倍以上）；若 Thinking trace 後段也如此，拉長 $T$ 時
   $\mu$ 同步下降，SNR 增益會縮水。(b) 失效時天花板本身會移動（方向不定——
   thinking trace 後段的結構分歧可能放大或縮小 item 間差異）。**跑 `-1` 前應
   先用前段資料外推後段 per-position delta，或收完後立刻做 position-bucket 檢查。**

4. **token 自相關使 $\hat\sigma_b^2$ 偏高（保守）**：§2 假設 $\varepsilon_{i,t}$ 獨立。
   實際上相鄰 token 的 $\delta$ 正自相關（同一段文字的條件分布相近），
   故 $\mathrm{Var}(\bar\varepsilon_i) \ge \sigma_w^2/T_i$，
   我們的 $\hat\sigma_b^2 = \mathrm{Var}(d_i) - \sigma_w^2/T$ 是**高估**。
   換言之真實天花板可能比表中更寬鬆——$T\to\infty$ 的潛在收益是表列數字的下界。

## 7. 實務決策規則

```
要提升檢定力時：
1. 先跑本文 §4 的分解（CPU、秒級、用既有 npz）
2. token 噪音占比 > 50%  →  拉長 --num-eval-tokens 有效，
   但先檢查 N80_min 是否 < 可用題數（否則白拉）
3. token 噪音占比 < 30%  →  只有加題數/換更高訊號的資料集有用
4. 拉長 T 之前，檢查 per-position delta 是否前段集中
   （集中 → 拉長反而稀釋 μ，考慮反向操作：截短 + 加題數）
```

對應工具：本文 §4 的分解已實作為 `variance_decomposition.py`（用法見其 docstring，
含 §7 決策規則的自動判讀與 --output-json）；
power 實證曲線見 `random_subsample_power.py`；
verdict 掃描當時由 `std10_report_sweep.py`（已移除；被報告 pipeline 的 SNR sweep 取代）產生於 `outputs/{vision,std10}-reports/SUMMARY.md`。
