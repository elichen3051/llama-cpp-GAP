# llama.cpp `perplexity` / `--kl-divergence` metrics 教學筆記

> 目的：整理 `llama.cpp` 的 `perplexity.cpp` 如何計算 PPL、KLD、Δp、RMS Δp、Same top p 等 metrics，並說明這些數值在 LLM quantization evaluation 中應如何解讀。
>
> 這份文件聚焦在 quantized model `Q` 與 base / FP model `B` 的比較。若比較兩個 quantization algorithms，可以把它們分別記為 `Q_A` 與 `Q_B`，並用同一套 block-level paired inference 進行比較。

---

## 0. 核心結論

在 llama.cpp 的 quantization 評估脈絡中，這些 metrics 其實可以分成三類：

1. **Gold-token likelihood metrics**：PPL、log PPL ratio、ΔPPL。\
   它們只看 corpus 中真實下一個 token 的機率。

2. **Full-distribution preservation metrics**：KLD。\
   它看的是 base model 與 quantized model 在整個 vocabulary 上的 next-token distribution 是否相近。

3. **Token-level perturbation diagnostics**：Mean Δp、percentile Δp、RMS Δp、Same top p。\
   它們用來回答「quantization 到底如何改變 token probability 與 top-1 decision」。

如果只選一個 formal primary metric，我會選：

\[
\Delta_{\mathrm{NLL}}
=
\overline{\ell_Q-\ell_B}
=
\log\frac{\mathrm{PPL}(Q)}{\mathrm{PPL}(B)}
\]

也就是 **mean NLL difference / log PPL ratio**。PPL ratio 可用來報告，但統計推論最好在 NLL / log-ratio scale 上做。

---

## 1. llama.cpp 的 evaluation protocol 概觀

### 1.1 PPL mode

在一般 PPL 計算中，llama.cpp 會：

1. 將文字 corpus tokenize。
2. 依照 context window 切成 chunks。
3. 對每個 token position 取得 logits。
4. 對真實下一個 token 計算 negative log-likelihood。
5. 對所有被計入的 token positions 平均 NLL。
6. 最後取 exponential 得到 PPL。

預設路徑中，程式會對每個 context window 只計算後半段 token 的 perplexity。也就是令：

\[
\texttt{first}=n_{ctx}/2
\]

每個 chunk 中只從 `first` 之後的 positions 累積 NLL。這樣做的用意是讓被評估的 token 有足夠前文 context。

> **注意（重現細節）**：若模型使用 BOS，llama.cpp 會把每個 chunk 的第一個 token 直接**替換**成 BOS（而非在前面插入，見 `perplexity.cpp` 的 `perplexity()` 與 `kl_divergence()`），因此 chunk 邊界與被評分的 token 集合不變，但第一個 context token 的內容不同——自行重現時若改用「插入 BOS」將無法對齊。另外在 `--kl-divergence` 模式下，token 序列是直接從 `.kld` 檔讀回、不重新 tokenize，兩個模型保證在完全相同的 token 上評分。

### 1.2 KL-divergence mode

若使用：

```bash
--kl-divergence-base path/to/base.kld
```

第一次通常是用 base / FP model 產生一個 `.kld` binary file。之後再用 quantized model 搭配：

```bash
--kl-divergence-base path/to/base.kld --kl-divergence
```

計算 quantized model 與 base model 的差異。

需要注意的是，這個 `.kld` file 並不是完整 FP32 logits dump。為了省空間，llama.cpp 會把 base log-probability 以 `uint16_t` 加上 scaling 的方式存起來。這代表 KLD、Same top p、base NLL 等比較 metrics 是相對於**壓縮後的 base log-prob approximation**，不是 exact FP32 base logits。

因此，如果看到非常小的 negative KLD percentile，不應解讀為數學上的 KL divergence 可以是負的；那通常是壓縮、截斷與數值近似造成的 artifact。

---

## 2. 基本符號

考慮 token position \(t=1,\dots,N\)。

- 真實下一個 token：\(y_t\)
- vocabulary：\(\mathcal V\)
- base / FP model：\(B\)
- quantized model：\(Q\)
- base logits：\(z_{B,t}(v)\)
- quantized logits：\(z_{Q,t}(v)\)

softmax probability：

\[
p_{M,t}(v)
=
\frac{\exp z_{M,t}(v)}{\sum_{u\in\mathcal V}\exp z_{M,t}(u)}
\]

其中 \(M\in\{B,Q\}\)。

log probability：

\[
\log p_{M,t}(v)
=
z_{M,t}(v)
-
\log\sum_{u\in\mathcal V}\exp z_{M,t}(u)
\]

negative log-likelihood：

\[
\ell_{M,t}
=
-
\log p_{M,t}(y_t)
\]

在 llama.cpp code 中，對 quantized model 的 NLL 會寫成：

\[
\ell_{Q,t}
=
C_{Q,t}-z_{Q,t}(y_t)
\]

其中：

\[
C_{Q,t}
=
\max_v z_{Q,t}(v)
+
\log\sum_v \exp\{z_{Q,t}(v)-\max_u z_{Q,t}(u)\}
\]

這是標準 numerically stable softmax / log-sum-exp 寫法。

---

## 3. llama.cpp accumulator 與數學對照

`perplexity.cpp` 中的 `kl_divergence_result` 主要累積以下量：

| Code variable | 數學量 | 用途 |
|---|---:|---|
| `sum_nll` | \(\sum_t \ell_{Q,t}\) | Quantized model mean NLL / PPL |
| `sum_nll2` | \(\sum_t \ell_{Q,t}^2\) | Quantized NLL uncertainty |
| `sum_nll_base` | \(\sum_t \ell_{B,t}\) | Base model mean NLL / PPL |
| `sum_nll_base2` | \(\sum_t \ell_{B,t}^2\) | Base NLL uncertainty |
| `sum_nll_nll_base` | \(\sum_t \ell_{Q,t}\ell_{B,t}\) | Paired covariance for log PPL ratio |
| `sum_kld` | \(\sum_t \mathrm{KL}_t(B\Vert Q)\) | Mean KLD |
| `sum_kld2` | \(\sum_t \mathrm{KL}_t^2\) | KLD uncertainty |
| `sum_p_diff` | \(\sum_t \Delta p_t\) | Mean Δp |
| `sum_p_diff2` | \(\sum_t \Delta p_t^2\) | RMS Δp |
| `sum_p_diff4` | \(\sum_t \Delta p_t^4\) | RMS Δp uncertainty |
| `n_same_top` | \(\sum_t \mathbf 1[\mathrm{top}_Q(t)=\mathrm{top}_B(t)]\) | Same top p |
| `count` | \(N\) | Number of evaluated token positions |

其中：

\[
\Delta p_t
=
p_{Q,t}(y_t)-p_{B,t}(y_t)
\]

---

## 4. Uncertainty 的計算方式

llama.cpp 的 `mean_and_uncertainty(sum, sum2, count)` 對 token-level values \(x_1,\dots,x_N\) 計算：

\[
\bar x
=
\frac{1}{N}\sum_{t=1}^N x_t
\]

以及 mean 的 standard error：

\[
\widehat{SE}(\bar x)
=
\sqrt{
\frac{
\frac{1}{N}\sum_t x_t^2-\bar x^2
}{N-1}
}
\]

這等價於：

\[
\widehat{SE}(\bar x)
=
\sqrt{
\frac{1}{N(N-1)}
\sum_t (x_t-\bar x)^2
}
\]

這是把 token-level values 視為近似 iid 後得到的 mean SE。

> **注意（輸出條件）**：只有在被評估的 token 數 `count ≥ 100` 時，llama.cpp 才會印出最終的 PPL / KLD / Δp 統計區塊（`perplexity.cpp` 中 `if (kld.count < 100) return;`）；另外當 `count ≤ 10` 時 `mean_and_uncertainty` 會直接回報 0 的 SE，covariance 亦視為 0。對短 corpus 或 per-item 的小型評估，這些 guard 會讓統計區塊整段消失或 `±` 顯示為 0。

### 4.1 重要 caveat：token 不是 iid

自然語言 token 有強烈序列相關，同一篇文章、同一個 chunk 裡的 token 也共享主題、語法與上下文。因此 llama.cpp 印出的 `±` 比較適合當作 engineering scoreboard 的 quick uncertainty，而不是完整統計推論。

若要做嚴格 model comparison，應改用：

- document-level paired bootstrap；或
- contiguous block / chunk-level paired bootstrap；或
- cluster-robust SE。

也就是不要把幾十萬或幾百萬 tokens 直接當成幾十萬或幾百萬個獨立樣本。

---

## 5. PPL 與 NLL

### 5.1 數學定義

mean NLL：

\[
L_M
=
\frac{1}{N}\sum_{t=1}^N \ell_{M,t}
\]

PPL：

\[
\mathrm{PPL}(M)
=
\exp(L_M)
\]

所以 PPL 不是「per-token PPL 的平均」，而是 **mean NLL 的 exponential**。

### 5.2 llama.cpp 計算方式

程式累積：

\[
\sum_t \ell_{M,t},\quad \sum_t \ell_{M,t}^2
\]

得到：

\[
\hat L_M
=
\frac{1}{N}\sum_t \ell_{M,t}
\]

再輸出：

\[
\widehat{\mathrm{PPL}}(M)
=
\exp(\hat L_M)
\]

PPL uncertainty 用 delta method：

\[
SE(\widehat{\mathrm{PPL}})
\approx
\exp(\hat L_M)\,SE(\hat L_M)
\]

### 5.3 在 quantization 中的解讀

PPL 越低，表示模型對 corpus 真實下一個 token 的平均預測越好。

但 PPL 有幾個限制：

1. **不可直接跨 tokenizer 比較**。LLaMA 2 與 LLaMA 3 tokenizer 不同時，absolute PPL 沒有乾淨的橫向意義。
2. **不可直接跨 corpus / implementation 比較**。context window、stride、BOS handling、batching、kernel precision 都可能影響 PPL。
3. **PPL 只看 gold token**。它不看整個 vocabulary distribution 是否被量化破壞。
4. **PPL 是 exponential scale**。小的 mean NLL 差異會被非線性放大。

因此 formal paired comparison 最好在 \(L_M\)、也就是 mean NLL scale 上做。

---

## 6. Mean log PPL ratio

### 6.1 數學定義

\[
\log\frac{\mathrm{PPL}(Q)}{\mathrm{PPL}(B)}
=
L_Q-L_B
\]

展開後：

\[
L_Q-L_B
=
\frac{1}{N}\sum_{t=1}^N
(\ell_{Q,t}-\ell_{B,t})
\]

這是非常重要的量。它是 token-level paired NLL difference 的平均。

### 6.2 llama.cpp 計算方式

llama.cpp 先計算：

\[
\hat L_Q,\quad \hat L_B
\]

再輸出：

\[
\widehat{\Delta}_{\log \mathrm{PPL}}
=
\hat L_Q-\hat L_B
\]

其 uncertainty 使用 paired covariance：

\[
SE(\hat L_Q-\hat L_B)
=
\sqrt{
SE(\hat L_Q)^2
+SE(\hat L_B)^2
-2\widehat{\mathrm{Cov}}(\hat L_Q,\hat L_B)
}
\]

其中 covariance 由：

\[
\sum_t \ell_{Q,t}\ell_{B,t}
\]

估計。

### 6.3 解讀

- \(\log\frac{\mathrm{PPL}(Q)}{\mathrm{PPL}(B)}=0\)：quantized 與 base 的 mean NLL 相同。
- 大於 0：quantized 較差。
- 小於 0：quantized 在該 corpus 上較好。

這比直接看 ΔPPL 更適合做統計推論，因為它是 additive paired difference。

---

## 7. Mean PPL ratio

### 7.1 數學定義

\[
\frac{\mathrm{PPL}(Q)}{\mathrm{PPL}(B)}
=
\exp(L_Q-L_B)
\]

### 7.2 llama.cpp 計算方式

程式先算 log PPL ratio：

\[
R_{\log}=L_Q-L_B
\]

再 exponentiate：

\[
R=\exp(R_{\log})
\]

uncertainty 同樣用 delta method：

\[
SE(R)
\approx
R\,SE(R_{\log})
\]

### 7.3 解讀

- \(R=1\)：無 PPL degradation。
- \(R=1.02\)：quantized PPL 約高 2%。
- \(R=1.50\)：quantized PPL 約高 50%。

在比較 quantization algorithms 時，PPL ratio 比 absolute PPL 更適合，因為它相對於同一個 base model。

但 formal test 仍建議在 \(R_{\log}\) 上做，報告時再轉回 PPL ratio。

---

## 8. Mean ΔPPL

### 8.1 數學定義

\[
\Delta \mathrm{PPL}
=
\mathrm{PPL}(Q)-\mathrm{PPL}(B)
\]

### 8.2 llama.cpp 計算方式

程式先取得：

\[
\widehat{\mathrm{PPL}}(Q),
\quad
\widehat{\mathrm{PPL}}(B)
\]

再取差：

\[
\widehat{\Delta\mathrm{PPL}}
=
\widehat{\mathrm{PPL}}(Q)-\widehat{\mathrm{PPL}}(B)
\]

uncertainty 使用 delta-method covariance：

\[
\mathrm{Cov}(\widehat{\mathrm{PPL}}(Q),\widehat{\mathrm{PPL}}(B))
\approx
\mathrm{PPL}(Q)\mathrm{PPL}(B)
\mathrm{Cov}(\hat L_Q,\hat L_B)
\]

再代入：

\[
SE(\Delta\mathrm{PPL})
=
\sqrt{
SE(\mathrm{PPL}_Q)^2
+SE(\mathrm{PPL}_B)^2
-2\mathrm{Cov}(\mathrm{PPL}_Q,\mathrm{PPL}_B)
}
\]

### 8.3 解讀

ΔPPL 直觀，但 scale 依賴 base PPL。

例如：

- PPL 從 5 到 5.1，ΔPPL = 0.1。
- PPL 從 50 到 50.1，ΔPPL = 0.1。

兩者的 relative degradation 完全不同。因此比較 quantization degradation 時，我會優先看 log PPL ratio / PPL ratio，而不是 ΔPPL。

---

## 9. PPL correlation

### 9.1 code 實際計算的是什麼

表格中叫做：

```text
Cor(ln(PPL(Q)), ln(PPL(base)))
```

從 code 來看，它使用的是：

\[
\ell_{Q,t},\quad \ell_{B,t}
\]

的 token-level correlation，也就是 per-token NLL / log-PPL contribution 的 correlation：

\[
\rho_{\ell_Q,\ell_B}
=
\frac{
\mathrm{Cov}(\ell_Q,\ell_B)
}{
\sqrt{\mathrm{Var}(\ell_Q)\mathrm{Var}(\ell_B)}
}
\]

雖然 `tools/perplexity/README.md` 的文字提到 correct-token probability correlation，但目前 code 印出的這個欄位更精確地說是 **per-token NLL correlation**。（本文引用的表格與說明文字皆出自 llama.cpp 的 `tools/perplexity/README.md`——本 repo 中的實際檔案；舊稱 `llama_cpp_ppl.md` 即指該檔。）

### 9.2 解讀

高 correlation 表示：

> base model 覺得難的 token，quantized model 也多半覺得難；base model 覺得簡單的 token，quantized model 也多半覺得簡單。

但它不是品質指標本身。兩個模型可以有很高 correlation，同時 quantized model 的 NLL 系統性偏高。

所以它適合回答：

> quantization 是否保留了 token difficulty ordering？

不適合單獨回答：

> quantization 是否退化？

---

## 10. KLD

### 10.1 數學定義

對每個 token position：

\[
\mathrm{KL}_t(B\Vert Q)
=
\sum_{v\in\mathcal V}
 p_{B,t}(v)
\log\frac{p_{B,t}(v)}{p_{Q,t}(v)}
\]

Mean KLD：

\[
\overline{\mathrm{KL}}
=
\frac{1}{N}\sum_{t=1}^N\mathrm{KL}_t(B\Vert Q)
\]

### 10.2 llama.cpp 計算方式

code 中會讀取 base 的 compressed log-prob：

\[
\widetilde{\log p}_{B,t}(v)
\]

並用 quantized logits 計算：

\[
\log p_{Q,t}(v)
=z_{Q,t}(v)-C_{Q,t}
\]

因此實際累積近似為：

\[
\widetilde{\mathrm{KL}}_t(B\Vert Q)
=
\sum_v
\widetilde p_{B,t}(v)
\left[
\widetilde{\log p}_{B,t}(v)-\log p_{Q,t}(v)
\right]
\]

其中：

\[
\widetilde p_{B,t}(v)=\exp(\widetilde{\log p}_{B,t}(v))
\]

程式中還有一個 practical cutoff：只有當：

\[
\widetilde{\log p}_{B,t}(v)>-16
\]

才納入 sum。這會忽略 base probability 極小的 vocabulary tail。

### 10.3 解讀

KLD 是 full-distribution metric。它不只看真實 token \(y_t\)，而是看 base model 的整個 next-token distribution 是否被 quantization 改變。

在 quantization evaluation 中，KLD 特別有用，因為好的 quantization 不只是要讓 corpus gold token probability 高，也要盡量保留原模型的行為分布。

- KLD 接近 0：quantized distribution 很接近 base distribution。
- KLD 變大：quantization 對 logits / probabilities 的擾動變大。
- KLD tail 很大：某些 contexts 上 distribution 被嚴重破壞。

### 10.4 KLD 與 PPL 的差異

PPL 看：

\[
p_Q(y_t)
\]

KLD 看：

\[
p_Q(v),\quad \forall v\in\mathcal V
\]

所以可能出現：

- PPL 變化小，但 KLD 大：gold token 沒太受傷，但其他 high-prob alternatives 被擾動。
- KLD 小，但 PPL 變化可見：整體 distribution 接近，但 corpus gold token 剛好被系統性壓低。
- Mean Δp 接近 0，但 KLD / RMS Δp 大：正負 probability perturbation 抵消，但 distribution noise 很大。

---

## 11. KLD percentiles

### 11.1 數學定義

對 token-level KLD values：

\[
k_t=\mathrm{KL}_t(B\Vert Q)
\]

排序後可計算：

\[
q_\tau(k)=\mathrm{Quantile}_\tau(\{k_t\}_{t=1}^N)
\]

llama.cpp 會印：

- Maximum KLD
- 99.9% KLD
- 99.0% KLD
- 95.0% KLD
- 90.0% KLD
- Median KLD
- 10.0% KLD
- 5.0% KLD
- 1.0% KLD
- 0.1% KLD
- Minimum KLD

percentile 使用排序後的 linear interpolation。

### 11.2 解讀

Mean KLD 告訴你平均 distribution drift。Percentiles 告訴你 drift 是否集中在少數 contexts。

在 quantization 中，我會特別看：

- 99.9% KLD：是否有極少數 catastrophic contexts。
- 99.0% / 95.0% KLD：高尾端是否普遍偏大。
- Median KLD：典型 token position 的 drift。

如果 mean KLD 不大，但 99.9% KLD 很大，代表大多數 token 沒事，但少數 contexts 可能被量化嚴重破壞。

---

## 12. Mean Δp

### 12.1 數學定義

對真實 token \(y_t\)：

\[
\Delta p_t
=
p_{Q,t}(y_t)-p_{B,t}(y_t)
\]

Mean Δp：

\[
\overline{\Delta p}
=
\frac{1}{N}\sum_t\Delta p_t
\]

### 12.2 llama.cpp 計算方式

code 中：

\[
p_B=\exp(-\ell_{B,t})
\]

\[
p_Q=\exp(-\ell_{Q,t})
\]

然後：

\[
\Delta p_t=p_Q-p_B
\]

並累積：

\[
\sum_t\Delta p_t,
\quad
\sum_t\Delta p_t^2
\]

用 `mean_and_uncertainty` 得到 mean 與 SE。

### 12.3 解讀

- \(\overline{\Delta p}>0\)：quantized model 平均上給 gold token 更高機率。
- \(\overline{\Delta p}<0\)：quantized model 平均上給 gold token 較低機率。
- \(\overline{\Delta p}\approx 0\)：平均 bias 小，但不代表沒有 noise。

Mean Δp 很直觀，但它會發生 cancellation。也就是：

\[
+0.1, -0.1
\]

平均是 0，但兩個 token 都被大幅擾動。

因此 Mean Δp 必須搭配 RMS Δp 與 percentile Δp 解讀。

---

## 13. Δp percentiles

### 13.1 數學定義

對：

\[
\Delta p_1,\dots,\Delta p_N
\]

排序後計算各分位數：

\[
q_\tau(\Delta p)
\]

llama.cpp 會印：

- Maximum Δp
- 99.9% Δp
- 99.0% Δp
- 95.0% Δp
- 90.0% Δp
- 75.0% Δp
- Median Δp
- 25.0% Δp
- 10.0% Δp
- 5.0% Δp
- 1.0% Δp
- 0.1% Δp
- Minimum Δp

### 13.2 解讀

Δp percentiles 很適合判斷 quantization 是「近似對稱 noise」還是「系統性品質損失」。

如果 positive tail 與 negative tail 大致對稱，可能表示 quantization 主要是在 probability 上加 noise。

如果 negative tail 明顯更大，例如：

\[
q_{0.001}(\Delta p) \ll -q_{0.999}(\Delta p)
\]

代表量化不只是加 noise，而是在部分 tokens 上大幅壓低 gold-token probability。

### 13.3 哪些 tail 最重要？

在 quantization context，我會特別看：

- 1.0% Δp
- 0.1% Δp
- Minimum Δp

因為這些反映 catastrophic probability drop。

但 minimum 很不穩定，容易受單一 token outlier 影響。研究報告中，比較建議使用：

\[
q_{0.001}(\Delta p),
\quad
q_{0.01}(\Delta p)
\]

或改成 tail probability：

\[
\Pr(\Delta p < -\eta)
\]

例如：

\[
\Pr(\Delta p < -0.05),
\quad
\Pr(\Delta p < -0.10)
\]

這會比 minimum 更穩定，也更容易做 paired bootstrap。

---

## 14. RMS Δp

### 14.1 數學定義

\[
\mathrm{RMS}\,\Delta p
=
\sqrt{
\frac{1}{N}\sum_t(\Delta p_t)^2
}
\]

### 14.2 llama.cpp 計算方式

code 先計算：

\[
\widehat{\mathrm{MSE}}_{\Delta p}
=
\frac{1}{N}\sum_t(\Delta p_t)^2
\]

再取平方根：

\[
\widehat{\mathrm{RMS}}_{\Delta p}
=
\sqrt{\widehat{\mathrm{MSE}}_{\Delta p}}
\]

uncertainty 使用 delta method：

\[
SE(\mathrm{RMS})
\approx
\frac{1}{2\sqrt{\mathrm{MSE}}}
SE(\mathrm{MSE})
\]

其中 \(SE(\mathrm{MSE})\) 是用 \((\Delta p_t)^2\) 與 \((\Delta p_t)^4\) 估出來的。

### 14.3 解讀

RMS Δp 是 probability perturbation 的 magnitude metric，不看方向。

- Mean Δp 很小、RMS Δp 也小：量化擾動小。
- Mean Δp 很小、RMS Δp 很大：正負方向抵消，但 token-level perturbation 大。
- Mean Δp 負、RMS Δp 大、negative tail 大：量化造成系統性傷害且有 catastrophic drops。

如果把 quantization 對 gold-token probability 的影響近似成 zero-mean Gaussian noise，RMS Δp 可以類比為 noise scale。但這只是診斷比喻，不應被過度解讀成真實 Gaussian model。

---

## 15. Same top p

### 15.1 數學定義

令：

\[
\mathrm{top}_M(t)
=
\arg\max_{v\in\mathcal V}p_{M,t}(v)
\]

Same top indicator：

\[
S_t
=
\mathbf 1[
\mathrm{top}_Q(t)=\mathrm{top}_B(t)
]
\]

Same top p：

\[
\widehat p_{\mathrm{same-top}}
=
\frac{1}{N}\sum_t S_t
\]

這裡的 `p` 是 proportion，不是 nucleus sampling 的 top-p。

### 15.2 llama.cpp 計算方式

code 會找：

- quantized logits 最大的 token id：`imax`
- base compressed log-prob 最大的 token id：`imax_base`

若兩者相同：

```cpp
++kld.n_same_top;
```

最後輸出：

\[
\frac{n_{same}}{N}
\]

uncertainty 用 binomial Gaussian approximation：

\[
SE
=
\sqrt{
\frac{
\hat p(1-\hat p)
}{N-1}
}
\]

### 15.3 解讀

Same top p 很適合衡量 greedy decoding stability。

- 高 Same top p：大多數 token positions 的 top-1 token 未改變。
- 低 Same top p：quantization 造成大量 top-1 decision flips。

但它有盲點：

1. 若 base top-1 與 top-2 margin 很小，top-1 flip 未必嚴重。
2. 若 top-1 沒變，但 probability distribution 大幅改變，Same top p 仍然很高。
3. 它不看 gold token 是否 top-1，只看 quantized 與 base 的 top-1 是否一致。

所以 Same top p 最好搭配：

- base top-1 margin
- KLD
- Δp tail
- NLL difference

一起看。

---

## 16. LLaMA 2 vs. LLaMA 3 Quantization comparison 的解讀

`tools/perplexity/README.md` 中的 LLaMA 2 vs. LLaMA 3 表格包含：

- Mean PPL
- Mean PPL ratio
- Mean ΔPPL
- PPL correlation
- Mean KLD
- Mean Δp
- Δp percentiles
- RMS Δp
- Same top p

最重要的原則是：

> 不要直接用 LLaMA 2 與 LLaMA 3 的 absolute Mean PPL 判斷誰比較好。更合理的是比較各自相對於自己的 base / FP model 的 degradation。

例如 q2_K：

| Metric | L2 7B q2_K | L3 8B q2_K | 解讀 |
|---|---:|---:|---|
| Mean PPL ratio | 1.107955 | 1.564849 | L3 q2_K relative PPL degradation 較大 |
| Mean KLD | 0.108903 | 0.445132 | L3 q2_K distribution drift 較大 |
| Mean Δp | -2.710% | -9.123% | L3 q2_K gold-token probability drop 較大 |
| RMS Δp | 9.762% | 21.421% | L3 q2_K probability perturbation magnitude 較大 |
| Same top p | 85.584% | 71.138% | L3 q2_K top-1 stability 較低 |

因此比較保守、精準的說法是：

> 在這個 llama.cpp / Wikitext-2 / revision / backend 設定下，LLaMA 3 8B 對相同名稱的 quantization scheme 看起來比 LLaMA 2 7B 更敏感，尤其在低 bit q2_K 時，relative PPL degradation、KLD、Δp tail、RMS Δp 與 Same top p 都顯示更大的擾動。

但這不能推出：

> LLaMA 3 本身比較差。

原因是兩者可能有不同 tokenizer、vocabulary、architecture details、training distribution 與 base PPL scale。

---

## 17. 如何對這些 metrics 做 paired inference

llama.cpp 目前輸出的 uncertainty 是 token-level quick SE。若要做研究級 comparison，我建議把 token positions 聚合成 blocks / documents。

令 block index：

\[
i=1,\dots,m
\]

block \(i\) 中有 \(T_i\) 個 evaluated token positions。

### 17.1 PPL / log PPL ratio

對 quantization algorithm \(A\) 與 \(C\)，定義 block-level mean NLL：

\[
L_i^A
=
\frac{1}{T_i}\sum_{t\in i}\ell_{A,t}
\]

\[
L_i^C
=
\frac{1}{T_i}\sum_{t\in i}\ell_{C,t}
\]

paired difference：

\[
D_i^{NLL}=L_i^C-L_i^A
\]

估計：

\[
\hat\Delta_{NLL}
=
\frac{1}{m}\sum_iD_i^{NLL}
\]

CI 可以用 paired t interval：

\[
CI
=
\bar D
\pm
 t_{m-1,1-\alpha/2}
\sqrt{
\frac{1}{m(m-1)}\sum_i(D_i-\bar D)^2
}
\]

或 block-level paired bootstrap。

最後再轉成 PPL ratio：

\[
CI\left(\frac{\mathrm{PPL}(C)}{\mathrm{PPL}(A)}\right)
=
\exp(CI(\Delta_{NLL}))
\]

### 17.2 KLD

若比較兩個 quantization algorithms \(A,C\) relative to same base \(B\)：

\[
K_i^A
=
\frac{1}{T_i}\sum_{t\in i}
\mathrm{KL}_t(B\Vert A)
\]

\[
K_i^C
=
\frac{1}{T_i}\sum_{t\in i}
\mathrm{KL}_t(B\Vert C)
\]

paired difference：

\[
D_i^{KLD}=K_i^C-K_i^A
\]

KLD 越低越好，所以：

- \(D_i^{KLD}<0\)：\(C\) 比 \(A\) 更接近 base。
- \(D_i^{KLD}>0\)：\(C\) 比 \(A\) 更遠離 base。

### 17.3 Mean Δp

\[
M_i^A
=
\frac{1}{T_i}\sum_{t\in i}
[p_{A,t}(y_t)-p_{B,t}(y_t)]
\]

\[
D_i^{\Delta p}=M_i^C-M_i^A
\]

Mean Δp 越高代表 gold-token probability loss 越小，但它會 cancellation，所以不應單獨使用。

### 17.4 RMS Δp

建議在 MSE scale 上做 inference：

\[
V_i^A
=
\frac{1}{T_i}\sum_{t\in i}
[p_{A,t}(y_t)-p_{B,t}(y_t)]^2
\]

\[
D_i^{MSE}=V_i^C-V_i^A
\]

因為 square root 是單調轉換：

\[
RMS_C<RMS_A
\quad\Longleftrightarrow\quad
MSE_C<MSE_A
\]

所以用 MSE 做 paired test 更乾淨，報告時再轉成 RMS。

### 17.5 Same top p

block-level：

\[
S_i^A
=
\frac{1}{T_i}\sum_{t\in i}
\mathbf 1[
\mathrm{top}_A(t)=\mathrm{top}_B(t)
]
\]

paired difference：

\[
D_i^{top}=S_i^C-S_i^A
\]

Same top p 越高，代表 top-1 preservation 越好。

若 token iid 假設可接受，也可以做 McNemar-style token-level paired binary test；但在 LLM evaluation 中，我仍建議用 block-level paired bootstrap。

### 17.6 Percentile metrics

對 KLD percentile 或 Δp percentile，不要直接平均 block percentiles。較好的方法是 paired block bootstrap：

1. 以 block 為單位 resample。
2. 每次 bootstrap 都重新組合 token-level values。
3. 重新計算 quantile。
4. 比較兩個 algorithms 的 quantile difference。

例如：

\[
D^*_{0.001}
=
q^*_{0.001}(\Delta p^C)-q^*_{0.001}(\Delta p^A)
\]

再用 bootstrap distribution 建 CI。

---

## 18. 我對 token-based metrics 的看法：如何獲得最大資訊

Token-based metrics 的優點是資訊量極大。一個 benchmark corpus 可以提供數十萬到數百萬個 token positions，每個 position 都能觀察到 quantization 對 logits、probabilities、ranking、NLL 的局部影響。

但它的陷阱也很明顯：token 數很多，不代表有效樣本數真的那麼大。若只看 token-level mean 與非常小的 SE，容易產生過度自信。

我會把 token-based analysis 設計成以下幾層。

### 18.1 第一層：primary degradation metric

使用：

\[
\overline{\ell_Q-\ell_B}
\]

也就是 log PPL ratio。

這回答：

> quantized model 在真實 corpus token 上平均退化多少？

這一層適合當 formal primary metric，並用 block-level paired CI 做推論。

### 18.2 第二層：distribution preservation

使用：

\[
\overline{\mathrm{KL}(B\Vert Q)}
\]

這回答：

> quantized model 是否保留 base model 的完整 next-token distribution？

這比 PPL 更接近 quantization 的本質，因為 quantization 理想上是保留原模型行為，而不是只在某個 corpus 上剛好提高 gold-token probability。

### 18.3 第三層：directional probability bias

使用：

\[
\overline{\Delta p}
\]

這回答：

> quantization 是否平均壓低 gold-token probability？

但它不能單獨使用，因為正負 perturbation 會互相抵消。

### 18.4 第四層：perturbation magnitude 與 tails

使用：

\[
\mathrm{RMS}\,\Delta p,
\quad
q_{0.01}(\Delta p),
\quad
q_{0.001}(\Delta p),
\quad
\Pr(\Delta p< -\eta)
\]

這回答：

> quantization 是否有少數 catastrophic probability drops？

我會特別重視 lower-tail Δp，因為 generation quality 常常不是由平均 token 決定，而是由少數關鍵 token 的錯誤分支決定。

### 18.5 第五層：decision stability

使用：

\[
\mathrm{SameTop}
\]

再搭配 base top-1 margin：

\[
\mathrm{margin}_t
=
\log p_{B,t}(v_{top1})-
\log p_{B,t}(v_{top2})
\]

Same top p 本身只知道 top-1 是否 flip，但不知道 flip 嚴不嚴重。

如果 base top-1 margin 很小，flip 可能只是 tie-breaking noise。\
如果 base top-1 margin 很大，flip 代表 quantization 嚴重改變 decision boundary。

因此我會報告：

\[
\Pr(\mathrm{top\ flip}\mid \mathrm{margin}>\gamma)
\]

例如：

- margin > 0.1
- margin > 0.5
- margin > 1.0

這比 raw Same top p 更有診斷價值。

### 18.6 第六層：stratified token analysis

為了知道 quantization 到底傷在哪裡，我會把 token positions 分層。

常見分層變數：

1. **base confidence**：\(p_B(y_t)\) 或 \(\ell_{B,t}\)
2. **base entropy**：\(H(p_{B,t})\)
3. **top-1 margin**：top1 與 top2 log-prob gap
4. **gold token rank under base**
5. **token frequency**：常見 token vs rare token
6. **token type**：數字、標點、空白、程式碼 token、專有名詞、多語 token
7. **position in context**：靠近 chunk 開頭、中間、結尾
8. **document / domain**：Wikipedia、code、math、dialogue、multilingual text

例如可以報告：

\[
\mathbb E[\ell_Q-\ell_B\mid p_B(y_t)>0.9]
\]

\[
\mathbb E[\mathrm{KL}(B\Vert Q)\mid H(p_B)<1.0]
\]

\[
\Pr(\Delta p<-0.1\mid \mathrm{gold\ rank}_B=1)
\]

這些分層分析能回答非常重要的工程問題：

> 量化是全面地增加小噪音，還是特別傷害高信心 token、rare token、數字 token、或低 entropy contexts？

### 18.7 第七層：sequence-level impact

Token-level metrics 仍然只是 local one-step prediction。Generation quality 是 sequential 的。某個 early token 的 small perturbation 可能讓整段 generation 進入不同 trajectory。

因此 token metrics 最好與 sequence-level eval 搭配：

- greedy exact match / accuracy
- sampling pass@1
- pass@k
- judge score
- task-level paired bootstrap

我會把 token metrics 當成：

> 高解析度診斷工具

而不是完整取代 downstream benchmark。

---

## 19. 建議的 quantization report 格式

如果要讓報告既簡潔又資訊量高，我會用以下格式。

### 19.1 Primary table

| Metric | Estimate | 95% paired CI | Direction |
|---|---:|---:|---|
| log PPL ratio | \(\overline{\ell_Q-\ell_B}\) | block bootstrap | lower better |
| PPL ratio | \(\exp(\overline{\ell_Q-\ell_B})\) | exponentiated CI | lower better |
| Mean KLD | \(\overline{\mathrm{KL}(B\Vert Q)}\) | block bootstrap | lower better |
| RMS Δp | \(\sqrt{\overline{\Delta p^2}}\) | block bootstrap | lower better |
| Same top p | \(\overline{S_t}\) | block bootstrap | higher better |

### 19.2 Tail diagnostics

| Metric | Meaning |
|---|---|
| 1% Δp | common lower-tail gold-token probability drop |
| 0.1% Δp | rare catastrophic probability drop |
| \(\Pr(\Delta p<-0.05)\) | interpretable failure-rate style diagnostic |
| 99.9% KLD | extreme distribution drift |
| top flip rate at margin > γ | severe decision-boundary instability |

### 19.3 Stratified diagnostics

Report each primary metric conditioned on:

- high-confidence base tokens
- low-confidence base tokens
- low-entropy contexts
- high-entropy contexts
- rare tokens
- numeric / code / multilingual tokens
- early vs late context positions

This often reveals more than a single aggregate score.

---

## 20. Practical recommendations

### 20.1 做 formal test 時

1. 對 PPL：用 NLL / log PPL ratio test，不要直接用 PPL difference 當 primary test。
2. 對 KLD：用 block-level paired bootstrap。
3. 對 Same top p：用 block-level paired bootstrap；若要 token-level test，至少要承認 token dependence。
4. 對 RMS Δp：在 MSE scale 上 test，報告時再轉 RMS。
5. 對 percentile Δp / KLD：用 paired block bootstrap 每次重新計算 quantile。

### 20.2 做工程 scoreboard 時

llama.cpp 原生輸出的 metrics 很有用，尤其適合：

- 快速比較多個 quantization schemes。
- 看 relative degradation。
- 找出明顯壞掉的 quantization。
- 判斷 noise-like perturbation vs systematic degradation。

但如果要寫 paper 或做嚴格 algorithm ranking，建議補上：

- block-level paired CI；
- practical equivalence / non-inferiority margin；
- 多 corpus validation；
- downstream task paired evaluation。

### 20.3 最小但高資訊量的 metric set

若只想保留少數 metrics，我建議：

1. **log PPL ratio / PPL ratio**：gold-token likelihood degradation。
2. **Mean KLD**：full distribution preservation。
3. **RMS Δp**：probability perturbation magnitude。
4. **0.1% or 1% Δp**：catastrophic gold-token probability drops。
5. **Same top p plus margin-stratified top flip rate**：greedy decision stability。

這五組合起來，可以同時回答：

- 平均 loss 是否變差？
- full distribution 是否被破壞？
- gold token probability 是否被壓低？
- 是否有 tail risk？
- greedy decoding 是否穩定？

---

## 21. 一句話總結

llama.cpp 的 PPL / KLD scoreboard 不只是單純的 perplexity 表格。它其實提供了三種互補視角：

\[
\text{gold-token loss}
\quad+
\text{full-distribution drift}
\quad+
\text{token-level perturbation diagnostics}
\]

在 quantization evaluation 中，我會用 **log PPL ratio** 做 primary paired inference，用 **KLD** 看 distribution preservation，用 **Δp tails / RMS Δp / Same top p** 找出量化造成的局部破壞模式。

最重要的是：token-level metrics 資訊量很大，但 token 不是 iid。嚴格比較 quantization algorithms 時，應把 token 聚合成 document / chunk / block，再做 paired confidence interval 或 paired bootstrap。
