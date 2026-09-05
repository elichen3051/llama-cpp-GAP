# LLM Quantization Benchmark 的配對統計推斷教學文件

版本：2026-04-27\
主題：如何在 LLM / VLM quantization evaluation 中，用合理的 estimand、item-level aggregation、paired confidence interval 與 bootstrap 判斷兩個模型或兩個 quantization algorithms 是否真的有差距。

---

## 0. 核心結論

在 LLM quantization evaluation 裡，最容易犯的錯不是公式寫錯，而是 **統計單位與 estimand 沒定義清楚**。

最重要的結論如下：

1. **統計推斷的基本單位應該是 benchmark item，不是 sampled response。**\
   同一題的多次 sampling 共享題目、ground truth、prompt、scoring rule，因此不能攤平成互相獨立的樣本。

2. **比較兩個模型時，應該做 paired comparison。**\
   因為 full-precision model、quantized model、quantization algorithm A、quantization algorithm B 通常是在同一批題目上評估。應先對每題建立差值，再對差值做統計推斷。

3. **正式判斷差異時，不應看兩個模型各自的 CI 是否 overlap。**\
   正確做法是直接建立 paired difference 的 CI：

   \[
   \Delta = \mu^B - \mu^A
   \]

   然後看：

   \[
   0 \in CI(\Delta)
   \]

   若 paired difference CI 不包含 0，才代表在該 confidence level 下有統計上可辨識的差異。

4. **Greedy、sampling pass@1、pass@k、majority vote、best-of-n 是不同 evaluation protocols，也通常是不同 estimands。**\
   repeated decoding 不一定只是降低 variance；有時它會改變你正在估計的目標量。

5. **固定推論預算下，如果目標是比較兩模型的整體平均 pass@1 差異，而且還可以增加題目數，通常應優先增加題目數，而不是增加每題 sampling 次數。**

6. **對 quantization 研究，建議至少報告兩組結果：**
   - benchmark-standard / greedy / official protocol result，用於 reproducibility 與 regression；
   - deployment-aligned stochastic result，用於回答實際 deployment decoding policy 下的表現。

---

## 1. 問題設定

我們考慮比較兩個模型或兩個 quantization algorithms：

- \(A\)：baseline，例如 full-precision model、FP16 model、或 quantization algorithm A。
- \(B\)：candidate，例如 quantized model、或 quantization algorithm B。

Benchmark task distribution 記為：

\[
X \sim P_{\text{task}}
\]

其中 \(X\) 是從任務分布中抽出的一個 benchmark item / problem。

實際 benchmark dataset 是固定的：

\[
\mathcal{D}=\{x_1,x_2,\dots,x_m\}
\]

其中 \(m\) 是題目數。

令 score function 為：

\[
S(x,y) \in \mathbb{R}
\]

它可以是：

- binary correctness：\(0/1\)
- exact match：\(0/1\)
- multiple-choice accuracy：\(0/1\)
- pass@k item score
- partial credit
- judge score
- reward score
- 任意 benchmark framework 已定義好的 continuous metric

在 stochastic decoding 下，第 \(i\) 題、第 \(j\) 次 decoding、模型 \(M\) 的 observed score 記為：

\[
Z_{ij}^{M}=S(x_i,Y_{ij}^{M})
\]

其中：

- \(i=1,\dots,m\)：item index
- \(j=1,\dots,r\)：同一題的 repeated decoding index
- \(M\in\{A,B\}\)：模型或 algorithm
- \(Y_{ij}^{M}\)：模型 \(M\) 在第 \(i\) 題第 \(j\) 次 decoding 的輸出
- \(r\)：每題每模型 decoding 次數

如果是 deterministic protocol，例如 greedy decoding 或 log-likelihood ranking，則通常可視為 \(r=1\)，且沒有 sampling randomness。

---

## 2. Estimand、Estimator、Estimate

做 evaluation 前要先分清楚三件事：

| 名稱 | 意思 | 例子 |
|---|---|---|
| Estimand | 真正想估計的目標量 | 任務分布上的平均 pass@1 |
| Estimator | 用資料計算 estimand 的方法 | 對 item score 取平均 |
| Estimate | 實際算出的數字 | accuracy = 0.732 |

### 2.1 Task-level estimand

若模型 \(M\) 在 item \(x\) 上的 protocol-level expected score 是：

\[
g^M(x)
\]

則 task-level performance 是：

\[
\mu^M
=
\mathbb{E}_{X\sim P_{\text{task}}}
\left[g^M(X)\right]
\]

兩模型的 task-level difference 是：

\[
\Delta
=
\mu^B-\mu^A
=
\mathbb{E}_{X\sim P_{\text{task}}}
\left[g^B(X)-g^A(X)\right]
\]

這是最常見的研究問題：

> 若 benchmark items 可視為任務分布的樣本，模型 B 在整體任務分布上是否比模型 A 好？

### 2.2 Fixed-dataset estimand

有時候你只關心這個固定 benchmark dataset：

\[
\mathcal{D}=\{x_1,\dots,x_m\}
\]

那麼 fixed-dataset target 是：

\[
\mu_{\mathcal D}^M
=
\frac{1}{m}\sum_{i=1}^{m}g_i^M
\]

其中：

\[
g_i^M=g^M(x_i)
\]

兩模型的 fixed-dataset difference 是：

\[
\Delta_{\mathcal D}
=
\frac{1}{m}\sum_{i=1}^{m}(g_i^B-g_i^A)
\]

### 2.3 重要提醒：CI 的解釋取決於你把 dataset 當成什麼

如果你已經跑完整個固定 benchmark，而且 decoding 是 deterministic，那麼在「這個固定 dataset」上，平均分數本身沒有 sampling uncertainty；它就是一個 deterministic number。

但研究上我們通常仍然會報 CI，因為我們把 benchmark items 視為從更大任務分布抽出的樣本，想知道結論是否能外推到類似題目。這時候 item-level CI 反映的是：

\[
\text{finite benchmark item uncertainty}
\]

也就是「抽到這批題目」造成的不確定性。

---

## 3. 統計單位：永遠先回到 item-level

不論 decoding protocol 多複雜，統計推斷前都應先整理成 item-level score：

\[
Y_i^M
\]

然後再做平均、差值、CI、bootstrap。

### 3.1 為什麼不能把 sampled responses 當 iid samples？

假設每題 sample \(r=10\) 次，共有 \(m=100\) 題。你會有 \(1000\) 個 generated responses，但有效的 benchmark item 數仍然是 \(100\)，不是 \(1000\)。

同一題的多次回答：

- 共享同一個 prompt / item difficulty；
- 共享同一個 ground truth；
- 共享同一套 scoring rule；
- 常常有高度相關的 failure mode。

如果把 \(m\times r\) 個 response 攤平成 iid observations，SE 會被低估，低估倍率是 design effect \(\sqrt{1+(r-1)\rho}\)，其中 \(\rho\) 是同題回答間的相關。當同題回答高度相關（\(\rho\to 1\)）時，有效樣本數確實只剩 \(m\)（即被錯誤放大 \(r\) 倍）；一般情況下有效樣本數介於 \(m\) 與 \(mr\) 之間，但不論如何，推斷單位都應是 item。

### 3.2 benchmark framework 已經算好的 item-level metric 應優先使用

如果使用 `lighteval`、`lm-evaluation-harness` 或其他 benchmark framework，且輸出中已經有 record-level / item-level metric，例如：

```text
metric.acc
metric.exact_match
metric.pass@1
metric.avg@n:n=1
__metric__.acc
```

應優先使用這些 metric，不要重新從 prediction text 解析答案。

原因是：

1. 避免重做 exact match normalization。
2. 避免 multiple-choice parsing rule 不一致。
3. 避免 free-form judge / rubric scoring 重現錯誤。
4. 能與 benchmark 官方結果對齊。
5. 更適合做 paired comparison。

---

## 4. 不同 decoding protocol 對應不同 item-level score

### 4.1 Greedy decoding

Greedy decoding 是 deterministic protocol。模型 \(M\) 在 item \(x_i\) 上的輸出記為：

\[
y_{i,\text{greedy}}^M
\]

item-level score 是：

\[
Y_i^M=S(x_i,y_{i,\text{greedy}}^M)
\]

整體 performance estimator 是：

\[
\hat\mu^M
=
\frac{1}{m}\sum_{i=1}^{m}Y_i^M
\]

Greedy 的優點：

- 幾乎沒有 seed noise；
- 可重現；
- 成本低；
- 適合 quantization regression；
- 能比較直接觀察 weight / logit perturbation 對輸出的影響。

但 greedy 只能估：

\[
\mu_{\text{greedy}}^M
\]

它不是 deployment sampling performance 的無偏估計。

### 4.2 Sampling pass@1

若 deployment 使用 sampling policy \(\lambda\)，例如 temperature、top-p、top-k、max tokens、stop sequence、chat template 等，模型輸出分布記為：

\[
Y^M \sim \pi_\lambda^M(\cdot\mid x)
\]

sampling protocol 的 expected score 是：

\[
g^M_{\text{sample}}(x)
=
\mathbb{E}_{Y\sim\pi_\lambda^M(\cdot\mid x)}[S(x,Y)]
\]

對 binary correctness，固定 item \(x_i\) 的 pass@1 probability 是：

\[
\theta_i^M
=
\Pr(Z_{ij}^M=1\mid x_i)
\]

若每題 sample \(r\) 次，並用 mean aggregation：

\[
\hat\theta_i^M
=
\frac{1}{r}\sum_{j=1}^{r}Z_{ij}^M
\]

這仍然是在估 pass@1 probability。也就是說：

> 若 aggregation 是 mean-of-r，增加 \(r\) 只是降低單題 pass@1 probability 的 Monte Carlo noise，不會把 estimand 變成 pass@r。

### 4.3 Single-sample stochastic run

若每題只 sample 一次，即 \(r=1\)，則：

\[
Y_i^M=Z_{i1}^M
\]

這是 sampling pass@1 的一個 noisy observation。

如果你固定 seed，這很適合做工程 regression test；但要注意：單次 stochastic run 同時包含 item uncertainty 與 decoding randomness，因此不要過度解讀非常小的差異。

### 4.4 Pass@k / Any-correct

若每題生成 \(k\) 次，只要一次答對就算該題成功：

\[
Y_i^M
=
\mathbf{1}\left(\sum_{j=1}^{k}Z_{ij}^{M}\ge 1\right)
\]

這時候 item-level estimand 不是 pass@1，而是：

\[
\phi_i^M(k)
=
\Pr(\text{k 次中至少一次正確}\mid x_i)
\]

如果每次 attempt iid，且單次正確率是 \(\theta_i^M\)，則：

\[
\phi_i^M(k)
=1-(1-\theta_i^M)^k
\]

code-generation benchmark 常見的 pass@k estimator 是：

\[
\widehat{\mathrm{pass@}k}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
\]

其中：

- \(n\)：實際生成 samples 數；
- \(c\)：其中正確的 samples 數；
- 需要 \(n\ge k\)。

此時 \(k\) 或 \(n\) 是 protocol 的一部分，不只是 variance reduction knob。

### 4.5 Majority vote / self-consistency

若每題 sample \(r\) 次，先抽取每次回答的 final answer，再取眾數：

\[
\tilde a_i^M
=\mathrm{mode}(a_{i1}^M,\dots,a_{ir}^M)
\]

item-level score 是：

\[
Y_i^M=S(x_i,\tilde a_i^M)
\]

這是 majority-vote protocol 的 performance，不是 pass@1，也不是 pass@r。

### 4.6 Best-of-n / judge-selected

若每題生成 \(n\) 個候選，然後由 verifier、reward model、judge 或 rule 選出最佳回答，則：

\[
Y_i^M
=\mathrm{aggregate}(Z_{i1}^M,\dots,Z_{in}^M)
\]

其中 `aggregate` 可以是：

- max；
- mean；
- verifier-selected；
- judge-selected；
- custom rubric aggregation。

重點仍然一樣：**先定義 item-level score，再做統計推斷。**

---

## 5. 單一模型 performance 與 confidence interval

假設已經有 item-level scores：

\[
Y_1^M,
Y_2^M,
\dots,
Y_m^M
\]

單一模型 performance estimator 是：

\[
\hat\mu^M
=
\frac{1}{m}\sum_{i=1}^{m}Y_i^M
\]

### 5.1 Parametric / t-based CI

令 sample variance：

\[
\hat\sigma_M^2
=
\frac{1}{m-1}\sum_{i=1}^{m}(Y_i^M-\hat\mu^M)^2
\]

standard error：

\[
\widehat{SE}(\hat\mu^M)
=
\frac{\hat\sigma_M}{\sqrt m}
\]

\((1-\alpha)\) confidence interval：

\[
\hat\mu^M
\pm
 t_{m-1,1-\alpha/2}\widehat{SE}(\hat\mu^M)
\]

這對 continuous score 或 item 數較多時通常可用。

> 注意：對 binary score，這個 t-based interval 本質上是 Wald interval。當 \(\hat\mu\) 接近 0 或 1（例如 >0.9 的高分 benchmark）或 \(m\) 偏小時，Wald 與 percentile bootstrap 都會 undercover（\(\hat\mu\in\{0,1\}\) 時退化成零寬度區間）。單一模型的 binary CI 建議改用 Wilson 或 Jeffreys interval；paired difference 的對應注意事項見 §6.5。

### 5.2 Item bootstrap CI

更通用的做法是 item bootstrap：

1. 從 \(\{1,\dots,m\}\) 有放回抽樣 \(m\) 個 item indices。
2. 用抽到的 items 重算：

   \[
   \hat\mu^{M,*(b)}
   \]

3. 重複 \(B\) 次。
4. 用 bootstrap distribution 取 percentile CI，例如 2.5% 與 97.5%。

這種方法適合：

- binary score；
- continuous score；
- heavy-tailed judge score；
- pass@k item score；
- majority vote item score；
- benchmark framework 預先輸出的任意 item-level metric。

---

## 6. 兩模型比較：paired difference 才是正式目標

### 6.1 Paired difference 的定義

對同一個 item \(i\)，定義：

\[
D_i
=
Y_i^B-Y_i^A
\]

整體差異估計量：

\[
\hat\Delta
=
\frac{1}{m}\sum_{i=1}^{m}D_i
\]

如果 metric 是 accuracy / score，通常越高越好：

- \(\hat\Delta>0\)：B 比 A 好。
- \(\hat\Delta<0\)：B 比 A 差。

如果 metric 是 loss / error rate，方向要反過來解讀。

### 6.2 Paired t CI

令：

\[
\bar D=\hat\Delta
\]

\[
\hat\sigma_D^2
=
\frac{1}{m-1}\sum_{i=1}^{m}(D_i-\bar D)^2
\]

則：

\[
\widehat{SE}(\hat\Delta)
=
\frac{\hat\sigma_D}{\sqrt m}
\]

paired difference 的 \((1-\alpha)\) CI 是：

\[
CI_{1-\alpha}(\Delta)
=
\bar D
\pm
 t_{m-1,1-\alpha/2}
\frac{\hat\sigma_D}{\sqrt m}
\]

正式判斷：

\[
0\notin CI_{1-\alpha}(\Delta)
\]

才表示在該 confidence level 下，差異可被統計上辨識。

### 6.3 Paired bootstrap CI

對 paired difference 做 bootstrap：

1. 對 item indices \(\{1,\dots,m\}\) 有放回抽樣 \(m\) 個。
2. 對每次 bootstrap sample 計算：

   \[
   \hat\Delta^{*(b)}
   =
   \frac{1}{m}\sum_{i\in\mathcal I_b^*}D_i
   \]

3. 重複 \(B\) 次。
4. 用 \(\{\hat\Delta^{*(b)}\}_{b=1}^{B}\) 的 quantiles 建立 CI。

這是實務上最推薦的做法，因為它直接保留 paired structure。

### 6.4 為什麼不能看兩個模型各自的 CI 是否 overlap？

很多人會先算：

\[
CI(\mu^A),\quad CI(\mu^B)
\]

然後看兩條 CI 是否重疊。這不是 paired comparison 的正式檢定。

原因是：

\[
\mathrm{Var}(\hat\mu^B-\hat\mu^A)
=
\mathrm{Var}(\hat\mu^B)
+
\mathrm{Var}(\hat\mu^A)
-2\mathrm{Cov}(\hat\mu^B,\hat\mu^A)
\]

在 paired benchmark 裡，兩模型是在同一批題目上測試。題目難度會同時影響兩模型，因此 \(\hat\mu^A\) 與 \(\hat\mu^B\) 通常正相關：

\[
\mathrm{Cov}(\hat\mu^B,\hat\mu^A)>0
\]

所以 paired difference 的 variance 會比 unpaired comparison 小。

結論：

- 兩個單獨 CI 不重疊，通常暗示差異很明顯。
- 兩個單獨 CI 重疊，不代表 paired difference 不顯著。
- 正式判斷應該看：

  \[
  CI(\mu^B-\mu^A)
  \]

  是否包含 0。

### 6.5 Binary paired score 的特殊情況：McNemar 觀點

如果 \(Y_i^A,Y_i^B\in\{0,1\}\)，則 paired difference：

\[
D_i\in\{-1,0,1\}
\]

可以定義 discordant counts：

\[
n_{10}=\#\{Y_i^A=1,Y_i^B=0\}
\]

\[
n_{01}=\#\{Y_i^A=0,Y_i^B=1\}
\]

差異估計為：

\[
\hat\Delta
=
\frac{n_{01}-n_{10}}{m}
\]

McNemar test 的 null 是：

\[
H_0:p_{01}=p_{10}
\]

也就是兩個模型「A 對 B 錯」與「A 錯 B 對」的機率相同。這是 binary paired testing 的經典做法。

不過在 general benchmark implementation 裡，paired bootstrap on \(D_i\) 通常更通用，因為它同時支援 binary 與 continuous metrics。

> 注意（2026-07-04 review 補充）：當 \(Y_i\in\{0,1\}\) 且兩模型高度一致時，paired t 與 percentile bootstrap 的有效資訊量是 discordant 題數 \(n_{01}+n_{10}\)，不是 \(m\)。discordant 數很小（例如 <25）時 CLT 近似不可靠：如 \(m=1000\)、\(n_{01}=5\)、\(n_{10}=0\)，paired t 與 bootstrap 會宣告顯著（p≈0.025），但 exact McNemar 雙尾 p=0.0625。此時應改用 exact McNemar（或 mid-p binomial test），並一律報告 \((n_{01},n_{10})\)。特別地，若所有 \(D_i=0\)，paired t 與 bootstrap 會給出寬度為 0 的 CI \([0,0]\)；這不能拿來主張 §10.3 的 equivalence——0 個 discordant 只能給出 discordance rate 約 \(3/m\) 的 95% 上界（rule of three），equivalence 判斷應以此上界評估 \(|\Delta|\) 的可能範圍。

---

## 7. Stochastic decoding 的數學建模

### 7.1 Pass@1 probability

在 stochastic decoding 下，對固定 item \(x_i\)，模型 \(M\) 的 pass@1 expected score 是：

\[
\theta_i^M
=
\mathbb E[Z_{ij}^M\mid x_i]
\]

binary correctness 時：

\[
Z_{ij}^M\mid x_i
\sim
\mathrm{Bernoulli}(\theta_i^M)
\]

因此：

\[
\mathrm{Var}(Z_{ij}^M\mid x_i)
=
\theta_i^M(1-\theta_i^M)
\]

### 7.2 Item-level true difference

定義固定 item 上的 true expected difference：

\[
d_i
=
\theta_i^B-\theta_i^A
\]

task-level difference 是：

\[
\Delta
=
\mathbb E_{X\sim P_{\text{task}}}[d(X)]
\]

fixed-dataset target 是：

\[
\Delta_{\mathcal D}
=
\frac{1}{m}\sum_{i=1}^{m}d_i
\]

### 7.3 用有限 repeated decoding 估 \(d_i\)

若每題每模型 decode \(r\) 次：

\[
\hat\theta_i^M
=
\frac{1}{r}\sum_{j=1}^{r}Z_{ij}^{M}
\]

\[
\hat d_i
=
\hat\theta_i^B-
\hat\theta_i^A
\]

展開：

\[
\hat d_i
=
\frac{1}{r}\sum_{j=1}^{r}Z_{ij}^{B}
-
\frac{1}{r}\sum_{j=1}^{r}Z_{ij}^{A}
\]

條件期望：

\[
\mathbb E[\hat d_i\mid x_i]
=
\theta_i^B-
\theta_i^A
=d_i
\]

所以 \(\hat d_i\) 是 \(d_i\) 的 unbiased estimator。

### 7.4 條件變異：允許兩模型 sampling noise 相關

原本最簡化的推導常假設兩模型 decoding noise 條件獨立。但實務上，你可能使用相同 seed、相同 prompt order、甚至 common random numbers。因此更一般地寫：

\[
\mathrm{Var}(\hat d_i\mid x_i)
=
\mathrm{Var}(\hat\theta_i^B\mid x_i)
+
\mathrm{Var}(\hat\theta_i^A\mid x_i)
-2\mathrm{Cov}(\hat\theta_i^B,\hat\theta_i^A\mid x_i)
\]

若兩模型各自有 \(r\) 次 repeated decoding，且第 \(j\) 次可配對，則可寫成：

\[
\mathrm{Var}(\hat d_i\mid x_i)
=
\frac{1}{r}
\left(
\sigma_{B,i}^2
+
\sigma_{A,i}^2
-2\sigma_{AB,i}
\right)
\]

其中：

\[
\sigma_{M,i}^2
=
\mathrm{Var}(Z_{ij}^{M}\mid x_i)
\]

\[
\sigma_{AB,i}
=
\mathrm{Cov}(Z_{ij}^{B},Z_{ij}^{A}\mid x_i)
\]

若兩模型 sampling noise 條件獨立，則：

\[
\sigma_{AB,i}=0
\]

binary correctness 下：

\[
\mathrm{Var}(\hat d_i\mid x_i)
=
\frac{
\theta_i^B(1-\theta_i^B)
+
\theta_i^A(1-\theta_i^A)
}{r}
\]

如果使用 common random numbers 或 seed-paired sampling，且 \(\sigma_{AB,i}>0\)，則 variance 會更小。這也是 paired randomization 的優點。

### 7.5 Law of total variance

令：

\[
\tau_i^2
=
\sigma_{B,i}^2+
\sigma_{A,i}^2-2\sigma_{AB,i}
\]

則：

\[
\mathrm{Var}(\hat d_i\mid x_i)
=
\frac{\tau_i^2}{r}
\]

使用 total variance：

\[
\mathrm{Var}(\hat d_i)
=
\mathrm{Var}(\mathbb E[\hat d_i\mid x_i])
+
\mathbb E[\mathrm{Var}(\hat d_i\mid x_i)]
\]

第一項：

\[
\mathrm{Var}(\mathbb E[\hat d_i\mid x_i])
=
\mathrm{Var}(d_i)
=
\sigma_d^2
\]

第二項：

\[
\mathbb E[\mathrm{Var}(\hat d_i\mid x_i)]
=
\mathbb E\left[\frac{\tau_i^2}{r}\right]
=
\frac{\sigma_\epsilon^2}{r}
\]

其中：

\[
\sigma_\epsilon^2
=
\mathbb E[\tau_i^2]
\]

因此：

\[
\mathrm{Var}(\hat d_i)
=
\sigma_d^2+
\frac{\sigma_\epsilon^2}{r}
\]

整體 estimator 是：

\[
\hat\Delta
=
\frac{1}{m}\sum_{i=1}^{m}\hat d_i
\]

若 items 之間近似獨立：

\[
\mathrm{Var}(\hat\Delta)
\approx
\frac{1}{m}
\left(
\sigma_d^2+
\frac{\sigma_\epsilon^2}{r}
\right)
\]

也就是：

\[
\boxed{
\mathrm{Var}(\hat\Delta)
\approx
\frac{\sigma_d^2}{m}
+
\frac{\sigma_\epsilon^2}{mr}
}
\]

這個式子說明兩種不確定性來源：

| 項目 | 來源 | 如何降低 |
|---|---|---|
| \(\sigma_d^2/m\) | 題目差異、finite benchmark uncertainty | 增加 item 數 \(m\) |
| \(\sigma_\epsilon^2/(mr)\) | finite repeated decoding noise | 增加 item 數 \(m\) 或 repeated decoding 次數 \(r\) |

---

## 8. 固定推論預算下的配置：為什麼通常先增加題目數

假設兩模型每次 inference 成本相同，總推論預算是 \(N\)，且每題每模型 decode \(r\) 次。總推論次數：

\[
N=2mr
\]

因此：

\[
m=\frac{N}{2r}
\]

把它代入 variance decomposition：

\[
\mathrm{Var}(\hat\Delta)
\approx
\frac{\sigma_d^2}{m}
+
\frac{\sigma_\epsilon^2}{mr}
\]

第一項：

\[
\frac{\sigma_d^2}{m}
=
\frac{\sigma_d^2}{N/(2r)}
=
\frac{2r\sigma_d^2}{N}
\]

第二項：

\[
\frac{\sigma_\epsilon^2}{mr}
=
\frac{\sigma_\epsilon^2}{(N/(2r))r}
=
\frac{2\sigma_\epsilon^2}{N}
\]

合併：

\[
\boxed{
\mathrm{Var}(\hat\Delta)
\approx
\frac{2}{N}
\left(
 r\sigma_d^2+
 \sigma_\epsilon^2
\right)
}
\]

在固定 \(N\) 下，\(\sigma_\epsilon^2\) 那項不隨 \(r\) 改變，但 \(r\sigma_d^2\) 會隨 \(r\) 增加。

只要：

\[
\sigma_d^2>0
\]

增加 \(r\) 就會犧牲 item 數，導致整體 comparison variance 變大。

因此，在以下條件成立時：

- 目標是比較整體平均 pass@1 difference；
- 還有更多 benchmark items 可以評估；
- 每個 item 成本相近；
- 每題每模型使用相同 \(r\)；
- \(r\) 不是 protocol 本身的一部分；

通常最佳策略是：

\[
\boxed{r=1，盡量增加 m}
\]

### 8.1 什麼時候 \(r>1\) 是合理的？

1. **benchmark 題目數已經用完。**\
   如果 \(m\le M_{\max}\)，且完整 benchmark 已經跑完，額外預算不能換成更多 independent items，就可以增加 \(r\)。

2. **你想估單題 behavior。**\
   例如找 quantization-sensitive items、分析某題是否只是 sampling accident、估 item-level pass@1 probability。

3. **你要估的 protocol 本身需要多次 generation。**\
   pass@k、majority vote、self-consistency、best-of-n 都屬於這種情況。

4. **decoding randomness 非常大，而且 deployment 本身就是 stochastic。**\
   這時可以用 repeated decoding 壓低每題的 Monte Carlo 噪音；對 superpopulation \(\Delta\) 的 CI 仍用聚合後的 item bootstrap（它已同時涵蓋兩種不確定性，見 §9.2 勘誤），如需單獨呈現 decoding noise，另報 fixed-items 的 inner-only / multi-seed 變異。

### 8.2 單題估計所需的 \(r\)

binary correctness 下：

\[
\mathrm{Var}(\hat\theta_i^M\mid x_i)
=
\frac{\theta_i^M(1-\theta_i^M)}{r}
\le
\frac{1}{4r}
\]

所以：

\[
SE(\hat\theta_i^M\mid x_i)
\le
\frac{1}{2\sqrt r}
\]

| \(r\) | 單模型 item-level pass@1 最壞情況 SE |
|---:|---:|
| 1 | 0.500 |
| 5 | 0.224 |
| 10 | 0.158 |
| 25 | 0.100 |
| 100 | 0.050 |

若比較兩模型 item-level difference，且兩模型 decoding noise 條件獨立：

\[
\mathrm{Var}(\hat d_i\mid x_i)
\le
\frac{1/4+1/4}{r}
=
\frac{1}{2r}
\]

\[
SE(\hat d_i\mid x_i)
\le
\sqrt{\frac{1}{2r}}
\]

| \(r\) | item-level difference 最壞情況 SE |
|---:|---:|
| 1 | 0.707 |
| 5 | 0.316 |
| 10 | 0.224 |
| 30 | 0.129 |
| 100 | 0.071 |

因此：

- 整體 model comparison：通常 \(r=1\) 最有效。
- 單題 behavior diagnosis：通常需要 \(r=10\sim30\)，甚至更高。

---

## 9. Bootstrap 應如何設計

### 9.1 Item bootstrap

適用場景：

- greedy / deterministic evaluation；
- 每題已經有 item-level score；
- repeated decoding 已經先聚合成 item-level score；
- 要對 superpopulation estimand \(\Delta\) 建 CI——注意：聚合分數的 item bootstrap **同時**反映 finite item uncertainty 與殘餘 decoding noise（\(\mathrm{Var}(\hat d_i)=\sigma_d^2+\sigma_\epsilon^2/r\)，兩者都在 \(\hat d_i\) 的離散度裡），不是「只有 item uncertainty」（2026-07-04 勘誤）。

演算法：

```text
Input:
  item-level paired differences D_1, ..., D_m
  bootstrap iterations B

For b = 1,...,B:
  sample m item indices with replacement
  compute Delta_star[b] = mean(D_i over sampled items)

CI = quantile(Delta_star, [alpha/2, 1-alpha/2])
```

這是 paired comparison 的預設做法。

### 9.2 Nested bootstrap

（2026-07-04 勘誤：本段已改寫）

若目標是 superpopulation 的 \(\Delta\)，對聚合後的 \(\hat d_i\) 做外層 item bootstrap 就已經同時反映兩種不確定性：由 §7.5，\(\mathrm{Var}(\hat d_i)=\sigma_d^2+\sigma_\epsilon^2/r\)，觀察到的 \(\hat d_i\) 之間的離散度本身就包含殘餘 decoding noise，因此外層 resampling 的 CI 是 \(\mathrm{Var}(\hat\Delta)\) 的一致估計，不需要再加內層。若在外層之外再對 item 內 attempts resample（nested bootstrap），within-item 成分會被重複計入（大 \(r\) 時 \(\sigma_\epsilon^2/r\) 約變成 \(2\sigma_\epsilon^2/r\)），得到偏保守、過寬的 CI。

內層 resampling 的正確用途是：

1. 只做內層、不動 item 集合，用來估「固定這批題目」時純 decoding noise 的不確定性（fixed-dataset estimand）；
2. 分解並分別報告兩種噪音成分；
3. aggregation 非線性（majority vote、pass@k 重算）時近似 protocol 在不同 random draws 下的變異。

多 seed 重跑整個 evaluation 是隔離 decoding variability 最乾淨的方法。

以下記錄 nested bootstrap 的演算法，供上述三種用途使用（勿用來替 superpopulation \(\Delta\) 建 CI）：

1. 外層 resample items。
2. 對每個被抽到的 item，在 item 內 resample decoding attempts。
3. 重算：

   \[
   \hat\theta_i^{M,*}
   =
   \frac{1}{r}\sum_{j\in\mathcal J_i^*}Z_{ij}^{M}
   \]

4. 重算：

   \[
   \hat d_i^*
   =
   \hat\theta_i^{B,*}-\hat\theta_i^{A,*}
   \]

5. 重算：

   \[
   \hat\Delta^*
   =
   \frac{1}{m}\sum_i\hat d_i^*
   \]

6. 重複多次，取 percentile CI。

### 9.3 Nested bootstrap 要保留 pairing structure

若兩模型的 samples 是 seed-paired，例如第 \(j\) 次用同一組 random seed 或 common random numbers，inner bootstrap 應該 resample paired attempts：

\[
(Z_{ij}^{A}, Z_{ij}^{B})
\]

而不是分別對 A、B 的 attempts 獨立 resample。這樣才能保留 conditional covariance。

若兩模型的 samples 本來就是獨立產生，則可以在 item 內對 A、B 分別 resample。

### 9.4 何時不需要 nested bootstrap？

如果你已經明確把 \(r\)-sample protocol 定義成 benchmark protocol，例如 pass@10 或 majority vote@20，而且只想比較「這個 protocol 的 observed item-level scores」，那麼可以先把每題聚合成 \(Y_i^M\)，再做 item bootstrap。

如果你還想估計這個 stochastic protocol 在不同 random draws 下的 variability，才需要 nested bootstrap 或多 seed evaluation。

---

## 10. Practical significance：不要只問是否顯著

在 quantization 研究中，樣本數很大時，非常小的差異也可能顯著。反過來，小 benchmark 上沒有顯著差異，也不代表兩者真的等價。

因此應該區分：

1. **Statistical significance**：差異是否可被統計上辨識？
2. **Practical significance**：差異是否大到有實務意義？
3. **Non-inferiority / equivalence**：quantized model 是否足夠接近 full-precision model？

### 10.1 Accuracy-like metric 的 non-inferiority

假設 metric 越高越好，且 B 是 quantized model。定義：

\[
\Delta = \mu^B - \mu^A
\]

若你能接受 quantized model 最多退化 \(\epsilon>0\)，則 non-inferiority 的目標是：

\[
\Delta > -\epsilon
\]

用 CI 判斷時，若：

\[
\mathrm{lower}(CI(\Delta)) > -\epsilon
\]

即可主張 B 在 threshold \(\epsilon\) 下 non-inferior。

### 10.2 Loss-like metric 的 non-inferiority

若 metric 越低越好，例如 loss、error rate、NLL，定義：

\[
\Delta = \mu^B - \mu^A
\]

此時 B 退化代表 \(\Delta>0\)。若最多接受退化 \(\epsilon>0\)，則希望：

\[
\Delta < \epsilon
\]

若：

\[
\mathrm{upper}(CI(\Delta)) < \epsilon
\]

即可主張 B 在 threshold \(\epsilon\) 下 non-inferior。

### 10.3 Equivalence

若想主張兩者等價，通常要事先定義 equivalence margin \(\epsilon\)，並要求：

\[
CI(\Delta)
\subset (-\epsilon,\epsilon)
\]

也就是整個 CI 都落在 practically negligible 的範圍內。

重要提醒：

> 沒有顯著差異不等於等價。\
> 要主張等價，需要 equivalence / non-inferiority 設計。

---

## 11. 任務類型的處理原則

### 11.1 Multiple-choice tasks

通常 item-level score 是：

\[
Y_i^M\in\{0,1\}
\]

如果 benchmark 已經有 `metric.acc` 或類似欄位，直接使用。

對 deterministic log-likelihood ranking 類 benchmark，通常沒有 decoding randomness，因此 paired item bootstrap 很適合。

### 11.2 Exact match tasks

exact match 常見於 QA、code、structured output。item score 通常也是：

\[
Y_i^M\in\{0,1\}
\]

如果 benchmark framework 已做 normalization，應直接使用 framework 的 item-level metric，不要自己重算。

### 11.3 Free-form generative tasks

free-form task 的 item score 可以是：

\[
Y_i^M\in\mathbb R
\]

例如：

- judge score；
- rubric score；
- partial credit；
- reward model score；
- human evaluation score。

paired framework 一樣成立，只要每個 item 最後有一個可比較的 item-level score。

---

## 12. Token usage 也可以做 paired analysis

如果 benchmark record 中有 output token usage，例如：

```text
model_response.output_tokens
```

可以把 token usage 也整理成 item-level quantity：

\[
T_i^M
\]

例如：

- 每題總 output tokens；
- 每題平均 output tokens；
- 每次 inference 平均 output tokens。

比較兩模型 token usage 時，也應用 paired difference：

\[
D_i^{\text{tok}}
=
T_i^B-T_i^A
\]

\[
\hat\Delta_{\text{tok}}
=
\frac{1}{m}\sum_{i=1}^{m}D_i^{\text{tok}}
\]

然後用 paired CI / paired bootstrap。

這對 quantization 很有價值，因為有些方法可能 accuracy 接近，但輸出長度、early stop behavior 或 generation cost 不同。

---

## 13. 建議的 quantization evaluation protocol

### 13.1 Primary benchmark-standard result

目的：

- 對齊既有 benchmark / leaderboard；
- 方便重現；
- 適合 regression test；
- 移除 stochastic decoding noise。

建議：

- 使用 greedy 或 benchmark 官方 protocol。
- 每題每模型 \(r=1\)。
- 使用 benchmark 儲存的 item-level score。
- 使用 paired item-level comparison。
- 用 paired item bootstrap 建立 \(CI(\Delta)\)。

報告：

\[
\hat\Delta_{\text{official}}
=
\frac{1}{m}\sum_{i=1}^{m}
\left(Y_i^B-Y_i^A\right)
\]

以及：

\[
CI(\Delta_{\text{official}})
\]

### 13.2 Deployment-aligned stochastic result

目的：

- 估計實際 deployment decoding policy 下的 performance；
- 避免 greedy protocol 與 deployment protocol mismatch。

建議：

- 固定完整 decoding policy \(\lambda\)：temperature、top-p、top-k、max tokens、stop、prompt template、system prompt。
- 若題目數足夠，優先增加 item 數，通常 \(r=1\)。
- 若題目數有限或 stochasticity 大，再增加 \(r\)。
- 若有 repeated decoding，報告聚合後的 item bootstrap CI（已同時涵蓋 item 與 decoding 不確定性，見 §9.2 勘誤）；如需單獨呈現 decoding noise，另報 fixed-items 的 inner-only / multi-seed 變異。

報告：

\[
\hat d_i
=
\frac{1}{r}\sum_{j=1}^{r}Z_{ij}^{B}
-
\frac{1}{r}\sum_{j=1}^{r}Z_{ij}^{A}
\]

\[
\hat\Delta_{\text{deploy}}
=
\frac{1}{m}\sum_{i=1}^{m}\hat d_i
\]

### 13.3 Robustness / diagnosis subset

目的：

- 找出 quantization-sensitive items；
- 觀察 degradation 是否集中在特定題型；
- 區分 true degradation 與 sampling accident；
- 分析 item-level variance。

建議：

- 選一個 subset；
- 每題每模型使用較大的 \(r\)，例如 \(10\sim30\)；
- 用 mean aggregation 估 item-level pass@1 probability；
- 使用 nested bootstrap 或 item-level variance 分析；
- 不要把 repeated responses 攤平成 independent items。

---

## 14. Data alignment 與 validation checklist

比較兩模型前，必須先確保 item 對齊。

### 14.1 建議對齊 key

優先使用：

```text
task_name + item_id
```

或：

```text
doc.task_name + doc.id
```

若 framework 沒有穩定 item id，應在資料讀取層建立 stable key。

### 14.2 必做 validation

1. 兩模型的 task set 一致。
2. 每個 task 的 item 數一致。
3. item key 完全對齊。
4. metric name 一致。
5. scoring direction 一致，例如越高越好或越低越好。
6. decoding protocol 一致，除非研究問題就是比較 protocol。
7. prompt template / chat template / system prompt 一致。
8. num samples / \(r\) 一致，或明確記錄不同 \(r_i\)。
9. 如果比較 token usage，需確認 output token 計算方式一致。
10. 若有 stochastic decoding，需記錄 seeds、temperature、top-p、top-k、max tokens。

---

## 15. 建議輸出格式

### 15.1 單一模型結果

```json
{
  "metric_name": "acc",
  "model": "B",
  "protocol": "greedy",
  "point_estimate": 0.733,
  "n_items": 1000,
  "ci": {
    "method": "item_bootstrap",
    "confidence_level": 0.95,
    "lower": 0.704,
    "upper": 0.761,
    "bootstrap_iters": 5000
  },
  "token_usage": {
    "total_output_tokens": 123456,
    "mean_output_tokens_per_item": 123.46,
    "mean_output_tokens_per_inference": 123.46
  }
}
```

### 15.2 兩模型 paired comparison

```json
{
  "metric_name": "acc",
  "baseline_model": "A",
  "candidate_model": "B",
  "protocol": "greedy",
  "score_direction": "higher_is_better",
  "score_A": 0.748,
  "score_B": 0.733,
  "delta_B_minus_A": -0.015,
  "n_items": 1000,
  "ci_delta": {
    "method": "paired_item_bootstrap",
    "confidence_level": 0.95,
    "lower": -0.031,
    "upper": 0.003,
    "bootstrap_iters": 5000
  },
  "decision": {
    "statistically_distinguishable_from_zero": false,
    "reason": "CI(delta) contains 0"
  },
  "token_usage_delta": {
    "delta_total_output_tokens_B_minus_A": -4200,
    "delta_mean_output_tokens_per_inference_B_minus_A": -4.2
  }
}
```

---

## 16. 實務決策表

| 研究目標 | 建議做法 |
|---|---|
| 比較兩個 quantization algorithms 的 benchmark-standard 表現 | greedy / official protocol，item-level paired bootstrap |
| 比較 deployment sampling 下的 pass@1 | 固定 deployment decoding policy，item-level paired comparison |
| 固定預算下最大化整體 comparison power | 若還能增加題目數，通常 \(r=1\)，優先增加 \(m\) |
| benchmark 題目數已用完但預算有剩 | 增加 \(r\)，用 nested bootstrap 反映 decoding noise |
| 想估 pass@k | \(k\) 是 protocol，本身需要足夠 samples |
| 想估 majority vote / self-consistency | \(r\) 是 protocol，本身不能視為單純 variance reduction |
| 想找 quantization-sensitive items | subset 上增加 \(r\)，做 item-level diagnosis |
| 想主張 quantized model 沒有明顯退化 | 做 non-inferiority / equivalence，而不是只說 non-significant |
| 同時比較很多 tasks / algorithms | 先指定 primary metric；必要時做 multiple-comparison control |

---

## 17. 常見錯誤與修正

### 錯誤 1：把每個 sampled response 當獨立樣本

錯誤做法：

\[
\{Z_{ij}\}_{i=1,j=1}^{m,r}
\]

全部攤平成 \(mr\) 個 iid samples。

修正：

先做 item-level aggregation：

\[
Y_i=\mathrm{aggregate}(Z_{i1},\dots,Z_{ir})
\]

再對 \(Y_i\) 做統計推斷。

### 錯誤 2：看兩個模型各自 CI 是否 overlap

錯誤做法：

```text
CI(A) overlaps CI(B), so no difference.
```

修正：

看 paired difference CI：

\[
CI(B-A)
\]

是否包含 0。

### 錯誤 3：沒有先定義 estimand

錯誤做法：

```text
我們 sample 10 次，所以算 pass@10 / pass@1 / majority vote 都差不多。
```

修正：

先說清楚 protocol：

- mean-of-10：估 pass@1 probability；
- any-correct among 10：估 pass@10；
- majority vote among 10：估 self-consistency / majority protocol；
- best-of-10：估 best-of-n protocol。

### 錯誤 4：沒有區分 statistical significance 與 practical significance

錯誤做法：

```text
p < 0.05，所以 B 一定比較好。
```

修正：

同時報告：

- point estimate；
- CI；
- practical threshold；
- non-inferiority / equivalence 判斷。

### 錯誤 5：只報 greedy，卻聲稱代表 deployment sampling

修正：

Greedy result 可以是 primary benchmark result，但若 deployment 使用 sampling，應另外做 deployment-aligned evaluation。

---

## 18. 最小可用演算法

### 18.1 Paired bootstrap for item-level metrics

```python
import numpy as np


def paired_bootstrap_ci(y_a, y_b, B=5000, alpha=0.05, rng=None):
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    assert y_a.shape == y_b.shape

    rng = np.random.default_rng(rng)
    d = y_b - y_a
    m = len(d)

    delta_hat = d.mean()
    boot = np.empty(B)

    for b in range(B):
        idx = rng.integers(0, m, size=m)
        boot[b] = d[idx].mean()

    lower, upper = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    return {
        "delta_hat": float(delta_hat),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_std": float(boot.std(ddof=1)),
        "significant": bool(lower > 0 or upper < 0),
    }
```

### 18.2 Paired t CI for item-level metrics

```python
import numpy as np
from scipy.stats import t


def paired_t_ci(y_a, y_b, alpha=0.05):
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    assert y_a.shape == y_b.shape

    d = y_b - y_a
    m = len(d)
    delta_hat = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(m)
    q = t.ppf(1 - alpha / 2, df=m - 1)

    return {
        "delta_hat": float(delta_hat),
        "se": float(se),
        "ci_lower": float(delta_hat - q * se),
        "ci_upper": float(delta_hat + q * se),
    }
```

---

## 19. 最終整理

可以把整份文件濃縮成一個 evaluation rule：

> 先定義 protocol 與 estimand；把每題整理成 item-level score；比較模型時對同一題做 paired difference；用 paired difference 的 CI 或 paired bootstrap 判斷差異；不要用兩個模型各自的 CI overlap 取代 paired test。

對 quantization evaluation，建議的主要報告方式是：

\[
\hat\Delta
=
\frac{1}{m}\sum_{i=1}^{m}(Y_i^B-Y_i^A)
\]

搭配：

\[
CI(\Delta)
\]

並清楚說明：

- metric 是越高越好還是越低越好；
- protocol 是 greedy、pass@1 sampling、pass@k、majority vote 或 deployment policy；
- CI 是 item bootstrap 還是 nested bootstrap；
- 是否有 practical threshold；
- 是否做 non-inferiority / equivalence 判斷。

這樣比較結果才不只是 leaderboard number，而是可以回答：

> 這個 quantization algorithm 的觀察差異，究竟是 benchmark sampling noise、decoding noise，還是真正可被統計上辨識且有實務意義的模型差異？
