# Sequential-prefix power analysis for skymizer

版本：2026-08-29

定位：說明 `stats/cli/power_analysis.py` 的 estimand、抽樣單位、模擬方法、
限制與使用方式。這份文件取代「把 `--num-eval-tokens` 當作 IID repeats，
所以 variance 按 1/K 收斂」的規劃方法。

---

## 1. 結論

`--num-eval-tokens = K` 是 ordered-prefix estimand 的定義，不是樣本數。
正確的獨立抽樣單位是 item；同一 item 內的 tokens 保留原順序和全部相依性。

skymizer 的事前規劃是二維問題：

    Power(N, K; Δ_K)

其中：

- N 是未來會收集的獨立 items 數；
- K 是每個 item 最多納入的 answer-token prefix；
- Δ_K 是事先指定的 cap-level SESOI；
- effect、variance、length saturation 與 power 都可以隨 K 改變；
- power 對 K 不保證單調。

工具的主方法：

1. 從 pilot 的逐 token VLMK records 重算每個 cap 的 item score；
2. 將 pilot residual 中心化到外部指定的 SESOI；
3. 有放回重抽完整 paired items，絕不重抽 tokens；
4. 對每個 future sample 執行正式預設的 paired Student-t endpoint；
5. 以「顯著且方向正確」的比例估 power；
6. 另作 null-centered simulation，檢查 Type-I error；
7. 報告 Monte-Carlo interval 與 pilot nuisance uncertainty。

若沒有可信 SESOI，工具只輸出 CI half-width 與 MDE，不會將 pilot observed
effect 偷換成 prospective power。

---

## 2. Estimand

令第 i 個 item 的 answer 長度為 T_i，第 t 個位置的 paired
candidate-minus-baseline metric difference 為：

    x_it = m_B,it - m_A,it

cap K 下保留的位置數：

    L_i(K) = min(K, T_i)

item prefix score：

    D_i(K) = [1 / L_i(K)] Σ_{t=1..L_i(K)} x_it

### 2.1 Item weighting

預設 confirmatory estimand：

    θ_K = E[D_i(K)]

items 是抽樣單位。tokens 可以在 item 內任意相關、異方差、非平穩。

### 2.2 Token weighting (descriptive only)

The observed token-weighted corpus summary is `sum(L_i(K) * D_i(K)) / sum(L_i(K))`. It has no paired-test CI, p-value, equivalence decision or power estimate. The earlier ratio-linearized Student-t proposal is superseded by the item-only inference policy in [the comparison guide](../docs/compare.md).

### 2.3 為什麼 variance 不服從 1/K

在 items 等長的簡化情形：

    Var[D_i(K)] = K^-2 Σ_{s=1..K} Σ_{t=1..K} Cov(x_is, x_it)

右側包含全部 cross-position covariance；position mean、variance 與 covariance
也可能隨位置改變。因此增加 K 可能：

- 降低 variance；
- 幾乎不改變 variance；
- 增加 variance；
- 稀釋前段 effect；
- 引入後段 effect；
- 讓 θ_K 翻號。

cap 是 estimator/endpoint choice，不是 precision knob。

---

## 3. 與 benchmark error-bar 文獻的關係

### 3.1 Miller：可借 paired item inference，不可借 repeated-answer 1/K

[Miller, Adding Error Bars to Evals](https://arxiv.org/html/2411.00640)
的 repeated sampling K 是「同一題獨立生成 K 次完整回答」。在 conditional
IID 且同一 latent question score 下，generation noise 才有 σ_i²/K。

skymizer 不同：

- teacher-forced token metric 在 model、item、target 和 knobs 固定後是確定的；
- position t 不是同一 scalar 的重複量測；
- autoregressive prefixes 重疊；
- K 改變 θ_K，不只改變 precision；
- answer lengths 不一。

Miller 可移植的是固定 K 後，以 paired item score D_i(K) 做推斷；不可移植
的是把 answer positions 當 repeated generations。

### 3.2 Baker：ordered-prefix contour 是直接先例

[Baker et al., Power Contours](https://arxiv.org/abs/1902.06122) 的 Iowa
Gambling Task ordered 分析固定取每位 participant 的 `1:K` prefix，只
bootstrap participants；[官方 OSF 程式](https://osf.io/b2wk3/) 也確認不會
重抽或重排 ordered trials。

該例的 power 隨 K 先下降到接近零，再上升，因為 prefix effect 本身在變。
skymizer 借用 whole-item resampling、ordered prefix 與 N × K contour。

但 Baker 將 bootstrap subset 的 observed Cohen's d 代回 analytic power。
這是 pilot-conditioned replication diagnostic，不是有外部 SESOI 的完整
prospective simulation。

### 3.3 Card：生成 future data 後跑正式 test

[Card et al., With Little Power](https://aclanthology.org/2020.emnlp-main.745/)
的框架是宣告 alternative DGP 和 production test，重複生成 future datasets，
再把顯著且方向正確的比例定義為 power。`power_analysis.py` 採用這個框架，
並以 pilot 的 empirical item residuals 作 nuisance distribution。

### 3.4 為何不是 token-axis HAC 或 block bootstrap

moving-block、stationary bootstrap 與一般 HAC 通常需要一條足夠長、
weakly dependent 或 locally stationary 的序列，並需要選 block length 或
bandwidth。skymizer 是多條短、不等長、start/end transient 明顯的 item
trajectories；重排 token blocks 還會破壞 position regime。

更根本的是，這些方法無法消除「K 改變 estimand」的問題。相關限制見
[Künsch 1989](https://doi.org/10.1214/aos/1176347265) 與非平穩 HAC
[Casini 2023](https://doi.org/10.1016/j.jeconom.2022.05.001)。

---

## 4. Prospective alternative

power 的條件事件是：

    P(reject H0 in the correct direction | Δ_K, N, K)

不指定 Δ_K 就沒有單一 power。SESOI 必須來自部署容忍度、品質門檻、
先前獨立研究或其他 domain decision，而不是本 pilot 的 observed effect。

小 pilot 的 observed effect 直接餵入 sample-size calculation 通常會讓後續
研究 underpowered；見
[Albers & Lakens 2018](https://doi.org/10.1016/j.jesp.2017.09.004)。
同批資料的 observed power 也沒有超越 p-value 的新資訊；見
[Hoenig & Heisey 2001](https://doi.org/10.1198/000313001300339897)。

### 4.1 `flat`

每個 cap 分別使用：

    D_i*(K) = D_i(K) - θ_hat_K + Δ

所有 caps 使用同一 cap-level SESOI。這適合比較「若各 endpoint 的實務重要
差異相同，哪個 cap 最有效率？」

限制：每個 K 是分開的 planning scenario，不是一條共同 token DGP。

### 4.2 `pilot`

選 reference cap K0：

    c = Δ_K0 - θ_hat_K0
    D_i*(K) = D_i(K) + c

這等同將整條 token trajectory 加同一常數，因此跨 caps coherent，且 K0
的 effect 正好等於 SESOI。

限制：其他 caps 的 effect profile 繼承 pilot 的 θ_hat_K shape；這是明確的
model-based sensitivity assumption。

---

## 5. Monte-Carlo algorithm

對每個 sample size N：

1. 從 n_pilot 個 items 有放回抽 N 個 indices；
2. 同一組 indices 同時套到所有 caps，形成 common random numbers；
3. 對 cap K 的 centered residual 加上 Δ_K；
4. 執行 paired Student-t interval；
5. 若 CI 排除 0 且 sign 與 Δ_K 相同，記為 detection；
6. 對 centered-at-zero residual 同時計算 null rejection；
7. 重複 `--reps` 次。

Per-cell schema v2 outputs include empirical directional power and wrong-sign rates, their pointwise Wilson MC intervals, null rejection, `plugin_ci_half_width`, `normal_model_mde`, and expected evaluated tokens. The width plugs in pilot SD; it is not an expected future interval width.

A detected null inflation (`null_rejection_mc_lower > nominal alpha`) excludes the cell from every N crossing. Remaining cells have no detected inflation, not a calibration guarantee. The pilot-p10 crossing also requires the ordinary MC lower-bound criterion. These are first evaluated-grid crossings, with no simultaneous guarantee across N or caps.

---

## 6. Pilot nuisance uncertainty

The empirical future-sample simulation conditions on the pilot empirical distribution. Wilson intervals describe finite simulation error, not uncertainty in the pilot distribution.

Outer item resamples estimate SD sensitivity. Gaussian directional power uses `scipy.stats.nct.sf` with a two-sided t critical value; only the assumed-direction rejection tail counts. MDE inverts that same Gaussian model using `scipy.optimize.brentq` in dimensionless noncentrality. These are separate model-based sensitivity quantities, not another empirical power interval. The JSON method is `outer_item_bootstrap_plus_directional_normal_model`.

Original and outer samples share an input-relative numerical-resolution guard. An unresolved original pilot aborts planning; any unresolved outer sample makes its whole nuisance band unavailable. Samples are not dropped or assigned power 1.

The signed SESOI is fixed externally, and the bands do not include dataset/model drift or uncertainty in that choice. See [statistical API references](../docs/statistical-apis.md) for exact calls, assumptions, and independent validation.

---

## 7. CLI

### 7.1 SESOI-driven prospective power

```bash
python3 stats/cli/power_analysis.py \
    --candidate-a outputs/vlm-kld-ref-vs-a \
    --candidate-b outputs/vlm-kld-ref-vs-b \
    --metric kld \
    --weighting item \
    --token-caps 16 32 64 128 256 512 1024 \
    --sample-sizes 25 50 75 100 150 200 300 500 \
    --sesoi -0.0001 \
    --effect-profile flat \
    --target-power 0.80 \
    --reps 5000 \
    --outer-reps 500 \
    --seed 7 \
    --out outputs/vlm-seq-power-kld.md \
    --output-json outputs/vlm-seq-power-kld.json
```

`--sesoi` 是 signed candidate-B minus candidate-A effect。對 lower-is-better
的 KLD，負值表示 B 比 A 更接近 reference。

### 7.2 Coherent pilot-profile sensitivity

```bash
python3 stats/cli/power_analysis.py \
    --candidate-a A --candidate-b B \
    --token-caps 32 64 128 256 512 \
    --sample-sizes 50 100 200 400 \
    --sesoi -0.0001 \
    --effect-profile pilot \
    --reference-cap 256 \
    --out power.md --output-json power.json
```

### 7.3 No SESOI: precision/MDE only

```bash
python3 stats/cli/power_analysis.py \
    --candidate-a A --candidate-b B \
    --token-caps 32 64 128 256 \
    --sample-sizes 50 100 200 400 \
    --out precision.md --output-json precision.json
```

### 7.4 Pipeline script

```bash
# Prospective power
SESOI=-0.0001 TOKEN_CAPS="32 64 128 256 512" \
    ./scripts/06_power_analysis.sh vlm

# Precision/MDE only
./scripts/06_power_analysis.sh vlm
```

---

## 8. Alignment and support contract

planner 重用 `saved_metrics_paired_compare.py` 的正式 guards：

- collection completeness；
- `collect_meta.json` reference/dataset/config alignment；
- common `SKIP_OVER_BUDGET` set；
- complete item-key alignment，除非明確 `--allow-interaction`；
- vocab、npos、n_prefill、n_past_actual、target alignment；
- bit-identical nll_ref、entropy_ref、argmax_ref；
- non-finite score hard failure。

collection-time cap 是 artifact identity。compare-time caps 只能從已保存的
prefix 重算：

- requested K 超過明確的 collection cap：hard fail；
- 所有 pilot items 都短於某個 K：輸出 saturated no-op warning；
- 不會從短 pilot 外推未觀測 token positions。

逐 token VLMK records 已足夠計算所有被保存的 prefixes，不需修改 C++ 格式。

---

## 9. Assumptions and failure modes

### 9.1 Items 必須是獨立抽樣單位

目前 production test 和 planner 都以 item 為 exchangeable unit。若多個 rows
共用同一 image、passage、conversation 或生成 family，它們可能是同一 cluster。
在加入 cluster-aware final inference 前，不應只傳一份 cluster map 給 simulator
就宣稱問題解決：future DGP 和正式 test 的 resampling unit 必須一致。

### 9.2 Pilot 必須能代表 future collection

whole-item bootstrap 保留 pilot 中的 token dependence、position
nonstationarity、answer-length distribution、item heterogeneity 與
length/outcome association。

但它無法保護新 dataset/subset、candidate family 改變、collection drift、
pilot 未涵蓋的極端 items 或 SESOI 本身選錯。

### 9.3 Cap selection 必須獨立

在 confirmatory sample 上掃描多個 caps，再選 power/verdict 最漂亮的 cap，
會膨脹 Type-I error。可接受的做法：

- 事先登記一個 primary cap；
- 用獨立 pilot 選 cap；
- 用 split/holdout 資料選擇後再確認。

不同 caps 改變 endpoint，不應把它們當同一 group-sequential test 的 interim looks。

### 9.4 目前只支援 production default t endpoint

planner 明確限制 `--ci-method t`。若正式報告使用 studentized、BCa 或
percentile bootstrap，這個版本不會宣稱已重放該 test。未來若支援，應讓
simulator直接呼叫同一 inference primitive，並分開控制 future-dataset reps
和每次 test 的 inner bootstrap reps。

---

## 10. 如何讀結果

1. observed pilot effect 只作 profile/sign-flip 診斷，不用它取代 SESOI。
2. 檢查 `fraction_reaching_cap`；大量 items 在 cap 前結束時，增加 cap 會飽和。
3. Exclude cells with detected null inflation; no detected inflation does not prove calibration.
4. The MC lower-bound crossing is pointwise, with no simultaneous grid-wide guarantee.
5. The Gaussian pilot-p10 criterion additionally requires the ordinary MC lower-bound crossing; it is not empirical nested-bootstrap power.
6. 比較 expected evaluated tokens，不要只比較 N 或 K。
7. 檢查 outer uncertainty band；若跨越 target，pilot 不足以精確規劃。
8. 不要假設更大的 K 一定更好，也不要自動輸出單一「最佳 K」。
9. 最終設計應寫下 metric、weighting、cap、SESOI、alpha、target power、
   dataset sampling rule 與 planned N。

---

## 11. Implementation map

- `stats/power.py`：paired-t parity、whole-item simulation、Wilson interval、
  outer nuisance bootstrap、precision/MDE surface。
- `stats/cli/power_analysis.py`：正式 artifact guards、cap panel、CLI validation、
  Markdown/JSON reporting。
- `scripts/06_power_analysis.sh`：有 `SESOI` 跑 prospective power；無
  `SESOI` 跑 precision/MDE；其他舊工具只作 diagnostics。
- `tests/test_power_analysis.py`：production parity、non-monotone caps、
  null calibration、effect profiles、guard parity 與 end-to-end artifacts。

---

## 12. References

- Miller, *Adding Error Bars to Evals*: <https://arxiv.org/abs/2411.00640>
- Baker et al., *Power Contours*: <https://doi.org/10.1037/met0000337>
- Baker ordered-trial source: <https://osf.io/b2wk3/>
- Card et al., *With Little Power*: <https://doi.org/10.18653/v1/2020.emnlp-main.745>
- Albers & Lakens, pilot effect/sample size: <https://doi.org/10.1016/j.jesp.2017.09.004>
- Hoenig & Heisey, observed power: <https://doi.org/10.1198/000313001300339897>
- Cameron & Miller, cluster-robust inference: <https://doi.org/10.3368/jhr.50.2.317>
- Morris et al., simulation-study MC error: <https://doi.org/10.1002/sim.8086>
