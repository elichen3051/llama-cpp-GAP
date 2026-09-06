# AWS: 收集 VLM KLD

更新日期: 2026-09-06。本文件只收集已生成 reference trajectories 上的 VLM per-token metrics，包括 KLD、EAR_64、EAR_64_normalized。Reference 的權重、生成、發布流程見 [RunPod reference](/home/ubuntu/llamacpp_reference_runpod_handover.md)；Wikitext2 / PG 的文字橋接另見 [llama-perplexity / LLM KLD](/home/ubuntu/llamacpp_perplexity_llm_kld_aws_handover.md)，目前延後處理。

Reference 先進行。Candidate 還在排除損壞檔案，目前沒有最終工作清單；任何用於容量測試的 candidate 都只是暫選，不能標為正式評估已核准。本文件提供可明確指定 8 threads 的直接 collector 入口，既有高階 study 預設仍需整合。

本 AI session 起初對 `tools/skymizer` 唯讀，後經使用者明確授權實作 generation 併行。容量與兩輪長度校準保留原 NVMe frozen binary；新 AR 並行功能另編譯 binary，保存來源 hashes 並獨立驗收。兩者的證據不混用。以下一般模型命令維持原校準協定；Qwen3.5-4B / Gemma E4B 改依 [小模型交接](/home/ubuntu/llamacpp_reference_runpod_handover_qwen_gemma_4b.md) 的 reference generation parallel=4 協定；KLD scorer 仍用 np1。後續來源或 binary 改動需另行驗收。

## 1. 範圍與固定原則

Model family 是同一基底 checkpoint 的所有量化版本。不同 size 可以選不同 runtime；同一基底的所有 candidate 必須使用完全相同的 VLM KLD runtime、reference 權重、reference dataset 與 scorer build。Kimi-VL-A3B-Instruct 與 Kimi-VL-A3B-Thinking-2506 分開處理。

先做 SNR 決策組: Gemma 31B、Gemma E4B、Kimi separate 的 Instruct / Thinking-2506、Qwen3.5-4B、Qwen3.6-35B-A3B。完成後才做最終評估組 InternVL3.5-30B-A3B、GLM-4.6V-Flash；Muse 暫緩，Gemma 26B 不在本次範圍。Pilot 與 500 題都遵守此順序。兩種 size 各自遵守 SNR 組先於最終評估組的順序。

AWS 使用 NVIDIA RTX PRO 6000 Blackwell Server Edition，96 GiB。每張 GPU 同時一個 scorer。同一基底的 candidates 固定在同一張卡或已驗證一致的同型環境，記錄 GPU identity、driver / CUDA、binary 與 backend hashes。現有鎖只涵蓋 metadata / output，沒有全機 GPU 排他鎖；派工者必須確保不同 study 也不會同卡重疊。

## 2. 輸入驗收與環境

Reference 模型與 mmproj 必須使用 [reference handover 的 provider 表](/home/ubuntu/llamacpp_reference_runpod_handover.md#2-權重與-provider)。分片 LLM 驗證全部分片，不能只核對入口檔名。Gemma 31B reference trajectories 使用 MTP 生成，但 KLD 使用普通 teacher forcing，無需載入 MTP head。

開始收集前固定以下 identity:

- Reference dataset 的 repo、config、split、精確 commit、eligible item IDs、完整內容 hashes、發布 receipt / audit manifest。
- 生成使用的 reference LLM 所有分片與 mmproj SHA256，並核對 AWS 實際檔案相同。
- 已核准 candidate 的來源 checkpoint、LLM / mmproj SHA256、量化格式、檔案完整性與相容性檢查結果。
- 每個基底凍結的 runtime、image budget、scorer binary / source / backend 環境。

目前 `prep_vlm_score_from_hf.py` 的 Hub 載入沒有傳 revision，collector 也沒有將 reference 全檔 SHA256 自動對回生成 manifest。正式流程先用指定 commit 將 dataset 下載並保存成 NVMe 上的本地 `save_to_disk` 目錄，人工核對 receipt / hashes，再傳 `--dataset <local path> --subset ''`。不要直接把可移動的 Hub HEAD 當作固定 reference。

例如本地輸入目錄可命名為 `/opt/dlami/nvme/skymizer-kld/reference-inputs/qwen3.5-4b/mmmu-pro-vision-subsample-100-ins/<reference-commit>/dataset`，旁邊保存 `source.json` 與發布 manifest。這裡的 commit 必須是生成後 reference dataset 的 commit，不能誤用 prepared input dataset 的 `6cb6a4d...`（或歷史 `c3f0e320...`）。

所有新 dataset、cache、temp、build、venv、logs、outputs 放 NVMe，模型沿用 `~/models`。在 checkout 執行:

```bash
cd ~/projects/llama.cpp
export SKYMIZER_KLD_ROOT=/opt/dlami/nvme/skymizer-kld
export TMPDIR="$SKYMIZER_KLD_ROOT/tmp"
export HF_DATASETS_CACHE="$SKYMIZER_KLD_ROOT/cache/datasets"
export HF_HUB_CACHE="$SKYMIZER_KLD_ROOT/cache/hub"
export HF_XET_CACHE="$SKYMIZER_KLD_ROOT/cache/xet"
export HF_ASSETS_CACHE="$SKYMIZER_KLD_ROOT/cache/assets"
export CUDA_CACHE_PATH="$SKYMIZER_KLD_ROOT/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$SKYMIZER_KLD_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$SKYMIZER_KLD_ROOT/venv"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$CUDA_CACHE_PATH" "$UV_CACHE_DIR"
cmake -S . -B "$SKYMIZER_KLD_ROOT/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build "$SKYMIZER_KLD_ROOT/build" -j 8 --target llama-vlm-kld
uv sync --project tools/skymizer --locked --no-dev
nvidia-smi
```

不更改 `HF_HOME`，沿用登入 token；新主機需要時執行 venv 的 `hf auth login`。單獨改 cache 路徑不會完成登入。

## 3. Runtime 與雙模型容量

以下固定要求適用所有本次 VLM KLD candidates:

| 設定 | 要求 |
| --- | --- |
| n_threads / n_threads_batch | 8 / 8；native scorer 由 `--n-threads 8` 同時設定兩者 |
| metric_threads | 8，明確傳入 |
| flash attention | enabled，明確 `--flash-attn` |
| GPU offload | 所有 layers，collector 使用 `--n-gpu-layers -2` |
| KV dtype | K/V 均 F16；目前 native scorer 使用這個預設，驗收 logs |
| n_seq | 1；native scorer 固定 `n_seq_max=1` |
| SWA | `swa_full=false`，不加 `--swa-full`；不代表關閉原生 sliding-window attention |
| fit | 不做自動 fit 或參數縮減；scorer 使用直接指定的 context / batch |
| MTP | 關閉；普通 teacher forcing |
| image budget | 沿用 reference 保存的設定，不逐 candidate 改 image-min/max |

All GPU layers 指 native 可 offload 的 layers 全開。Gemma31 兩側日誌皆為 61/61，input embeddings 仍依原生設計位於 CPU_Mapped；不把這個預期行為誤報為層數不足或所有 tensor 都在 GPU。

`n_ctx / n_batch / n_ubatch / tf_chunk` 依每個基底的容量調查凍結，同一基底的所有量化版本完全一致。優先與 reference 的容量設定對齊，必須容納相同圖片 / prompt；VLM 不需要與文字橋接的 512-token runtime 一致。

起點為 `ctx=32768, batch=2048, ubatch=512, tf_chunk=2048`。Gemma 31B 使用 `ubatch=2048`，因 non-causal image chunk 必須同時放入 batch 與 ubatch。其餘五個 SNR checkpoint 的已測起點為 `ubatch=512`。ctx / batch / ubatch 已寫入 reference profiles；高階 study 的 tf_chunk 由 batch 導出為 2048。每個基底的正式 KLD runtime 仍須通過已核准候選清單的容量驗收。InternVL 的 strict replay 缺口另外處理，不能只從圖片 pixels 推定就緒。

每個基底先使用 7 subset 各 3 列的壓力樣本量測圖片 token / position 與峰值記憶體；資料與 native probes 見 `/opt/dlami/nvme/skymizer-reference-audit-20260906/dataset/`、`capacity/`。這些樣本是 heuristic，完整來源仍有 3500 列，不能宣稱 21 列代表所有 native 最大值。既有 12-thread 探測保留作歷史；新的 8-thread 結果位於 `/opt/dlami/nvme/skymizer-capacity8-audit-20260906/`，使用獨立輸出與真實 runtime metadata。

Scorer 同時載入 reference + candidate，兩側各有 projector、KV 與 workspace。容量驗證必須包含已核准清單中最大的 candidate，不能只測單獨載入 reference。F16 KV、flash attention 與全 GPU layers 都固定；OOM 時記錄失敗，不對單一 candidate 自動降 ubatch、context、offload 或圖片解析度。若需要改 protocol，整個基底的 candidates 重新使用一致設定與新的 collection identity。調低 tf_chunk 不能解決權重本身已超出 VRAM。

內建工具 profile 按 mode 收 2048 / 4096，不作本次 collection 預設。全部六個 SNR checkpoint 已用獨立 cohort profiles 表達新上限：pilot instruct/thinking 為 2048/8192，500 為 1024/4096；generation_caps 與 kld_eval_tokens 一致。小模型使用[專用 profiles](/home/ubuntu/llamacpp_reference_runpod_handover_qwen_gemma_4b.md)，其餘四個 checkpoint 使用 [reference handover 第 7 節](/home/ubuntu/llamacpp_reference_runpod_handover.md#7-其餘-snr-決策模型的-100--500-操作) 的 profiles。所有 collection / study 必須選到對應 cohort 的同一份 JSON，並核對實際每列 targets / horizon。這是考量成本後凍結的起始協定，尚未由 native paired SNR 證明充分；較少題目的 4096/16384 長度敏感度是後續研究選項，本輪 synthetic 容量測試不是該研究。舊 vLLM 的 512/2048 SNR 平台不能直接套到 native reference。若 pilot 的 paired metrics 支持更改協定，另立版本並保留原始 profiles，不能改寫已收集的 cohort。Candidate 清單未定，目前不宣稱 KLD production 或 SNR 驗收完成。

每側省略 `--image-min-tokens` / `--image-max-tokens`，包含不顯式傳 -1，採選定 mmproj 自身預設。舊資料若保存非預設 budget，不可當成本輪新 reference；實際命令與原圖 / projector identity 一起核對。較長 metric collection 必須保留到相應 reference cap，否則無法事後向上切 prefix。短答案只計既有 tokens，prompt 不計分但仍占 context；使用 `--num-eval-tokens`，不以 `--max-total-tokens` 取代。

接受 budget 截斷、repetition 排除與容量失敗，不在個別候選上臨時改 runtime。保存每格到達各 prefix 的題數、失敗與排除 IDs、實際 n_eval / 時間。Repetition 的整列排除可能隨生成上限改變 eligible cohort，跨 cap 比較需報告共同 IDs 與選樣限制。舊 prepared pin c3f0e320 已確認七組 pilot 都是對應 500 的原始前 100 列，全部 IDs、文字與有序原圖 hashes 一致。使用者已確認 pilot 用於估計 SNR，500 可以包含 pilot，不要求獨立確認集；舊版剩餘 400 仍有 23 列與同來源 pilot 共用原圖；新版 6cb6a4d6 的替換與重疊須見 [更新稽核](/opt/dlami/nvme/skymizer-length-audit-20260906/dataset-update)，不沿用舊版此數字。MMMU Standard / Vision 在 100 與 500 configs 各有完整 100 / 500 組相同 item_id + origin_id，不能直接當作獨立題源。共同 IDs / 圖片 / 題源群聚均需在分析中處理；零原始 bytes 匹配不等於獨立。[逐列重疊稽核](/opt/dlami/nvme/skymizer-length-audit-20260906/cohort-overlap/REPORT.md)。另有 21 個預先選定的長度校準 IDs，其中 17 個位於 500 的第 101 列之後。實際參與上限選擇的 calibration IDs 依逐輪紀錄標記；本次不因這些紀錄要求刪題或另建獨立確認集。

8-thread 短容量測試已完成: 七個模型共 19 個 4096-target synthetic rows 通過，InternVL 0/3。這不是正式 KLD 品質結果。Gemma31 peak=96435 MiB，只餘 1452 MiB；Qwen3.6 peak=92351 MiB。完整表見 reference handover。InternVL peak=84387 MiB，兩側均可載入；`vlm-kld.cpp` 的 prefix checker 把單個 256-token tile 與多個連續 tiles 所形成的 placeholder run 比較，導致 strict mismatch。上述為舊測試結果。此 Skymizer checker 已修正並通過原 3 列 / 48 個真實 targets strict replay，metrics 全 finite，peak84905MiB；未改 upstream model / mtmd。另修正 Python source-image / tile count 驗證。詳見 [最終評估組交接](reference-runpod-final-handover.md)；短 replay 不構成所有 candidates 的數值品質或全量容量驗收。

既有短 artifacts 的額外 CPU 診斷：19 列、292 個原始 target IDs 完全相符時，generation 的 -raw_logprob 與 scorer nll_ref 仍有最大 0.06698 nats 差異（Qwen3.6），Gemma31 MTP 為 0.01879。Reference 生成與 teacher forcing 的 decode 形狀不同，runtime 數值對齊不等於逐位一致；本次未隔離原因，也未預設通過閾值。此 selected-token NLL 差不是完整分布 KLD 或量化誤差，不把它當 candidate 品質結論。[逐列數值證據](/opt/dlami/nvme/skymizer-length-audit-20260906/short-reference-replay-logprob-comparison.json)。

## 4. 直接收集一個 reference / candidate 配對

以下 Qwen3.5-4B 命令使用本地已驗收的 reference dataset。先將 `SKYMIZER_REFERENCE_DATASET` 設為實際帶有 `state.json` 的 `save_to_disk` 目錄，並將 `SKYMIZER_KLD_CAP` 設為已確認的 cohort / mode 計分上限。Q4_K_M 路徑只示範如何指定候選，尚未宣告它已通過 candidate 核准；正式執行前換成已核准的檔案與 collection 名稱。

```bash
CUDA_VISIBLE_DEVICES=0 "$UV_PROJECT_ENVIRONMENT/bin/python" tools/skymizer/cli/collect_kld.py \
  --dataset "${SKYMIZER_REFERENCE_DATASET:?set the verified local reference dataset path}" --subset '' --split train \
  --ref-model "$HOME/models/qwen3.5-4b/bartowski/Qwen_Qwen3.5-4B-bf16.gguf" \
  --ref-mmproj "$HOME/models/qwen3.5-4b/bartowski/mmproj-Qwen_Qwen3.5-4B-bf16.gguf" \
  --cand-model "$HOME/models/qwen3.5-4b/bartowski/Qwen_Qwen3.5-4B-Q4_K_M.gguf" \
  --cand-mmproj "$HOME/models/qwen3.5-4b/bartowski/mmproj-Qwen_Qwen3.5-4B-bf16.gguf" \
  --llama-vlm-kld "$SKYMIZER_KLD_ROOT/build/bin/llama-vlm-kld" \
  --n-ctx 32768 --n-batch 2048 --n-ubatch 512 --tf-chunk 2048 \
  --n-gpu-layers -2 --n-threads 8 --metric-threads 8 --flash-attn \
  --keep-prep --sort-by num_images --num-eval-tokens "${SKYMIZER_KLD_CAP:?set the confirmed KLD cap for this size and mode}" \
  --out "$SKYMIZER_KLD_ROOT/outputs/qwen35-pilot-mmmu-vision-ins-Q4_K_M-attempt-0001"
```

上面的容量仍是起點，正式執行須換成該基底的凍結值。Thinking 使用對應 reference dataset、已確認的 cohort / mode 計分上限與新輸出名稱。500 題使用已發布 500 cohort 的 reference，不能把 pilot 延伸幾列就視為同一資料集。量化 LLM 比較預設兩側用相同 reference mmproj；測 projector 量化時才更換 candidate mmproj，並在候選 identity 中記錄。

直接 collector 不建立高階 `study.json`，不會自動形成完整 candidate 派工表。每次保存完整 shell 環境與命令，並將 checkpoint / subset / mode / size / candidate / frozen runtime / result path 記入外部工作清單。

## 5. 高階入口目前的限制

`collect_model_kld.py` 可建立 AWS 自己的 study、封存 scripts 並派一個 collection，但不能直接視為本次正式預設:

- 舊 profile 仍為六模型且含 Gemma 26B，缺 Kimi / InternVL。Profile 的 threads / threads_batch 與 `lib/reference_study.py` 的 metric_threads 均為 8，已符合本次 thread 要求。
- `lib/reference_study.py` 依所選 profile 讀取 mode 上限；各 cohort 使用獨立 profile 時要同時凍結 collection 命令與 study identity。不能只修改分析 prefix cap，或誤用內建舊 cap。
- 沒有 SNR 組先於最終組的 gate，也沒有預先列出全部待跑 candidates。根層 `idle` 只表示已提交工作目前沒有在跑。
- Study 沒有充分固定 scorer / GPU identity；比較檢查目前也漏核對 `n_ctx / n_gpu_layers / n_threads / metric_threads / flash_attn`。啟動前與交付前必須依外部 frozen manifest 完整核對，不能只依 comparator 通過判定對齊。

整合完成前使用上節明確指定參數的直接 collector。未來切換高階入口時開新 study，不修改已封存設定來延續舊 study。

儲存量可按目前 v5 格式估算：NPZ bytes = 76 x scored tokens + 6044 x 成功列數，並非壓縮後浮動比率。若每 lane 均有 C 個 candidates，pilot 2048/8192、每格 10 題 4096/16384 長度抽查，加上 500 的 1024/4096 成本情境，全部達 cap 時約為 13.477 GiB x C。不同 lane 的 candidate 數不同時應逐 lane 加總。另留 reference / 權重、prep、轉換中 bin + npz.tmp、失敗與重試保留空間。Collector 未加 --keep-prep 時通常刪除成功列的 prep 圖片；本文件範例明確加 --keep-prep，保留要求的原圖 / tokens / formatted chat。失敗與 rejected 亦保留；沒有已確認的自動保留政策。[公式與原始影像統計](/opt/dlami/nvme/skymizer-length-audit-20260906/storage-audit/REPORT.md)。

## 6. 完成判定與失敗處理

保留整個 output，至少包含 `metrics/*.npz`、`manifest.csv`、`collect_meta.json`、prep 原圖 / tokens / formatted chat，以及 logs 和 attempt 狀態。每個 item 的 OK / failure、實際 n_prefill / n_eval、reference 身分與 runtime 都需驗收。Paired report、N80 與 SNR 分析不在本收集文件執行範圍。

同一路徑不覆寫。中斷後不能只因部分 metrics 存在就判定完成；需要重做的工作用新的 study 或明確 retry 目錄，保留已產生的證據。可以繼續其他獨立工作，但跨群組 gate 以完整派工表與驗收結果判定。Candidate 尚未定案期間只能標示目前已核准範圍的進度，不能宣告整個 KLD campaign 完成。


已追加 Gemma31 / Qwen3.6 各一列 16384 synthetic-target 容量測試，均通過，peak 96411 / 92317 MiB。[完整 16k metrics CPU 檢查](/opt/dlami/nvme/skymizer-length-audit-20260906/reserve-capacity-full-metrics-validation.json) 確認全長 finite，仍不代表品質或所有 candidates 合格。

數值異常：完全相同 prompt / targets 的 4k 與 16k scorer，Gemma31 前 4096 positions 的 19 fields 全逐位一致；Qwen3.6 在 positions 3585..4095 的 511 個 targets 有差異，KLD max abs=0.002700676 nats、前 4096 平均 abs=0.0000113999，nll_ref max abs=0.000995162。nll_cand 與兩側 argmax 一致。尾端 microbatch 511 / 512 是可能解釋，未做隔離試驗；相同 CLI runtime 不保證跨 scoring horizon 逐位一致。同基底所有候選須使用相同每列 targets、num_eval、tf_chunk 與有效 batch 形狀；要比較多個 prefix，優先一次收最大已凍結 horizon，再離線切同一份 metrics，避免重新計算每個 cap 引入邊界差異。[逐 field 證據](/opt/dlami/nvme/skymizer-length-audit-20260906/capacity-4k-vs-16k-prefix-comparison.json)。

完整跨流程異常與證據見 [本輪稽核清單](/opt/dlami/nvme/skymizer-length-audit-20260906/ALL_FINDINGS.md)。

兩輪 native 長度校準已完成196/196列、252010 tokens：183 EOS、11 cap stops、2 repetition exclusions，沒有失敗或timeout。[完整長度結果](/opt/dlami/nvme/skymizer-length-audit-20260906/NATIVE_LENGTH_RESULTS.md)。此為舊pin/單序列長度證據，與新np4端到端驗收分開。

小模型最終驗收：Qwen3.5-4B / Gemma E4B 使用同一generator內4個sequences、每sequence ctx32768；210相關測試與413個native/Python/MTP/replay assertions均通過。兩模型各一列新np4 reference已由np1 scorer通過strict KLD replay；VLM KLD仍固定單sequence。同基底各candidate須使用同一份reference與完全相同scorer runtime。正式100/500依[小模型專用交接](/home/ubuntu/llamacpp_reference_runpod_handover_qwen_gemma_4b.md)，此次沒有代跑全量或發布Hub。
