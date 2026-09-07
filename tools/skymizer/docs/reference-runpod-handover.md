# RunPod: 單純產生 reference

更新日期: 2026-09-06。本文件只處理 reference 生成、驗收與交付。[VLM KLD](/home/ubuntu/llamacpp_kld_aws_handover.md) 與 [llama-perplexity / LLM KLD](/home/ubuntu/llamacpp_perplexity_llm_kld_aws_handover.md) 分開；文字橋接與文章／連續區塊統計已完成驗收，操作方式見該專用文件。

Qwen3.5-4B / Gemma E4B 的 100 / 500 題已另備可執行的 [小模型交接文件](/home/ubuntu/llamacpp_reference_runpod_handover_qwen_gemma_4b.md) 與兩份外部 profiles。其餘四個 SNR checkpoint 的外部 profiles、MTP 還原修正與 84-job 操作步驟在第 7 節，GPU 驗收狀態列於第 8 節。InternVL / GLM / Muse 的 100 / 500 題另見 [最終評估組交接](reference-runpod-final-handover.md)，以該文件與補充包的最終驗收狀態為準。先依第 3 節設定環境，再依第 7 節核對來源、建置、還原與生成。不要直接執行舊版預設六模型 campaign。

本 AI session 起初對 `tools/skymizer` 唯讀，後經使用者明確授權實作 generation 併行。容量與兩輪長度校準保留原 NVMe frozen binary；新 AR 並行功能另編譯 binary，保存來源 hashes 並獨立驗收。兩者的證據不混用。以下一般模型命令維持原校準協定；Qwen3.5-4B / Gemma E4B 改依 [小模型交接](/home/ubuntu/llamacpp_reference_runpod_handover_qwen_gemma_4b.md) 的 parallel=4 協定。後續來源或 binary 改動需另行驗收。

## 1. 模型與順序

Model family 是同一基底 checkpoint 的所有量化版本，不同 size 不必使用相同 runtime。Kimi Instruct 與 Thinking-2506 是不同 checkpoint，reference 與 hashes 必須分開。

| 階段 | checkpoints |
| --- | --- |
| 先做 SNR 決策 | gemma-4-31b-it、gemma-4-e4b-it、kimi-vl-a3b-instruct、kimi-vl-a3b-thinking-2506、qwen3.5-4b、qwen3.6-35b-a3b |
| 後做最終評估 | internvl3.5-30b-a3b、glm-4.6v-flash、muse-glimmer-30b；使用專用交接與外部 profiles |

Gemma 26B 不在本次範圍。Pilot 為正式 `-subsample-100` config，全量為 `-subsample-500`。Reference 與 KLD 都先完成 SNR 組再做最終組。Pilot 與 500 題各自遵守此群組順序。既有 launcher 沒有這個群組 gate，派工表須先控管。

## 2. 權重與 provider

以 `~/models/<family>/<provider>` 的實際 provider 配對 LLM 與 mmproj，優先 BF16，同 provider 沒有 BF16 才採 F16 並記錄例外。本次 8 組都有 BF16 LLM / mmproj，不需 fallback。下表路徑相對於 `~/models`，分片指定第一片，所有分片都需存在並驗證 SHA256。

| Checkpoint | Provider | LLM entry path relative to model root | MM projector path relative to model root | Dtype |
|---|---|---|---|---|
| gemma-4-31b-it | bartowski | `gemma-4-31b-it/bartowski/bf16/google_gemma-4-31B-it-bf16/google_gemma-4-31B-it-bf16-00001-of-00002.gguf` | `gemma-4-31b-it/bartowski/mmproj-google_gemma-4-31B-it-bf16.gguf` | BF16 LLM and BF16 mmproj |
| gemma-4-e4b-it | bartowski | `gemma-4-e4b-it/bartowski/google_gemma-4-E4B-it-bf16.gguf` | `gemma-4-e4b-it/bartowski/mmproj-google_gemma-4-E4B-it-bf16.gguf` | BF16 LLM and BF16 mmproj |
| kimi-vl-a3b-instruct | llamacpp | `kimi-vl-a3b-separate/kimi-vl-a3b-instruct/llamacpp/Kimi-VL-A3B-Instruct-bf16.gguf` | `kimi-vl-a3b-separate/kimi-vl-a3b-instruct/llamacpp/mmproj-Kimi-VL-A3B-Instruct-bf16.gguf` | BF16 LLM and BF16 mmproj |
| kimi-vl-a3b-thinking-2506 | llamacpp | `kimi-vl-a3b-separate/kimi-vl-a3b-thinking-2506/llamacpp/Kimi-VL-A3B-Thinking-2506-bf16.gguf` | `kimi-vl-a3b-separate/kimi-vl-a3b-thinking-2506/llamacpp/mmproj-Kimi-VL-A3B-Thinking-2506-bf16.gguf` | BF16 LLM and BF16 mmproj |
| qwen3.5-4b | bartowski | `qwen3.5-4b/bartowski/Qwen_Qwen3.5-4B-bf16.gguf` | `qwen3.5-4b/bartowski/mmproj-Qwen_Qwen3.5-4B-bf16.gguf` | BF16 LLM and BF16 mmproj |
| qwen3.6-35b-a3b | bartowski | `qwen3.6-35b-a3b/bartowski/BF16/Qwen_Qwen3.6-35B-A3B-bf16/Qwen_Qwen3.6-35B-A3B-bf16-00001-of-00002.gguf` | `qwen3.6-35b-a3b/bartowski/mmproj-Qwen_Qwen3.6-35B-A3B-bf16.gguf` | BF16 LLM and BF16 mmproj |
| internvl3.5-30b-a3b | bartowski | `internvl3.5-30b-a3b/bartowski/bf16/OpenGVLab_InternVL3_5-30B-A3B-bf16-00001-of-00002.gguf` | `internvl3.5-30b-a3b/bartowski/mmproj-OpenGVLab_InternVL3_5-30B-A3B-bf16.gguf` | BF16 LLM and BF16 mmproj |
| glm-4.6v-flash | bartowski | `glm-4.6v-flash/bartowski/zai-org_GLM-4.6V-Flash-bf16.gguf` | `glm-4.6v-flash/bartowski/mmproj-zai-org_GLM-4.6V-Flash-bf16.gguf` | BF16 LLM and BF16 mmproj |

Gemma 31B 生成必須使用 MTP，head 為 `~/models/gemma-4-31b-it/bartowski/mtp-google_gemma-4-31B-it-Q8_0.gguf`；主模型 / mmproj 保持 BF16。

S3 備份目的地 `s3://research-kld-benchmark/reference_model`。本次已核對 8 checkpoints、20 GGUF (308,550,736,128 bytes) 與 18 個 manifest / checksum 物件，選定項目沒有 missing / error。七個 GGUF 新傳，其餘十三個重用；逐檔 identity 與完整性層級見 [獨立驗證收據](/opt/dlami/nvme/skymizer-reference-audit-20260906/inventory/independent-sync-verification.json)。Muse 與範圍外權重保留。核心 `manifest.json` 保存 llm / mmproj，Gemma 31B head 另存 `mtp-manifest.json`。`0ecc643d9` 版 restore CLI 只處理 llm / mmproj；本輪提交補齊 MTP head，使其與主模型使用相同 SHA256 / 不覆寫流程。第 7 節保留舊 base 加 patch 的還原方式。[完整 20 檔還原 mapping](/opt/dlami/nvme/skymizer-reference-audit-20260906/inventory/complete-restore-map.json) 包含所有 S3 URI、相對本機路徑、bytes 與完整 SHA256。

本機 inventory: `/opt/dlami/nvme/skymizer-reference-audit-20260906/inventory/selected-models.json`、`SELECTED.md`。跨機器還原應攜帶 S3 manifest 的完整 identity。

## 3. 環境與儲存

RunPod / AWS 都使用 NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GiB。本機既有模型沿用 `~/models`；新 dataset、cache、temp、build、venv、logs、outputs 全部放 `/opt/dlami/nvme`。新主機若需還原模型，也將 `--models-dir` 放 NVMe；restore CLI 的下載暫存位於 destination.parent，不受 TMPDIR 控制。舊容量 / 長度校準使用每 GPU 一個 np1 generator；新 AR 小模型路徑已驗收每 GPU 一個 generator process、process 內 np4。Gemma31 MTP 仍用 np1。KLD 維持每 GPU 同時一個 np1 scorer。先檢查 NVMe mount / 空間與 GPU 空閒。需要 Python >=3.12、uv、CMake / C++ / CUDA build toolchain、AWS CLI 與 S3 read 權限；HF 來源需可讀取，正式發布另需寫入權限。

本節只設定環境。第 7 節核對 accepted checkout 與 patch 後才建置，避免沿用其他 checkout 的執行檔。記錄實際 Git revision、dirty diff、binary / backend hashes、driver / CUDA 與 GPU identity。小模型使用其專用交接的獨立建置流程。

```bash
cd ~/projects/llama.cpp
export AWS_DEFAULT_REGION=us-east-2
export AWS_REGION=us-east-2
export SKYMIZER_RUN_ROOT=/opt/dlami/nvme/skymizer-reference
export TMPDIR="$SKYMIZER_RUN_ROOT/tmp"
export HF_DATASETS_CACHE="$SKYMIZER_RUN_ROOT/cache/datasets"
export HF_HUB_CACHE="$SKYMIZER_RUN_ROOT/cache/hub"
export HF_XET_CACHE="$SKYMIZER_RUN_ROOT/cache/xet"
export HF_ASSETS_CACHE="$SKYMIZER_RUN_ROOT/cache/assets"
export CUDA_CACHE_PATH="$SKYMIZER_RUN_ROOT/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$SKYMIZER_RUN_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$SKYMIZER_RUN_ROOT/venv"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$CUDA_CACHE_PATH" "$UV_CACHE_DIR"
nvidia-smi
```

不改 `HF_HOME`，沿用既有登入 token；新機器需要時用 venv 的 `hf auth login`。若自行改 HF_HOME，必須重新登入，不能假設 cache 含有 token。不要將 token 寫進 handover / logs / manifest。

儲存規劃：14 個有效 lanes 的一份原圖 payload，pilot 合計 2.582 GiB、500 合計 13.838 GiB；目前 generator / uploader 同時保留 inputs 圖片、嵌圖 Arrow、上傳 Parquet 及部分重複生成紀錄，上傳成功不自動清理。實際峰值不能只算 HF 最終 parquet；重試、原始證據與保留週期需入帳。建立工作根目錄前核對 `/opt/dlami/nvme` 的實際 mount 與 `df -h`，不能只依目錄名稱認定是 NVMe。[逐項儲存稽核](/opt/dlami/nvme/skymizer-length-audit-20260906/storage-audit/REPORT.md)。

## 4. 來源與容量調查

正式來源 `elichen-skymizer/vlm-prepared-dataset` 固定更新後 revision `6cb6a4d65fcb2c8d788f68aaa0aa92c5ceca408b`，split=train。小模型與其餘四個 SNR checkpoint 的外部 profiles 均使用新 pin；內建舊 profile 不作本次正式派工來源。以下圖片容量與 GPU 長度校準固定舊版 `c3f0e320b7f0360a3a10ee4c74aca0b93d697dc5`，不中途替換。舊版七組正式 100 題逐列精確等於各 500 的前 100 列，14 個 parquet 已核對上游 LFS hashes。新版完成逐列與原圖核對，為 pilot 替換 7/700 (1.0%)、500 替換 91/3500 (2.6%)；長度21與 stress21 各一筆被更換，其餘各20筆相同。新版 pilot/剩400 原圖 bytes 共享為0；MMMU 題源重疊仍在。MMStar 最大寬度與 OCRv2 最長問題增加，不能將舊 GPU 結果稱為新版全量驗收。[新版逐列稽核](/opt/dlami/nvme/skymizer-length-audit-20260906/dataset-update)。

| 全量 config | 已掃描列數 | 最大單圖 pixels | 最大圖數 |
| --- | ---: | ---: | ---: |
| blink-val-subsample-500 | 500 | 4,463,860 | 4 |
| mathvision-test-subsample-500 | 500 | 11,990,160 | 1 |
| mmmu-pro-standard-10-subsample-500 | 500 | 4,915,200 | 7 |
| mmmu-pro-vision-subsample-500 | 500 | 8,896,520 | 1 |
| mmstar-subsample-500 | 500 | 4,741,120 | 1 |
| ocrbench-v1-subsample-500 | 500 | 34,806,376 | 1 |
| ocrbench-v2-subsample-500 | 500 | 34,806,376 | 1 |

已逐列讀原圖 header，依最大單圖、最多圖片、極端長寬比或長問題選 3 列 / subset，共 21 列。最大原圖 4961x7016；MMMU vision 所有 question 都是空字串但有圖片，屬有效輸入。原始 bytes / hashes、`requests.jsonl`、`selected.json`、每 subset 的 `all_500_dimensions.jsonl` 位於 `/opt/dlami/nvme/skymizer-reference-audit-20260906/dataset/`。

選樣是 heuristic，不是全量 native token 最大值證明。模型的 resize / tiling / per-image cap 不同，7 張中型圖可能比一張大圖更吃 context。每個 reference checkpoint 需測 21 列的 prompt tokens / positions、最大 non-causal image chunk、prefill / generation 峰值 VRAM，並為完整 generation cap 留 context。原 12-thread 產物保留在 `/opt/dlami/nvme/skymizer-reference-audit-20260906/capacity/`，不改寫舊測試紀錄。新的 8-thread 探測位於 `/opt/dlami/nvme/skymizer-capacity8-audit-20260906/`，八個 reference 共 294/294 列通過；實際生成上限僅 16，不能作為完整回答或 SNR 驗收。

容量同時考慮 KLD 的 reference + candidate + 兩側 projector / KV / workspace。單模型生成通過不代表雙模型 scorer 能放進 96 GiB。Candidate 清單未定案，現有較大候選只供估算，正式凍結前仍需覆蓋核准候選中的最壞情況。

8-thread 壓力樣本結果如下，ctx=32768、batch=2048、Gemma31 ubatch=2048，其餘 ubatch=512。GPU 總量 97887 MiB；KLD 使用暫選 candidate 及 4096 個 synthetic repeated targets，只作容量測試。

| Checkpoint | 最大 prompt tokens | 最大 image chunk | Reference peak MiB | Dual KLD peak MiB | KLD 結果 |
| --- | ---: | ---: | ---: | ---: | --- |
| gemma-4-31b-it | 3211 | 1102 | 68139 | 96435 | 3 rows pass |
| gemma-4-e4b-it | 3211 | 1102 | 12509 | 18169 | 3 rows pass |
| kimi-vl-a3b-instruct | 2996 | 1015 | 33089 | 45599 | 2 rows pass |
| kimi-vl-a3b-thinking-2506 | 2996 | 1015 | 33089 | 45599 | 2 rows pass |
| qwen3.5-4b | 8865 | 4067 | 11693 | 17477 | 3 rows pass |
| qwen3.6-35b-a3b | 8865 | 4067 | 68769 | 92351 | 3 rows pass |
| internvl3.5-30b-a3b | 9617 | 256 | 62541 | 84387 | 0/3, prefix replay failure |
| glm-4.6v-flash | 10374 | 4067 | 22251 | 32373 | 3 rows pass |

Gemma31 雙模型 4096-target 測試只餘 1452 MiB；追加 16384-target 單列測試已通過，peak 96411 MiB、餘 1476 MiB。Qwen3.6 同樣通過 16384 targets，peak 92317 MiB、餘 5570 MiB。兩個完整 metrics artifacts 的 19 fields 均有 16384 positions，float 全 finite、token IDs 合法；這是 synthetic capacity probe，其他候選與圖片 workspace 仍需驗證。[16k probe](/opt/dlami/nvme/skymizer-length-audit-20260906/RESERVE_CAPACITY_REPORT.md)。InternVL 兩模型可載入，但其相鄰 256-token 圖片 tiles 形成連續 placeholder run，現行 checker 將單一 tile 與整段 run 比較而拒絕；並非已證明 preprocessing drift 或 OOM。此舊 checker 問題已修正並以原 3 列、48 個實際 targets 完成 strict replay；所有 metric 有限，峰值 84905 MiB。另修正 Python R12 將 source images 與 image chunks 混為一談的檢查。修正僅在 Skymizer，未改 upstream model / mtmd；證據見最終評估組交接。詳見 [8-thread summary](/opt/dlami/nvme/skymizer-capacity8-audit-20260906/summary.json)。

## 5. 明確指定參數的本地生成

固定 decode threads=8、batch threads=8、FA=on、所有 LLM layers GPU、F16 K/V、fit=off、swa_full=false。Generation 不要求固定n_seq=1。新授權的AR實作支援--parallel；launcher ctx是每題容量，native總ctx乘slots。兩小模型np4各保留32768 context的native與完整pilot上限Python驗收皆通過。舊長度校準維持np1，MTP也仍為np1。VLM KLD 另維持 n_seq=1。不加 `--swa-full` 指不配置 full-size SWA cache，不是關掉模型原生 sliding-window attention。此處「所有 layers」指 native 可 offload 的 layers 全開；Gemma31 日誌為 61/61，但 llama.cpp 仍刻意將 input embeddings 放在 CPU_Mapped buffer，不代表所有 tensor / operation 都駐 GPU，也不是偷偷降低 ngl。

100 題 pilot 與 500 題使用不同的 generated-token 上限。新值由本輪調查提出建議，不沿用歷史生成上限。歷史 vLLM 並非統一協定：部分 campaign 有 16384 / 65536，早期 Qwen3.6 為 2048 / 4096，另有 30 格完整 cap / sampling 日誌未找到，不能由 KLD scoring cap 推定。舊 vLLM 實驗在 instruct 約 512、thinking 約 2048 tokens 出現 SNR 收益趨緩，但 runtime、reference trajectories 與 vision budget 均已改變，不能直接當成 native llama.cpp 的飽和點或固定 primary endpoint。

| 用途 | instruct generation cap | thinking generation cap | 狀態 |
| --- | ---: | ---: | --- |
| Pilot 起始協定 | 2048 | 8192 | 全部六個 SNR checkpoint 已有獨立外部 profiles；不是 SNR 充分性證明 |
| 預先選定的長 prefix 驗證 | 4096 | 16384 | 建議每格選 10 個 pilot IDs，與正式 cohort 分開存放 |
| 500 題起始協定 / 成本情境 | 1024 | 4096 | 全部六個 SNR checkpoint 使用各自 collect500 profile；最終評估組仍待整合 |

每個 checkpoint 使用自身 mmproj 預設: 不傳 `--image-min-tokens` / `--image-max-tokens`，連 -1 也不顯式傳入。Reference 與 KLD 使用相同 projector identity / 預設，保存實際 image tokens、tiles、prompt tokens / positions。不能人工將不同模型設成同一 token budget；以前 Gemma 的 280 與 Qwen 的 16384 單圖上限不套用本輪。

GGUF 未必保存明確的最大 token 數；本輪選定的 projector 會使用 native family fallback。查得預設如下，並以凍結 scorer 的 loader logs 中 image_min/max_pixels / tile 設定及實際 prompt layout 交叉核對，因此 mmproj hash 與 binary identity 必須一起固定。

| 模型 | 無 override 的預設 | 單位 |
| --- | --- | --- |
| Gemma4 31B / E4B | 70 - 1120 | image tokens; default pooling3 |
| Qwen3.5 / Qwen3.6 | 8 - 4096 | image tokens; merge2 |
| Kimi Instruct / Thinking | 8 - 1024 | image tokens; merge2 |
| GLM4.6VFlash | 8 - 4096 | image tokens; merge2 |
| InternVL3.5 | 1 - 12 tiles | 256tokens/tile; overview can add another tile |

[原始 mmproj headers](/opt/dlami/nvme/skymizer-length-audit-20260906/mmproj-default-headers.json)、[fallback 原始碼快照](/opt/dlami/nvme/skymizer-length-audit-20260906/vision-default-source-evidence.json)。這些是預設限制，實際 emitted tokens 隨圖形幾何而變，不等於每張圖一定產生上限數量。

新上限需分 checkpoint / semantic mode / source 評估。既有每 subset 三筆圖片極端樣本只驗證容量，不用來估計一般回答長度。新增可重現的分層抽樣以 dataset 為 strata，各 subset 按固定 SHA256 排序預選三題，與生成結果和圖片大小無關；第一輪先每格一題，再按預算擴樣。來源、21 個 IDs、原始 image hashes 與每輪七個 requests 位於 `/opt/dlami/nvme/skymizer-length-audit-20260906/dataset/`。本輪原始單 GPU 預算7200秒；後經使用者明確延長同一起點的截止至2026-09-06 17:00 UTC。兩輪196列長度校準已完成，第三輪取消以優先驗收新並行功能，不自動延長。

面向 ICLR / CVPR 的長度敏感度設計應先登記 prefix ladder、長度抽查 IDs 與選擇規則。建議 pilot 檢查 instruct 128 / 256 / 512 / 1024 / 2048，thinking 512 / 1024 / 2048 / 4096 / 8192，再以同組長度抽查 IDs 檢查 4096 / 16384。報告每格到達各位置的題數、自然停止 / 撞 cap / repetition / failure、實測時間，以及候選確定後的 paired mean、CI、SD / MDE。只量 reference 長度不能證明 SNR 飽和，不能依最有利的 observed SNR 或 verdict 選上限。短答案自然結束也會讓 prefix 曲線變平；須區分新增 tokens 已很少與後段 metric 沒有增益，並另報共同長答案的敏感度及其條件選樣限制。官方審查指引強調實驗嚴謹性與可重現性，並沒有規定應收多少 generated tokens；此處 cap 是本研究設計，不保證 reviewer 接受。[ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)、[CVPR 2026 Reviewer Guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)。CVPR 此處引用已公開的 2026 版審查原則，不將它當作 2027 投稿規則。

若每格 100 題用 2048 / 8192，14 lanes x 7 subsets 的最大 generated-token 預算為 50,176,000；每格另外 10 題從頭跑 4096 / 16384，增加 10,035,200，即最大 token 預算增加 20%。這不是 GPU 時間估計，未包含 prefill、重試及各 KLD candidates。500 的 1024 / 4096 全模型成本情境最多 125,440,000 generated tokens；六個 SNR checkpoint 已採此起始協定；最終評估組仍待整合。

接受 budget 限制所造成的回答截斷與 repetition 排除，不為保留每題而持續加大 context 或重跑。非重複的 length-capped answers 保留並標記；自然 EOG 可提早停止。現行 native generator 對 `max(prompt tokens, prompt positions) + requested cap > ctx` 直接拒絕該列，尚無自動將回答上限縮至 context 餘量的行為。圖片 / prompt 不偷偷截短；容量失敗記錄在 cohort partition。Repetition 目前會剔除整列，生成越長可能排除更多列，跨 cap 比較需報告 eligible IDs 差異及共同 IDs 的敏感度分析。七組 pilot 都包含在 500 內。使用者已確認 pilot 用於估計 SNR 訊號，500 題可以包含 pilot，不要求獨立確認集；不因包含關係額外拆 holdout 或刪題。在舊 c3f0e320 版本，其餘 400 題仍有原圖重用：MMMU Standard 5、MMStar 7、OCRBench v1 10、OCRBench v2 1 列，共 23 列。MMMU Standard / Vision 的 100 與 500 configs 分別有 100 / 500 組相同 item_id + origin_id，須視為相關題源的不同呈現。分析保留題源 / 圖片群聚資訊，並如實報告資料關係；原始 bytes 無匹配不證明獨立。[逐列重疊稽核](/opt/dlami/nvme/skymizer-length-audit-20260906/cohort-overlap/REPORT.md)。另有 21 個預先選定的長度校準 IDs，其中 17 個位於 500 的第 101 列之後。實際參與上限選擇的 calibration IDs 依逐輪紀錄標記即可，本次不要求因此排除或建立獨立確認集。

長度抽查與 pilot 的共同 token prefix 必須實際核對，相同 seed 不保證完全相同 trajectory；不同時不拼接。長度抽查產物放 NVMe 獨立 audit，不混入聲稱統一 cap 的正式 HF config。舊 generation / KLD caps 與 `-pivot` 遠端資料僅作歷史來源，不能當作新協定或 native 長度證據。

重新查核的完整 model-card 表、native defaults 與 Kimi / InternVL 差異見 [本機 decoding 紀錄](/home/ubuntu/models/DECODING_PARAMS.md)。既有 Gemma / GLM profiles 未指定 min_p，effective 值為 0.05；不能將官方未指定解讀為停用。Seed=1234 是實驗設定。Kimi Thinking card 推薦 temperature=0.8，但來源 generation_config 與本機 GGUF 為 0.6，本機驗模 smoke 又採 greedy；本輪長度探測採 card 的 0.8，並保存 effective sampler；這不是沿用轉換 smoke 的 greedy。

Kimi Instruct 與 Thinking-2506 各只收自己的模式；兩個本機 template 都沒有 enable_thinking 分支。InternVL thinking 需要官方 R1 system prompt，不能只加 enable_thinking。高階 model launcher 現已支援 profile 的逐模式 system_prompt / chat_template_kwargs，publisher 也核對實際 run / row 設定。Muse 固定 thinking/high 與 current_date=2026-09-06，沒有虛設 instruct 模式。完整九個 checkpoints 合計 15 個有效 lanes，每個 size 為 105 個 jobs；先前 14-lane 成本表不含 Muse。

以下 Gemma 31B pilot 完整 config 指令產生本地資料並使用 MTP。先將 `SKYMIZER_GENERATION_CAP` 設為該 cohort / mode 已凍結的上限；本次 pilot profile 為 instruct=2048 / thinking=8192，未設定時此直接呼叫範例不會執行。第 7 節的 profile launcher 會自動選擇對應 cap。`ctx=32768, batch=2048, ubatch=2048` 是容量調查起點，並非全量容量合格宣告。輸出目錄必須不存在。

```bash
CUDA_VISIBLE_DEVICES=0 "$UV_PROJECT_ENVIRONMENT/bin/python" tools/skymizer/cli/generate_reference.py \
  --dataset elichen-skymizer/vlm-prepared-dataset \
  --subset mmmu-pro-vision-subsample-100 --split train \
  --revision 6cb6a4d65fcb2c8d788f68aaa0aa92c5ceca408b \
  --out "$SKYMIZER_RUN_ROOT/outputs/gemma31-pilot-mmmu-vision-ins-attempt-0001" \
  --no-enable-thinking --llama-reference "$SKYMIZER_RUN_ROOT/build/bin/llama-reference" \
  --row-timeout 1800 --native-timeout 172800 -- \
  -m "$HOME/models/gemma-4-31b-it/bartowski/bf16/google_gemma-4-31B-it-bf16/google_gemma-4-31B-it-bf16-00001-of-00002.gguf" \
  --mmproj "$HOME/models/gemma-4-31b-it/bartowski/mmproj-google_gemma-4-31B-it-bf16.gguf" \
  -c 32768 -b 2048 -ub 2048 -t 8 -tb 8 -ngl all \
  -fa on -ctk f16 -ctv f16 -np 1 --fit off \
  -n "${SKYMIZER_GENERATION_CAP:?set the confirmed generation cap for this size and mode}" --seed 1234 --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.05 \
  --spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-n-min 0 --spec-draft-p-min 0 \
  -ngld all --model-draft "$HOME/models/gemma-4-31b-it/bartowski/mtp-google_gemma-4-31B-it-Q8_0.gguf"
```

Thinking 改 `--enable-thinking`、對應的 `SKYMIZER_GENERATION_CAP` 與新輸出名稱。500 題改對應 `-subsample-500`，並切換成 500 題 / 該模式的生成上限，不加 `--num-samples`。其他 subset 改 `--subset`，其他 checkpoint 使用各自凍結後的權重 / sampling / 容量，不照抄 Gemma MTP。

Gemma 31B 的完整 non-causal image chunk 須同時放入 batch / ubatch；測得 1102 tokens，因此不可任意降 ubatch=512。2048 是已測值，不是精確最小值。Gemma E4B 與 Qwen 使用 causal 圖片 decode，允許拆批；Qwen 的 4067-token 圖片區塊可拆成 2048 + 2019 logical tokens，再以 ubatch=512 執行。此差異見 `tools/mtmd/mtmd.cpp` 的 `mtmd_decode_use_non_causal`、`mtmd-helper.cpp` 與 `tools/skymizer/reference.cpp` 的容量檢查。Context 需容納 `max(prompt tokens, prompt positions) + generation cap`，包含圖片與文字；不能只用 Qwen 較小的 M-RoPE positions 計算容量。

## 6. 驗收、失敗與發布

直接 `generate_reference.py` 只產生本地資料，不建立 campaign、不自動上傳，也沒有跨呼叫 `--resume`。內部 row / startup retry 不等於整個任務可恢復。保留整個輸出: `dataset/`、`metadata.json`、`requests.jsonl`、`run_state.json`、`native/`、`attempts/`、`inputs/`、`scripts/`。

每任務核對 requested / success / failure / repetition-excluded IDs，不只看 exit code。排除保留理由，不補抽、不偷偷改圖 / cap / runtime。預設 row watchdog 從每列 row_started 起算 1800 秒；startup / idle 無 journal 進度也受限，其他列完成不會延長該列期限。Native 全程序 timeout 是選填；本節直接呼叫範例另設 48 小時，第 7 節高階 launcher 沒有此固定總期限，仍須由操作者依正式 collection 預算監控。

既有 campaign resume 將 `complete_with_failures` 視為已處理，不會補失敗列。重跑用新 attempt / study 並保留舊產物。HF repo/config 禁止覆寫成不同內容，需先定版本 / supersede 命名；新本地 study 不代表新遠端版本。舊 `elichen-skymizer/qwen3.6-35b-a3b-pivot` 的 14 個 configs 沒有現行 uploader 的 audit manifest；只作歷史來源，不覆寫或宣稱是本輪結果。新 100 題輸出使用 -pilot namespace，500 題仍用 -collect-500。改名前的 native campaign 若需恢復，使用當時封存的 controller / scripts 與原始命名；新 controller 會驗證 -pilot receipt，不直接接手舊 -pivot study。

正式批次尚缺:

- 舊 profile 仍是六模型，含 Gemma 26B，缺 Kimi / InternVL；threads=8 已符合要求，但仍需更新模型、容量與 sampling 才能使用預設 campaign。
- 內建 profile 的 generation_caps 只按 mode 設定；可用各 cohort 獨立 JSON 配合 generator / uploader 相同 `--profiles`，無需修改工具即可保持兩端 cap 一致。小模型 56 與其餘 SNR 84 命令建構檢查已通過。只用 --max-new-tokens override 仍會與發布驗證不一致；study 同樣要凍結所選 profile。Generator 自動封存的是內建 JSON，外部實際 profile 必須另存副本與 SHA256。
- 發布驗證沒有完整核對 ctx / batch / ubatch / threads / MTP。發布前另核對 effective metadata、MTP method / draft_max / head SHA256、主模型所有分片與 mmproj hashes、凍結 profile。
- 派工表需要 checkpoint / source / mode / cohort / attempt / status 與 SNR gate，不能把 queue idle 當成全部完成。

HF dataset repo 命名為 `elichen-skymizer/<checkpoint>-pilot` (100 題) 或 `elichen-skymizer/<checkpoint>-collect-500` (500 題)。Subset 就是 Hub config，名稱為 `<source>-subsample-<100|500>-<ins|think>`，split 固定 `train`。

```text
elichen-skymizer/gemma-4-31b-it-collect-500
  README.md
  mmmu-pro-vision-subsample-500-ins/
    train-00000-of-00001.parquet
  mmmu-pro-vision-subsample-500-think/
    train-00000-of-00001.parquet
  audit/<config>/
    manifest.json
    metadata.json
    ...
```

每個 source / 有效 mode 各有獨立 config。Parquet 保存原始 image bytes、prompt / answer token IDs、raw target logprobs、生成 metadata 與原圖 hashes，不依賴 RunPod 暫存圖片路徑。Local campaign 存於 `<out>/artifacts/<checkpoint>/<config>/attempt-0001/dataset/`；直接 generator 存於 `<out>/dataset/`，outputs 都位於 NVMe。

Kimi repo 名稱分別使用 `kimi-vl-a3b-instruct` 與 `kimi-vl-a3b-thinking-2506`，Instruct repo 只發布 -ins configs，Thinking-2506 repo 只發布 -think configs。`kimi-vl-a3b-separate` 只是本機分組目錄，不是 checkpoint 或 HF 模型來源。Kimi 的外部 profile 已按各自唯一有效模式凍結；不能對每個 Kimi checkpoint 都派兩種模式。S3 保存權重，HF 保存生成的 reference datasets，manifest 分開。

交付 AWS 必須有 reference dataset 的 repo / config / split / 精確 commit、內容 hashes、發布 receipt / audit manifest、生成權重全分片 / mmproj / MTP head SHA256、effective runtime / sampling / image budget、binary / source provenance 與列狀態。驗收完成才可進入 VLM KLD。


歷史 vLLM 本機調查涵蓋 68 格、6717 個 retained rows；其中較晚 58 格只有 5717/5800 列留存，部分 campaign 排除 capped 與 repetition / fuzzy rows。Thinking retained p95 為 Gemma31 11892、E4 5246、Qwen3.5 27692、Qwen3.6 23739 tokens，但來源分布與選擇偏差不同，不能直接作 native 所需上限。舊 Gemma min_p=0，本輪為 0.05；HF processor 與 GGUF mmproj 也不同，歷史 model revision 未固定。原始輸入相同仍不足以隔離 backend 因果。[完整歷史調查](/opt/dlami/nvme/skymizer-length-audit-20260906/vllm-history/REPORT.md)。

完整跨流程異常與證據見 [本輪稽核清單](/opt/dlami/nvme/skymizer-length-audit-20260906/ALL_FINDINGS.md)。

兩輪 native 長度校準已完成196/196列、252010 tokens：183 EOS、11 cap stops、2 repetition exclusions，沒有失敗或timeout。[完整長度結果](/opt/dlami/nvme/skymizer-length-audit-20260906/NATIVE_LENGTH_RESULTS.md)。此為舊pin/單序列長度證據，與新np4端到端驗收分開。

小模型最終驗收：Qwen3.5-4B / Gemma E4B 使用同一generator內4個sequences、每sequence ctx32768；210相關測試與413個native/Python/MTP/replay assertions均通過。兩模型各一列新np4 reference已由np1 scorer通過strict KLD replay；VLM KLD仍固定單sequence。同基底各candidate須使用同一份reference與完全相同scorer runtime。正式100/500依[小模型專用交接](/home/ubuntu/llamacpp_reference_runpod_handover_qwen_gemma_4b.md)，此次沒有代跑全量或發布Hub。

## 7. 其餘 SNR 決策模型的 100 / 500 操作

本節涵蓋 Gemma31、Qwen3.6、Kimi Instruct、Kimi Thinking-2506，共 6 個有效 model/mode lanes。每個 size 為 42 jobs，合計 84 jobs、25,200 個名義請求。配合小模型的 56 jobs，SNR 組合計 140 jobs / 42,000 個名義請求。各 cohort 仍須先完成整個 SNR 組，才可啟動 InternVL / GLM / Muse 最終評估組；這份 loop 不包含最終組，也不代表其他主機上的小模型已完成。

補充包位於 [~/to_runpod_supplements/skymizer-remaining-snr-reference-20260906](/home/ubuntu/to_runpod_supplements/skymizer-remaining-snr-reference-20260906)。整包帶到 RunPod 的相同 home-relative 路徑；`HANDOVER.md` 是本文件副本。包內有 profiles、size/profile mapping、MTP restore patch、協定驗收腳本、來源與測試收據，不含模型 / dataset / credentials / 完整 Git history。正式輸出仍全部在 NVMe。

| Profile | Instruct cap | Thinking cap | SHA256 |
| --- | ---: | ---: | --- |
| reference-model-profiles-pilot100.json | 2048 | 8192 | db30699ec6316067a832a31a7493b520a6b1909a57088990b571e8464f3a45c1 |
| reference-model-profiles-collect500.json | 1024 | 4096 | 07a336b177444eed11c5728dead1990b2465b245b3f2e4521cc84085bb94f196 |

這四個 checkpoint 使用已量測的 np1 / ctx32768；AR 的 np1 是本次選定容量配置，工具本身已支援更多 sequences。Gemma31 MTP 仍需要 np1，draft_max=8、Q8_0 sidecar。其他三個 checkpoint 是 AR、MTP off。共同 batch2048、threads8/8、FA on、f16 KV、all layers、fit off；Gemma31 ubatch2048，其餘512。省略 image-token flags 和 --swa-full。

Gemma31 明確加 min_p=0.05，完整 effective sampler 的 user_sampling_config=78；不能將舊內建 profile 的 config70 和這份結果混用。Qwen3.6 使用 config334，instruct temp0.7/top_p0.8、thinking temp1/top_p0.95、top_k20/min_p0/presence1.5。Kimi 只指定 card 的 temp0.2 /0.8，其他 native 預設保留並完整凍結：top_k40、top_p0.95、min_p0.05、config64。不沿用舊 Kimi greedy smoke 或 HF processor 設定。

在第 3 節環境準備完成後，取得補充包 `source-provenance.json` 的 `accepted_commit` checkout；新提交由使用者 push。也可使用既有 base `0ecc643d97a43050668c2efa452f8466289890fb` 加包內 patch。以下只在 patch 尚未存在時套用，之後核對每個指定來源檔案；若有不相符的其他修改會停止，勿跳過檢查。兩條路徑驗證同一組來源 SHA256。原始 4B 部署仍固定其專用包的已驗收版本。

```bash
set -euo pipefail
cd ~/projects/llama.cpp
export SKYMIZER_REMAINING_BUNDLE="$HOME/to_runpod_supplements/skymizer-remaining-snr-reference-20260906"
export SKYMIZER_REMAINING_ROOT="$SKYMIZER_RUN_ROOT/remaining-snr-a01"
export SKYMIZER_REMAINING_PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export SKYMIZER_REMAINING_BINARY="$SKYMIZER_RUN_ROOT/build/bin/llama-reference"
(cd "$SKYMIZER_REMAINING_BUNDLE" && sha256sum -c SHA256SUMS)
python3 - "$SKYMIZER_REMAINING_BUNDLE" <<'SOURCE_CHECK'
import hashlib, json, subprocess, sys
from pathlib import Path
bundle = Path(sys.argv[1])
manifest = json.loads((bundle / 'source-provenance.json').read_text())
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if head not in {manifest['base_commit'], manifest.get('accepted_commit')}:
    raise SystemExit('Git revision differs from the accepted source revisions: ' + head)
def matches():
    return all(Path(r['path']).is_file() and hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256'] for r in manifest['files'])
if not matches():
    patch = bundle / manifest['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest['patch_sha256']:
        raise SystemExit('restore patch checksum differs')
    subprocess.run(['git', 'apply', '--check', str(patch)], check=True)
    subprocess.run(['git', 'apply', str(patch)], check=True)
if not matches():
    raise SystemExit('source differs from accepted snapshot')
changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], text=True).splitlines()
accepted = {r['path'] for r in manifest['files']}
unexpected = [p for p in changed if p not in accepted and p != 'tools/skymizer/README.md' and not (p.startswith('tools/skymizer/docs/') and p.endswith('.md'))]
if unexpected:
    raise SystemExit('additional source changes require separate acceptance: ' + ', '.join(unexpected))
print('Accepted source files verified')
SOURCE_CHECK
cmake -S . -B "$SKYMIZER_RUN_ROOT/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build "$SKYMIZER_RUN_ROOT/build" -j 8 --target llama-reference
uv sync --project tools/skymizer --locked --no-dev
mkdir -p "$SKYMIZER_REMAINING_ROOT/protocol" "$SKYMIZER_REMAINING_ROOT/logs" "$SKYMIZER_REMAINING_ROOT/frozen-tools"
cp -a "$SKYMIZER_REMAINING_BUNDLE/." "$SKYMIZER_REMAINING_ROOT/protocol/"
test ! -e "$SKYMIZER_REMAINING_ROOT/frozen-tools/skymizer"
cp -a tools/skymizer "$SKYMIZER_REMAINING_ROOT/frozen-tools/skymizer"
export SKYMIZER_REMAINING_TOOLS="$SKYMIZER_REMAINING_ROOT/frozen-tools/skymizer"
export SKYMIZER_REMAINING_MODELS="$SKYMIZER_REMAINING_ROOT/models"
git rev-parse HEAD > "$SKYMIZER_REMAINING_ROOT/protocol/git-head.txt"
git diff --binary HEAD > "$SKYMIZER_REMAINING_ROOT/protocol/working-tree.patch"
sha256sum "$SKYMIZER_REMAINING_BINARY" > "$SKYMIZER_REMAINING_ROOT/protocol/llama-reference.sha256"
"$SKYMIZER_REMAINING_PY" "$SKYMIZER_REMAINING_TOOLS/cli/restore_reference_models.py" \
  --profiles "$SKYMIZER_REMAINING_ROOT/protocol/reference-model-profiles-pilot100.json" \
  --models-dir "$SKYMIZER_REMAINING_MODELS" --download \
  | tee "$SKYMIZER_REMAINING_ROOT/protocol/model-restore.jsonl"
```

S3 bucket region 固定 `us-east-2`；第 3 節同時設定 AWS_DEFAULT_REGION / AWS_REGION。還原器使用 profile 中的完整 object key 逐檔 aws s3 cp，沒有 ls / sync / recursive / ListObjects 呼叫。明確 region 可避免 SDK 自動以 HeadBucket 探測 region；本次沒有更改 IAM、bucket policy 或 ACL。下載所需為對應 object 的 GetObject 權限及適用的加密金鑰權限。

還原計畫共 11 檔 / 200,779,298,208 bytes，包括 Gemma31 MTP head；兩份 profiles 的 identity 相同。新 CLI 在任何下載前先檢查整個 plan / 既有檔案，已存在且 hash 不符會停止並保留，不覆寫。下載暫存位於 models-dir 下的對應目錄，因此此處明確使用 NVMe。若已有本機模型，改 models-dir 後先省略 --download，只有全部 `verified_existing` 才算已驗證，`planned` 不算。此次實際新主機完整下載仍未演練。

以下先跑四個 checkpoint 的 pilot，再跑500。Kimi各只派一個模式。每 job 串行、同卡一個 generator，並保存成功和失敗 run 的實際 profile。`verify_generation_profile.py` 補上現有 uploader 未強制的 runtime / Gemma31 MTP8 檢查；uploader 的 dry-run 再驗完整 cohort 並產生本地 Parquet，不會發布。

```bash
lanes=(gemma-4-31b-it:instruct gemma-4-31b-it:thinking qwen3.6-35b-a3b:instruct qwen3.6-35b-a3b:thinking kimi-vl-a3b-instruct:instruct kimi-vl-a3b-thinking-2506:thinking)
sources=(blink-val mathvision-test mmmu-pro-standard-10 mmmu-pro-vision mmstar ocrbench-v1 ocrbench-v2)
for size in 100 500; do
  if [ "$size" = 100 ]; then cohort=pilot100; else cohort=collect500; fi
  profile="$SKYMIZER_REMAINING_ROOT/protocol/reference-model-profiles-$cohort.json"
  for lane in "${lanes[@]}"; do
    model="${lane%:*}"
    mode="${lane#*:}"
    if [ "$mode" = instruct ]; then suffix=ins; else suffix=think; fi
    for source in "${sources[@]}"; do
      config="$source-subsample-$size-$suffix"
      run="$SKYMIZER_REMAINING_ROOT/runs/size$size/$model/$config"
      log="$SKYMIZER_REMAINING_ROOT/logs/size$size-$model-$config"
      if [ -e "$run" ]; then
        printf 'existing_run\t%s\n' "$run" >> "$SKYMIZER_REMAINING_ROOT/job-status.tsv"
        continue
      fi
      if "$SKYMIZER_REMAINING_PY" "$SKYMIZER_REMAINING_TOOLS/cli/generate_model_reference.py" \
        --profiles "$profile" --model "$model" --mode "$mode" --source "$source" --size "$size" \
        --hardware pro6000 --gpu 0 --parallel 1 --models-dir "$SKYMIZER_REMAINING_MODELS" \
        --llama-reference "$SKYMIZER_REMAINING_BINARY" --out "$run" > "$log.generate.log" 2>&1; then
        cp "$profile" "$run/selected-profile.json"
        sha256sum "$profile" > "$run/selected-profile.sha256"
        if "$SKYMIZER_REMAINING_PY" "$SKYMIZER_REMAINING_ROOT/protocol/verify_generation_profile.py" \
          --tools "$SKYMIZER_REMAINING_TOOLS" --run "$run" --profiles "$profile" --model "$model" --mode "$mode" \
          > "$log.verify.log" 2>&1; then
          if "$SKYMIZER_REMAINING_PY" "$SKYMIZER_REMAINING_TOOLS/cli/upload_reference.py" \
            --run "$run" --profiles "$profile" --model "$model" --mode "$mode" --private --dry-run > "$log.prepare.log" 2>&1; then
            printf 'prepared\t%s\n' "$run" >> "$SKYMIZER_REMAINING_ROOT/job-status.tsv"
          else
            printf 'prepare_failed\t%s\n' "$run" >> "$SKYMIZER_REMAINING_ROOT/job-status.tsv"
          fi
        else
          printf 'verification_failed\t%s\n' "$run" >> "$SKYMIZER_REMAINING_ROOT/job-status.tsv"
        fi
      else
        if [ -d "$run" ]; then
          cp "$profile" "$run/selected-profile.json"
          sha256sum "$profile" > "$run/selected-profile.sha256"
        fi
        printf 'generation_failed\t%s\n' "$run" >> "$SKYMIZER_REMAINING_ROOT/job-status.tsv"
      fi
    done
  done
done
```

`existing_run` 不代表完成，也不是跨呼叫 resume。失敗、repetition exclusion 與 cap 截斷依第 5 / 6 節保存；不補抽或自動縮 cap。若 eligible=0，uploader 拒絕發布。若在建立 run 前失敗，保留 protocol/、generate log 與 job-status。每個正式 job 的完整 requested/eligible/excluded/failed partition 與 published receipt 都是交付的一部分。

發布時重新核對同一份 profile。例如 Gemma31 pilot instruct 的 MMMU Vision：

```bash
set -euo pipefail
model=gemma-4-31b-it
mode=instruct
profile="$SKYMIZER_REMAINING_ROOT/protocol/reference-model-profiles-pilot100.json"
run="$SKYMIZER_REMAINING_ROOT/runs/size100/$model/mmmu-pro-vision-subsample-100-ins"
"$SKYMIZER_REMAINING_PY" "$SKYMIZER_REMAINING_ROOT/protocol/verify_generation_profile.py" \
  --tools "$SKYMIZER_REMAINING_TOOLS" --run "$run" --profiles "$profile" --model "$model" --mode "$mode"
"$SKYMIZER_REMAINING_PY" "$SKYMIZER_REMAINING_TOOLS/cli/upload_reference.py" --private \
  --run "$run" --profiles "$profile" --model "$model" --mode "$mode" --dry-run
"$SKYMIZER_REMAINING_PY" - "$run/upload/manifest.json" <<'HF_PRIVATE'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi
repo = json.loads(Path(sys.argv[1]).read_text())["repo"]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
if api.repo_info(repo, repo_type="dataset").private is not True:
    api.update_repo_settings(repo, repo_type="dataset", private=True)
if api.repo_info(repo, repo_type="dataset").private is not True:
    raise SystemExit("HF dataset is not private; publication stopped")
print("Verified private HF dataset: " + repo)
HF_PRIVATE
"$SKYMIZER_REMAINING_PY" "$SKYMIZER_REMAINING_TOOLS/cli/upload_reference.py" --private \
  --run "$run" --profiles "$profile" --model "$model" --mode "$mode"
```

500 使用對應 collect500 profile、size500 與 -subsample-500 config；所有 reference datasets 必須為 private。上述流程先重新 dry-run 驗收同一組參數，再依 upload/manifest.json 建立 private repo，或將既有 public repo 改為 private，再次核對成功後才執行 uploader --private；權限不足或仍非 private 時停止。Repository 命名沿用第 6 節，每 model/mode/source 分 config，split=train。不覆寫既有不同內容；發布結果以 upload/receipt.json 的精確 Hub commit 與內容核對為準。這份交接沒有代執行 Hub 發布。

## 8. 其餘 SNR 模型的驗收狀態

84 個命令建構、12 個 publisher model-identity adapter 檢查、11 檔 S3 restore plan 對照均通過。216 個相關回歸 tests、5 項獨立 restore 檢查也通過。另有 20 組 VLM KLD profile/runtime/命令的 CPU 對照通過，這不代表候選核准或正式 KLD 收集完成。

六個 model/mode lanes 各使用新 pin 的兩個 BLINK 樣本，走 high-level launcher 與完整 pilot generation cap，完成 12 列、13176 generated tokens。Driver 170 checks 與補充 108 checks 全通過；另外六份真實 metadata 均通過包內 verify_generation_profile.py。停止結果為 eos=11, limit=1，沒有失敗、timeout 或 repetition exclusion。

| Checkpoint / mode | 兩列 generated tokens | 停止原因 | Job wall seconds | Peak VRAM MiB |
| --- | --- | --- | ---: | ---: |
| gemma-4-31b-it/instruct | 282, 193 | eos, eos | 70.281 | 68109 |
| gemma-4-31b-it/thinking | 880, 1150 | eos, eos | 94.698 | 68105 |
| kimi-vl-a3b-instruct/instruct | 4, 4 | eos, eos | 124.702 | 33071 |
| kimi-vl-a3b-thinking-2506/thinking | 96, 247 | eos, eos | 126.497 | 33071 |
| qwen3.6-35b-a3b/instruct | 455, 395 | eos, eos | 290.004 | 68599 |
| qwen3.6-35b-a3b/thinking | 1278, 8192 | eos, limit | 151.135 | 68599 |

Qwen3.6 thinking 的 limit stop 是有效的 budget 截斷，不能當自然 EOS；保留該列與截斷狀態。Job wall 包含載入、生成、完整權重 hashing 與輸出處理，不是純 decoding 時間。Gemma31 已核對 Q8_0 head 完整 SHA256、MTP draft=8、target raw logits；其他 lanes 為 AR。

本輪六 jobs 自 2026-09-06T16:31:15.189367+00:00 開始，最後一個 job 於 2026-09-06T16:45:33.117137+00:00 結束，早於 2026-09-06 17:00 UTC 硬截止。GPU 清理與 deadline 收據另存於包內 evidence/gpu/；此後沒有追加 GPU probes。正式 100 / 500 collection 預算另計。

可攜證據見補充包 evidence/gpu/FINAL_ACCEPTANCE_REPORT.md、acceptance-summary.json、row-results.json 與各 lane 的驗收 / metadata / command。完整本機 artifacts 位於 `/opt/dlami/nvme/skymizer-reference-next-20260906/gpu-acceptance`。獨立 agent 最終複查見包內 evidence/final-independent-review/。

外部 profiles 的 validation 文字保留凍結當時狀態；後續驗收以包內 evidence 的帶時間收據為準，不改寫已用於生成的 profile bytes。這十二列不代表完整 100 / 500 題、七個 subsets 全量、新主機完整還原、Hub 發布、全部 candidates 或 native SNR 已驗收。
