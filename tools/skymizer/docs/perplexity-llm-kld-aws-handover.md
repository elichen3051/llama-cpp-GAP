# AWS: llama-perplexity 與 LLM KLD 文字收集

更新日期: 2026-09-06。此路線依目前決策延後，先完成 [reference 生成](/home/ubuntu/llamacpp_reference_runpod_handover.md) 與 [VLM KLD 收集](/home/ubuntu/llamacpp_kld_aws_handover.md)。本文件保存已確認的評估契約與缺口，不代表文字橋接已可直接正式執行，也不包含本輪的新實作。

## 1. 目的與比較範圍

LLM KLD 作為 llama-perplexity 與 VLM KLD 之間的文字橋接，是因為 llama-perplexity 保存 reference logits 時使用較低精度表示。要分辨模型量化差異與該儲存路徑引入的偏差，兩個文字 scorer 必須先使用同一份 corpus、完全相同 token windows、BOS 處理與計分 targets。

Model family 指同一基底 checkpoint 的所有量化版本，不包含同系列不同 size。每個基底內的 candidates 使用完全相同 runtime；不同基底可分別凍結。LLM KLD 與 llama-perplexity 對齊本文件的 512-token 協定。VLM KLD 的 context / batch / ubatch 依圖片容量另訂，不需要跟這條文字路線相同。

模型與 SNR / 最終評估先後順序沿用 reference handover；Kimi Instruct 與 Thinking-2506 分開。Candidate 清單仍未定案，不能因本文件列出協定就宣告所有工作已規劃完成。

## 2. Corpus identity 與 512-token 切窗

預計測 Wikitext2 與 PG。正式版本、split、下載 revision 與 PG 的完整資料集名稱尚待固定，不能自行把 PG 解讀成某個特定 corpus。

每份 corpus 需要保存來源、精確 revision、檔案 SHA256、文件順序、換行 / 文件分隔、任何前處理，以及原始 bytes。兩個文字 scorer 讀同一份凍結文字檔，使用同一基底 GGUF tokenizer / vocabulary。不要各自下載、清理或串接內容。

採 `ppl_stride=0` 的 classic chunk protocol:

1. 依 llama-perplexity 的 `common_tokenize(..., add_special=true)` 處理整份 corpus，保存產生的完整 token IDs 與 tokenizer identity。
2. 按來源順序切不重疊 512-token windows。`chunks=-1` 代表所有完整 windows；不足 512 的尾段不另補齊、不計分。目前 PPL 也要求輸入至少有 1024 tokens。
3. 每個 window 開始清空 KV。模型要求 BOS 時，依目前 PPL 實作替換 window 位置 0 的 token，不是在 window 前插入一個額外 token。
4. 以位置 256..510 的 logits 計分位置 257..511 的 next-token targets，每個完整 window 正好 255 個 targets。

整份 corpus 的 add-special 與每窗 BOS 替換是兩個步驟，兩者都要對齊。不能只看兩邊總 token 數相同就判定切窗一致。每個 window 至少保存來源 offset、實際輸入 token IDs、BOS 決策、255 個 target IDs 與對應 logits positions。

若轉成現有 LLM KLD 的 prompt / answer 邊界，`n_prefill=257, n_eval=255` 才與 PPL 的 targets 相同。這只對齊計分位置，仍未對齊 decode 形狀。

## 3. 固定 runtime

| 設定 | llama-perplexity | LLM KLD |
| --- | --- | --- |
| n_ctx | 512 | 512 |
| n_batch | 512 | 512 |
| n_ubatch | 512 | 512 |
| n_seq | `max(1, n_batch/n_ctx)=1` | 1 |
| n_threads / n_threads_batch | 8 / 8 | 8 / 8 |
| metric_threads | 目標 8，但目前 CLI 不能固定 PPL metric worker 數 | 8 |
| flash attention | on | enabled |
| GPU offload | 所有 layers | 所有 layers |
| KV dtype | F16 K/V | F16 K/V |
| SWA | `swa_full=false` | `swa_full=false` |
| fit | off | 不使用自動 fit |
| MTP | 關閉 | 關閉 |

`-c 512 -b 512` 使 PPL 推導 `n_seq=1`，因此只有一個 512-token window，不再是舊設定 `b=2048` 時的四個並行 windows。`swa_full=false` 指不配置 full-size SWA cache，不是停用模型原生 sliding-window attention。

硬體為 NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GiB，每張卡同時只跑一個 scorer。同一基底全部 candidates 固定 binary、backend、driver / CUDA、硬體與 runtime。LLM KLD 同時載入 reference 和 candidate，仍須確認雙模型與 KV / workspace 的容量；PPL 能單獨跑某個模型不代表雙模型 scorer 能容納。

目前 PPL 的 metric workers 來自 `std::thread::hardware_concurrency()`，classic 路徑建立 `hardware_concurrency()-1` 個 workers，再由主執行緒參與計算；`--threads 8 --threads-batch 8` 只控制 inference threads，不會將 metric workers 固定為 8。這是正式要求完全對齊前必須處理的缺口，不能在 metadata 中聲稱已符合。

## 4. 尚未完成的橋接

目前 `collect_llm_kld.py` 接受已有 prompt / answer 邊界的資料，不會直接把 Wikitext2 / PG 轉成 PPL windows。`collect_model_kld.py` 是 VLM collector 的高階入口，也不是文字 corpus adapter。

正式橋接至少需要:

- 一個共用 corpus / tokenizer / window 準備流程，固定每窗 512 個輸入與 255 個 targets，保存上述來源 identity。
- 與 PPL 相同的整窗 decode 路徑。現在 PPL 在 `c=b=ub=512` 時一次 decode 512 tokens；既有 LLM KLD 對 `n_prefill=257, n_eval=255` 先 decode 257，再 teacher-force 254 個 tokens。雖然 targets 相同，浮點計算的 batch 形狀仍不同。
- 明確固定 PPL metric worker 數的方式，以及完整 runtime / target identity 驗證。
- 對同一組 logits 分別使用完整精度 metric 與 PPL 的編碼 / 解碼 / 尾端處理，才能將儲存誤差、尾端截取與 decode 分段誤差分開量測。

PPL 保存路徑不是單純將每個 float 換成 uint16: 它先將表示範圍限制到最大 logit 以下 16，再做 uint16 編碼；讀取後的 KLD 計算又略過 reference log-probability <= -16 的項。一般 PPL 的 NLL 則直接由當下 logits 計算。LLM KLD 在兩側 logits 同時存在記憶體時直接算 metrics，因此可以作為精度橋接，但在上述差異隔離前不能把結果差異全歸因於「儲存精度」。

本輪只記錄設計，暫不實作 adapter、整窗 decode 或 PPL worker 控制。

## 5. 延後執行的 PPL 命令範本

完成 corpus 與 binary 凍結後，PPL baseline 可使用下列參數。`SKYMIZER_PPL_BINARY` 指向 NVMe build 的 `llama-perplexity`，`SKYMIZER_TEXT_CORPUS` 指向 NVMe 上已核對 SHA256 的同一份 corpus。這不是已完成的 paired bridge，也不會將 PPL metric workers 固定成 8。

```bash
CUDA_VISIBLE_DEVICES=0 "${SKYMIZER_PPL_BINARY:?set the pinned NVMe llama-perplexity path}" \
  -m "$HOME/models/qwen3.5-4b/bartowski/Qwen_Qwen3.5-4B-bf16.gguf" \
  -f "${SKYMIZER_TEXT_CORPUS:?set the frozen NVMe corpus path}" \
  -c 512 -b 512 -ub 512 -t 8 -tb 8 -ngl all \
  -fa on -ctk f16 -ctv f16 --fit off \
  --ppl-stride 0 --chunks -1
```

所有新建 corpus、token windows、logits、metrics、logs、cache、build 與暫存檔都放 `/opt/dlami/nvme`，模型沿用 `~/models`。執行前沿用另外兩份 handover 的 `TMPDIR / HF_DATASETS_CACHE / HF_HUB_CACHE / UV_CACHE_DIR` 設定。不要把預計很大的 logits 檔留在 root disk。

LLM KLD 的正式收集命令等共用 corpus adapter 與整窗 decode 完成後再補；現在列出既有 collector 命令會讓操作人員誤以為協定已一致。

## 6. 正式收集前的驗收

同一 window 的實際輸入 IDs、BOS、255 個 targets 必須逐一一致，並驗證全 corpus 完整 window 數與丟棄尾段。基底、candidate、corpus、tokenizer、runtime、binary、metric worker、編碼路徑 identity 一起保存。各候選使用相同資料順序，輸出不覆寫，失敗使用新 attempt 並保留舊紀錄。

驗證需分開報告 PPL NLL、直接 logits KLD、PPL 編碼後 KLD，以及 decode 路徑是否一致。只有這些因素都可追溯時，才可將 LLM KLD 當作本次量化比較的正式文字橋樑。

完整跨流程異常與證據見 [本輪稽核清單](/opt/dlami/nvme/skymizer-length-audit-20260906/ALL_FINDINGS.md)。
