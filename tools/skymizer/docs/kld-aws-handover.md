# AWS：收集 KLD metrics

使用共用 AMI 中的 `~/projects/llama.cpp` 與 `~/models`，交付版本 `6851898e6`。輸入是已發布的 llama.cpp reference dataset。本階段只收 per-token metrics，包含 EAR_64、EAR_64_normalized；不執行 paired test、N80 或分析報告。

## 1. 準備 scorer

```bash
cd ~/projects/llama.cpp
git checkout 6851898e6
cmake --build build -j 8 --target llama-vlm-kld llama-llm-kld
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project tools/skymizer --locked --no-dev
nvidia-smi
```

若 reference dataset 需要登入權限，執行 `.venv/bin/hf auth login`。模型直接使用 `~/models`。RunPod 與 AWS 的 CUDA 可以不同；**同一 model/subset 的所有 candidate 固定在同一張 AWS GPU、同一 scorer build／CUDA 環境收集**，以保留未來可比較性。

## 2. 收一個 model/subset/candidate

下例收 qwen3.5-4b、MMMU vision instruct、bartowski Q4_K_M。第一次執行會自動建立 AWS 自己的 study、封存 scripts；後續工作使用封存版本，不需要從 RunPod 搬 scripts 或 build。

```bash
cd ~/projects/llama.cpp
.venv/bin/python tools/skymizer/cli/collect_model_kld.py \
  --study "$HOME/gap/kld-pivot" --size 100 \
  --models-dir "$HOME/models" --gpu 0 \
  --model qwen3.5-4b --source mmmu-pro-vision --mode instruct \
  --candidate bartowski-Q4_K_M \
  --cand-model "$HOME/models/qwen3.5-4b/bartowski/Qwen_Qwen3.5-4B-Q4_K_M.gguf" \
  --llama-vlm-kld "$HOME/projects/llama.cpp/build/bin/llama-vlm-kld"
```

`--candidate` 是此候選組合的目錄名稱；`--cand-model` 必須指定實際 GGUF，分片模型指定第一片。預設 candidate 與 reference 使用相同 BF16 mmproj；要測量化 projector 才加 `--cand-mmproj <檔案>`，並給該組合不同 candidate 名稱。

其他工作只換 `--model`、`--source`、`--mode`、candidate 名稱／檔案。來源可選 `blink-val`、`mathvision-test`、`mmmu-pro-standard-10`、`mmmu-pro-vision`、`mmstar`、`ocrbench-v1`、`ocrbench-v2`。混有文字和圖片的 subset 都使用此 VLM 入口，它也接受零圖片的文字題。

- `--mode thinking` 自動使用 `-think` config，評估前 **4096** 個生成 tokens；instruct 評估前 **2048** 個。
- `--size 100` 讀取 `elichen-skymizer/<model>-pivot`；500 題改 `--size 500 --study "$HOME/gap/kld-collect-500"`，讀取 `<model>-collect-500`。
- 加 `--dry-run` 可以先看完整 collector 命令與 `study.json`，不載入權重或資料。
- 每 GPU 同時只跑一個 scorer。AI agent 每次派一個工作；成功或失敗後都接下一個，保留失敗紀錄供稍後處理。

## 3. 固定參數

入口自動沿用對應 reference profile 的 `ctx`、`batch`、`ubatch`；不要逐 candidate 自行調整。

| 模型 | ctx | batch | ubatch | tf_chunk |
|---|---:|---:|---:|---:|
| qwen3.5-4b | 32768 | 2048 | 512 | 2048 |
| qwen3.6-35b-a3b | 32768 | 2048 | 512 | 2048 |
| gemma-4-e4b-it | 32768 | 2048 | 512 | 2048 |
| glm-4.6v-flash | 32768 | 2048 | 512 | 2048 |
| gemma-4-26b-a4b-it | 32768 | 2048 | 2048 | 2048 |
| gemma-4-31b-it | 32768 | 2048 | 2048 | 2048 |

共同設定：`threads=8`、`metric_threads=8`、flash attention 開啟、F16 KV、SWA full 關閉、`n_gpu_layers=-2`（所有 layers）、**MTP 關閉**。圖片 budget 自動採用 reference dataset 保存的設定，不另加 image-min/max overrides。Scorer 不使用生成 sampler。

`tf_chunk=2048` 可以大於 `ubatch=512`：一般 causal teacher forcing 會再切 micro-batch。Gemma 26B/31B 的完整 non-causal image chunk 則需要 `batch/ubatch=2048/2048` 的容量；不要縮小圖片來避開容量錯誤。

KLD 只評估生成位置，prompt 不計分，但圖片／prompt 仍佔 context。Thinking tokens 包含在 4096 上限內。請使用上述 answer cap，不要拿 `max-total-tokens=2048/4096` 代替。短答案只計已有 tokens。

這是固定收集設定，不保證任意大小的 candidate 都放得進單卡；reference、candidate、KV 和工作空間必須同時容納。遇到 OOM／context 錯誤先保留失敗並換下一個工作，不對個別 candidate 自動降參數。需要新容量或新 protocol 時，為整組 candidate 開新 study；降低 tf_chunk 無法解決權重本身超過 VRAM 的情況。

## 4. 結果位置與狀態

```text
~/gap/kld-pivot/
  study.json                     所有模型的固定設定總覽
  status.json                    已提交 KLD 工作的狀態
  scripts/manifest.json
  scripts/skymizer/{cli,lib,scripts,...}
  scripts/provenance/
  artifacts/<model>/<subset>/kld/<candidate>/
    metrics/*.npz                KLD、EAR_64、EAR_64_normalized 等 metrics
    manifest.csv                 每題 OK／失敗狀態、實際評估長度
    collect_meta.json            實際模型、資料與 scorer 環境
    collect.log                  完整 collector 輸出
    collect.command.json         實際執行指令
```

例如：`~/gap/kld-pivot/artifacts/qwen3.5-4b/mmmu-pro-vision-subsample-100-ins/kld/bartowski-Q4_K_M/metrics/`。

保留整個 study。`status.json` 每個 collection 的 `complete` 才代表收集完成；根層 `idle` 只表示目前沒有正在執行的已提交工作，不代表所有候選都已派完。中斷與失敗會保留既有 metrics、logs 和內部 attempt 狀態。

同一路徑不覆寫，也不把未完成結果當完成。需要重做失敗工作時使用新的 study 或新的明確 retry 目錄名稱；agent 繼續派其他工作。完成後交付此 study 目錄供後續分析。
