# RunPod：產生 reference

專案 `/workspace/llama.cpp`，交付版本 `6851898e6`。使用 RTX PRO 6000 Blackwell 96 GiB；每張 GPU 同時一個生成工作。模型由資料擁有者預先放在 `/workspace/models`，包含 profile 指定的 BF16 shards、mmproj 與啟用的 MTP head。

## 1. 建置與登入

在已取得交付版本的 checkout 執行。`/workspace` 是永久儲存空間，專案、模型與輸出都放在這裡。

```bash
cd /workspace/llama.cpp
git checkout 6851898e6
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 8 --target llama-reference
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project tools/skymizer --locked --no-dev
.venv/bin/hf auth login
nvidia-smi
```

使用 RunPod 當地的 CUDA toolkit 建置即可，不必與 AWS 同版；toolkit 必須支援 Blackwell（CUDA 12.8+）。若有多個 CUDA 安裝，配置 CMake 時明確指定 `CMAKE_CUDA_COMPILER` 與 `CUDAToolkit_ROOT`，使用新的 build 目錄。HF 帳號需有 `elichen-skymizer` dataset 寫入權限。

## 2. 開始生成

在 tmux 中執行下列指令。四張卡會共用工作佇列，完成或失敗一個任務後自動接下一個。

```bash
cd /workspace/llama.cpp
.venv/bin/python tools/skymizer/cli/run_reference_campaign.py \
  --out /workspace/reference-pivot \
  --models-dir /workspace/models --gpus 0,1,2,3 \
  --size 100 --hardware pro6000
```

六張卡改 `--gpus 0,1,2,3,4,5`；單卡改 `--gpus 0`。多台 RunPod 各自分配不同模型，例如加 `--models gemma-4-31b-it`，不要在不同機器重複派同一 model/subset/mode。

500 題用新輸出目錄並改 `--size 500`：

```bash
.venv/bin/python tools/skymizer/cli/run_reference_campaign.py \
  --out /workspace/reference-collect-500 \
  --models-dir /workspace/models --gpus 0,1,2,3 \
  --size 500 --hardware pro6000
```

來源固定為 `elichen-skymizer/vlm-prepared-dataset` 的七個 subset：BLINK、MathVision、MMMU standard、MMMU vision、MMStar、OCRBench v1/v2。完整預設是六模型 × 七 subset × instruct/thinking，共 84 個任務；每個任務請求 100 或 500 題。Muse 不產生 reference。

## 3. 固定參數

入口自動從 `reference_model_profiles.json` 取值；不需要手動組裝 native 參數。生成長度上限：instruct **8192**、thinking **16384**；遇到 EOG 提早停止。

| 模型 | batch | ubatch | 生成方式 |
|---|---:|---:|---|
| qwen3.5-4b | 2048 | 512 | AR |
| qwen3.6-35b-a3b | 2048 | 512 | AR |
| gemma-4-e4b-it | 2048 | 512 | AR |
| glm-4.6v-flash | 2048 | 512 | AR |
| gemma-4-26b-a4b-it | 2048 | 2048 | MTP，最多提案 3 tokens |
| gemma-4-31b-it | 2048 | 2048 | MTP，最多提案 8 tokens |

兩種 mode 使用相同 runtime 表。共同設定：`ctx=32768`、`threads=8`、`threads_batch=8`、flash attention 開啟、F16 KV、所有 layers 放 GPU、單 sequence、`fit=off`、`seed=1234`。image token budget 使用 GGUF/mtmd 預設；sampling 使用 profile 中的 model-card 設定。`study.json` 會逐模型列出完整 runtime、sampling args 與輸出位置。

Gemma 26B/31B 的整個 non-causal image chunk 必須同時放得進 batch 與 ubatch，**不要把 ubatch 降回 512**。Context 包含圖片、prompt 與生成內容；32768 是總容量，不是額外可生成 32768 tokens。不要為單題錯誤偷偷改 cap、圖片大小或 sampling。

## 4. 產物與完成判定

預設自動上傳 HF：

- 100 題：`elichen-skymizer/<model-name>-pivot`
- 500 題：`elichen-skymizer/<model-name>-collect-500`
- config：原來源 config 加 `-ins` 或 `-think`，例如 `mmmu-pro-vision-subsample-100-think`；split=`train`。

重複文本會停止生成並排除，失敗題也會記錄；上傳的是 eligible rows，所以最終筆數可能少於 100/500，不補抽。只有已驗證上傳的任務才算交付完成。

保留整個輸出目錄即可，主要看這些檔案：

```text
/workspace/reference-pivot/
  study.json                     設定總覽
  status.json                    生成進度與各任務結果
  scripts/manifest.json           scripts 的 checksum
  scripts/skymizer/{cli,lib,scripts,...}
  scripts/provenance/             實際原始碼與環境紀錄
  artifacts/<model>/<subset>/     該任務的 logs、狀態與上傳紀錄
    upload-status.json           verified 表示該任務的 HF 上傳已驗證
    attempt-0001/dataset/         本地 reference dataset
```

單題／單任務錯誤會保留結果並繼續其他工作。終端中斷後，在**同一環境**執行原命令加 `--resume`；換 CUDA、重建執行檔或換機器，請開新 study。`status.json` 最終 `complete` 表示全部任務與要求的上傳成功；`complete_with_failures` 表示仍有例外要處理。磁碟滿、機器關閉等環境問題需先排除。

AWS 執行者直接從已發布的 HF config 收 KLD，不需要搬 RunPod 的 build 或整個工作目錄。
