# Spec: register the google/gemma-4 VLM family in prep_vlm_score_from_hf.py

> STATUS: DONE — the family is registered (MODEL_FAMILIES / IMAGE_PAD_TOKENS
> carry `google/gemma-4`) and `smoke_vlm_gemma4.sh` now drives the KLD-only
> pipeline against real GAP rows. Kept as the historical record of the
> verified wrapper facts; file references below describe the tree as it was
> in 2026-07 (the dense-logits lane and its checkers have since been
> removed).

Goal: `smoke_vlm_gemma4.sh`(同目錄)能以 gemma-4-E4B 跑完六步。目前卡在
`PrepError: unsupported model family: 'google/gemma-4-E4B-it'`。

## Verified facts(2026-07-28 session;實作時不必重新推導,但歡迎覆核)

1. **Dataset 解碼**(`elichen-skymizer/ground-truth-mmmu-pro-vision` /
   `gemma-4-E4B-it-enable-thinking-gen-4096`,以 gemma tokenizer 解碼
   `input_ids[:n_prefill_tokens]`):影像區塊 =
   `<|image>` (1 tok) + `<|image|>` × N + `<image|>` (1 tok),
   N 動態(row 0 = 270)。`generation_model_name_or_path` =
   `google/gemma-4-E4B-it`。chat 標記為 `<|turn>`/`<turn|>`,thinking 為
   `<|think|>` —— 對 prep 無關(decode-and-collapse 原樣保留)。
2. **mtmd 端**(`../mtmd/mtmd.cpp:479-485`):`PROJECTOR_TYPE_GEMMA4V` 自行
   注入 `img_beg="<|image>"`、`img_end="<image|>"`,preprocessor 為
   `mtmd_image_preprocessor_dyn_size` → 現行 `MEDIA_WRAPPER_MODE="stripped"`
   收合語義對 gemma-4 成立(wrapper 必須剝掉,mtmd 會補回)。
   mmproj metadata:`clip.vision.projector_type=gemma4v`、`image_size=224`、
   `patch_size=16`。
3. **`image_processor_config`(per-row JSON)schema**(與 Qwen 完全不同,
   無 `merge_size`/`min_pixels`/`max_pixels`):

   ```json
   {
     "gemma4_processor_config": {
       "boi_token": "<|image>", "eoi_token": "<image|>",
       "image_seq_length": 280, "image_token": "<|image|>",
       "image_token_id": 258880, "tokenizer_image_token": "<|image|>" },
     "gemma4_vision_token_config": {
       "max_soft_tokens": 280, "patch_size": 16, "pooling_kernel_size": 3 },
     "gemma4_vision_tokens_override_by_vllm": false,
     "hf_image_processor": {
       "image_processor_type": "Gemma4ImageProcessor",
       "image_seq_length": 280, "max_soft_tokens": 280,
       "patch_size": 16, "pooling_kernel_size": 3, ... }
   }
   ```

4. 現行 `IMAGE_PAD_TOKEN = "<|image_pad|>"` 是模組級常數,經
   `resolve_image_pad_id(tokenizer)` 被 `make_mini_ground_truth.py:153-154`
   與 `check_hf_vs_dump.py:130-131` 消費。gemma-4 的 pad token 是
   `<|image|>`(id 258880)。

## Required changes(全在 tools/skymizer;不碰 C++、不碰 collectors)

### A. `MODEL_FAMILIES` 註冊

`prep_vlm_score_from_hf.py:141` 加:

```python
_GEMMA4_IMAGE_BLOCK_RE = re.compile(r"<\|image>(?:<\|image\|>)+<image\|>")
MODEL_FAMILIES = { ..., "google/gemma-4": _GEMMA4_IMAGE_BLOCK_RE }
```

多影像相鄰時每個區塊獨立匹配(regex 不可能跨過 `<image|>`)。更新模組
docstring 與 registry 註解(記入上面 fact 1/2 的出處)。

### B. `IMAGE_PAD_TOKEN` per-family 化

- 以 per-family 對照表取代單一常數,並提供
  `image_pad_token_for(model_name)`(longest-prefix-wins、未知 family 拋
  `PrepError` —— 與 `image_block_regex_for` 同語義;可抽共用 helper)。
- `resolve_image_pad_id(tokenizer)` → `resolve_image_pad_id(tokenizer,
  model_name)`,round-trip 驗證邏輯保留。
- 更新兩個呼叫端(`make_mini_ground_truth.py`、`check_hf_vs_dump.py` ——
  兩處 scope 內都有 model name,確認後接上)。不保留舊常數;一併更新引用
  它的測試。

### C. `derive_image_token_limits()` per-schema dispatch

- Qwen 路徑:`merge_size` 存在 → 現行邏輯不變。
- gemma-4 路徑:`hf_image_processor.image_processor_type ==
  "Gemma4ImageProcessor"`(或 `gemma4_vision_token_config` 存在)→ 回傳
  `{"patch_size", "pooling_kernel_size", "max_soft_tokens",
  "image_min_tokens": -1, "image_max_tokens": <max_soft_tokens>}`,
  docstring 註明:min=-1 表示交給模型 metadata;這些欄位是 provenance
  記錄,collector 預設 -1/-1、只有顯式傳遞時才生效。
- 未知 schema → `PrepError`(fail-closed),不得 KeyError。

### D. Hermetic tests(`tests/test_prep_vlm_score_from_hf.py`,numpy-only)

- gemma regex:單影像、多影像相鄰各自收合;裸 pad run(無 wrapper)不產生
  marker(交由 count check 攔截)。
- `image_pad_token_for`:兩個 Qwen prefix、gemma prefix、未知 family
  PrepError、longest-prefix-wins。
- `derive_image_token_limits`:gemma schema 正確輸出;未知 schema PrepError;
  Qwen 路徑回歸不變。
- `resolve_image_pad_id` 新簽名的 round-trip / 拒絕測試更新。

### E. 明確不做

- 不改 `MEDIA_WRAPPER_MODE` 語義(gemma 同為 stripped,fact 2)。
- 不碰 C++、collectors、exporter。
- 不 git commit。

### F. 附帶調查(報告即可,不實作)

mtmd `gemma4v` 的 dyn_size 前處理對每張圖產生的 vision token 數,必須與
HF `Gemma4ImageProcessor` 完全一致(row 0 = 270),否則 teacher-forcing
整體錯位。請閱讀 `vlm-score.cpp` 回報:scorer 在哪裡(file:line)偵測
「mtmd 組裝出的 token 序列」與 `tokens.bin` 的長度/內容不一致?若沒有這
道檢查,明確指出 —— 這決定驗收時錯位是 fail-loud 還是 silent。

## 驗收(由 review 方執行)

1. 全套 hermetic pytest 綠(現況 484 passed / 1 skipped,不得變紅)。
2. CLI 單 row prep:`prep_vlm_score_from_hf.py --row 0 --out <tmp>`(gemma
   dataset)→ `formatted_chat.txt` 恰 1 個 `<__media__>`、meta.json 的
   image_token_limits 正確。
3. `smoke_vlm_gemma4.sh` 六步全過(GPU)。
