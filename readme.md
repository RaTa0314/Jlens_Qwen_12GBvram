# J-Lens on Qwen2-Audio (12GB VRAM Optimization)

模型：Qwen2-Audio-7B-Instruct
---

## 硬體限制與壓縮技術權衡

為了在 12GB 顯存下塞得下一個 7B 模型 + Jacobian 擬合所需的反向傳播，本專案採用：

- **4-bit (NF4) 量化**：模型權重約壓到 5.9GB，留下空間給擬合階段的計算圖
- **梯度凍結 + `start_graph_at` 機制**：讓 J-Lens 官方函式庫的計算圖只從你要探測的那一層開始建，不用整條網路都保留梯度
- **音訊特徵快取**：Whisper 音訊編碼器的前向計算只跑一次（`no_grad`），快取結果重複使用，避免每次反向傳播的 batch 複製都要重跑一次音訊編碼器（這是 12GB 顯卡上最容易爆記憶體的地方）

代價：
- 4-bit 壓縮會讓模型的數值精度打折，J-Lens 學到的特徵映射是基於「量化過的」模型內部狀態，不是原始 bf16/fp16 精度下的狀態
- 反向傳播需要即時反量化權重，運算比全精度慢
- 受限於時間預算，fit 語料規模控制在 100 筆量級（官方文件建議這個量級足以 fit 出「堪用」的 lens，但跟論文裡用大規模算力 fit 出的版本相比，雜訊會明顯更多）

---
## 環境

| 項目 | 版本 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA GeForce RTX 3060 (12GB / 實測可用 ~11.6GB) |
| NVIDIA 驅動 | ≥ 550.x |
| CUDA | 12.1 |
| Python | 3.10 (conda/venv 隔離環境) |
| PyTorch | 2.5.1+cu121 |
| transformers | 5.16.0.dev0（從 source 安裝：`pip install git+https://github.com/huggingface/transformers`，Qwen2-Audio 在較新版本才穩定支援）|
| bitsandbytes | 0.50.1（4-bit NF4 量化）|
| accelerate | 1.14.0 |
| librosa | 0.11.0 |
| soundfile | 0.14.0 |
| jacobian-lens | Apache-2.0，來自 `anthropics/jacobian-lens`，`pip install -e .` 本地安裝 |

---
## 實驗設計

| 實驗 | 訓練語料 | 目的 |
|---|---|---|
| **A：ASR** | 100 筆 FLEURS 短語音（50筆英文 `en_us` + 50筆中文 `cmn_hans_cn`，2-5 秒）| 純語音轉寫任務下，J-Lens 能否比傳統 logit lens 更早讀出正確概念 |
| **C：QA** | 100 筆 ESC-50 環境音問答（5 秒動物/環境音，「請描述這段音訊裡的聲音是什麼」）| 驗證 J-Lens 在非語言類音訊理解任務上的表現 |
| **B：50/50 混合** | 從 A、C 各抽 50 筆混合 | 測試「窄語料分開 fit」vs「多樣語料合併 fit」哪個更好，同時檢驗合併 fit 是否會發生災難性遺忘 |

擬合細節：`dim_batch=4`、全部 32 層一起 fit（`source_layers=0..30`，`target_layer=31`）、每筆 prompt 用「音訊 prompt + ground-truth 逐字稿」teacher-forcing 組成完整輸入序列。

---

## 目前的發現

> 以下數字來自 Phase 5（實驗 A，n=100，**注意：這批數字是在 fit 用過的樣本上測的，不是嚴格 held-out**，僅供內部除錯時的健檢用）。嚴格的 held-out 驗證結果見 Phase 6。

**1. J-Lens 讀出的 top-1 一致率贏過傳統 logit lens：**

| 樣本 | J-Lens (layer 30) | 傳統 logit lens (layer 30) |
|---|---|---|
| EN_0 | 58.6% | 55.9% |
| ZH_0 | 50.0% | 41.8% |

**2. 更有意思的是「早期層提前浮現正確概念」的現象。** 以 EN_0（transcript: *"it is the biggest acquisition in ebay's history"*）為例，在預測 `"ebay"` 這個詞的位置：

| Layer | J-Lens top-1 | 傳統 logit lens top-1 |
|---|---|---|
| 4 | **eBay** | （雜訊）|
| 8 | eBay / ebay / eb | （雜訊）|
| 16 | eBay | 開始有意義（`'s`）|

J-Lens 在第 4 層就讀出「eBay」這個概念，傳統 logit lens 要到第 16 層才開始有意義的輸出——這正是 J-Lens 這類技術存在的核心價值：修正淺層基底空間的偏移，讓早期層的資訊變得可讀。

**3. 跨任務泛化與多任務混合實驗 (Phase 6 嚴格驗證)**

- **Zero-shot 跨任務解碼能力 (以 ASR-Lens 解 QA 資料)：**
  實驗結果展現了令人驚豔的 Out-of-Distribution 泛化能力。即使 J-Lens 在訓練階段 (Phase 4) 僅看過人類語音的逐字稿 (ASR)，完全沒有接觸過 ESC-50 環境音標籤，它依然能看穿模型底層的通用聲音表徵。在 QA 驗證集上：
  - Layer 30：J-Lens 準確率達 **83.9%** 
  - Layer 26：J-Lens 準確率達 **52.0%**（傳統 Logit Lens 在此層通常毫無意義）
  這證明 J-Lens 並非單純死背 ASR 的文字映射，而是真正學到了 Qwen2-Audio 的通用多模態對齊空間。

- **多任務混合訓練 (50/50 實驗 B) 的效能天花板：**
  <TODO: 貼上 mixed_lens 在 ASR 與 QA 的最終準確率。例如：QA 準確率在第 X 層提早突破 80%，而 ASR 準確率維持在 X%>

- **災難性遺忘 (Catastrophic Forgetting) 觀察：**
  <TODO: 根據數據填寫。例如：觀察實驗 B 的結果，單一的 Jacobian 矩陣是否能同時處理兩種截然不同的聲音特徵，而不會因為學習了環境音 (QA) 就導致語音辨識 (ASR) 的準確率大幅下降。>

---

## 實做步驟

| Phase | 內容 | 主要產出 |
|---|---|---|
| **Phase 0** | 環境建置：Ubuntu + CUDA + conda env + 套件安裝，`nvidia-smi` / `torch.cuda.is_available()` 驗證 | 可用的開發環境 |
| **Phase 1** | 模型載入測試：4-bit 載入 Qwen2-Audio-7B-Instruct，跑通單筆音訊+文字推論，量測 VRAM 峰值 | 估計實驗時間+ VRAM 基準數據 |
| **Phase 2** | 資料準備 | 前處理後的資料集 |
| **Phase 3** | J-Lens 擬合基礎設施：改寫/適配 `anthropics/jacobian-lens` 的 `jlens.fitting`，在 decoder 各層 hook 殘差流，實作「音訊編碼器 no_grad + decoder 開梯度」的分段前向邏輯 | 可執行的擬合 pipeline |
| **Phase 4** |fitting| 儲存的 J_l（各層）|
| **Phase 5** | validate on fitting data (WTF) | data |
| **Phase 6** | validate on validation data : )  | data |

---

## 檔案架構

```text
Jlens_Qwen_12GBvram/
├── checkpoints/              # 訓練好的模型權重（.ckpt 訓練中 / .pt 最終封裝）
├── g_data/                   # 資料庫
│   ├── jlens_dataset/        # 實驗 A：100 筆 ASR 訓練集（FLEURS）
│   ├── jlens_dataset_qa/     # 實驗 C：100 筆 QA 訓練集（ESC-50）
│   ├── jlens_dataset_mixed/  # 實驗 B：50/50 混合訓練集
│   └── jlens_val_dataset/    # Phase 6：嚴格隔離、不與任何訓練集重疊的驗證集
├── logs/                     # 終端機輸出紀錄（tee）
├── outputs/                  # 產出的 .json 成績單與 .png 折線圖
│   ├── phase5/               # 基準測試結果
│   └── phase6/               # 泛化與跨任務評估結果
├── archive/                  # 舊版除錯腳本存檔（fix_unembed.py 等）
├── phase2_5050.py            # 生成 50/50 混合訓練集
├── phase4_fit_asr.py         # 核心擬合腳本（4-bit NF4 + 音訊特徵快取防 OOM）
├── phase4_finalize.py        # .ckpt（訓練進度）→ .pt（正式 J-Lens 權重）
├── phase6_evaluation.py      # 雙軌評估：J-Lens vs 傳統 logit lens 的 layer-wise 一致率
└── plot_results.py           # 把 JSON 結果畫成折線圖
```

---

## 各個檔案負責的功能

* **`g_data裡面的檔案`**：生data，可以不動，基本都下載好了
* **`phase4_fit_asr.py`**：fitting，會生成checkpoint(.ckpt)存在 checkpoints/
* **`phase4_finalize.py`**：大容量的訓練進度檔（`.ckpt`）-> 正式 J-Lens 權重檔（`.pt`），同樣存在checkpoints/
* **`phase6_evaluation.py`**：雙軌評估系統。自動載入驗證集，分別計算 J-Lens 與 Baseline (Logit Lens) 在各個 Layer 的預測一致率，並輸出綜合報表。
* **`plot_results.py`**：讀取評估輸出的 JSON 數據，繪製並排的 ASR、QA、Combined 三合一學術級折線圖。

## 操作指南

確保已啟動虛擬環境並位於專案根目錄。

**1. 生成資料（如需要）**

```bash
python g_data/phase2_5050.py
```

**2. 執行擬合**

```bash
python phase4_fit_asr.py \
  --jsonl-name <資料集路徑，例如 jlens_dataset_mixed/prompts.jsonl> \
  --checkpoint-path <輸出 .ckpt 路徑> \
  2>&1 | tee <log 路徑>
```

支援中斷續跑：直接重新執行同一條指令，會自動從最後一次 checkpoint 接續。

**3. 封裝權重**

```bash
python phase4_finalize.py \
  --jsonl-name <同上> \
  --checkpoint-path <上一步的 .ckpt> \
  --output-path <輸出 .pt 路徑>
```

**4. 執行驗證評估**

```bash
python phase6_evaluation.py \
  --lens-path <上一步的 .pt> \
  --out-json <輸出 .json 路徑> \
  2>&1 | tee <log 路徑>
```

**5. 繪圖**

```bash
python plot_results.py
```

---

## 已知限制

- 4-bit 量化造成的精度損失，J-Lens 學到的是「量化後模型」的內部結構，不完全等同原始精度模型
- 訓練語料規模（100 / 100 / 50+50 筆）遠小於一般用大規模算力 fit 的規模，結果雜訊較大
- 實驗 A/C 的 prompt 用 ground-truth 逐字稿做 teacher-forcing 拼接，不是模型真正 `generate()` 出來的回應，兩者的殘差流分布可能有落差
- FLEURS 資料集本身有同句子多說話者重複朗讀的特性，實際文字內容多樣性比「100 筆」這個數字看起來要低



