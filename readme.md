# J-Lens on Qwen2-Audio (12GB VRAM Optimization)

模型：Qwen2-Audio-7B-Instruct
---
## 這個專案在做什麼（from claude)

**J-Lens（Jacobian Lens）** 是 Anthropic 提出的一種模型內部可解釋性技術。語言模型每一層的殘差流（residual stream）活在不同的「基底空間」裡，如果直接把中間層的激發值丟進最後的 unembed（也就是傳統的 *logit lens*），讀出來的東西大多是雜訊，因為中間層根本沒有活在跟輸出層一樣的座標系裡。

J-Lens 的做法：對每一層 fit 出一個線性轉換矩陣 `J_l = E[∂h_final/∂h_l]`（該層激發值對最終輸出的平均一階效應），再用它把中間層的激發值「轉正」到輸出層的座標系，才做 unembed 讀出。理論上能比傳統 logit lens 更早、更準地讀出模型「心裡在想什麼」。

這個專案做的事：把這套原本針對純文字 LLM 設計的技術，**搬到一個語音輸入、文字輸出的多模態模型（LALM）上**，並且全程限制在一張消費級 12GB 顯卡上完成。

參考實作：[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)（Apache-2.0）

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
驗證集（`jlens_val_dataset`）與三個訓練集完全隔離、無重疊。

---

## 目前的發現

### 1. Phase 5 健檢（in-sample，僅供除錯參考，不是正式結果）

用實驗 A 的 lens，在 2 筆訓練語料本身（EN_0、ZH_0）上做的健檢：

| 樣本 | J-Lens (layer 30) | 傳統 logit lens (layer 30) |
|---|---|---|
| EN_0 | 58.6% | 55.9% |
| ZH_0 | 50.0% | 41.8% |

質化上還觀察到「eBay」這個詞在 layer 4 就被 J-Lens 讀出，比傳統 logit lens（要到 layer 16 才有意義）早了 12 層。**這是個真實但挑出來的個案**，不能代表整體趨勢——正式的量化結論看下面 Phase 6。

### 2. Phase 6 正式驗證（嚴格 held-out，n 為驗證集裡的有效文字位置數）

![J-Lens vs Baseline: 純 ASR vs 50/50 混合](./compared_asr_vs_mixed.png)

| Layer | ASR 驗證集<br>J-Lens(A) / J-Lens(B) / Baseline | QA 驗證集（zero-shot for A）<br>J-Lens(A) / J-Lens(B) / Baseline |
|---|---|---|
| 0 | 2.3% / 1.1% / 0.7% | 6.1% / 0.0% / 0.0% |
| 10 | 3.1% / 5.4% / 2.4% | 12.2% / 11.4% / 0.0% |
| 20 | 24.2% / 25.9% / 15.7% | 26.7% / 17.9% / 2.4% |
| 26 | 39.1% / 42.3% / 35.7% | 52.1% / 47.1% / 7.9% |
| 29 | 64.5% / 63.5% / 53.8% | 67.2% / 70.2% / 38.8% |
| 30 | 80.5% / 82.7% / 72.4% | 84.0% / 85.4% / 71.1% |

（A = 純 ASR lens，B = 50/50 混合 lens，n(ASR)=2740、n(QA)=823 個有效文字位置）

**觀察一：沒有偵測到災難性遺忘。** B（混合）在 ASR 驗證集上的表現幾乎跟 A（純 ASR）打平，多數層甚至略高一點點。加入 QA 語料在這個規模下沒有稀釋掉 ASR 能力。

**觀察二：zero-shot 跨任務泛化比預期強。** A 從沒看過 QA 資料，卻在 QA 驗證集上逼近甚至偶爾超過 B（layer 26、20 兩處 A 明顯贏過 B）。這暗示 J_l 在中後段層學到的線性結構本身就有一定的任務無關性，不完全是靠混資料才泛化出來的。

**觀察三：J-Lens 相對 baseline 的優勢主要集中在中後段層（約 layer 20-29），不是最早期層。** layer 0-15 這段兩者差距很小（多在 1-3 個百分點內，且數值本身很低，訊噪比差），差距在 layer 20 之後才明顯拉開（+7 到 +11 個百分點），layer 30 仍有約 +8 個百分點的優勢。Phase 5 那個「eBay 在 layer 4 就出現」的例子，是好例子但不能代表整體趨勢。

**已查證：Phase 5（單筆 58.6%/55.9%）跟 Phase 6（全驗證集平均 80.5%/72.4%）數字量級差很多，但不是統計口徑不一致造成的。** 對照過 `phase6_evaluation.py` 的原始碼，排除 `<|AUDIO|>` 佔位符位置的篩選邏輯（`mask = (input_ids[0] != audio_token_id)`）跟 `phase5b_layerwise_accuracy.py` 完全一致。真正的原因是樣本規模：Phase 5 只看 2 筆樣本（有效位置數僅約數百個），Phase 6 是 n=2740 個位置的整體平均，小樣本雜訊本來就大。關鍵佐證是 **baseline 也同步從 41.8%/55.9% 跳到 72.4%**——如果是只影響 J-Lens 路徑的 bug，baseline 應該原地不動，但兩者同步跳升，代表拉高的是「整體資料的可預測性隨樣本數變穩定」，不是計算邏輯有誤。


---

## 實做步驟

| Phase | 內容 | 主要產出 |
|---|---|---|
| **Phase 0** | 環境建置：Ubuntu + CUDA + conda env + 套件安裝 | 可用的開發環境 |
| **Phase 1** | 模型載入測試：4-bit 載入 Qwen2-Audio-7B-Instruct，量測 VRAM 峰值 | VRAM 基準數據 |
| **Phase 2** | 資料準備（ASR / QA / 50-50 混合三組） | 前處理後的資料集 |
| **Phase 3** | J-Lens 擬合基礎設施：適配  `jlens.fitting`，實作「音訊編碼器 no_grad + decoder 開梯度」的分段前向邏輯 | 可執行的擬合 pipeline |
| **Phase 4** |正式擬合（`phase4_fit.py`）| 儲存的 J_l（各層），`.ckpt` |
| **Phase 4.5** | 權重封裝（`phase4_finalize.py`） | 正式 `.pt` 權重檔 |
| **Phase 5** | 在 fit 用過的樣本上健檢（in-sample sanity check）(WTF) | 快速確認 pipeline 沒壞掉，不是正式結果 |
| **Phase 6** | validate on validation data : )  | 正式的 layer-wise accuracy 報表與圖表 |

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
├── phase4_fit.py         # 核心擬合腳本（4-bit NF4 + 音訊特徵快取防 OOM）
├── phase4_finalize.py        # .ckpt（訓練進度）→ .pt（正式 J-Lens 權重）
├── phase6_evaluation.py      # 雙軌評估：J-Lens vs 傳統 logit lens 的 layer-wise 一致率
└── plot_results.py           # 把 JSON 結果畫成折線圖
└── plot_compared_results.py  # 疊加兩組JSON 實驗結果與 Baseline 進行對比
```

---

## 各個檔案負責的功能

* **`g_data/`**：產生並存放實驗資料集
* **`phase4_fit.py`**：fitting，會生成checkpoint(.ckpt)存在 checkpoints/
* **`phase4_finalize.py`**：大容量的訓練進度檔（`.ckpt`）-> 正式 J-Lens 權重檔（`.pt`），同樣存在checkpoints/
* **`phase6_evaluation.py`**：雙軌評估系統。自動載入驗證集，分別計算 J-Lens 與 Baseline (Logit Lens) 在各個 Layer 的預測一致率，並輸出綜合報表。
* **`plot_results.py`**：讀取評估輸出的 JSON 數據，繪製並排的 ASR、QA、Combined 三合一學術級折線圖。
* **`plot_compared_results.py`**：將兩個不同實驗 (如純 ASR 與 混合) 的 JSON 結果繪製在同一張圖上，共用 Baseline 以利觀察任務干擾。
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

**5. 繪製單一實驗結果圖**

```bash
python plot_results.py \
  --json-path <輸入 .json 路徑1> \
  --output-image <輸出.png>
```

**6. 繪製實驗對比圖**

```bash
python plot_compared_results.py \
  --json1 <輸入 .json 路徑1> \
  --label1 "<標籤1名稱>" \
  --json2 <輸入 .json 路徑2> \
  --label2 "<標籤1名稱>" \
  --output-image <輸出.png>
```

---

## 已知限制

- 4-bit 量化造成的精度損失，J-Lens 學到的是「量化後模型」的內部結構，不完全等同原始精度模型
- 訓練語料規模（100 / 100 / 50+50 筆）遠小於一般用大規模算力 fit 的規模，結果雜訊較大
- 實驗 A/C 的 prompt 用 ground-truth 逐字稿做 teacher-forcing 拼接，不是模型真正 `generate()` 出來的回應，兩者的殘差流分布可能有落差
- FLEURS 資料集本身有同句子多說話者重複朗讀的特性，實際文字內容多樣性比「100 筆」這個數字看起來要低
- `phase6_evaluation.py` 對每筆樣本呼叫兩次 `lens.apply()`（`use_jacobian=True/False` 各一次），跑了兩次完整 forward pass，評估時間比理論上多一倍——不影響正確性，但之後驗證集規模變大時可以優化


