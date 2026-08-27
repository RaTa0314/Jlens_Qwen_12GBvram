"""
phase4_fit_asr.py
==================
Phase 4: 正式在 100 筆純 ASR (FLEURS EN+ZH) 語料上 fit J-Lens。

沿用 Phase 3 驗證過、不會在 RTX 3060 (12GB) 上 OOM 的架構：
  1. 手動凍結全部參數 (model.eval() + requires_grad_(False))
  2. 每筆樣本的音訊特徵只跑一次 forward 並快取 (_compute_audio_features_once)
  3. dim_batch=4、checkpoint_every=1，確保掛機時可隨時中斷 / 續跑

NOTE: 本檔案假設你本地有一個 `jlens` 套件，其中提供
      `jlens.fit(model=..., dataset=..., ...)` 這個介面 (对應你先前討論中提到的
      `fit()` / checkpoint_path / checkpoint_every 參數)。
      如果你實際的模組名稱、函式簽名不同，把下面標了 `# ADAPT:` 的地方
      換成你自己專案裡真正的 import / 呼叫方式即可，其餘邏輯不受影響。
"""

import os
import json
import argparse
import logging

# 必須在 import torch / 任何 CUDA context 建立之前設定，
# 用來緩解長時間掛機、反覆配置/釋放不同大小 tensor 造成的記憶體碎片化。
# 注意：這只解決「碎片化型」OOM，若模型 + activation 本身就超過顯存容量
# （capacity OOM），這個變數無法解決，仍需靠量化/凍結梯度等方式縮減用量。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# jlens.fit()'s real signature (anthropics/jacobian-lens, jlens/fitting.py):
#   fit(model, prompts, *, source_layers=None, target_layer=None, dim_batch=8,
#       max_seq_len=128, skip_first=16, checkpoint_path=None,
#       checkpoint_every=1, resume=True) -> JacobianLens
# Note: the first arg is named `model`, not `lens_model`.
from jlens.fitting import fit as jlens_fit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase4_fit_asr")


# ---------------------------------------------------------------------------
# 1. Adapter：沿用 Phase 3 架構，補完 encode()
# ---------------------------------------------------------------------------
class Qwen2AudioLensModel:
    def __init__(self, full_model, processor, data_root: str, device: str = "cuda",
                 model_dtype: torch.dtype = torch.bfloat16):
        self._model = full_model
        self._inner = full_model.model
        self._processor = processor
        self._sampling_rate = processor.feature_extractor.sampling_rate
        self._data_root = data_root
        self._device = device
        # 4-bit 量化後，模型參數本身的 dtype 是量化用的儲存格式（不是 bf16），
        # 所以 input_features 要轉的 dtype 必須明確傳入 compute_dtype，不能用
        # next(full_model.parameters()).dtype 推斷，否則會轉錯型別。
        self._model_dtype = model_dtype

        self.tokenizer = processor.tokenizer
        self.n_layers = self._inner.language_model.config.num_hidden_layers
        self.d_model = self._inner.language_model.config.hidden_size
        self.layers = self._inner.language_model.layers

        self._cached_audio_features = None

    def _compute_audio_features_once(self, input_ids_batch1, input_features, feature_attention_mask):
        """只在 batch=1 時跑一次完整 forward，把 projector 輸出的 audio embedding 快取起來。
        之後 forward() 裡對這份快取做 .expand(dim_batch, ...)，
        避免每個 batch step 都重新跑一次昂貴的音訊 encoder。"""
        captured = {}

        def _hook(module, inputs, output):
            captured["audio_features"] = output

        handle = self._inner.multi_modal_projector.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self._inner(
                    input_ids=input_ids_batch1,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()
        self._cached_audio_features = captured["audio_features"].detach()

    def encode(self, json_line: str, *, max_length: int = 300):
        """
        解析 prompts.jsonl 的其中一行 (JSON string)，動態載入對應的 wav，
        並把該筆樣本的音訊特徵算一次、快取下來。

        輸入格式範例 (一行 JSONL)：
          {"id": "EN_000", "language": "EN",
           "audio_path": "jlens_dataset/audio/EN_000.wav",
           "transcript": "...",
           "prompt": "<|audio_bos|><|AUDIO|><|audio_eos|>請寫出這段音訊的逐字稿："}

        Returns
        -------
        input_ids : torch.LongTensor, shape (1, seq_len)
            prompt + transcript 的完整 token 序列 (teacher forcing 用)，
            padding 到 max_length，可以直接丟進 forward()。

        Notes
        -----
        這裡刻意不使用 truncation=True：若真的截斷，`transcript` 尾巴或甚至
        `<|AUDIO|>` 佔位符本身都可能被切掉，前者會讓 teacher-forcing 目標
        悄悄損毀而不報錯，後者會讓 input_ids 裡的音訊 token 數量與
        `_cached_audio_features` 的數量對不上，導致 forward() 裡的
        masked_scatter 直接崩潰。
        正確做法是在呼叫 fit() 之前，用 main() 裡的 filter_by_length() 先把
        超過 max_length 的樣本篩掉；下面的檢查只是最後一道防線，理論上不該
        被觸發 —— 一旦觸發，代表離線過濾邏輯本身有 bug，需要用明確的例外
        讓你知道，而不是靜默截斷資料。
        """
        record = json.loads(json_line)
        audio_path = os.path.join(self._data_root, record["audio_path"])
        prompt = record["prompt"]
        transcript = record["transcript"]

        # 5 秒以內的音訊，載入成本不高，直接同步讀取即可
        audio_array, _ = librosa.load(audio_path, sr=self._sampling_rate)

        # prompt 內含 <|AUDIO|> 佔位符；transcript 接在後面做 teacher forcing 目標
        full_text = prompt + transcript

    # 1. 保持最乾淨的 Processor 呼叫（不加任何 padding，保護文字長度）
        inputs = self._processor(
            text=full_text,
            audio=[audio_array],
            sampling_rate=self._sampling_rate,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > max_length:
            raise ValueError(
                f"樣本 {record.get('id', '<unknown>')} 長度 {input_ids.shape[1]} "
                f"超過 max_length={max_length}，安全跳過。"
            )

        # ==========================================
        # 🚀 2. 手動將音訊特徵補齊到 3000 滿足模型要求
        # ==========================================
        input_features = inputs["input_features"]
        feature_attention_mask = inputs["feature_attention_mask"]
 
        if input_features.shape[-1] < 3000:
            pad_len = 3000 - input_features.shape[-1]
            # PyTorch pad 語法：(左邊補0, 右邊補pad_len)
            input_features = torch.nn.functional.pad(input_features, (0, pad_len), value=0.0)
            feature_attention_mask = torch.nn.functional.pad(feature_attention_mask, (0, pad_len), value=0)
        # ==========================================

	# 這樣改才對：使用上面 pad 好的變數
        input_ids = input_ids.to(self._device)
        input_features = input_features.to(self._device, dtype=self._model_dtype)
        feature_attention_mask = feature_attention_mask.to(self._device)
        # 關鍵省顯存設計：這筆樣本的音訊特徵只算這一次，後面 forward() 全部重複使用
        self._compute_audio_features_once(
            input_ids_batch1=input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )

        return input_ids

    def forward(self, input_ids):
        dim_batch = input_ids.shape[0]
        text_embeds = self._inner.get_input_embeddings()(input_ids)
        audio_features = self._cached_audio_features.expand(dim_batch, -1, -1).to(text_embeds.dtype)

        special_audio_mask = (input_ids == self._model.config.audio_token_id).unsqueeze(-1)
        inputs_embeds = text_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self._inner.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def unembed(self, residual):
        # transport()/select() 回傳的 residual 是 float32（JacobianLens 為了精度
        # 刻意這麼做），但 norm/lm_head 的權重是 bf16，dtype 對不上會直接報錯，
        # 所以這裡先 cast 回模型實際在用的 dtype 再送進去。
        target_dtype = self._inner.language_model.norm.weight.dtype

	# 加上 device=self._device，把資料送去 GPU！
        residual = residual.to(device=self._device, dtype=target_dtype)

        normed = self._inner.language_model.norm(residual.to(target_dtype))
        return self._model.lm_head(normed)
# ---------------------------------------------------------------------------
# 2. JSONL Loader
# ---------------------------------------------------------------------------
def load_jsonl_lines(path: str):
    """回傳原始 JSON 字串的 list（不預先 parse），因為 encode() 本身就吃 json_line。"""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                lines.append(raw)
    return lines


def filter_by_length(records, processor, data_root: str, sampling_rate: int, max_length: int):
    """在正式 fit() 之前，先用一次輕量 tokenize（不含音訊 encoder forward）
    把長度超過 max_length 的樣本篩掉。

    這一步刻意獨立於 encode()：
    - encode() 拿掉 truncation 之後，若遇到超長樣本要嘛靜默損毀資料、
      要嘛在 forward() 階段才 shape mismatch 崩潰——兩者都會發生在
      12 小時 fit() 迴圈「途中」，讓整個 job 中止或產出有問題的結果。
    - 這裡先用 100 筆規模、成本很低的一次性掃描（只做 tokenize，
      不跑音訊 encoder），把有問題的樣本在跑之前就排除掉，
      風險轉移到「你現在就能看到 log 並決定」而不是「掛機途中才發現」。
    """
    kept, dropped = [], []
    for line in records:
        record = json.loads(line)
        audio_path = os.path.join(data_root, record["audio_path"])
        audio_array, _ = librosa.load(audio_path, sr=sampling_rate)
        full_text = record["prompt"] + record["transcript"]

        # 不 pad、不截斷，純粹拿到真實 token 長度
        inputs = processor(
            text=full_text,
            audio=[audio_array],
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        length = inputs["input_ids"].shape[1]
        if length > max_length:
            logger.warning(
                "跳過樣本 %s：長度 %d 超過 max_length=%d",
                record.get("id", "<unknown>"), length, max_length,
            )
            dropped.append(record.get("id", "<unknown>"))
        else:
            kept.append(line)

    logger.info(
        "長度過濾完成：保留 %d / %d 筆%s",
        len(kept), len(records),
        f"，捨棄: {dropped}" if dropped else "",
    )
    return kept


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: fit J-Lens on 100 ASR samples")
    # 1. 直接改成相對路徑 "g_data"
    parser.add_argument("--data-root", default="g_data",
                        help="工作目錄，內含 jlens_dataset/prompts.jsonl 與 jlens_dataset/audio/")
    parser.add_argument("--jsonl-name", default="jlens_dataset/prompts.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2-Audio-7B-Instruct")
    # 2. 直接改成相對路徑 "checkpoints/..."
    parser.add_argument("--checkpoint-path",
                        default="checkpoints/phase4_asr.ckpt")
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    jsonl_path = os.path.join(args.data_root, args.jsonl_name)
    logger.info("Loading metadata from %s", jsonl_path)
    records = load_jsonl_lines(jsonl_path)
    logger.info("Loaded %d samples", len(records))

    logger.info("Loading model %s in 4-bit (nf4) ...", args.model_name)
    # 4-bit 量化是 RTX 3060 (12GB) 能塞下 7B 模型的關鍵，沒有這個會直接 OOM。
    # compute_dtype 用 bfloat16，配合 double_quant 進一步壓縮記憶體。
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # bitsandbytes 量化模型要用 device_map，不能事後 .to(device)
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # 在 12 小時 fit() 正式開始之前，先做一次性、成本很低的長度掃描，
    # 把超過 max_seq_len 的樣本篩掉，避免任務跑到一半才因為單一樣本
    # 讓 masked_scatter shape mismatch 崩潰，或悄悄截斷 teacher-forcing 目標。
    logger.info("Filtering samples by length (max_seq_len=%d) before fitting...", args.max_seq_len)
    records = filter_by_length(
        records,
        processor=processor,
        data_root=args.data_root,
        sampling_rate=processor.feature_extractor.sampling_rate,
        max_length=args.max_seq_len,
    )
    if not records:
        raise RuntimeError("長度過濾後沒有剩下任何樣本，請檢查 max_seq_len 設定或資料本身。")

    # 手動凍結全部參數：整個模型只做 forward，不需要梯度，
    # 這是 Phase 3 驗證過在 RTX 3060 上不 OOM 的關鍵之一
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)

    adapter = Qwen2AudioLensModel(
        full_model=full_model,
        processor=processor,
        data_root=args.data_root,
        device=args.device,
        model_dtype=torch.bfloat16,  # 要跟 bnb_4bit_compute_dtype 一致
    )

    n_layers = adapter.n_layers  # 應為 32
    source_layers = list(range(n_layers - 1))  # 0..30，共 31 層
    target_layer = n_layers - 1  # 31

    logger.info(
        "Fitting all %d layers together: source_layers=%s..%s, target_layer=%s",
        n_layers, source_layers[0], source_layers[-1], target_layer,
    )
    logger.info(
        "dim_batch=%d, max_seq_len=%d, checkpoint_every=1, checkpoint_path=%s",
        args.dim_batch, args.max_seq_len, args.checkpoint_path,
    )

    jlens_fit(
        model=adapter,
        prompts=records,                 # 傳原始 json 字串 list，encode() 內部會 json.loads
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=1,               # 每個 step 都存檔，掛機隨時可中斷續跑
    )

    logger.info("Phase 4 fitting finished (or checkpoint saved if interrupted).")


if __name__ == "__main__":
    main()
