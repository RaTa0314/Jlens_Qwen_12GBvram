"""
Phase 3 (v0): Qwen2AudioLensModel adapter + 端到端 jacobian_for_prompt 測試
================================================================================
設計要點（對應 jlens/protocol.py 與 jlens/hooks.py 的真實介面）：

  - 一個 adapter instance 綁定「一段音訊」。文字 prompt 在 encode() 時才給，
    這樣同一段音訊可以搭配不同文字重複使用（如果你的資料集有這種情境）。
  - forward(input_ids) 直接呼叫 model.model(...)（Qwen2AudioModel），它已經內建
    audio_tower -> multi_modal_projector -> masked_scatter 合併邏輯，且不含 lm_head，
    正好符合 protocol 要求。
  - 刻意「不」呼叫 model.enable_input_require_grads()。讓權重維持完全凍結、
    輸入也不 require_grad，這樣 ActivationRecorder 的 start_graph_at 機制才能真正
    只從 source layer 開始建圖，把 audio tower + 前面所有層排除在反向傳播記憶體之外。
  - encode() 裡如果超過 max_length 直接 raise，不做事後截斷——事後截斷會讓
    音訊 placeholder token 數量和 audio_features 數量對不上，觸發庫內的一致性檢查錯誤。

使用前：
  cd ~/jacobian-lens && pip install -e .
"""

import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402  (放在 env var 設定之後，import 順序才有效)

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
TEST_AUDIO_PATH = "test_audio.wav"  # 沿用 Phase 1 的測試音訊，或換成真實語音檔


class Qwen2AudioLensModel:
    """LensModel adapter：一個 instance 綁定一段音訊。

    設計重點：audio_tower 的前向計算只在 encode() 時、以 batch=1、no_grad 跑一次，
    結果快取下來。forward(input_ids) 之後不管被複製成幾份 batch（dim_batch），
    都只重複使用這個小小的快取音訊特徵去做 embedding 合併，不重跑 audio_tower ——
    否則 audio_tower 的中間激發值會隨 dim_batch 線性爆炸，這正是上一輪 OOM 的原因。
    """

    def __init__(self, full_model, processor, audio, sampling_rate: int):
        self._model = full_model              # Qwen2AudioForConditionalGeneration
        self._inner = full_model.model        # Qwen2AudioModel (audio_tower + projector + language_model)
        self._processor = processor
        self._audio = audio
        self._sampling_rate = sampling_rate

        self.tokenizer = processor.tokenizer
        self.n_layers = self._inner.language_model.config.num_hidden_layers
        self.d_model = self._inner.language_model.config.hidden_size
        self.layers = self._inner.language_model.layers  # Sequence[nn.Module]，ActivationRecorder 會 hook 這個

        # 由 encode() 填入
        self._cached_audio_features = None  # [1, num_audio_tokens, d_model]，no_grad 算好快取

    def _compute_audio_features_once(self, input_ids_batch1, input_features, feature_attention_mask):
        """batch=1、no_grad 跑一次完整 forward，用 forward hook 攔截 multi_modal_projector
        的輸出（就是最終要合併進 inputs_embeds 的 audio_features），完全重用官方的合併邏輯，
        不用自己重寫 audio_tower 的 mask 計算細節。"""
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

        if "audio_features" not in captured:
            raise RuntimeError("multi_modal_projector 的 hook 沒有被觸發，請確認這個 forward 真的有音訊輸入。")

        self._cached_audio_features = captured["audio_features"].detach()  # [1, num_audio_tokens, d_model]

    def encode(self, text: str, *, max_length: int = 256):
        """Tokenize「這段綁定音訊 + text」，回傳 input_ids [1, seq_len]，並順便快取音訊特徵。"""
        inputs = self._processor(
            text=text,
            audio=[self._audio],
            sampling_rate=self._sampling_rate,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > max_length:
            raise ValueError(
                f"encode 後長度 {input_ids.shape[1]} 超過 max_length={max_length}；"
                "請縮短音訊或文字，不要在這裡截斷（會讓音訊 placeholder token 數與 "
                "audio_features 數量對不上）。"
            )
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        input_features = inputs["input_features"].to(device)
        feature_attention_mask = inputs["feature_attention_mask"].to(device)

        self._compute_audio_features_once(input_ids, input_features, feature_attention_mask)
        return input_ids

    def forward(self, input_ids):
        """跑 residual stack（不含 lm_head）。input_ids 可能是複製 dim_batch 份的同一個 prompt，
        但音訊特徵只用快取的那份 expand，不重跑 audio_tower。"""
        dim_batch = input_ids.shape[0]

        text_embeds = self._inner.get_input_embeddings()(input_ids)  # [dim_batch, seq, d_model]
        audio_features = self._cached_audio_features.expand(dim_batch, -1, -1).to(text_embeds.dtype)

        special_audio_mask = (input_ids == self._model.config.audio_token_id).unsqueeze(-1)
        inputs_embeds = text_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self._inner.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def unembed(self, residual):
        """final norm + lm_head。"""
        normed = self._inner.language_model.norm(residual)
        return self._model.lm_head(normed)


def main():
    import librosa
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    # 讓 python 找得到 clone 下來的 jacobian-lens repo（假設在 ~/jacobian-lens）
    repo_path = os.path.expanduser("~/jacobian-lens")
    if os.path.isdir(repo_path) and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from jlens.fitting import jacobian_for_prompt

    # ------------------------------------------------------------------
    # 載入模型：注意，這裡刻意「不」呼叫 model.enable_input_require_grads()
    # ------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # 明確凍結全部參數：embed_tokens / lm_head / 各層 norm / audio_tower 的 bias 等
    # 預設不會是 requires_grad=False，只有這樣 ActivationRecorder 的 start_graph_at
    # 才能讓計算圖真的只從 source layer 開始，而不是從 embedding 就整條保留。
    for p in model.parameters():
        p.requires_grad_(False)

    n_requires_grad = sum(p.requires_grad for p in model.parameters())
    print(f"[檢查] requires_grad=True 的參數數量: {n_requires_grad}（預期為 0）")
    assert n_requires_grad == 0, "凍結失敗，請檢查是否有參數在 from_pretrained 之後才被建立"

    # ------------------------------------------------------------------
    # 準備測試音訊
    # ------------------------------------------------------------------
    if not os.path.exists(TEST_AUDIO_PATH):
        raise FileNotFoundError(f"找不到 {TEST_AUDIO_PATH}，請先跑過 Phase 1 產生測試音訊，或換成真實語音檔路徑")
    audio, sr = librosa.load(TEST_AUDIO_PATH, sr=processor.feature_extractor.sampling_rate)

    lens_model = Qwen2AudioLensModel(model, processor, audio, sr)
    print(f"[Adapter] n_layers={lens_model.n_layers}, d_model={lens_model.d_model}")

    # ------------------------------------------------------------------
    # 端到端測試：對最後幾層做真正的 jacobian_for_prompt
    # ------------------------------------------------------------------
    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>請描述這段音訊的內容："
    # 最壞情況測試：包含第 0 層 -> start_graph_at=0 -> 保留全部 32 層的計算圖。
    # 同時混入中間層 (15) 和倒數第二層 (30)，一次驗證「全部層一起 fit」在時間上
    # 是否真的跟只 fit 2 層一樣只需要 1024 次 backward（差別應該只在每次 backward
    # 要多算過幾層，而不是 pass 次數變多）。
    source_layers = [0, 15, 30]
    dim_batch = 8  # 官方預設；跟上次 dim_batch=4 的全深度結果比較 VRAM/時間 trade-off
    max_seq_len = 256

    print(f"[測試] source_layers={source_layers}（含第 0 層 = 全深度計算圖）, dim_batch={dim_batch}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    jacobians, seq_len, n_valid = jacobian_for_prompt(
        lens_model,
        prompt,
        source_layers,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
    )

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print()
    print("=" * 70)
    print("[結果]")
    print("=" * 70)
    print(f"  seq_len={seq_len}, n_valid_positions={n_valid}")
    print(f"  耗時: {elapsed:.1f}s")
    print(f"  VRAM 峰值: {peak_vram:.2f} GB / 11.6 GB")
    for layer, J in jacobians.items():
        print(f"  J_{layer} shape={tuple(J.shape)}, norm={J.norm().item():.3f}, device={J.device}")

    d_model = lens_model.d_model
    import math
    n_passes_per_prompt = math.ceil(d_model / dim_batch)
    print(f"\n  完整一個 prompt 需要 {n_passes_per_prompt} 次 backward（dim_batch={dim_batch}）")
    print("  由於是同一張保留圖，上面量到的峰值應該已經代表跑完整個 prompt 的真實情況。")

    headroom = 11.6 - peak_vram
    print(f"  剩餘空間: {headroom:.2f} GB")
    if headroom > 3.0:
        print("  空間充裕，可嘗試增加 source_layers 數量或提高 dim_batch。")
    else:
        print("  空間偏緊，正式跑 Phase 4 時建議維持目前的 dim_batch，並分批處理 source_layers。")


if __name__ == "__main__":
    main()
