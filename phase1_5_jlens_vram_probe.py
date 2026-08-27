"""
Phase 1.5: J-Lens 核心 VJP 迴圈 —— VRAM 壓力測試（純文字，先不含音訊）
================================================================
目的：
  在投入時間寫完整的 jlens 適配器（Phase 3）之前，先用官方 jlens/fitting.py
  裡 jacobian_for_prompt() 的核心邏輯（複製 batch -> 保留計算圖 -> 多次
  one-hot cotangent backward）跑一個縮小版，量測：
    1. model.language_model（Qwen2-Audio 的文字 decoder）正確的存取路徑
    2. 4-bit 凍結權重下，backward 是否真的能取得梯度（enable_input_require_grads）
    3. 在 retain_graph=True 的連續多次 backward 情境下，VRAM 真實峰值

尚未涵蓋：
  - 音訊條件輸入（audio-conditioned forward）——這是 Phase 3 才處理的複雜度，
    這裡刻意先隔離掉，避免同時除錯兩個變因
  - 完整符合 jlens.protocol.LensModel 介面的正式 adapter

執行後請把印出的數字回報，我們再依此決定 Phase 3 的 dim_batch / 分層策略。
"""

import os
import sys
import math

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DIM_BATCH = 4          # 保守起點，官方預設是 8
MAX_SEQ_LEN = 128      # 對齊官方 fitting.py 預設
N_PROBE_PASSES = 5     # 只跑 5 次 backward 來估算，不用真的跑完 ceil(d_model/dim_batch) 次


def main():
    import torch
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    # ------------------------------------------------------------------
    # Step A: 載入模型（同 Phase 1 設定）
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

    # ------------------------------------------------------------------
    # Step B: 找出 decoder 子模組的正確路徑（不要盲猜）
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[Step B] 模型頂層子模組")
    print("=" * 70)
    for name, _ in model.named_children():
        print(f"  - {name}")

    decoder = getattr(model, "language_model", None)
    if decoder is None:
        print("\n[FATAL] 找不到 model.language_model。")
        print("請執行 `print(model)` 找出正確的 decoder 屬性路徑，並修改本腳本的 `decoder = ...` 那一行。")
        sys.exit(1)

    n_layers = decoder.config.num_hidden_layers
    d_model = decoder.config.hidden_size
    print(f"\n  decoder = model.language_model")
    print(f"  n_layers = {n_layers}, d_model (hidden_size) = {d_model}")
    print()

    # ------------------------------------------------------------------
    # Step C: 讓梯度可以流過凍結的 4-bit 權重
    # ------------------------------------------------------------------
    model.enable_input_require_grads()
    print("[Step C] 已呼叫 model.enable_input_require_grads()（QLoRA 常見手法，"
          "讓凍結量化權重之間的中間激發值仍可反向傳播）\n")

    # ------------------------------------------------------------------
    # Step D: 核心 VJP 迴圈壓力測試（純文字，模擬 jacobian_for_prompt 的記憶體行為）
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"[Step D] VJP 壓力測試: dim_batch={DIM_BATCH}, 只跑 {N_PROBE_PASSES} 次 backward 估算峰值")
    print("=" * 70)

    prompt = "請用一句話描述今天的天氣狀況，並簡短說明理由。" * 4
    tok = processor.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    )
    input_ids = tok["input_ids"].to(model.device)
    replicated_ids = input_ids.expand(DIM_BATCH, -1)
    seq_len = replicated_ids.shape[1]
    print(f"  序列長度(截斷後): {seq_len} tokens")

    target_layer_idx = n_layers          # hidden_states 索引: 0=embedding, 1..n_layers=各層輸出
    source_layer_idx = n_layers // 2     # 先測中間層，代表較深的保留圖情境

    torch.cuda.reset_peak_memory_stats()

    with torch.enable_grad():
        out = decoder(
            replicated_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states  # tuple, 長度 = n_layers + 1

        target_activation = hidden_states[target_layer_idx]  # [dim_batch, seq, d_model]
        source_activation = hidden_states[source_layer_idx]

        vram_after_forward = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  forward (batch={DIM_BATCH}) 完成後 VRAM 峰值: {vram_after_forward:.2f} GB")

        cotangent = torch.zeros_like(target_activation)

        for pass_idx in range(N_PROBE_PASSES):
            cotangent.zero_()
            dim_start = pass_idx * DIM_BATCH
            for b in range(DIM_BATCH):
                dim = (dim_start + b) % d_model
                cotangent[b, :, dim] = 1.0

            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activation,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < N_PROBE_PASSES - 1),
            )
            del grads

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  連續 {N_PROBE_PASSES} 次 retain_graph backward 後 VRAM 峰值: {peak_vram:.2f} GB")

    n_full_passes = math.ceil(d_model / DIM_BATCH)
    print()
    print("=" * 70)
    print("[結論]")
    print("=" * 70)
    print(f"  完整擬合一個 prompt（單一 source layer）需要 {n_full_passes} 次 backward，")
    print(f"  由於是同一張保留的圖，記憶體峰值應該已經在上面 {N_PROBE_PASSES} 次測試中穩定出現，")
    print(f"  不會隨 backward 次數線性增加（主要成本是 forward 保留的 activation，而非每次 backward 本身）。")
    print(f"  目前峰值 {peak_vram:.2f} GB / 11.6 GB 總 VRAM。")
    headroom = 11.6 - peak_vram
    print(f"  估計剩餘空間: {headroom:.2f} GB")
    if headroom < 2.0:
        print("  [警告] 空間非常緊繃，建議 dim_batch 降到 2，並且一次只對 1-2 個 source layer 擬合。")
    elif headroom < 4.0:
        print("  [提醒] 有一定緊繃度，建議正式擬合時 source_layers 分批處理（例如每次 4-8 層），"
              "而非一次對全部 28 層擬合。")
    else:
        print("  空間看起來足夠，可以嘗試 dim_batch=8（官方預設）並對多層同時擬合。")


if __name__ == "__main__":
    main()
