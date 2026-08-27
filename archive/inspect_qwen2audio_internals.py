"""
內省腳本：找出目前安裝版本 (transformers 5.16.0.dev0) 的 Qwen2-Audio 真實結構
============================================================================
目的：不用猜 attribute 路徑，直接印出：
  1. model.model 底下的子模組（audio_tower / multi_modal_projector / language_model 等）
  2. language_model 的內部路徑，特別是 decoder layers 在哪
  3. Qwen2AudioForConditionalGeneration.forward 與 model.model.forward 的「真實原始碼」，
     讓我確認音訊特徵是怎麼被合併進 inputs_embeds 的（merge 邏輯、audio_token_id 等）

跑完後請把完整輸出貼給我。
"""

import inspect
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


def safe_getsource(obj, label):
    print(f"\n{'=' * 70}\n[原始碼] {label}\n{'=' * 70}")
    try:
        print(inspect.getsource(obj))
    except Exception as e:
        print(f"  (無法取得原始碼: {e})")


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("=" * 70)
    print("[結構] model.model 的子模組")
    print("=" * 70)
    for name, mod in model.model.named_children():
        print(f"  - {name}: {type(mod).__name__}")

    inner = model.model
    lm = getattr(inner, "language_model", None)
    print("\n" + "=" * 70)
    print("[結構] language_model 內部路徑")
    print("=" * 70)
    if lm is None:
        print("  model.model 沒有 language_model 屬性，請檢查上面列出的子模組名稱。")
    else:
        print(f"  type(model.model.language_model) = {type(lm).__name__}")
        print(f"  hasattr(.model.language_model, 'model') = {hasattr(lm, 'model')}")
        if hasattr(lm, "model"):
            core = lm.model
            print(f"  hasattr(.language_model.model, 'layers') = {hasattr(core, 'layers')}")
            if hasattr(core, "layers"):
                print(f"  len(layers) = {len(core.layers)}")
                print(f"  layer[0] type = {type(core.layers[0]).__name__}")
        if hasattr(lm, "config"):
            print(f"  hidden_size = {getattr(lm.config, 'hidden_size', '?')}")
            print(f"  num_hidden_layers = {getattr(lm.config, 'num_hidden_layers', '?')}")

    print("\n" + "=" * 70)
    print("[結構] 頂層 lm_head")
    print("=" * 70)
    print(f"  type(model.lm_head) = {type(model.lm_head).__name__}")

    print("\n" + "=" * 70)
    print("[結構] audio 相關 config")
    print("=" * 70)
    print(f"  audio_token_id (或類似欄位) = {getattr(model.config, 'audio_token_id', getattr(model.config, 'audio_token_index', '?'))}")

    # ---- 原始碼：這是重點，讓我看到真實的 forward / 合併邏輯 ----
    safe_getsource(type(model).forward, "Qwen2AudioForConditionalGeneration.forward")
    safe_getsource(type(model.model).forward, "model.model (內層) .forward")

    # 嘗試找出音訊合併相關的 helper method（不同版本命名可能不同，全部嘗試印出）
    for name in [
        "_merge_input_ids_with_audio_features",
        "get_placeholder_mask",
        "get_audio_features",
        "get_input_embeddings",
    ]:
        obj = getattr(type(model.model), name, None) or getattr(type(model), name, None)
        if obj is not None:
            safe_getsource(obj, f"{name}")


if __name__ == "__main__":
    main()
