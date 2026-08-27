"""
Phase 1: 環境依賴建立與模型載入測試
================================================
目標：
  1. 檢查關鍵依賴（torch / transformers / bitsandbytes / librosa 等）是否就緒
  2. 在 12GB VRAM (RTX 3060) 限制下，以 4-bit (NF4) 量化安全載入 Qwen2-Audio-7B-Instruct
  3. 跑一次最小可行的「音訊 + 文字指令 -> 文字回應」推論，驗證模型可正常運作
  4. 回報 VRAM 使用峰值，作為後續 Phase (資料準備 / J-Lens 擬合) 的記憶體預算基準

使用前準備：
  - 已依「環境建構建議」安裝好 CUDA / PyTorch / transformers(source) / bitsandbytes 等套件
  - 準備一段測試音訊檔（16kHz, wav, 3~8 秒），修改下方 TEST_AUDIO_PATH
    若暫無音訊檔，腳本會自動產生一段合成正弦波作為 smoke test（僅驗證管線可跑通，
    不代表模型輸出有意義）

執行：
    python phase1_env_and_model_load_test.py
"""

import os
import sys
import time

# 建議在 import torch 之前設定，減少記憶體碎片化造成的假性 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TEST_AUDIO_PATH = "test_audio.wav"  # 請替換成你自己的 3~8 秒 16kHz 語音檔
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


# ----------------------------------------------------------------------
# Step 0. 依賴檢查
# ----------------------------------------------------------------------
def check_dependencies():
    print("=" * 70)
    print("[Step 0] 依賴檢查")
    print("=" * 70)

    missing = []
    versions = {}

    try:
        import torch
        versions["torch"] = torch.__version__
        if not torch.cuda.is_available():
            print("[FATAL] 找不到可用的 CUDA GPU，請確認驅動與 CUDA 安裝。")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU: {gpu_name} | 總 VRAM: {total_vram_gb:.1f} GB")
        if total_vram_gb < 11:
            print("  [警告] 偵測到的 VRAM 低於 11GB，本腳本的記憶體預算可能不適用，請調整 batch/長度設定。")
    except ImportError:
        missing.append("torch")

    for pkg in ["transformers", "bitsandbytes", "accelerate", "librosa", "soundfile"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[FATAL] 缺少套件: {missing}")
        print("  請參考環境建構建議安裝，例如：")
        print("  pip install git+https://github.com/huggingface/transformers")
        print("  pip install bitsandbytes accelerate librosa soundfile")
        sys.exit(1)

    print("  已安裝套件版本：")
    for k, v in versions.items():
        print(f"    - {k}: {v}")
    print()


# ----------------------------------------------------------------------
# Step 1. 準備測試音訊（若無現成檔案，合成一段 smoke-test 用的音訊）
# ----------------------------------------------------------------------
def ensure_test_audio(path: str, duration_sec: float = 5.0, sr: int = 16000):
    if os.path.exists(path):
        print(f"[Step 1] 使用既有測試音訊: {path}")
        return path

    print(f"[Step 1] 找不到 {path}，合成一段 {duration_sec}s 正弦波作為管線 smoke test（非語義測試）")
    import numpy as np
    import soundfile as sf

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    sf.write(path, tone, sr)
    return path


# ----------------------------------------------------------------------
# Step 2. 以 4-bit 量化安全載入模型
# ----------------------------------------------------------------------
def load_model_4bit():
    print("=" * 70)
    print("[Step 2] 以 4-bit (NF4) 量化載入 Qwen2-Audio-7B-Instruct")
    print("=" * 70)

    import torch
    from transformers import (
        Qwen2AudioForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # attn_implementation="flash_attention_2" 可選用（需另外安裝 flash-attn，
    # RTX 3060 屬 Ampere 架構相容）；若未安裝則保留預設 "sdpa" 即可。
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    load_time = time.time() - t0
    vram_after_load = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  模型載入完成，耗時 {load_time:.1f}s")
    print(f"  載入後 VRAM 峰值: {vram_after_load:.2f} GB")
    print()

    return model, processor


# ----------------------------------------------------------------------
# Step 3. 最小可行推論測試（音訊 + 文字指令 -> 文字回應）
# ----------------------------------------------------------------------
def run_inference_smoke_test(model, processor, audio_path: str):
    print("=" * 70)
    print("[Step 3] 音訊 + 文字指令推論測試")
    print("=" * 70)

    import torch
    import librosa

    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": "請描述這段音訊的內容。"},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(
        text=text_prompt,
        audio=[audio],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=128)

    gen_time = time.time() - t0
    vram_after_gen = torch.cuda.max_memory_allocated() / (1024 ** 3)

    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"  生成耗時: {gen_time:.1f}s")
    print(f"  推論階段 VRAM 峰值: {vram_after_gen:.2f} GB")
    print(f"  模型輸出: {response!r}")
    print()

    return vram_after_gen


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    check_dependencies()
    audio_path = ensure_test_audio(TEST_AUDIO_PATH)
    model, processor = load_model_4bit()
    peak_vram = run_inference_smoke_test(model, processor, audio_path)

    print("=" * 70)
    print("[結論] Phase 1 完成")
    print("=" * 70)
    print(f"  本次推論階段 VRAM 峰值: {peak_vram:.2f} GB / 12 GB")
    headroom = 12.0 - peak_vram
    print(f"  預估剩餘可用 VRAM（供 Phase 3/4 J-Lens 反向傳播使用）: 約 {headroom:.2f} GB")
    if headroom < 3.0:
        print("  [提醒] 剩餘空間偏緊，J-Lens 擬合時建議：")
        print("    - 縮短音訊長度 / 文字長度")
        print("    - decoder 開啟 gradient checkpointing")
        print("    - 分層擬合（用 JacobianLens.merge() 合併多次結果）")
    else:
        print("  空間看起來足夠進入 Phase 2/3，但正式擬合時仍需持續監控 VRAM。")


if __name__ == "__main__":
    main()
