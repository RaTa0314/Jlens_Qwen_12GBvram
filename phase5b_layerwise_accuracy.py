"""
phase5b_layerwise_accuracy.py
================================
把 Phase 5 質化觀察到的「J-Lens 在早期層就比傳統 logit-lens 更早讀出正確概念」
變成量化證據：對全部 100 筆樣本、每一層都算 top-1 一致率，排除 <|AUDIO|>
佔位符位置（那些位置「預測下一個 token」不是有意義的語言建模任務，混進來
會稀釋訊號）。

輸出一張表：layer -> (J-Lens 一致率, baseline 一致率, 樣本數)，
可以直接畫成「一致率 vs layer 深度」的折線圖。

放在跟 phase4_fit_asr.py 同一個資料夾底下執行：
    python phase5b_layerwise_accuracy.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from phase4_fit_asr import Qwen2AudioLensModel, load_jsonl_lines
from jlens.lens import JacobianLens

DATA_ROOT = os.path.expanduser("~/jacobian-lens/g_data")
JSONL_NAME = "jlens_dataset/prompts.jsonl"
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
LENS_PATH = os.path.expanduser("~/jacobian-lens/checkpoints/phase4_asr_lens.pt")
MAX_SEQ_LEN = 300


def main():
    lens = JacobianLens.load(LENS_PATH)
    print(f"[Lens] {lens!r}")
    layers = lens.source_layers  # 0..30

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)

    audio_token_id = full_model.config.audio_token_id

    jsonl_path = os.path.join(DATA_ROOT, JSONL_NAME)
    records = load_jsonl_lines(jsonl_path)
    print(f"共 {len(records)} 筆樣本")

    correct_jlens = {l: 0 for l in layers}
    correct_baseline = {l: 0 for l in layers}
    total_positions = {l: 0 for l in layers}

    n_ok, n_failed = 0, 0

    for i, json_line in enumerate(records):
        record = json.loads(json_line)
        sample_id = record.get("id", f"idx{i}")
        try:
            adapter = Qwen2AudioLensModel(
                full_model=full_model,
                processor=processor,
                data_root=DATA_ROOT,
                device="cuda",
                model_dtype=torch.bfloat16,
            )

            lens_logits, model_logits, input_ids = lens.apply(
                adapter, json_line, layers=layers, positions=None,
                max_seq_len=MAX_SEQ_LEN, use_jacobian=True,
            )
            baseline_logits, _, _ = lens.apply(
                adapter, json_line, layers=layers, positions=None,
                max_seq_len=MAX_SEQ_LEN, use_jacobian=False,
            )

            input_ids = input_ids.cpu()

            is_text_position = (input_ids[0] != audio_token_id)
            model_top1 = model_logits.argmax(dim=-1)

            for layer in layers:
                jlens_top1 = lens_logits[layer].argmax(dim=-1)
                base_top1 = baseline_logits[layer].argmax(dim=-1)

                mask = is_text_position
                correct_jlens[layer] += ((jlens_top1 == model_top1) & mask).sum().item()
                correct_baseline[layer] += ((base_top1 == model_top1) & mask).sum().item()
                total_positions[layer] += mask.sum().item()

            n_ok += 1

        except Exception as e:
            n_failed += 1
            print(f"[{sample_id}] 失敗，跳過: {e}")
            continue

        finally:
            # 每筆音訊長度不同，重複配置/釋放不同大小的 tensor 長期下來會讓 VRAM
            # 碎片化，最後導致 bitsandbytes 觸發 CPU fallback、產生裝置不一致的錯誤。
            # 下一輪迴圈變數重新賦值時，Python 的參照計數就會釋放這一輪的 tensor，
            # 這裡只需要定期呼叫 empty_cache() 把已釋放的記憶體真的還給 CUDA allocator。
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"  [記憶體] allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

        if (i + 1) % 10 == 0:
            print(f"已處理 {i + 1}/{len(records)} 筆 (成功 {n_ok}, 失敗 {n_failed})")

    print(f"\n完成，成功 {n_ok} 筆，失敗 {n_failed} 筆\n")

    print(f"{'layer':>5} | {'J-Lens 一致率':>12} | {'baseline 一致率':>14} | {'差距':>8} | n_positions")
    print("-" * 65)
    results = []
    for layer in layers:
        n = total_positions[layer]
        j_acc = correct_jlens[layer] / n if n > 0 else float("nan")
        b_acc = correct_baseline[layer] / n if n > 0 else float("nan")
        results.append({"layer": layer, "jlens_acc": j_acc, "baseline_acc": b_acc, "n": n})
        print(f"{layer:>5} | {j_acc:>11.1%} | {b_acc:>13.1%} | {j_acc - b_acc:>+7.1%} | {n}")

    out_path = os.path.expanduser("~/phase5b_layerwise_accuracy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果存到 {out_path}")


if __name__ == "__main__":
    main()
