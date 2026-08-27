"""
phase5_apply_lens.py
======================
套用 Phase 4 fit 好的 JacobianLens：
  1. 質化檢查：挑幾筆樣本，逐 layer/position 印出 J-Lens vs 傳統 logit-lens
     (use_jacobian=False) vs 模型真實預測的 top-k 解碼結果。
  2. 量化檢查：在最靠近輸出的 fitted layer 上，算「J-Lens 讀出的 top-1」跟
     「模型真實 top-1 預測」的一致率，跟傳統 logit-lens 的一致率對比——
     J-Lens 應該要明顯更準，這是驗證整套流程有沒有用的第一道量化指標。

注意：目前 SAMPLE_IDS_TO_INSPECT 裡的樣本都在 fit 用過的 100 筆之內（in-sample），
還不是嚴格意義的 held-out 泛化測試。如果要驗證 lens 對「沒看過的音訊」是否
一樣有效，需要另外找幾筆沒放進 Phase 4 訓練語料的音訊來測。

放在跟 phase4_fit_asr.py 同一個資料夾底下執行：
    python phase5_apply_lens.py
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

# 依你 prompts.jsonl 裡實際的 id 命名調整（先前看到的樣本是 EN_0, EN_1... ZH_...）
SAMPLE_IDS_TO_INSPECT = ["EN_0", "ZH_0"]
LAYERS_TO_SHOW = [0, 4, 8, 12, 16, 20, 24, 28, 30]
TOP_K = 3
MAX_SEQ_LEN = 300


def decode_top_tokens(tokenizer, logits_row, k):
    topk = torch.topk(logits_row, k)
    return [tokenizer.decode([tid]).strip() for tid in topk.indices.tolist()]


def top1_agreement(lens_logits_layer, model_logits):
    """兩個 [n_positions, vocab_size] logits，算 top-1 token 的一致率。"""
    lens_top1 = lens_logits_layer.argmax(dim=-1)
    model_top1 = model_logits.argmax(dim=-1)
    return (lens_top1 == model_top1).float().mean().item()


def main():
    lens = JacobianLens.load(LENS_PATH)
    print(f"[Lens] {lens!r}")

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

    jsonl_path = os.path.join(DATA_ROOT, JSONL_NAME)
    records = {json.loads(l)["id"]: l for l in load_jsonl_lines(jsonl_path)}

    layers_available = [l for l in LAYERS_TO_SHOW if l in lens.source_layers]
    missing = [l for l in LAYERS_TO_SHOW if l not in lens.source_layers]
    if missing:
        print(f"[注意] 這些層沒有被 fit 到，略過: {missing}")
    deepest_layer = max(layers_available)  # 離輸出最近的 fitted layer，量化比較用

    for sample_id in SAMPLE_IDS_TO_INSPECT:
        if sample_id not in records:
            print(f"\n[跳過] 找不到 id={sample_id}，請確認 prompts.jsonl 裡實際的 id 命名。")
            continue
        json_line = records[sample_id]
        record = json.loads(json_line)
        print("\n" + "=" * 78)
        print(f"樣本 {sample_id} | transcript(ground truth) = {record['transcript']!r}")
        print("=" * 78)

        adapter = Qwen2AudioLensModel(
            full_model=full_model,
            processor=processor,
            data_root=DATA_ROOT,
            device="cuda",
            model_dtype=torch.bfloat16,
        )

        lens_logits, model_logits, input_ids = lens.apply(
            adapter, json_line, layers=layers_available, positions=None,
            max_seq_len=MAX_SEQ_LEN, use_jacobian=True,
        )
        baseline_logits, _, _ = lens.apply(
            adapter, json_line, layers=layers_available, positions=None,
            max_seq_len=MAX_SEQ_LEN, use_jacobian=False,
        )

        # ---------------- 量化：top-1 一致率（全部位置，最深的 fitted layer）----------------
        jlens_agree = top1_agreement(lens_logits[deepest_layer], model_logits)
        baseline_agree = top1_agreement(baseline_logits[deepest_layer], model_logits)
        print(f"\n[量化] layer {deepest_layer} 對「模型真實預測」的 top-1 一致率:")
        print(f"    J-Lens:        {jlens_agree:.1%}")
        print(f"    傳統 logit-lens: {baseline_agree:.1%}")
        if jlens_agree > baseline_agree:
            print("    -> J-Lens 比傳統 logit-lens 更貼近模型真實行為，符合預期。")
        else:
            print("    -> J-Lens 沒有贏過傳統 logit-lens，這一點值得深入檢查"
                  "（樣本數太少 / 該層本身線性度不足 / fit 語料跟這筆樣本差異太大 都有可能）。")

        # ---------------- 質化：挑幾個位置，逐層印出 top-k 解碼 ----------------
        seq_len = input_ids.shape[1]
        tokenizer = adapter.tokenizer
        input_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        positions_to_show = sorted(set([seq_len // 4, seq_len // 2, seq_len - 5, seq_len - 3, seq_len - 1]))
        positions_to_show = [p for p in positions_to_show if 0 <= p < seq_len]

        for pos in positions_to_show:
            print(f"\n--- position {pos} (輸入 token: {input_tokens[pos]!r}) ---")
            actual_top = decode_top_tokens(tokenizer, model_logits[pos], TOP_K)
            print(f"  模型真實預測 (最後一層): {actual_top}")
            for layer in layers_available:
                jlens_top = decode_top_tokens(tokenizer, lens_logits[layer][pos], TOP_K)
                base_top = decode_top_tokens(tokenizer, baseline_logits[layer][pos], TOP_K)
                print(f"  layer {layer:>2} | J-Lens: {jlens_top}  |  plain logit-lens: {base_top}")


if __name__ == "__main__":
    main()
