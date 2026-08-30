"""
verify_qa_encode.py
======================
不等 12 小時，直接用現有的 adapter 對「一筆」QA 訓練樣本呼叫 encode()，
印出中間過程：target_text 抓到了什麼、组出來的 full_text 長什麼樣、
最後 tokenize 出來的長度是多少。

這能在幾秒內回答一個問題：這次 fit 到底有沒有真的把 category 標籤
（例如 "chainsaw"、"door_wood_knock"）接進 teacher-forcing 目標，
還是又意外變回空字串。

放在跟 phase4_fit.py 同一個資料夾底下執行：
    python verify_qa_encode.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from phase4_fit import Qwen2AudioLensModel, load_jsonl_lines

DATA_ROOT = "g_data"
JSONL_NAME = "jlens_dataset_qa/prompts.jsonl"
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"


def main():
    jsonl_path = os.path.join(DATA_ROOT, JSONL_NAME)
    records = load_jsonl_lines(jsonl_path)
    print(f"讀到 {len(records)} 筆 QA 訓練樣本")

    # 只看前 3 筆的原始 target_text 抓取邏輯，不用載入模型就能先確認一半
    print("\n===== Step 1: 純邏輯層面檢查（不需要模型）=====")
    for line in records[:3]:
        record = json.loads(line)
        target_text = (
            record.get("transcript")
            or record.get("category")
            or record.get("answer")
            or record.get("target")
        )
        print(f"  id={record.get('id')!r}  抓到的 target_text={target_text!r}")
        if not target_text:
            print("  !!! target_text 是空的，encode() 這裡會直接 raise KeyError !!!")

    # ---------------------------------------------------------------
    # Step 2: 真的載入模型，跑一次 encode()，看實際組出來的 full_text
    # ---------------------------------------------------------------
    print("\n===== Step 2: 實際呼叫 adapter.encode()（會載入模型，約需 1 分鐘）=====")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)

    adapter = Qwen2AudioLensModel(
        full_model=full_model, processor=processor, data_root=DATA_ROOT,
        device="cuda", model_dtype=torch.bfloat16,
    )

    line = records[0]
    record = json.loads(line)
    print(f"  測試樣本 id={record.get('id')!r}")
    print(f"  原始 prompt 欄位: {record.get('prompt')!r}")
    print(f"  原始 category 欄位: {record.get('category')!r}")

    input_ids = adapter.encode(line, max_length=300)
    decoded_full_text = adapter.tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f"\n  encode() 回傳 input_ids.shape = {tuple(input_ids.shape)}")
    print(f"  decode 回去的完整文字:\n    {decoded_full_text!r}")

    if record.get("category") and record["category"] in decoded_full_text:
        print(f"\n  ✅ 確認：category 的值 {record['category']!r} 有出現在最終的 full_text 裡，encode() 抓對欄位了。")
    else:
        print(f"\n  ❌ 警告：category 的值沒有出現在最終 decode 出來的文字裡，encode() 可能還是沒抓到正確欄位！")


if __name__ == "__main__":
    main()
