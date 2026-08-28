"""
phase4_finalize.py
====================
Phase 4 的 main() 呼叫完 jlens_fit() 之後沒有接住回傳值、也沒存檔——
但 checkpoint 裡的 jacobian_sum / n_done / next_idx 資訊完整，代表 12 小時的
計算成果都還在。這支腳本重新呼叫一次 fit()（resume=True 會偵測到已經跑完，
瞬間回傳，不會重算），把回傳的 JacobianLens 存成正式的結果檔。
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from phase4_fit import (
    Qwen2AudioLensModel,
    load_jsonl_lines,
    filter_by_length,
)
from jlens.fitting import fit as jlens_fit

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Finalize checkpoint into .pt model")
    
    # 全部換成相對路徑，並改為透過終端機參數接收
    parser.add_argument("--data-root", default="g_data")
    parser.add_argument("--jsonl-name", required=True, help="訓練使用的 JSONL : --jsonl-name")
    parser.add_argument("--model-name", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--checkpoint-path", required=True, help="要讀取的 ckpt : --checkpoint-path")
    parser.add_argument("--output-path", required=True, help="輸出的 pt : --output-path")
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=300)
    args = parser.parse_args()

    jsonl_path = os.path.join(args.data_root, args.jsonl_name)
    records = load_jsonl_lines(jsonl_path)
    print(f"讀到 {len(records)} 筆原始資料")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)

    records = filter_by_length(
        records,
        processor=processor,
        data_root=args.data_root,
        sampling_rate=processor.feature_extractor.sampling_rate,
        max_length=args.max_seq_len,
    )
    print(f"過濾後剩餘 {len(records)} 筆（應該跟當初正式跑的時候一致）")

    adapter = Qwen2AudioLensModel(
        full_model=full_model,
        processor=processor,
        data_root=args.data_root,
        device="cuda",
        model_dtype=torch.bfloat16,
    )

    n_layers = adapter.n_layers
    source_layers = list(range(n_layers - 1))
    target_layer = n_layers - 1

    print("重新呼叫 fit()（resume=True，checkpoint 已完成的話應該幾秒內就回傳）...")
    lens = jlens_fit(
        model=adapter,
        prompts=records,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=1,
        resume=True,
    )

    print(repr(lens))
    lens.save(args.output_path)
    print(f"✅ 已存成正式結果檔: {args.output_path}")

if __name__ == "__main__":
    main()
