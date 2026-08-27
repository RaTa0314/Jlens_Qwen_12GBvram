import os
import json
import argparse
import logging
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from jlens import JacobianLens

# 直接從同目錄下的 phase4 匯入 Adapter
from phase4_fit_asr import Qwen2AudioLensModel

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase6_eval")

# ==============================================================================
# 評估核心邏輯
# ==============================================================================
def load_jsonl_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [raw.strip() for raw in f if raw.strip()]

def evaluate_dataset(records, adapter, lens, max_seq_len, layers):
    audio_token_id = adapter._model.config.audio_token_id
    stats = {layer: {"j_corr": 0, "b_corr": 0, "total": 0} for layer in layers}
    
    success_count = 0
    for idx, json_line in enumerate(records):
        try:
            baseline_logits, _, _ = lens.apply(
                adapter, json_line, layers=layers, positions=None,
                max_seq_len=max_seq_len, use_jacobian=False,
            )
            
            lens_logits, model_logits, input_ids = lens.apply(
                adapter, json_line, layers=layers, positions=None,
                max_seq_len=max_seq_len, use_jacobian=True,
            )
            
            # 搬回 CPU 防護
            input_ids = input_ids.cpu()
            mask = (input_ids[0] != audio_token_id)
            total_valid = mask.sum().item()
            
            m_top1 = model_logits.argmax(dim=-1)
            if m_top1.dim() > 1: m_top1 = m_top1[0]
            
            for layer in layers:
                j_top1 = lens_logits[layer].argmax(dim=-1)
                b_top1 = baseline_logits[layer].argmax(dim=-1)
                
                if j_top1.dim() > 1: j_top1 = j_top1[0]
                if b_top1.dim() > 1: b_top1 = b_top1[0]
                
                stats[layer]["j_corr"] += ((j_top1 == m_top1) & mask).sum().item()
                stats[layer]["b_corr"] += ((b_top1 == m_top1) & mask).sum().item()
                stats[layer]["total"] += total_valid
                
            success_count += 1
            if success_count % 10 == 0:
                logger.info(f"  已評估 {success_count}/{len(records)} 筆...")
                
        except Exception as e:
            logger.warning(f"樣本 {idx} 評估失敗跳過: {e}")
            
    return stats


def print_and_format_stats(title, stats, layers):
    print(f"\n{title}")
    print("layer |   J-Lens 一致率 |   baseline 一致率 |       差距 | n_positions")
    print("-" * 65)
    
    formatted_results = []
    for layer in layers:
        s = stats[layer]
        if s["total"] == 0:
            continue
            
        j_acc = s["j_corr"] / s["total"]
        b_acc = s["b_corr"] / s["total"]
        diff = j_acc - b_acc
        
        print(f"   {layer:2d} |        {j_acc*100:4.1f}% |          {b_acc*100:4.1f}% |    {diff*100:+5.1f}% | {s['total']}")
        
        formatted_results.append({
            "layer": layer,
            "jlens_acc": j_acc,
            "baseline_acc": b_acc,
            "n": s["total"]
        })
    return formatted_results


# ==============================================================================
# Main 流程
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--lens-path", default="checkpoints/phase4_asr_lens.pt")
    parser.add_argument("--asr-jsonl", default="g_data/jlens_val_dataset/val_prompts_asr.jsonl")
    parser.add_argument("--qa-jsonl", default="g_data/jlens_val_dataset/val_prompts_qa.jsonl")
    parser.add_argument("--out-json", default="outputs/phase6/phase6_validation_results.json")
    parser.add_argument("--max-seq-len", type=int, default=1024) 
    args = parser.parse_args()

    logger.info("載入 Qwen2-Audio 7B 模型 (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)

    # 🚀 data_root 也換成絕對路徑！
    adapter = Qwen2AudioLensModel(
        full_model, processor, 
        data_root=os.path.expanduser("~/jacobian-lens/g_data"), 
        device="cuda", model_dtype=torch.bfloat16
    )
    
    logger.info(f"掛載 J-Lens 放大鏡 ({args.lens_path})...")
    lens = JacobianLens.load(args.lens_path)
    layers = sorted(lens.source_layers)

    asr_records = load_jsonl_lines(args.asr_jsonl) if os.path.exists(args.asr_jsonl) else []
    qa_records = load_jsonl_lines(args.qa_jsonl) if os.path.exists(args.qa_jsonl) else []
    
    logger.info(f"準備評估: {len(asr_records)} 筆 ASR, {len(qa_records)} 筆 QA")

    final_report = {}

    asr_stats = None
    if asr_records:
        logger.info(">>> 開始評估 ASR 任務 (聽寫泛化測試)...")
        asr_stats = evaluate_dataset(asr_records, adapter, lens, args.max_seq_len, layers)
        final_report["ASR"] = print_and_format_stats("🎤 【ASR 聽寫泛化測試成績單】", asr_stats, layers)

    qa_stats = None
    if qa_records:
        logger.info(">>> 開始評估 QA 任務 (環境音零樣本跨任務測試)...")
        qa_stats = evaluate_dataset(qa_records, adapter, lens, args.max_seq_len, layers)
        final_report["QA"] = print_and_format_stats("🐾 【QA 環境音測試成績單 (Zero-shot)】", qa_stats, layers)

    if asr_stats and qa_stats:
        logger.info(">>> 計算綜合表現...")
        combined_stats = {layer: {"j_corr": 0, "b_corr": 0, "total": 0} for layer in layers}
        for layer in layers:
            combined_stats[layer]["j_corr"] = asr_stats[layer]["j_corr"] + qa_stats[layer]["j_corr"]
            combined_stats[layer]["b_corr"] = asr_stats[layer]["b_corr"] + qa_stats[layer]["b_corr"]
            combined_stats[layer]["total"] = asr_stats[layer]["total"] + qa_stats[layer]["total"]
            
        final_report["Combined"] = print_and_format_stats("🏆 【綜合表現總表】", combined_stats, layers)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"✅ 所有評估完成！完整成績單已儲存至: {args.out_json}")


if __name__ == "__main__":
    main()
