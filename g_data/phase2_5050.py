import os
import json
import random
import shutil

# ==========================================
# 參數設定
# ==========================================
ASR_JSONL = "jlens_dataset/prompts.jsonl"
QA_JSONL = "jlens_dataset_qa/prompts.jsonl"

OUT_DIR = "jlens_dataset_mixed"
OUT_AUDIO_DIR = os.path.join(OUT_DIR, "audio")
OUT_JSONL = os.path.join(OUT_DIR, "prompts.jsonl")

# 固定亂數種子，確保每次抽出來的 50/50 組合都一樣
random.seed(42)

def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        print(f"⚠️ 找不到檔案: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def process_and_copy(records, task_name, sample_size=50):
    """隨機抽取指定數量的資料，複製音訊，並標準化 JSON 格式"""
    if len(records) < sample_size:
        print(f"⚠️ {task_name} 資料不足 {sample_size} 筆，將全數使用 ({len(records)} 筆)")
        sample_size = len(records)
        
    sampled_records = random.sample(records, sample_size)
    processed_records = []
    
    for record in sampled_records:
        old_audio_path = record["audio_path"]
        file_name = os.path.basename(old_audio_path)
        
        # 為了避免 ASR 和 QA 檔名剛好重複，我們加上 task 前綴
        new_file_name = f"{task_name}_{file_name}"
        new_audio_path = os.path.join(OUT_AUDIO_DIR, new_file_name)
        
        # 複製音訊檔
        if os.path.exists(old_audio_path):
            shutil.copy2(old_audio_path, new_audio_path)
        else:
            print(f"❌ 找不到音訊檔: {old_audio_path}，已跳過。")
            continue
            
        # 標準化 JSON 格式
        new_record = {
            "id": f"MIXED_{record['id']}",
            "task": task_name,
            "audio_path": new_audio_path,
            "prompt": record["prompt"],
        }
        
        # 統一將答案欄位命名為 transcript (解決 QA 原本叫 category 的問題)
        if "transcript" in record:
            new_record["transcript"] = record["transcript"]
        elif "category" in record:
            new_record["transcript"] = record["category"]
            
        processed_records.append(new_record)
        
    return processed_records

def main():
    print("=" * 60)
    print("🧪 Phase 2: 建立 50/50 混合訓練資料集 (ASR + QA)")
    print("=" * 60)
    
    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    
    # 1. 讀取並抽取 ASR 資料
    print(f"\n[1/3] 讀取 ASR 資料 ({ASR_JSONL})...")
    asr_records = load_jsonl(ASR_JSONL)
    asr_mixed = process_and_copy(asr_records, "ASR", 50)
    print(f"  ✓ 成功抽取並複製 {len(asr_mixed)} 筆 ASR 資料。")
    
    # 2. 讀取並抽取 QA 資料
    print(f"\n[2/3] 讀取 QA 資料 ({QA_JSONL})...")
    qa_records = load_jsonl(QA_JSONL)
    qa_mixed = process_and_copy(qa_records, "QA", 50)
    print(f"  ✓ 成功抽取並複製 {len(qa_mixed)} 筆 QA 資料。")
    
    # 3. 混合並打亂 (Shuffle) 資料
    print("\n[3/3] 混合並打亂訓練資料...")
    final_mixed_records = asr_mixed + qa_mixed
    random.shuffle(final_mixed_records)
    
    # 儲存成新的 JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for record in final_mixed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("=" * 60)
    print(f"🎉 50/50 混合資料集建立完成！共 {len(final_mixed_records)} 筆。")
    print(f"📁 混合音訊儲存至: {OUT_AUDIO_DIR}/")
    print(f"📄 訓練清單儲存至: {OUT_JSONL}")
    print("=" * 60)

if __name__ == "__main__":
    main()