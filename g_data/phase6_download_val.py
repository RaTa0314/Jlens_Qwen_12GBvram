import os
import json
import librosa
import soundfile as sf
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 參數設定
# ==========================================
TARGET_ASR_EN = 25
TARGET_ASR_ZH = 25
TARGET_QA = 50

MIN_DUR = 4.0
MAX_DUR = 15.0
TARGET_SR = 16000

# 黑名單檔案路徑 (包含 ASR 與 QA)
PHASE4_ASR_JSONL = "jlens_dataset/prompts.jsonl"
PHASE2_QA_JSONL = "jlens_dataset_qa/prompts.jsonl"

# 輸出路徑 (Phase 6 專用，分離儲存清單與音訊資料夾)
OUT_DIR = "jlens_val_dataset"
AUDIO_ASR_DIR = os.path.join(OUT_DIR, "audio_asr")
AUDIO_QA_DIR = os.path.join(OUT_DIR, "audio_qa")
JSONL_ASR_PATH = os.path.join(OUT_DIR, "val_prompts_asr.jsonl")
JSONL_QA_PATH = os.path.join(OUT_DIR, "val_prompts_qa.jsonl")

# 建立獨立的音訊目錄
os.makedirs(AUDIO_ASR_DIR, exist_ok=True)
os.makedirs(AUDIO_QA_DIR, exist_ok=True)

PROMPT_ASR = "<|audio_bos|><|AUDIO|><|audio_eos|>請寫出這段音訊的逐字稿："
PROMPT_QA = "<|audio_bos|><|AUDIO|><|audio_eos|>請描述這段音訊裡的聲音是什麼："


def get_blacklist_info():
    """
    雙重防禦機制：
    1. blacklist_text: ASR 專用，用過的 transcript 絕對不抓。
    2. old_qa_count: QA 專用，記錄之前抓了幾筆，直接在串流時跳過。
    """
    blacklist_text = set()
    old_qa_count = 0

    # 1. 讀取 Phase 4 ASR 舊資料
    if os.path.exists(PHASE4_ASR_JSONL):
        with open(PHASE4_ASR_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if "transcript" in record:
                    blacklist_text.add(record["transcript"])
        print(f"🛡️ 載入 ASR 黑名單完畢，共 {len(blacklist_text)} 句。")
    else:
        print(f"⚠️ 找不到 {PHASE4_ASR_JSONL}")

    # 2. 讀取 Phase 2 QA 舊資料
    if os.path.exists(PHASE2_QA_JSONL):
        with open(PHASE2_QA_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                old_qa_count += 1
        print(f"🛡️ 載入 QA 黑名單完畢，發現舊資料 {old_qa_count} 筆。")
    else:
        print(f"⚠️ 找不到 {PHASE2_QA_JSONL}")

    return blacklist_text, old_qa_count


def save_sample(idx, prefix, task, audio_array, sr, target_text, prompt, target_dir):
    """將音訊存成本地 wav，並回傳 metadata 字典 (加入 target_dir 指定資料夾)"""
    if sr != TARGET_SR:
        audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=TARGET_SR)
    
    file_name = f"{prefix}_{idx:03d}.wav"
    file_path = os.path.join(target_dir, file_name)
    sf.write(file_path, audio_array, TARGET_SR)
    
    return {
        "id": f"{prefix}_{idx:03d}",
        "task": task,
        "audio_path": file_path,
        "transcript": target_text,
        "prompt": prompt
    }


def main():
    print("=" * 60)
    print("🚀 Phase 6: 準備終極 Validation 資料集 (嚴格防污染 + 完全分離目錄)")
    print("=" * 60)
    
    blacklist_text, old_qa_count = get_blacklist_info()
    
    asr_records = []
    qa_records = []
    
    # ---------------------------------------------------------
    # 1. 抓取 ASR (英文)
    # ---------------------------------------------------------
    print("\n[1/3] 正在搜尋 FLEURS (英文) 作為 ASR Validation...")
    try:
        print("  -> 準備呼叫 load_dataset (正在連線 HF 伺服器)...")
        en_dataset = load_dataset("google/fleurs", "en_us", split="train", streaming=True)
        print("  -> 連線成功！準備讀取第一筆音訊...")
        
        en_count = 0
        for sample in en_dataset:
            # ... 下面的邏輯不變 ...
            if en_count >= TARGET_ASR_EN:
                break
                
            text = sample["transcription"]
            if text in blacklist_text:
                continue  # 防污染機制生效！

            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            duration = len(audio) / sr
            
            if MIN_DUR <= duration <= MAX_DUR:
                record = save_sample(en_count, "VAL_ASR_EN", "ASR", audio, sr, text, PROMPT_ASR, AUDIO_ASR_DIR)
                asr_records.append(record)
                en_count += 1
                print(f"  ✓ 取得 ASR-EN {en_count}/{TARGET_ASR_EN} | {text[:15]}...")
    except Exception as e:
        print(f"❌ 讀取 FLEURS(EN) 失敗: {e}")

    # ---------------------------------------------------------
    # 2. 抓取 ASR (中文)
    # ---------------------------------------------------------
    print("\n[2/3] 正在搜尋 FLEURS (中文) 作為 ASR Validation...")
    try:
        zh_dataset = load_dataset("google/fleurs", "cmn_hans_cn", split="train", streaming=True)
        zh_count = 0
        for sample in zh_dataset:
            if zh_count >= TARGET_ASR_ZH:
                break
                
            text = sample["transcription"]
            if text in blacklist_text:
                continue  # 防污染機制生效！

            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            duration = len(audio) / sr
            
            if MIN_DUR <= duration <= MAX_DUR:
                clean_text = text.replace('\n', ' ')
                record = save_sample(zh_count, "VAL_ASR_ZH", "ASR", audio, sr, clean_text, PROMPT_ASR, AUDIO_ASR_DIR)
                asr_records.append(record)
                zh_count += 1
                print(f"  ✓ 取得 ASR-ZH {zh_count}/{TARGET_ASR_ZH} | {clean_text[:15]}...")
    except Exception as e:
        print(f"❌ 讀取 FLEURS(ZH) 失敗: {e}")

    # ---------------------------------------------------------
    # 3. 抓取 QA (環境音) - 使用 Iterator Skip 技術
    # ---------------------------------------------------------
    print(f"\n[3/3] 正在搜尋 ESC-50 作為 QA Validation (將強制跳過前 {old_qa_count} 筆舊資料)...")
    try:
        qa_dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
        qa_iterator = iter(qa_dataset)
        
        # 🚀 精準跳過之前抓過的所有資料，避免類別被誤殺
        for _ in range(old_qa_count):
            next(qa_iterator, None)
            
        qa_count = 0
        for sample in qa_iterator:
            if qa_count >= TARGET_QA:
                break
                
            text = sample["category"]
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            
            record = save_sample(qa_count, "VAL_QA", "QA", audio, sr, text, PROMPT_QA, AUDIO_QA_DIR)
            qa_records.append(record)
            qa_count += 1
            print(f"  ✓ 取得 QA {qa_count}/{TARGET_QA} | 標籤: {text}")
    except Exception as e:
        print(f"❌ 讀取 ESC-50 失敗: {e}")

    # ---------------------------------------------------------
    # 4. 獨立儲存 Metadata
    # ---------------------------------------------------------
    print("\n💾 正在生成 Phase 6 Validation Metadata...")
    with open(JSONL_ASR_PATH, "w", encoding="utf-8") as f:
        for record in asr_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    with open(JSONL_QA_PATH, "w", encoding="utf-8") as f:
        for record in qa_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("=" * 60)
    print(f"🎉 終極 Validation 資料集建立完成！")
    print(f"📁 ASR 音訊儲存至: {AUDIO_ASR_DIR}/")
    print(f"📄 ASR 清單儲存至: {JSONL_ASR_PATH}")
    print(f"📁 QA  音訊儲存至: {AUDIO_QA_DIR}/")
    print(f"📄 QA  清單儲存至: {JSONL_QA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
