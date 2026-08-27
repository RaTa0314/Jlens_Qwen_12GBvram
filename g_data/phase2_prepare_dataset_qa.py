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
TARGET_COUNT = 100   # 需要 100 筆環境音
TARGET_SR = 16000    # Qwen2-Audio 的標準採樣率

# ⚠️ 注意：存到不同的資料夾，避免覆蓋剛才辛苦抓好的語音資料！
OUT_DIR = "jlens_dataset_qa"
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
JSONL_PATH = os.path.join(OUT_DIR, "prompts.jsonl")

os.makedirs(AUDIO_DIR, exist_ok=True)

# 針對音訊問答的專屬提示詞
PROMPT_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>請描述這段音訊裡的聲音是什麼："

def save_sample(idx, audio_array, sr, category):
    if sr != TARGET_SR:
        audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=TARGET_SR)
    
    file_name = f"env_{idx:03d}.wav"
    file_path = os.path.join(AUDIO_DIR, file_name)
    
    sf.write(file_path, audio_array, TARGET_SR)
    
    return {
        "id": f"env_{idx}",
        "audio_path": file_path,
        "category": category,
        "prompt": PROMPT_TEMPLATE
    }

def main():
    print("=" * 60)
    print("🐾 Phase 2 (備用): Qwen2-Audio J-Lens 環境音 QA 資料集 (ESC-50)")
    print("=" * 60)
    
    dataset_records = []
    
    print("\n正在串流搜尋 ESC-50 (環境/動物/日常聲)...")
    try:
        # ESC-50 完全開源免登入
        qa_dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
        
        count = 0
        for sample in qa_dataset:
            if count >= TARGET_COUNT:
                break
                
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            category = sample["category"] # 這是聲音的標籤，例如 "dog", "rain"
            
            # ESC-50 原生就是 5 秒，我們直接存
            record = save_sample(count, audio, sr, category)
            dataset_records.append(record)
            count += 1
            print(f"  ✓ 取得環境音樣本 {count}/{TARGET_COUNT} | 聲音類型: {category}")
            
    except Exception as e:
        print(f"\n❌ 讀取 ESC-50 失敗！錯誤細節: {e}")

    # 儲存 Metadata JSONL
    if len(dataset_records) > 0:
        print("\n💾 正在生成 Phase 4 所需的 Metadata...")
        with open(JSONL_PATH, "w", encoding="utf-8") as f:
            for record in dataset_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print("=" * 60)
        print(f"🎉 環境音 QA 資料集建立完成！共 {len(dataset_records)} 筆。")
        print(f"📁 音訊檔已儲存至: {AUDIO_DIR}/")
        print(f"📄 清單檔已儲存至: {JSONL_PATH}")
        print("=" * 60)

if __name__ == "__main__":
    main()
