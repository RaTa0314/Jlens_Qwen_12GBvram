import os
import json
import librosa
import soundfile as sf
from datasets import load_dataset
import warnings

# 忽略 librosa 的一些警告
warnings.filterwarnings("ignore")

# ==========================================
# 參數設定
# ==========================================
TARGET_EN = 50       # 英文需求數量
TARGET_ZH = 50       # 中文需求數量
MIN_DUR = 2.0        # 最短 2 秒
MAX_DUR = 5.0        # 最長 5 秒
TARGET_SR = 16000    # Qwen2-Audio 的標準採樣率

OUT_DIR = "jlens_dataset"
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
JSONL_PATH = os.path.join(OUT_DIR, "prompts.jsonl")

os.makedirs(AUDIO_DIR, exist_ok=True)

PROMPT_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>請寫出這段音訊的逐字稿："

def save_sample(idx, lang, audio_array, sr, text):
    """將音訊存成本地 wav，並回傳 metadata 字典"""
    if sr != TARGET_SR:
        audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=TARGET_SR)
    
    file_name = f"{lang}_{idx:03d}.wav"
    file_path = os.path.join(AUDIO_DIR, file_name)
    
    sf.write(file_path, audio_array, TARGET_SR)
    
    return {
        "id": f"{lang}_{idx}",
        "language": lang,
        "audio_path": file_path,
        "transcript": text,
        "prompt": PROMPT_TEMPLATE
    }

def main():
    print("=" * 60)
    print("🎧 Phase 2: Qwen2-Audio J-Lens 雙語資料集 (FLEURS 多樣化百科)")
    print("=" * 60)
    
    dataset_records = []
    
    # ---------------------------------------------------------
    # 1. 抓取英文 (google/fleurs - en_us 子集)
    # ---------------------------------------------------------
    print("\n[1/2] 正在串流搜尋 FLEURS (美式英文 en_us)...")
    try:
        en_dataset = load_dataset("google/fleurs", "en_us", split="train", streaming=True)
        
        en_count = 0
        for sample in en_dataset:
            if en_count >= TARGET_EN:
                break
                
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            text = sample["transcription"]
            duration = len(audio) / sr
            
            if MIN_DUR <= duration <= MAX_DUR:
                record = save_sample(en_count, "EN", audio, sr, text)
                dataset_records.append(record)
                en_count += 1
                print(f"  ✓ 取得 EN 樣本 {en_count}/{TARGET_EN} (長度: {duration:.1f}s) | 內容: {text[:15]}...")
    except Exception as e:
        print(f"\n❌ 讀取 FLEURS(EN) 失敗！錯誤細節: {e}")

    # ---------------------------------------------------------
    # 2. 抓取中文 (google/fleurs - cmn_hans_cn 子集)
    # ---------------------------------------------------------
    print("\n[2/2] 正在串流搜尋 FLEURS (簡體中文 cmn_hans_cn)...")
    try:
        # 修改這裡：使用官方存在的簡體中文子集
        zh_dataset = load_dataset("google/fleurs", "cmn_hans_cn", split="train", streaming=True)
        
        zh_count = 0
        for sample in zh_dataset:
            if zh_count >= TARGET_ZH:
                break
                
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            text = sample["transcription"]
            duration = len(audio) / sr
            
            if MIN_DUR <= duration <= MAX_DUR:
                record = save_sample(zh_count, "ZH", audio, sr, text)
                dataset_records.append(record)
                zh_count += 1
                # 為了避免終端機排版亂掉，把換行符號拿掉
                clean_text = text.replace('\n', ' ')
                print(f"  ✓ 取得 ZH 樣本 {zh_count}/{TARGET_ZH} (長度: {duration:.1f}s) | 內容: {clean_text[:15]}...")
                
    except Exception as e:
        print(f"\n❌ 讀取 FLEURS(ZH) 失敗！錯誤細節: {e}")

    # ---------------------------------------------------------
    # 3. 儲存 Metadata JSONL
    # ---------------------------------------------------------
    if len(dataset_records) > 0:
        print("\n💾 正在生成 Phase 4 所需的 Metadata...")
        with open(JSONL_PATH, "w", encoding="utf-8") as f:
            for record in dataset_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print("=" * 60)
        print(f"🎉 高品質多樣化資料集建立完成！共 {len(dataset_records)} 筆。")
        print(f"📁 音訊檔已儲存至: {AUDIO_DIR}/")
        print(f"📄 清單檔已儲存至: {JSONL_PATH}")
        print("=" * 60)
    else:
        print("\n⚠️ 未能成功取得任何樣本。")

if __name__ == "__main__":
    main()
