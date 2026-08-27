import json
import matplotlib.pyplot as plt

# ==========================================
# 參數設定
# ==========================================
JSON_PATH = "phase6_validation_results.json"
OUTPUT_IMAGE = "jlens_results_plot.png"

def main():
    print(f"📊 正在讀取 {JSON_PATH}...")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到檔案 {JSON_PATH}，請確認是否在正確的目錄執行！")
        return

    # 設定畫布大小 (寬 18 吋, 高 5 吋)，1 橫排 3 直排
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('J-Lens vs Baseline (Logit Lens) Accuracy per Layer', fontsize=18, fontweight='bold', y=1.05)

    tasks = ['ASR', 'QA', 'Combined']
    titles = ['ASR (Length Generalization)', 'QA (Zero-Shot Environmental)', 'Combined Performance']

    for i, task in enumerate(tasks):
        if task not in data:
            print(f"⚠️ 找不到 {task} 的數據，跳過繪製。")
            continue

        layers = [item['layer'] for item in data[task]]
        # 將小數點轉成百分比 (%)
        jlens_acc = [item['jlens_acc'] * 100 for item in data[task]]
        baseline_acc = [item['baseline_acc'] * 100 for item in data[task]]

        ax = axes[i]
        
        # 繪製 J-Lens (藍色實線 + 圓點)
        ax.plot(layers, jlens_acc, label='J-Lens', color='#1f77b4', linewidth=2.5, marker='o', markersize=5)
        # 繪製 Baseline (紅色虛線 + 叉叉)
        ax.plot(layers, baseline_acc, label='Baseline (Logit Lens)', color='#d62728', linewidth=2.5, linestyle='--', marker='x', markersize=5)
        
        # 設定標題與軸標籤
        ax.set_title(titles[i], fontsize=14, pad=10)
        ax.set_xlabel('Layer Depth', fontsize=12)
        if i == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12)
        
        # 設定座標軸範圍與刻度
        ax.set_ylim(-5, 105)
        ax.set_xlim(0, 30)
        ax.set_xticks(range(0, 31, 5))
        
        # 增加網格線 (更有學術感)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=11, loc='upper left')

    # 自動調整排版並存檔
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"🎉 繪圖完成！折線圖已成功儲存為: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()