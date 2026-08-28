import os
import json
import argparse
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 參數設定 (Argparse)
    # ==========================================
    parser = argparse.ArgumentParser(description="繪製 J-Lens 評估結果折線圖")
    parser.add_argument("--json-path", required=True, help="輸入的評估結果 JSON 檔案路徑")
    parser.add_argument("--output-image", required=True, help="輸出的圖表圖片路徑 (例如: .png)")
    args = parser.parse_args()

    print(f"📊 正在讀取 {args.json_path}...")
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到檔案 {args.json_path}，請確認是否在正確的目錄執行！")
        return

    # 取得單純的檔案名稱 (例如 phase6_mixed_results.json)
    filename = os.path.basename(args.json_path)

    # 設定畫布大小 (寬 18 吋, 高 5 吋)，1 橫排 3 直排
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 將檔名動態加入主標題中
    fig.suptitle(f'J-Lens vs Baseline Accuracy per Layer\n(Data: {filename})', 
                 fontsize=16, fontweight='bold', y=1.08)

    # 修改：拿掉 Zero-Shot 等不適用的字眼，保持純粹的任務名稱
    tasks = ['ASR', 'QA', 'Combined']
    titles = ['ASR Performance', 'QA Performance', 'Combined Performance']

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
        ax.plot(layers, baseline_acc, label='Baseline', color='#d62728', linewidth=2.5, linestyle='--', marker='x', markersize=5)
        
        # 設定標題與軸標籤
        ax.set_title(titles[i], fontsize=14, pad=10)
        ax.set_xlabel('Layer Depth', fontsize=12)
        if i == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12)
        
        # 設定座標軸範圍與刻度
        ax.set_ylim(-5, 105)
        ax.set_xlim(0, 30)
        ax.set_xticks(range(0, 31, 5))
        
        # 增加網格線
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(fontsize=11, loc='upper left')

    # 自動調整排版並存檔
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=300, bbox_inches='tight')
    print(f"🎉 繪圖完成！折線圖已成功儲存為: {args.output_image}")

if __name__ == "__main__":
    main()
