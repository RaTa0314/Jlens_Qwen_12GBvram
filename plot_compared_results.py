import os
import json
import argparse
import matplotlib.pyplot as plt

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到檔案 {path}，請確認路徑是否正確！")
        return None

def main():
    # ==========================================
    # 參數設定 (Argparse)
    # ==========================================
    parser = argparse.ArgumentParser(description="比較三組 J-Lens 評估結果並與 Baseline 繪製在同一張圖")
    
    parser.add_argument("--json1", required=True, help="第一組評估結果 JSON (例如: ASR)")
    parser.add_argument("--label1", default="J-Lens (ASR-Only)", help="第一組在圖例上的名稱")
    
    parser.add_argument("--json2", required=True, help="第二組評估結果 JSON (例如: Mixed)")
    parser.add_argument("--label2", default="J-Lens (Mixed 50/50)", help="第二組在圖例上的名稱")
    
    parser.add_argument("--json3", required=True, help="第三組評估結果 JSON (例如: QA)")
    parser.add_argument("--label3", default="J-Lens (QA-Only)", help="第三組在圖例上的名稱")
    
    parser.add_argument("--output-image", required=True, help="輸出的圖表圖片路徑 (例如: compared_plot.png)")
    args = parser.parse_args()

    print(f"📊 正在讀取並比較:\n 1️⃣ {args.json1} ({args.label1})\n 2️⃣ {args.json2} ({args.label2})\n 3️⃣ {args.json3} ({args.label3})")
    
    data1 = load_json(args.json1)
    data2 = load_json(args.json2)
    data3 = load_json(args.json3)
    
    if not data1 or not data2 or not data3:
        print("❌ 由於部分 JSON 檔案讀取失敗，繪圖中斷。")
        return

    # 設定畫布大小 (寬 18 吋, 高 5 吋)，1 橫排 3 直排
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 標題稍微拉高一點 (y=1.12)，避免被三組 Label 擠壓
    fig.suptitle(f'J-Lens Comparison:\n{args.label1} vs {args.label2} vs {args.label3}', 
                 fontsize=16, fontweight='bold', y=1.12)

    tasks = ['ASR', 'QA', 'Combined']
    titles = ['ASR Performance', 'QA Performance', 'Combined Performance']

    for i, task in enumerate(tasks):
        ax = axes[i]
        
        # 檢查三份資料是否都有該任務
        if task not in data1 and task not in data2 and task not in data3:
            ax.set_title(f"{titles[i]}\n(No Data)", fontsize=14)
            ax.axis('off')
            continue

        # 提取 Baseline 和 Layers (智慧尋找任何一份有效的資料提取 Baseline)
        valid_data = data1 if task in data1 else (data2 if task in data2 else data3)
        layers = [item['layer'] for item in valid_data[task]]
        baseline_acc = [item['baseline_acc'] * 100 for item in valid_data[task]]

        # 繪製 Baseline (共用基準線：紅色虛線 + 叉叉)
        ax.plot(layers, baseline_acc, label='Baseline', color='#d62728', linewidth=2.5, linestyle='--', marker='x', markersize=5)

        # 繪製第一組 J-Lens (藍色實線 + 圓點)
        if task in data1:
            jlens1_acc = [item['jlens_acc'] * 100 for item in data1[task]]
            ax.plot(layers, jlens1_acc, label=args.label1, color='#1f77b4', linewidth=2.5, marker='o', markersize=5)
        
        # 繪製第二組 J-Lens (綠色實線 + 方塊)
        if task in data2:
            jlens2_acc = [item['jlens_acc'] * 100 for item in data2[task]]
            ax.plot(layers, jlens2_acc, label=args.label2, color='#2ca02c', linewidth=2.5, marker='s', markersize=5)

        # 繪製第三組 J-Lens (橘色實線 + 三角形)
        if task in data3:
            jlens3_acc = [item['jlens_acc'] * 100 for item in data3[task]]
            ax.plot(layers, jlens3_acc, label=args.label3, color='#ff7f0e', linewidth=2.5, marker='^', markersize=5)

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
        # 設定 legend 字體大小避免擁擠
        ax.legend(fontsize=10, loc='upper left')

    # 自動調整排版並存檔
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=300, bbox_inches='tight')
    print(f"🎉 三方比較圖表已成功儲存為: {args.output_image}")

if __name__ == "__main__":
    main()
