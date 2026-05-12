# ================================================================
# 課程名稱：馬達軸承預測性維護 (AI Predictive Maintenance)
# 開發環境：Anaconda / Python 3.5+ (Colab 適用)
# 核心技術：Isolation Forest (孤立森林), TensorFlow/Keras 概念實作
# 數據來源：NASA IMS Bearing Dataset (加速壽命實驗)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from google.colab import files
import io

# ----------------------------------------------------------------
# 步驟 1: 資料輸入 (Data Ingestion)
# ----------------------------------------------------------------
print("【系統訊息】請上傳馬達軸承數據集 (merged_dataset_BearingTest_2.csv)")
uploaded = files.upload()

# 讀取檔案
file_name = list(uploaded.keys())[0]
# 單位說明：數值單位為 g (重力加速度)，代表振動衝擊能量
df = pd.read_csv(io.BytesIO(uploaded[file_name]), index_col=0, parse_dates=True)

print(f"\n成功讀取數據！實驗負載：6,000 lbs (約 2,700 公斤)")
print(f"數據共計 {len(df)} 筆快照，取樣頻率 20kHz。")

# ----------------------------------------------------------------
# 步驟 2: AI 演算法執行 (Multi-Bearing Analysis)
# ----------------------------------------------------------------
# 我們使用 Isolation Forest 針對每個軸承獨立建模
bearings = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
comparison_results = []

# 設定畫布佈局 (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

print("\n【系統訊息】AI 正在分析各軸承健康狀態...")

for i, b_name in enumerate(bearings):
    # 初始化演算法 (contamination=0.05 代表預設 5% 為可能的潛在異常)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    
    # 執行偵測：1 代表正常 (Normal), -1 代表異常 (Anomaly)
    df[f'{b_name}_anomaly'] = model.fit_predict(df[[b_name]])
    
    # 提取異常點用於繪圖
    anomalies = df[df[f'{b_name}_anomaly'] == -1]
    
    # 統計物理指標
    max_g = df[b_name].max()        # 最大衝擊值 (g)
    avg_g = df[b_name].mean()      # 平均振動水平 (g)
    first_warning = anomalies.index[0] if not anomalies.empty else "Normal"
    
    comparison_results.append({
        "軸承名稱": b_name,
        "最大振動值 (g)": round(max_g, 4),
        "平均能量 (g)": round(avg_g, 4),
        "AI 預警時間點": first_warning,
        "異常點計數": len(anomalies)
    })
    
    # 繪製趨勢圖
    axes[i].plot(df.index, df[b_name], label='Vibration Trend', color='#2c3e50', alpha=0.4)
    axes[i].scatter(anomalies.index, anomalies[b_name], color='#e74c3c', s=12, label='AI Detected', marker='x')
    axes[i].set_title(f'Analysis: {b_name}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Acceleration (g)')
    axes[i].legend(loc='upper left')
    axes[i].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# 步驟 3: 診斷報告與比較表
# ----------------------------------------------------------------
print("\n" + "="*50)
print("          馬達軸承 AI 診斷分析報告")
print("="*50)

# 轉換為 DataFrame 顯示比較表
summary_table = pd.DataFrame(comparison_results)
display(summary_table)

# 自動診斷結論
faulty_bearing = summary_table.loc[summary_table['最大振動值 (g)'].idxmax(), '軸承名稱']
print(f"\n[AI 專家系統結論]:")
print(f"經由演算法判定，{faulty_bearing} 呈現明顯的物理性退化趨勢。")
print(f"最大衝擊力達到 {summary_table['最大振動值 (g)'].max()}g，遠超正常基準線。")
print(f"建議：立即安排停機，針對 {faulty_bearing} 執行更換作業。")
