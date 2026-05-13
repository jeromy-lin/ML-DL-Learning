# ================================================================
# 課程名稱：馬達軸承預測性維護 II (AI Predictive Maintenance)
# 演算法  ：Isolation Forest (孤立森林), TensorFlow/Keras 概念實作
# 數據來源：NASA IMS Bearing Dataset (加速壽命實驗)
# 請學員載入相關馬達軸承診斷數據集
# 作者 : 國立雲林科技大學電機系 林家仁
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from google.colab import files
import io

# 1. 上傳檔案
print("--- 請上傳 'merged_dataset_BearingTest_2.csv' ---")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[file_name]), index_col=0, parse_dates=True)

# 2. 演算法核心：針對每個軸承單獨進行建模分析
bearings = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
comparison_data = []

# 建立畫布：四個軸承的對比圖
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, b_name in enumerate(bearings):
    # 初始化孤立森林 (針對單一軸承維度)
    model = IsolationForest(contamination=0.05, random_state=42)

    # 執行偵測 (-1 為異常, 1 為正常)
    df[f'{b_name}_anomaly'] = model.fit_predict(df[[b_name]])

    # 提取異常點
    anomalies = df[df[f'{b_name}_anomaly'] == -1]

    # 計算該軸承的統計特徵，用於比較表
    first_anom_time = anomalies.index[0] if not anomalies.empty else "N/A"
    max_vibration = df[b_name].max()
    avg_vibration = df[b_name].mean()

    comparison_data.append({
        "軸承名稱": b_name,
        "最大振動值 (g)": round(max_vibration, 4),
        "平均振動值 (g)": round(avg_vibration, 4),
        "首次偵測異常時間": first_anom_time,
        "異常點總數": len(anomalies)
    })

    # 繪製各別軸承圖表
    axes[i].plot(df.index, df[b_name], label='Vibration', color='gray', alpha=0.5)
    axes[i].scatter(anomalies.index, anomalies[b_name], color='red', s=10, label='Anomaly')
    axes[i].set_title(f'Analysis: {b_name}', fontsize=12, fontweight='bold')
    axes[i].legend()

plt.tight_layout()
plt.show()

# 3. 產出分析比較表
print("\n" + "="*30)
print("   馬達軸承健康診斷比較表")
print("="*30)
comparison_df = pd.DataFrame(comparison_data)
display(comparison_df) # 在 Colab 中以表格形式美化輸出

# 4. 自動化診斷結論
faulty_bearing = comparison_df.loc[comparison_df['最大振動值 (g)'].idxmax(), '軸承名稱']
print(f"\n[AI 診斷結果]: 系統偵測到 {faulty_bearing} 表現最為異常，建議立即執行更換或潤滑維護。")
