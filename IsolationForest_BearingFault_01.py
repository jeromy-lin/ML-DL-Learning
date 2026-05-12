# ================================================================
# 課程名稱：馬達軸承預測性維護 (AI Predictive Maintenance)
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

# ==========================================
# 1. 手動上傳檔案
# ==========================================
print("【步驟 1】請上傳您的 CSV 數據檔案")
uploaded = files.upload()

# 讀取上傳的檔案
file_name = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[file_name]), index_col=0, parse_dates=True)

print(f"成功載入檔案：{file_name}")

# ==========================================
# 2. 執行 AI 異常檢測 (僅針對 Bearing 1)
# ==========================================
# 初始化孤立森林演算法
# contamination=0.05 代表我們假設資料中約有 5% 的異常點
model = IsolationForest(contamination=0.05, random_state=42)

# 訓練模型並預測 (-1 為異常, 1 為正常)
df['anomaly_label'] = model.fit_predict(df[['Bearing 1']])

# 提取異常點用於後續繪圖
anomalies = df[df['anomaly_label'] == -1]

# ==========================================
# 3. 繪圖與結果展示
# ==========================================
plt.figure(figsize=(12, 6))

# 繪製 Bearing 1 的原始振動趨勢 (單位: g)
plt.plot(df.index, df['Bearing 1'], label='Bearing 1 Vibration', color='steelblue', alpha=0.7)

# 標註 AI 偵測到的異常點 (紅色叉號)
plt.scatter(anomalies.index, anomalies['Bearing 1'], color='red', label='Detected Anomaly', s=20, marker='x')

# 圖表美化
plt.title('Bearing 1 - Fault Detection Analysis', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Vibration Amplitude (g)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

# 輸出簡單報告
print(f"--- 分析報告 ---")
print(f"1. 監測數據總筆數: {len(df)}")
print(f"2. AI 偵測到的異常點數量: {len(anomalies)}")
if not anomalies.empty:
    print(f"3. 首次偵測到異常的時間: {anomalies.index[0]}")
