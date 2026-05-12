import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Load the dataset
df = pd.read_csv('merged_dataset_BearingTest_2.csv', index_col=0, parse_dates=True)

# 2. Initialize and Train Isolation Forest
# contamination: 預期異常比例，我們設 5% 左右觀察轉折點
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df)

# 將 -1 (異常) 轉為方便繪圖的標記，1 為正常
# 我們在圖上標示出偵測為 -1 的點
anomalies = df[df['anomaly'] == -1]

# 3. Visualization
plt.figure(figsize=(14, 7))

# 畫出主角 Bearing 1 的趨勢
plt.plot(df.index, df['Bearing 1'], label='Bearing 1 (RMS)', color='royalblue', alpha=0.8)

# 標註異常點
plt.scatter(anomalies.index, anomalies['Bearing 1'], color='crimson', label='Detected Anomaly', s=15, marker='x')

plt.title('Bearing Fault Detection using Isolation Forest', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Vibration Amplitude (RMS)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('anomaly_detection_result.png')

# 找出第一次偵測到持續異常的時間點
first_anomaly = anomalies.index[0]
print(f"First anomaly detected at: {first_anomaly}")
