# ===========================================================
#  主題 : Rgression 迴歸分析練習 PART II
#  目標 : 使學員了解機器學習進行迴歸分析的方法與概念
#  模擬電廠發電 設計 𝒙1 氣溫, 𝒙𝟐 : 濕度、𝒙𝟑 : 風速 , 𝝐 : 誤差項
#   𝒚 ̂  : "預測的發電量", 𝜷 : 對應特徵的回歸係數
#  增加 每一個輸入項目與輸出項目的 數值 以及 誤差
#  以顏色表示預測準確性 , 誤差定義為預測值與實際值之絕對差：Error = |y - ŷ| 
#  其中 ŷ 為線性迴歸模型輸出，y 為實際觀測值
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ===========================================================
# 1. 產生模擬資料
# ===========================================================
np.random.seed(0)
data_size = 150

temperature = np.random.normal(25, 4, data_size)
humidity = np.random.normal(60, 10, data_size)
wind_speed = np.random.normal(3, 1, data_size)

power_output = (
    5
    + 2.5 * temperature
    - 1.2 * humidity
    + 4.8 * wind_speed
    + np.random.normal(0, 3, data_size)
)

df = pd.DataFrame({
    "Temperature": temperature,
    "Humidity": humidity,
    "Wind Speed": wind_speed,
    "Power Output": power_output
})

# ===========================================================
# 2. feature / label
# ===========================================================
X = df[['Temperature', 'Humidity', 'Wind Speed']]
y = df['Power Output']

# ===========================================================
# 3. train/test split
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================================================
# 4. model training
# ===========================================================
model = LinearRegression()
model.fit(X_train, y_train)

# ===========================================================
# 5. prediction
# ===========================================================
y_pred = model.predict(X_test)

# ===========================================================
# 6. 去除不合理值（y_pred <= 0）
# ===========================================================
mask = y_pred > 0

X_test_filtered = X_test.iloc[mask].reset_index(drop=True)
y_test_filtered = y_test.iloc[mask].reset_index(drop=True)
y_pred_filtered = pd.Series(y_pred[mask]).reset_index(drop=True)

# ===========================================================
# 7. error / metrics
# ===========================================================
error = np.abs(y_test_filtered - y_pred_filtered)

mse = mean_squared_error(y_test_filtered, y_pred_filtered)
r2 = r2_score(y_test_filtered, y_pred_filtered)

print("\n================ MODEL INFO ================")
print("Coefficients:", model.coef_)
print("Intercept   :", model.intercept_)
print("============================================")
print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")
print("============================================\n")

# ===========================================================
# 8.  RESULT TABLE 
# ===========================================================
result_df = pd.DataFrame({
    "Temperature": X_test_filtered["Temperature"],
    "Humidity": X_test_filtered["Humidity"],
    "Wind Speed": X_test_filtered["Wind Speed"],
    "True Power": y_test_filtered,
    "Predict Power": y_pred_filtered,
    "Error": error
})

result_df = result_df.round(2).reset_index(drop=True)

print("\n" + "="*85)
print("                 REGRESSION RESULT TABLE")
print("="*85)
print(result_df.to_string(index=True))
print("="*85)
print(f"Total Samples: {len(result_df)}")
print("="*85)

# ===========================================================
# 9.  VISUALIZATION（綠/紅分類 + 藍線）
# ===========================================================
threshold = np.mean(error)

good_mask = error <= threshold
bad_mask = error > threshold

plt.figure(figsize=(10,6))

# ===== Good points（綠色）=====
plt.scatter(
    y_test_filtered[good_mask],
    y_pred_filtered[good_mask],
    color='green',
    label='Good Prediction',
    alpha=0.7
)

# ===== Bad points（紅色）=====
plt.scatter(
    y_test_filtered[bad_mask],
    y_pred_filtered[bad_mask],
    color='red',
    label='Bad Prediction',
    alpha=0.7
)

# ===== Ideal line（藍色）=====
plt.plot(
    [y_test_filtered.min(), y_test_filtered.max()],
    [y_test_filtered.min(), y_test_filtered.max()],
    color='blue',
    linewidth=2,
    label='Ideal Line (y=x)'
)

plt.xlabel("True Power Output")
plt.ylabel("Predicted Power Output")
plt.title("Regression Result: Good vs Bad Predictions")

# ===== metrics =====
plt.text(
    0.05, 0.95,
    f"MSE = {mse:.3f}\nR² = {r2:.3f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

plt.legend()
plt.show()
