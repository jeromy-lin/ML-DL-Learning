# ===========================================================
#  主題 : Rgression 迴歸分析練習
#  目標 : 使學員了解機器學習進行迴歸分析的方法與概念
#  模擬電廠發電 設計 𝒙1 氣溫, 𝒙𝟐 : 濕度、𝒙𝟑 : 風速 , 𝝐 : 誤差項
#   𝒚 ̂  : "預測的發電量", 𝜷 : 對應特徵的回歸係數
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 產生模擬資料
np.random.seed(0)
data_size = 150

temperature = np.random.normal(25, 4, data_size)
humidity = np.random.normal(60, 10, data_size)
wind_speed = np.random.normal(3, 1, data_size)

power_output = 5 + 2.5 * temperature - 1.2 * humidity + 4.8 * wind_speed + np.random.normal(0, 3, data_size)

df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind Speed': wind_speed,
    'Power Output': power_output
})

X = df[['Temperature', 'Humidity', 'Wind Speed']]
y = df['Power Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.maximum(0, y_pred)  # 非負限制

# 移除預測值為0的點（連同真實值一起移除）
mask = y_pred > 0
y_pred_filtered = y_pred[mask]
y_test_filtered = y_test.values[mask]

print("Regression coefficients:", model.coef_)
print("Intercept:", model.intercept_)

mse = mean_squared_error(y_test_filtered, y_pred_filtered)
r2 = r2_score(y_test_filtered, y_pred_filtered)

print(f"Mean Squared Error (MSE) after filtering zeros: {mse:.2f}")
print(f"R-squared (R2) after filtering zeros: {r2:.2f}")

plt.figure(figsize=(10,6))
plt.scatter(y_test_filtered, y_pred_filtered, color='blue')
plt.plot([y_test_filtered.min(), y_test_filtered.max()], 
         [y_test_filtered.min(), y_test_filtered.max()], 'r--')
plt.xlabel('True Power Output')
plt.ylabel('Predicted Power Output (filtered)')
plt.title('Linear Regression Predictions (Filtered Non-zero)')
plt.show()
