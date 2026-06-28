# ===========================================================
#  主題 : Regression 迴歸分析練習
#  目標 : 使學員了解機器學習進行物流配送時間預測的方法與概念
#  情境 : 以宅配物流配送為主軸，預測單一路線完成配送所需時間
#
#  模擬物流配送資料：
#  x1 : 配送距離 Distance
#  x2 : 配送點數 Delivery Points
#  x3 : 包裹數量 Package Count
#  x4 : 交通壅塞指數 Traffic Index
#  x5 : 天氣影響 Weather Impact
#  ε  : 誤差項
#
#  ŷ  : 預測的配送完成時間
#  β  : 對應特徵的回歸係數
#
#  作者 : 國立雲林科技大學電機系 林家仁
# ===========================================================

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

# 配送距離，單位：公里
distance = np.random.normal(35, 10, data_size)
distance = np.maximum(5, distance)

# 配送點數，例如一條路線要送幾個地址
delivery_points = np.random.normal(45, 12, data_size)
delivery_points = np.maximum(10, delivery_points)

# 包裹數量，通常會比配送點數多，因為同一地址可能有多件
package_count = delivery_points + np.random.normal(15, 8, data_size)
package_count = np.maximum(delivery_points, package_count)

# 交通壅塞指數
# 1.0 表示正常交通，越大代表越塞
traffic_index = np.random.normal(1.2, 0.25, data_size)
traffic_index = np.maximum(0.8, traffic_index)

# 天氣影響指數
# 0 表示天氣良好，1 表示下雨或天候不佳
weather_impact = np.random.choice([0, 1], size=data_size, p=[0.75, 0.25])


# ===========================================================
# 2. 建立配送時間模型
# ===========================================================
# 假設真實配送時間受到以下因素影響：
# 基本作業時間        : 20 分鐘
# 每公里配送距離      : 增加 2.2 分鐘
# 每個配送點          : 增加 3.5 分鐘
# 每件包裹            : 增加 0.8 分鐘
# 交通壅塞指數        : 增加 25 分鐘
# 天氣不佳            : 增加 18 分鐘
# 誤差項              : 模擬實際配送中的不確定因素

delivery_time = (
    20
    + 2.2 * distance
    + 3.5 * delivery_points
    + 0.8 * package_count
    + 25 * traffic_index
    + 18 * weather_impact
    + np.random.normal(0, 12, data_size)
)

# 配送時間不能小於 0
delivery_time = np.maximum(0, delivery_time)


# ===========================================================
# 3. 建立 DataFrame
# ===========================================================

df = pd.DataFrame({
    'Distance': distance,
    'Delivery Points': delivery_points,
    'Package Count': package_count,
    'Traffic Index': traffic_index,
    'Weather Impact': weather_impact,
    'Delivery Time': delivery_time
})

print(df.head())


# ===========================================================
# 4. 準備訓練資料
# ===========================================================

X = df[['Distance', 'Delivery Points', 'Package Count', 'Traffic Index', 'Weather Impact']]
y = df['Delivery Time']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ===========================================================
# 5. 建立與訓練線性迴歸模型
# ===========================================================

model = LinearRegression()
model.fit(X_train, y_train)


# ===========================================================
# 6. 預測配送時間
# ===========================================================

y_pred = model.predict(X_test)

# 配送時間不應該為負數，因此加入非負限制
y_pred = np.maximum(0, y_pred)


# ===========================================================
# 7. 顯示迴歸係數
# ===========================================================

print("\nRegression coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print(f"\nIntercept: {model.intercept_:.2f}")


# ===========================================================
# 8. 模型評估
# ===========================================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# ===========================================================
# 9. 視覺化：真實配送時間 vs 預測配送時間
# ===========================================================

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, color='blue')

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel('True Delivery Time')
plt.ylabel('Predicted Delivery Time')
plt.title('Linear Regression for Logistics Delivery Time Prediction')
plt.grid(True)
plt.show()
