# ===========================================================
#  主題 : Regression 迴歸分析練習 PART II
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
#  增加每一個輸入項目與輸出項目的數值，以及預測誤差
#  以顏色表示預測準確性
#  誤差定義為預測值與實際值之絕對差：
#  Error = |y - ŷ|
#
#  其中 ŷ 為線性迴歸模型輸出，y 為實際配送完成時間
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
weather_impact = np.random.choice(
    [0, 1],
    size=data_size,
    p=[0.75, 0.25]
)


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
    "Distance": distance,
    "Delivery Points": delivery_points,
    "Package Count": package_count,
    "Traffic Index": traffic_index,
    "Weather Impact": weather_impact,
    "Delivery Time": delivery_time
})


# ===========================================================
# 4. Feature / Label
# ===========================================================

X = df[[
    "Distance",
    "Delivery Points",
    "Package Count",
    "Traffic Index",
    "Weather Impact"
]]

y = df["Delivery Time"]


# ===========================================================
# 5. Train / Test Split
# ===========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===========================================================
# 6. 建立與訓練線性迴歸模型
# ===========================================================

model = LinearRegression()
model.fit(X_train, y_train)


# ===========================================================
# 7. 預測配送時間
# ===========================================================

y_pred = model.predict(X_test)


# ===========================================================
# 8. 去除不合理值 y_pred <= 0
# ===========================================================
# 配送時間不應該小於 0，因此移除不合理預測值

mask = y_pred > 0

X_test_filtered = X_test.iloc[mask].reset_index(drop=True)
y_test_filtered = y_test.iloc[mask].reset_index(drop=True)
y_pred_filtered = pd.Series(y_pred[mask]).reset_index(drop=True)


# ===========================================================
# 9. Error / Metrics
# ===========================================================

error = np.abs(y_test_filtered - y_pred_filtered)

mse = mean_squared_error(y_test_filtered, y_pred_filtered)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_filtered, y_pred_filtered)

print("\n================ MODEL INFO ================")
print("Coefficients:", model.coef_)
print("Intercept   :", model.intercept_)
print("============================================")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")
print("============================================\n")


# ===========================================================
# 10. Result Table
# ===========================================================

result_df = pd.DataFrame({
    "Distance": X_test_filtered["Distance"],
    "Delivery Points": X_test_filtered["Delivery Points"],
    "Package Count": X_test_filtered["Package Count"],
    "Traffic Index": X_test_filtered["Traffic Index"],
    "Weather Impact": X_test_filtered["Weather Impact"],
    "True Delivery Time": y_test_filtered,
    "Predicted Delivery Time": y_pred_filtered,
    "Error": error
})

result_df = result_df.round(2).reset_index(drop=True)

print("\n" + "=" * 120)
print("                         LOGISTICS REGRESSION RESULT TABLE")
print("=" * 120)
print(result_df.to_string(index=True))
print("=" * 120)
print(f"Total Samples: {len(result_df)}")
print("=" * 120)


# ===========================================================
# 11. Visualization
#     綠色：預測誤差較小
#     紅色：預測誤差較大
#     藍線：理想預測線 y = x
# ===========================================================

# 使用平均誤差作為好壞預測的分界
threshold = np.mean(error)

good_mask = error <= threshold
bad_mask = error > threshold

plt.figure(figsize=(10, 6))

# Good points：誤差小於等於平均誤差
plt.scatter(
    y_test_filtered[good_mask],
    y_pred_filtered[good_mask],
    color='green',
    label='Good Prediction',
    alpha=0.7
)

# Bad points：誤差大於平均誤差
plt.scatter(
    y_test_filtered[bad_mask],
    y_pred_filtered[bad_mask],
    color='red',
    label='Bad Prediction',
    alpha=0.7
)

# Ideal line：理想預測線
plt.plot(
    [y_test_filtered.min(), y_test_filtered.max()],
    [y_test_filtered.min(), y_test_filtered.max()],
    color='blue',
    linewidth=2,
    label='Ideal Line (y=x)'
)

plt.xlabel("True Delivery Time")
plt.ylabel("Predicted Delivery Time")

# pad=20：讓標題與圖表保持距離
plt.title(
    "Logistics Regression Result: Good vs Bad Predictions",
    pad=20
)

# 將方框往下移，避免與標題重疊
plt.text(
    0.05,
    0.78,
    f"MSE = {mse:.3f}\nRMSE = {rmse:.3f}\nR² = {r2:.3f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
