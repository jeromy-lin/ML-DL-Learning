# ===========================================================
#  主題 : Pareto Front最佳解練習
#  目標 : 使學員了解多目標最佳化方法與概念
#  產生 300 筆 (time, cost) 隨機點 找到最佳解
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================



import numpy as np
import matplotlib.pyplot as plt

# 產生輸入
n_samples = 300
x1 = np.random.uniform(0, 5, n_samples)
x2 = np.random.uniform(0, 3, n_samples)

def f1(x1, x2):
    return (x1 - 1)**2 + (x2 - 2)**2 + 1  # 成本

def f2(x1, x2):
    return (x1 - 3)**2 + (x2 - 1)**2 + 2  # 時間

obj1 = f1(x1, x2)
obj2 = f2(x1, x2)

# 找 Pareto Front 函數
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

cost_time = np.vstack([obj1, obj2]).T
pareto_mask = is_pareto_efficient(cost_time)
pareto_points = cost_time[pareto_mask]

# 選擇權重 (可自行調整偏好)
w1, w2 = 0.6, 0.4
weighted_obj = w1 * obj1 + w2 * obj2

# 找到加權目標最小的輸入 (最佳解)
best_idx = np.argmin(weighted_obj)
best_point = (obj1[best_idx], obj2[best_idx])

# 繪圖
plt.figure(figsize=(8,6))
plt.scatter(obj1, obj2, alpha=0.2, label='All solutions')
plt.scatter(pareto_points[:,0], pareto_points[:,1], color='red', s=30, label='Pareto Front')
plt.scatter(best_point[0], best_point[1], color='blue', s=100, label='Best Compromise')
plt.xlabel('Objective 1 (Cost)')
plt.ylabel('Objective 2 (Time)')
plt.title(f'Pareto Front with Best Compromise (w1={w1}, w2={w2})')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best compromise solution at: Cost={best_point[0]:.3f}, Time={best_point[1]:.3f}")
print(f"Corresponding inputs: x1={x1[best_idx]:.3f}, x2={x2[best_idx]:.3f}")

