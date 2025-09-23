# ============================================
#  Cluster_Kmeans
#  目標 : 使學員了解K Means 分群法以及圖形繪製概念
#  作者：國立雲林科技大學 林家仁
# ============================================
# !pip install scikit-learn -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. 產生 30 筆 (time, cost) 隨機點 ── 三個聚落
np.random.seed(0)  # 讓結果可重現
cluster_centers = [(10, 100), (50, 300), (80, 150)]
points_per_cluster = 10

data = []
for cx, cy in cluster_centers:
    times = np.random.normal(loc=cx, scale=3,  size=points_per_cluster)
    costs = np.random.normal(loc=cy, scale=30, size=points_per_cluster)
    data.extend(zip(times, costs))

df = pd.DataFrame(data, columns=["time", "cost"])

# 2. K‑Means 分群 (k = 3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
df["cluster"] = kmeans.fit_predict(df[["time", "cost"]])

# 3. 顯示資料表
display(df)

# 4. 視覺化：散點圖
plt.figure(figsize=(8, 5))
plt.scatter(df["time"], df["cost"], c=df["cluster"], s=60)
plt.title("Time vs Cost – K‑Means Clusters")
plt.xlabel("Time")
plt.ylabel("Cost")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

