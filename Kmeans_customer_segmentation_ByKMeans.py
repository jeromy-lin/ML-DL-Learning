# ===========================================================
#  主題 : K_Means分群練習
#  目標 : 使學員了解機器學習進行分群的方法與概念
#  模擬顧客資料以使用 K-Means 找出潛在分群
#  X軸  : 每月訪問頻率
#  Y軸  : 平均單筆消費金額
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 三群顧客的中心位置
customer_centers = [
    [3, 300],      # 低頻率、低消費
    [8, 900],      # 中頻率、中消費
    [15, 1800]     # 高頻率、高消費
]

# 數值越小，資料點越集中
cluster_std = [
    [0.6, 50],
    [0.8, 80],
    [1.0, 110]
]

# 增加至600筆資料，使畫面更密集
X, _ = make_blobs(
    n_samples=600,
    centers=customer_centers,
    cluster_std=cluster_std,
    random_state=42
)

# 避免出現負值
X[:, 0] = np.clip(X[:, 0], 1, None)
X[:, 1] = np.clip(X[:, 1], 50, None)

# K-Means 分成3群
kmeans = KMeans(
    n_clusters=3,
    random_state=0,
    n_init=10
)

labels = kmeans.fit_predict(X)

# 繪製分群結果
plt.figure(figsize=(10, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="Set1",
    s=25,              # 
    alpha=0.75
)

# 顯示群中心
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=220,
    label="Cluster Centroids"
)

plt.title("Customer Segmentation by K-Means", fontsize=16)
plt.xlabel("Monthly Visit Frequency", fontsize=12)
plt.ylabel("Average Purchase Amount", fontsize=12)

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
