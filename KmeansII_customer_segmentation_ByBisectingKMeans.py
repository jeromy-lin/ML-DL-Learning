# ===========================================================
#  主題 : K-Means vs Bisecting K-Means 顧客分群實作
#  本範例用於教學機器學習分群方法之概念，透過顧客行為資料 訪問頻率 vs 單筆消費金額）比較兩種分群演算法之差異：
#  1. K-Means Clustering vs 2. Bisecting K-Means Clustering
#   ■ K-Means：
#      - 一次性分群（Flat clustering）
#      - 以距離最小化為目標
#      - 容易忽略商業結構與群內層次
##    ■ Bisecting K-Means：
#      - 階層式二分法（Hierarchical splitting）
#      - 逐步拆解最異質群體
#      - 更容易形成由粗到細的結構
#  X軸 : 年消費金額（Spending）& #  Y軸 : 訪問頻率（Frequency） 
#  作者 : 國立雲林科技大學 電機系 林家仁
# ===========================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, BisectingKMeans


# ============================================================
# 建立商業意義資料
# ============================================================
# X: 年消費金額（萬元）
# Y: 購買頻率（次/年）

X, _ = make_blobs(
    n_samples=3000,
    # 模擬 3 種顧客族群
    centers=[
        [2, 3],   # 低消費 + 低頻率（潛在流失）
        [6, 8],   # 中高消費 + 高頻率（活躍顧客）
        [9, 3],   # 高消費 + 低頻率（偶爾大戶）
    ],

    cluster_std=0.8,
    random_state=42
)

# 拆解 X/Y 
spending = X[:, 0]   # 年消費金額
frequency = X[:, 1]  # 購買頻率

# ============================================================
# 視覺化原始資料
# ============================================================

plt.figure(figsize=(6, 5))
plt.scatter(spending, frequency, s=8)
plt.xlabel("Annual Spending (k NTD)")
plt.ylabel("Purchase Frequency (TIMES/YEARS)")
plt.title("Customer Data (Before Clustering)")
plt.show()


# ============================================================
# 設定分群目標
# ============================================================

# 假設公司要分成 4 群顧客
n_clusters = 4
random_state = 42


# ============================================================
#  K-Means（一次性分群）
# ============================================================

kmeans = KMeans(
    n_clusters=n_clusters,
    n_init=10,
    random_state=random_state
)

kmeans_labels = kmeans.fit_predict(X)
kmeans_centers = kmeans.cluster_centers_


# ============================================================
# Bisecting K-Means（逐步拆群）
# ============================================================

bisect = BisectingKMeans(
    n_clusters=n_clusters,
    n_init=10,
    random_state=random_state
)

bisect_labels = bisect.fit_predict(X)
bisect_centers = bisect.cluster_centers_


# ============================================================
# 視覺化比較
# ============================================================

fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# ----------------------------
# K-Means
# ----------------------------
axs[0].scatter(spending, frequency, c=kmeans_labels, s=8)
axs[0].scatter(
    kmeans_centers[:, 0],
    kmeans_centers[:, 1],
    c="red", s=80, marker="x"
)

axs[0].set_title("K-Means Clustering")
axs[0].set_xlabel("Annual Spending (K NTD)")
axs[0].set_ylabel("Purchase Frequency")


# ----------------------------
# Bisecting K-Means
# ----------------------------
axs[1].scatter(spending, frequency, c=bisect_labels, s=8)
axs[1].scatter(
    bisect_centers[:, 0],
    bisect_centers[:, 1],
    c="red", s=80, marker="x"
)

axs[1].set_title("Bisecting K-Means Clustering")
axs[1].set_xlabel("Annual Spending (K NTD)") 
axs[1].set_ylabel("Purchase Frequency")

plt.tight_layout()
plt.show()
