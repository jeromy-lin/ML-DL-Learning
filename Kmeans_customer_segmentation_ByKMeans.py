# ===========================================================
#  主題 : K_Means分群練習
#  目標 : 使學員了解機器學習進行分群的方法與概念
#  模擬顧客資料以使用 K-Means 找出潛在分群
#  X軸: 訪問頻率 vs Y軸 : 單筆消費金額
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#  模擬顧客資料（300人，分成3群）
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

#  使用 K-Means 找出潛在分群
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)

#  X軸: 訪問頻率 vs Y軸 : 單筆消費金額
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Set1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='black', marker='x', s=100, label='Centroids')
plt.title("Customer Segmentation by K-Means")
plt.xlabel("Monthly Visit Frequency")
plt.ylabel("Average Purchase Amount")
plt.legend()
plt.grid(True)
plt.show()
