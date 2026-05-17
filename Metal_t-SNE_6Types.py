import os
import zipfile
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage.feature import local_binary_pattern
from google.colab import files

# ==========================================
# 1. 解壓大雜燴壓縮檔
# ==========================================
print("=== 步驟 1: 解壓混合瑕疵資料集 ===")
extract_dir = '/content/extracted_defects'
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir, exist_ok=True)

zip_files = glob.glob('/content/*.zip')
if len(zip_files) == 0:
    uploaded = files.upload()
    zip_files = [os.path.join('/content', f) for f in uploaded.keys()]

with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
all_image_paths = []
for ext in image_extensions:
    all_image_paths.extend(glob.glob(os.path.join(extract_dir, '**', ext), recursive=True))

image_paths = [p for p in all_image_paths if "__MACOSX" not in p and ".ipynb_checkpoints" not in p]

# ==========================================
# 依據 crazing_39 規則精準切出真實類別
# ==========================================
image_labels = []
for p in image_paths:
    filename = os.path.basename(p)
    label = filename.split('_')[0] # 遇到 crazing_39 會精準切出 crazing
    image_labels.append(label)

unique_labels = sorted(list(set(image_labels)))
print(f"✅ 成功載入 {len(image_paths)} 張影像。")
print(f"解析檔名後，成功識別出的瑕疵標籤為: {unique_labels}")

# ==========================================
# 2. 工業級特徵改造：提取 LBP 局部紋理特徵
# ==========================================
print("\n=== 步驟 2: 正在提取 LBP 工業紋理特徵指紋 (跳脫貓狗模型的盲點)... ===")

features_list = []
valid_labels = []

# LBP 參數設定
radius = 3
n_points = 24

for img_path, label in zip(image_paths, image_labels):
    try:
        # 轉成灰階影像，因為紋理只跟明暗有關
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = cv2.resize(img, (256, 256))
        
        # 計算 LBP 紋理圖
        lbp = local_binary_pattern(img_resized, n_points, radius, method='uniform')
        
        # 將紋理統計成直方圖作為這張圖的「特徵向量」
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        features_list.append(hist)
        valid_labels.append(label)
    except Exception:
        continue

features_array = np.array(features_list)
valid_labels = np.array(valid_labels)
print(f"紋理特徵提取完畢！特徵矩陣維度: {features_array.shape}")

# ==========================================
# 3. 執行 t-SNE 降維
# ==========================================
print("\n=== 步驟 3: 執行 t-SNE 降維演算法 ===")
tsne = TSNE(n_components=2, perplexity=10, random_state=42, n_iter=1000, init='random')
tsne_features = tsne.fit_transform(features_array)

# ==========================================
# 4. 繪製彩色分佈圖
# ==========================================
print("\n=== 步驟 4: 繪製 6 大瑕疵全新紋理分佈圖 ===")
plt.figure(figsize=(12, 7))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
markers = ['o', 's', '^', 'D', 'X', 'P', '*']

for idx, label in enumerate(unique_labels):
    indices = np.where(valid_labels == label)[0]
    points = tsne_features[indices]
    if len(points) > 0:
        plt.scatter(points[:, 0], points[:, 1], 
                    c=colors[idx % len(colors)], 
                    marker=markers[idx % len(markers)],
                    label=f'{label} (n={len(points)})', s=100, edgecolors='k', alpha=0.9)

plt.title('Texture-Based AI Diagnostic: Defect Categories in LBP Space', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=11)
plt.ylabel('t-SNE Dimension 2', fontsize=11)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="True Defect Types")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
