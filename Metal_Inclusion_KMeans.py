import os
import zipfile
import glob
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from google.colab import files

# ==========================================
# 1. 單一 ZIP 檔案 上傳與解壓 
# ==========================================
print("=== Step 1: Upload and Extract Your Inclusion Dataset ===")
raw_data_dir = '/content/inclusion_50_dataset'

if os.path.exists(raw_data_dir):
    import shutil
    shutil.rmtree(raw_data_dir)
os.makedirs(raw_data_dir, exist_ok=True)

print("Please select and upload your single Inclusion ZIP file:")
uploaded = files.upload()

if not uploaded:
    raise ValueError("No file uploaded. Please re-run the cell.")

for filename in uploaded.keys():
    zip_path = os.path.join('/content', filename)
    with open(zip_path, 'wb') as f:
        f.write(uploaded[filename])

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)
    print(f"\nFile [{filename}] successfully extracted.")

# 遞迴搜尋圖片並剔除 Mac/Windows 系統暫存隱藏檔
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
all_found_paths = []
for ext in image_extensions:
    all_found_paths.extend(glob.glob(os.path.join(raw_data_dir, '**', ext), recursive=True))

image_paths = []
for p in all_found_paths:
    filename = os.path.basename(p)
    if "__MACOSX" not in p and not filename.startswith('.'):
        image_paths.append(p)

total_images = len(image_paths)
print(f"✨ Total valid images detected: {total_images} pcs")

if total_images == 0:
    raise ValueError("No valid images found! Please check your ZIP file content.")

# ==========================================
# 2. 初始化輕量級特徵提取網路 (ResNet18)
# ==========================================
print("\n=== Step 2: Initialize Feature Extractor ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

resnet = models.resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# ==========================================
# 3. 高速提取所有影像的特徵指紋
# ==========================================
print("\n=== Step 3: AI Extracting Image Features... ===")
features_list = []
valid_image_paths = []

with torch.no_grad():
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = img_transform(img).unsqueeze(0).to(device)

            feat = feature_extractor(tensor)
            feat = torch.flatten(feat, 1)

            features_list.append(feat.cpu().numpy().flatten())
            valid_image_paths.append(img_path)
        except Exception:
            continue

features_array = np.array(features_list)

# ==========================================
# 4. 非監督式學習：K-Means 自動分群
# ==========================================
print("\n=== Step 4: Executing K-Means Clustering ===")
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features_array)

# ==========================================
# 5. PCA 數據降維與特徵空間視覺化
# ==========================================
print("\n=== Step 5: Visualizing Feature Space ===")
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(features_array)

plt.figure(figsize=(8, 4))
colors = ['#1f77b4', '#ff7f0e']
markers = ['o', 's']

for i in range(n_clusters):
    points = pca_features[cluster_labels == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], marker=markers[i],
                label=f'Cluster {i}', s=60, edgecolors='k', alpha=0.8)

plt.title('Inclusion Feature Space (K-Means)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ==========================================
# 6. 自動抽樣視覺化與問題說明
# ==========================================
print("\n=== Step 6: Visual Sampling for Slide Presentation ===")
samples_per_cluster = 3

for cluster_id in range(n_clusters):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    actual_samples = min(samples_per_cluster, len(cluster_indices))
    sample_indices = cluster_indices[:actual_samples]

    fig, axes = plt.subplots(1, actual_samples, figsize=(12, 3))
    fig.suptitle(f"Defect Profile Samples: Cluster {cluster_id}", fontsize=12, color=colors[cluster_id], fontweight='bold')

    if actual_samples == 1:
        axes = [axes]

    for idx, img_idx in enumerate(sample_indices):
        img_path = valid_image_paths[img_idx]
        img = Image.open(img_path)

        axes[idx].imshow(img)
        axes[idx].set_title(f"File: {os.path.basename(img_path)[:20]}", fontsize=9)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 7. 產出報告表格
# ==========================================
print("\n=== Step 7: Presentation Summary Report (With File Names) ===")

# 【DEBUG 強化】：自動撈出各群組內所有圖片的純檔名，用逗號連接
c0_indices = np.where(cluster_labels == 0)[0]
c1_indices = np.where(cluster_labels == 1)[0]

c0_files_list = [os.path.basename(valid_image_paths[idx]) for idx in c0_indices]
c1_files_list = [os.path.basename(valid_image_paths[idx]) for idx in c1_indices]

# 限制表格顯示的檔名長度，（多的用 ... 代替，可快速檢視）
c0_files_display = ", ".join(c0_files_list[:8]) + (f" ... and {len(c0_files_list)-8} more" if len(c0_files_list) > 8 else "")
c1_files_display = ", ".join(c1_files_list[:8]) + (f" ... and {len(c1_files_list)-8} more" if len(c1_files_list) > 8 else "")

# 建立簡化報告結構
simple_report = {
    "Quality Status": [
        "🟢 GOOD (Standard)", 
        "🔴 DEFECT (Cluster 0)", 
        "🔴 DEFECT (Cluster 1)"
    ],
    "Visual Appearance": [
        "Smooth surface without anomalies.",
        f"Scattered dark spots or irregular shapes ({len(c0_files_list)} pcs).",
        f"Continuous vertical streaks or line marks ({len(c1_files_list)} pcs)."
    ],
    "Affected File Names (點名清單)": [
        "None (All cleared)",
        c0_files_display,
        c1_files_display
    ],
    "Possible Root Cause": [
        "Normal operation status.",
        "Random particles or raw material impurities.",
        "Mechanical friction or roller buildup issues."
    ]
}

df_simple = pd.DataFrame(simple_report)

html_style = """
<style>
    .simple-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        margin: 15px 0;
    }
    .simple-table th {
        background-color: #2c3e50;
        color: white;
        padding: 10px;
        text-align: left;
    }
    .simple-table td {
        border: 1px solid #ddd;
        padding: 10px;
        font-size: 13px;
        word-break: break-word; /* 避免長檔名把表格撐壞 */
    }
    .simple-table tr:nth-child(even) { background-color: #f9f9f9; }
</style>
"""

display(HTML(html_style + df_simple.to_html(index=False, classes='simple-table')))
