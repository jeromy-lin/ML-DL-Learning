# ===========================================================
#  主題 : CNN瑕疵檢測
#  目標 : 使學員了解深度學習進行瑕疵檢測的方法與概念
#  作者 : 國立雲林科技大學電機系 林家仁
# ===========================================================
# Colab ：Runtime ▸ Change runtime type ▸ GPU → Save
# ============================================================

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. 載入 Beans 資料集（第一次執行時會自動下載到 Colab 快取）
# ------------------------------------------------------------
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    "beans",
    split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True)

print("Raw classes :", ds_info.features["label"].names)

# ------------------------------------------------------------
# 2. 標籤轉換與影像預處理
#     0,1 → defect  (angular_leaf_spot, bean_rust)
#     2   → ok        (healthy)
# ------------------------------------------------------------
def map_labels(image, label):
    defect = tf.cast(label < 2, tf.float32)   # 0/1 → defect
    ok     = tf.cast(label == 2, tf.float32)  # 2   → ok
    return image, tf.where(ok == 1, 0., 1.)   # 最終標籤：0=ok, 1=defect

IMG_SIZE = 128
BATCH    = 32
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # 固定 resize 128x128
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def make_ds(ds, training=False):
    ds = ds.map(map_labels, num_parallel_calls=AUTOTUNE)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(1000).repeat()
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_ds(ds_train, training=True)
val_ds   = make_ds(ds_val)
test_ds  = make_ds(ds_test)

# ------------------------------------------------------------
# 3. 顯示部分影像
# ------------------------------------------------------------
plt.figure(figsize=(10, 3))
for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        lbl = "Defect" if labels[i] else "OK"
        plt.title(lbl, fontsize=9, color="red" if lbl == "Defect" else "green")
        plt.axis("off")
plt.suptitle("Beans - OK / Defect Example", fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. 建立簡易 CNN 模型
# ------------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")          # 二分類
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# ------------------------------------------------------------
# 5. 訓練模型
# ------------------------------------------------------------
EPOCHS = 10
train_count     = int(ds_info.splits["train"].num_examples * 0.70)  # 70% 訓練樣本數
steps_per_epoch = train_count // BATCH                              # 取整數

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds
)

# ------------------------------------------------------------
# 6. 評估 + 學習曲線
# ------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f" Test accuracy: {test_acc:.3f}")

plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Curve (Beans OK vs Defect)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 7. 預測示範（前 12 張測試影像）
# ------------------------------------------------------------
for images, labels in test_ds.take(1):
    preds = (model.predict(images[:12]) > 0.5).astype(int).flatten()
    plt.figure(figsize=(12, 3))
    for i in range(12):
        ax = plt.subplot(2, 6, i + 1)
        plt.imshow(images[i])
        true_lbl = "Defect" if labels[i] else "OK"
        pred_lbl = "Defect" if preds[i] else "OK"
        color = "green" if true_lbl == pred_lbl else "red"
        plt.title(f"T:{true_lbl}\nP:{pred_lbl}", color=color, fontsize=8)
        plt.axis("off")
    plt.suptitle("Prediction vs Ground Truth (Beans)", fontsize=14)
    plt.tight_layout()
    plt.show()

