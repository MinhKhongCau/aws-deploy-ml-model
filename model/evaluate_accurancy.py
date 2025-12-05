import cv2
import os
from until import predict_batch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

X_test = []  # List of test images
y_test = []  # List of true labels for test images

for root, dir, _ in os.walk("processed_dataset/test"):
    print(root, dir)
    for d in dir:
        label = d  # "with_mask", "without_mask", "incorrect_mask"
        dir_path = os.path.join(root, d)
        for filename in os.listdir(dir_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print(f"[❌] Không thể đọc ảnh: {img_path}")
                continue
            X_test.append(img)
            y_test.append(label)

print(f"[ℹ️] Đã tải {len(X_test)} ảnh test.")
# Example usage:
preds = predict_batch(X_test, y_test) 


# Các class bạn đang dùng
class_names = ["with_mask", "without_mask", "incorrect_mask"]

# Tạo confusion matrix
cm = confusion_matrix(y_test, preds, labels=class_names)

# Vẽ confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Face Mask Classification")
plt.show()

# Báo cáo chi tiết precision, recall, f1
print("\n🔹 Classification Report:")
print(classification_report(y_test, preds, target_names=class_names))