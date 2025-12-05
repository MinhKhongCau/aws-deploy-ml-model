import cv2
import os
import time
import torch
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import until
import numpy as np

# Tạo thư mục lưu dữ liệu
class_names = ['with_mask', 'without_mask', 'incorrect_mask']
svm_model = until.load_svm_model()
resnet50_fe_model = until.load_resnet50_fe_model()
yolo_model = until.load_yolo_model()

# Dataset
labels = []
images = []

# Cấu hình webcam và tham số
cap = cv2.VideoCapture(0)
frames_per_label = 30
capture_time = 5  # giây
wait_time = 5     # giây giữa các nhãn

for idx, label in enumerate(class_names):
    print(f"Chuẩn bị lấy ảnh cho nhãn: {label}. Đợi {wait_time} giây...")
    time.sleep(wait_time)
    print(f"Bắt đầu lấy ảnh cho nhãn: {label}")
    count = 0
    start = time.time()
    while count < frames_per_label and (time.time() - start) < capture_time:
        ret, frame = cap.read()
        if not ret:
            continue
        face, bbox = until.crop_face(frame, yolo_model)
        if face is None:
            continue
        x1, y1, x2, y2 = bbox
        face_resized = cv2.resize(face, (224, 224))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_resized = np.array(face_resized)
        labels.append(label)
        images.append(face_resized)
        count += 1
        cv2.imshow('Frame', face_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f"Đã lấy xong {count} ảnh cho nhãn: {label}")

cap.release()
cv2.destroyAllWindows()

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Số ảnh train: {len(X_train)}, Số ảnh val: {len(X_val)}")
print(f"Phân bố nhãn train: {np.unique(y_train, return_counts=True)}")
print(f"Phân bố nhãn val: {np.unique(y_val, return_counts=True)}")
print(f"Ảnh train: {X_train[0].shape}, Ảnh val: {X_val[0].shape}")

for i in range(len(X_train)):
    print(f"Ảnh train {i}: {y_train[i]}")
    cv2.imshow(f"{y_train[i]}", X_train[i])
    cv2.waitKey(500)  # hiển thị 0.5s mỗi ảnh

for i in range(len(X_val)):
    print(f"Ảnh val {i} label {y_val[i]}: {y_val[i]}")
    cv2.imshow(f"{y_val[i]}", X_val[i])
    cv2.waitKey(100)  # hiển thị 0.3s mỗi ảnh

print("Bắt đầu fine-tune ResNet50...")
print("Press any key to continue...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Data augmentation ---
print("Data augmentation...")
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# Fine-tune 
print("Fine-tuning ResNet50...")
finetune_model, history = until.finetune_model(
    X_train=X_train,
    y_train=y_train,
    X_test=X_val,
    y_test=y_val)

