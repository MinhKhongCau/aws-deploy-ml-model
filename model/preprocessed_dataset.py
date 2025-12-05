import os
import cv2
import random

DATASET_DIR = "clear_dataset"
PREPROCESSED_DIR = "processed_dataset"

CLASSES = ["with_mask", "without_mask", "incorrect_mask"]

os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def get_image_paths(subdir):
    """Lấy tất cả đường dẫn ảnh trong thư mục con."""
    dir_path = os.path.join(DATASET_DIR, subdir)
    if not os.path.exists(dir_path):
        print(f"[⚠️] Thư mục không tồn tại: {dir_path}")
        return []
    return [
        os.path.join(dir_path, f) 
        for f in os.listdir(dir_path) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def save_preprocessed_image(image, dataset_type, subdir, filename):
    """Lưu ảnh đã tiền xử lý vào thư mục tương ứng."""
    save_dir = os.path.join(PREPROCESSED_DIR, dataset_type, subdir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    preprocessed_img = preprocess_image(image)
    cv2.imwrite(save_path, preprocessed_img)
    print(f"[💾] Đã lưu ảnh vào {dataset_type}/{subdir} <- {filename}")

def preprocess_image(image):
    """Tiền xử lý ảnh (ví dụ: chuẩn hóa, thay đổi kích thước)."""
    # Ví dụ: Chuyển đổi sang kích thước 224x224 và chuẩn hóa
    return cv2.resize(image, (224, 224))

def split_dataset_80_10_10(subdir):
    """Chia tập dữ liệu thành 80% train, 10% test, 10% val."""
    image_paths = get_image_paths(subdir)
    total_images = len(image_paths)
    if total_images == 0:
        print(f"[⚠️] Không có ảnh để chia trong: {subdir}")
        return [], [], []
    

    random.shuffle(image_paths)
    train_end = int(0.8 * total_images)
    test_end = int(0.9 * total_images)

    train_paths = image_paths[:train_end]
    test_paths = image_paths[train_end:test_end]
    val_paths = image_paths[test_end:]

    return train_paths, test_paths, val_paths

for subdir in CLASSES:
    print(f"[ℹ️] Xử lý thư mục: {subdir}")
    train_paths, test_paths, val_paths = split_dataset_80_10_10(subdir)
    for dataset_type, paths in zip(["train", "test", "val"], [train_paths, test_paths, val_paths]):
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[❌] Không thể đọc ảnh: {img_path}")
                continue
            filename = os.path.basename(img_path)
            print(f"[ℹ️] Xử lý ảnh: {filename} vào tập {dataset_type}")
            save_preprocessed_image(img, dataset_type=dataset_type, subdir=subdir, filename=filename)

    print(f"✅ {subdir}: {len(train_paths)} train, {len(test_paths)} test, {len(val_paths)} val\n")
