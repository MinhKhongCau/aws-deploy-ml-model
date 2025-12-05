import os
import cv2

PREPROCESSED_DIR = "processed_dataset"

CLASSES = ["with_mask", "without_mask", "incorrect_mask"]

def verify_dataset():
    """Kiểm tra tính toàn vẹn của tập dữ liệu đã tiền xử lý."""
    for subdir in CLASSES:
        for dataset_type in ["train", "test", "val"]:
            dir_path = os.path.join(PREPROCESSED_DIR, dataset_type, subdir)
            if not os.path.exists(dir_path):
                print(f"[❌] Thiếu thư mục: {dir_path}")
                continue

            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(image_files) == 0:
                print(f"[❌] Không có ảnh trong thư mục: {dir_path}")
                continue

            for img_file in image_files:
                img_path = os.path.join(dir_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[❌] Ảnh hỏng hoặc không thể đọc: {img_path}")
                else:
                    print(f"[✅] Ảnh hợp lệ: {img_path}")

def count_images():
    """Đếm số lượng ảnh trong mỗi thư mục con."""
    for subdir in CLASSES:
        for dataset_type in ["train", "test", "val"]:
            dir_path = os.path.join(PREPROCESSED_DIR, dataset_type, subdir)
            if not os.path.exists(dir_path):
                print(f"[❌] Thiếu thư mục: {dir_path}")
                continue

            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"[ℹ️] {dataset_type}/{subdir}: {len(image_files)} ảnh")

if __name__ == "__main__":
    print("[ℹ️] Bắt đầu kiểm tra tập dữ liệu...")
    verify_dataset()
    print("\n[ℹ️] Đếm số lượng ảnh trong mỗi thư mục...")
    count_images()
    print("\n[ℹ️] Kiểm tra hoàn tất.")