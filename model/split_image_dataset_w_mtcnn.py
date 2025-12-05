from mtcnn import MTCNN
import cv2
import os

# --------- CONFIG ---------
INPUT_DIR = "dataset"           # Thư mục chứa ảnh gốc
OUTPUT_DIR = "clear_dataset"    # Thư mục lưu ảnh khuôn mặt
FACE_SIZE = (320, 320)          # Kích thước khuôn mặt sau khi cắt

# Các thư mục con trong dataset
MASK_PATH = os.path.join(INPUT_DIR, "with_mask")
WITHOUT_MASK_PATH = os.path.join(INPUT_DIR, "without_mask")
INCORRECT_MASK_PATH = os.path.join(INPUT_DIR, "incorrect_mask")

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Khởi tạo MTCNN detector
detector = MTCNN()

def preprocess_image(image_path):
    """Đọc ảnh bằng cv2 và chuyển sang RGB để detect."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[❌] Không thể đọc ảnh: {image_path}")
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

def detect_faces(image_rgb):
    """Phát hiện khuôn mặt trong ảnh RGB."""
    return detector.detect_faces(image_rgb)

def save_cropped_face(image_bgr, box, save_path):
    """Cắt, resize và lưu khuôn mặt bằng cv2."""
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    cropped = image_bgr[y:y+h, x:x+w]
    if cropped.size == 0:
        print(f"[⚠️] Khuôn mặt rỗng, bỏ qua: {save_path}")
        return
    face_resized = cv2.resize(cropped, FACE_SIZE)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, face_resized)
    print(f"[💾] Đã lưu: {save_path}")

def save_faces_from_directory(input_dir, output_subdir):
    """Xử lý tất cả ảnh trong thư mục input_dir."""
    if not os.path.exists(input_dir):
        print(f"[⚠️] Không tìm thấy thư mục: {input_dir}")
        return

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(input_dir, filename)
        img_bgr, img_rgb = preprocess_image(image_path)
        if img_bgr is None:
            continue

        faces = detect_faces(img_rgb)
        if len(faces) == 0:
            print(f"[⚠️] Không tìm thấy khuôn mặt trong {filename}")
            continue

        for i, face in enumerate(faces):
            box = face['box']
            save_path = os.path.join(
                OUTPUT_DIR, output_subdir, f"{os.path.splitext(filename)[0]}_face_{i+1}.jpg"
            )
            save_cropped_face(img_bgr, box, save_path)

# ------------ RUN ------------
save_faces_from_directory(MASK_PATH, "with_mask")
save_faces_from_directory(WITHOUT_MASK_PATH, "without_mask")
save_faces_from_directory(INCORRECT_MASK_PATH, "incorrect_mask")
