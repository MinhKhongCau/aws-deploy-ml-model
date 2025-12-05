import cv2

# Đọc ảnh từ đường dẫn và trả về ảnh BGR và RGB
def read_image(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[❌] Không thể đọc ảnh: {image_path}")
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb 

# Ví dụ sử dụng
if __name__ == "__main__":
    bgr, rgb = read_image("test_mask.jpg")
    if bgr is not None and rgb is not None:
        print("Ảnh BGR shape:", bgr.shape)
        print("Ảnh RGB shape:", rgb.shape)