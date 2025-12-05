import os
import json
import io
import joblib
import numpy as np
import cv2
from PIL import Image

# Các thư viện mô hình
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from sklearn.svm import SVC # Import để type hinting, dù joblib không cần

# 
# 1. Được gọi KHI endpoint khởi động (load 1 lần)
#
def model_fn(model_dir):
    """
    Tải TẤT CẢ các mô hình cần thiết 1 LẦN DUY NHẤT.
    model_dir sẽ là /opt/ml/model/
    Các file model (ví dụ: svm_model.pkl, yolov8n-face.pt, resnet50_feature_extractor.h5)
    phải nằm trực tiếp bên trong file .tar.gz của bạn.
    """
    print("--- [INFO] Đang tải các mô hình...")
    
    # Đường dẫn tới các file model
    MODEL_SVM_PATH = os.path.join(model_dir, "svm_model.pkl")
    MODEL_RESNET50_FE_PATH = os.path.join(model_dir, "resnet50_feature_extractor.h5")
    MODEL_YOLO_PATH = os.path.join(model_dir, "yolov8n-face.pt")

    # Tải các mô hình
    try:
        yolo_model = YOLO(MODEL_YOLO_PATH)
        svm_model = joblib.load(MODEL_SVM_PATH)
        resnet_model = load_model(MODEL_RESNET50_FE_PATH)
        
        print("--- [INFO] Đã tải thành công 3 mô hình (YOLO, ResNet, SVM).")
        
        # Trả về một dict chứa tất cả mô hình
        models = {
            "yolo": yolo_model,
            "svm": svm_model,
            "resnet": resnet_model
        }
        return models
        
    except Exception as e:
        print(f"--- [ERROR] Lỗi khi tải mô hình: {e}")
        return None

#
# 2. Được gọi TRƯỚC khi dự đoán (tiền xử lý)
#
def input_fn(request_body, content_type):
    """
    Giải mã dữ liệu đầu vào. Client sẽ gửi ảnh (bytes).
    Chỉ giải mã, không chạy mô hình ở đây.
    """
    print(f"--- [INFO] Nhận request. Content-Type: {content_type}")
    if content_type == 'application/octet-stream':
        try:
            # Đọc ảnh từ bytes
            image_pil = Image.open(io.BytesIO(request_body))
            
            # Chuyển từ PIL (RGB) sang OpenCV (BGR) để xử lý
            image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            return image_cv2
        except Exception as e:
            print(f"--- [ERROR] Lỗi giải mã ảnh: {e}")
            raise ValueError(f"Không thể giải mã ảnh đầu vào: {e}")
            
    raise ValueError(f"Content-Type không được hỗ trợ: {content_type}")

#
# 3. Được gọi KHI dự đoán (chạy toàn bộ pipeline)
#
def predict_fn(input_object, models):
    """
    Chạy toàn bộ pipeline inference.
    - input_object: là ảnh image_cv2 từ input_fn.
    - models: là dict {"yolo": ..., "svm": ..., "resnet": ...} từ model_fn.
    """
    print("--- [INFO] Bắt đầu pipeline dự đoán...")
    
    if models is None:
        return {"error": "Mô hình không được tải."}
        
    if input_object is None:
        return {"error": "Dữ liệu ảnh đầu vào bị lỗi."}

    # Lấy các mô hình
    yolo_model = models["yolo"]
    svm_model = models["svm"]
    resnet_model = models["resnet"]
    
    # 1. Chạy YOLO để phát hiện khuôn mặt (Logic từ crop_face)
    # input_object là ảnh BGR
    results_yolo = yolo_model.predict(input_object, conf=0.5, verbose=False)
    
    bboxes = results_yolo[0].boxes
    if len(bboxes) == 0:
        print("--- [INFO] Không phát hiện khuôn mặt.")
        return [] # Trả về danh sách rỗng nếu không có mặt

    # Danh sách để lưu kết quả
    predictions = []
    
    print(f"--- [INFO] Phát hiện {len(bboxes)} khuôn mặt. Đang xử lý...")

    # 2. Vòng lặp qua từng khuôn mặt
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # bounding box
        
        # 3. Cắt khuôn mặt
        face = input_object[y1:y2, x1:x2]
        
        if face.size == 0:
            print(f"--- [WARN] Bỏ qua bbox không hợp lệ: {[x1, y1, x2, y2]}")
            continue

        # 4. Tiền xử lý khuôn mặt cho ResNet (Logic từ predict_image)
        try:
            face_resized = cv2.resize(face, (224, 224))
            # ResNet/TensorFlow mong đợi ảnh RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB) 
            face_expanded = np.expand_dims(face_rgb, axis=0) # (1, 224, 224, 3)
            
            # 5. Trích xuất đặc trưng
            features = resnet_model.predict(face_expanded)
            
            # 6. Dự đoán bằng SVM
            label = svm_model.predict(features)[0]
            
            # Thêm kết quả vào danh sách
            predictions.append({
                "label": label,
                "bbox": [x1, y1, x2, y2] # Gửi lại bbox để client biết vị trí
            })
        except Exception as e:
            print(f"--- [ERROR] Lỗi khi xử lý 1 khuôn mặt: {e}")
            predictions.append({
                "label": "error",
                "bbox": [x1, y1, x2, y2]
            })

    print("--- [INFO] Hoàn tất dự đoán.")
    return predictions

#
# 4. Được gọi SAU khi dự đoán (hậu xử lý)
#
def output_fn(prediction, accept):
    """
    Định dạng kết quả trả về (thường là JSON)
    """
    print(f"--- [INFO] Định dạng output sang {accept}")
    if accept == 'application/json':
        # 'prediction' là danh sách (list) các dict từ predict_fn
        results_json = json.dumps(prediction)
        return results_json, 'application/json'
        
    raise ValueError(f"Accept type không được hỗ trợ: {accept}")