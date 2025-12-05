import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

# Import các hàm tiện ích và hàm tải model từ until.py
# Đảm bảo file until.py nằm cùng thư mục với server.py
import until 

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- Tải các model MỘT LẦN KHI SERVER KHỞI ĐỘNG ---
# Giảm thiểu thời gian chờ cho mỗi request
print("--- [INFO] Đang tải các mô hình...")
try:
    model_yolo = until.load_yolo_model()
    model_svm = until.load_finetuned_svm_model() # Hoặc load_finetuned_svm_model() nếu bạn muốn dùng model fine-tuned
    model_resnet50_fe = until.load_resnet50_fe_model()
    print("--- [INFO] Đã tải thành công 3 mô hình (YOLO, ResNet, SVM).")
except Exception as e:
    print(f"--- [ERROR] Lỗi nghiêm trọng khi tải mô hình: {e}")
    # Có thể thêm xử lý thoát hoặc báo lỗi ở đây nếu cần
    model_yolo = None
    model_svm = None
    model_resnet50_fe = None

# --- Định nghĩa Endpoint cho việc dự đoán ---
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra xem model đã được tải chưa
    if not all([model_yolo, model_svm, model_resnet50_fe]):
        return jsonify({"error": "Mô hình chưa sẵn sàng, vui lòng thử lại sau."}), 503 # Service Unavailable

    # Kiểm tra xem có file 'image' được gửi lên không
    if 'image' not in request.files:
        return jsonify({"error": "Không tìm thấy file image trong request"}), 400 # Bad Request

    file = request.files['image']

    # Kiểm tra xem có file không và có tên file không
    if file.filename == '':
        return jsonify({"error": "Chưa chọn file nào"}), 400

    if file:
        try:
            # Đọc dữ liệu ảnh từ request
            img_bytes = file.read()
            
            # Chuyển bytes thành ảnh OpenCV (định dạng BGR mà hàm crop_face_and_predict có vẻ đang dùng)
            # 1. Đọc bytes thành mảng numpy
            nparr = np.frombuffer(img_bytes, np.uint8)
            # 2. Decode mảng numpy thành ảnh OpenCV
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_cv2 is None:
                 return jsonify({"error": "Không thể decode file ảnh"}), 400

            print("--- [INFO] Nhận được ảnh, bắt đầu dự đoán...")
            # Gọi hàm xử lý và dự đoán từ until.py
            # Truyền các model đã tải sẵn vào hàm
            predictions = until.crop_face_and_predict(
                img_cv2, 
                model_yolo=model_yolo, 
                svm_model=model_svm, 
                resnet50_fe_model=model_resnet50_fe
            )
            print(f"--- [INFO] Kết quả dự đoán: {predictions}")

            # Format lại kết quả một chút cho dễ đọc hơn dạng JSON
            results_list = []
            if predictions: # predictions là list of tuples (label, bbox)
                for label, bbox in predictions:
                     results_list.append({
                         "label": label,
                         "bbox": list(bbox) # Chuyển tuple bbox thành list
                     })

            # Trả về kết quả dạng JSON
            return jsonify({"predictions": results_list})

        except Exception as e:
            print(f"--- [ERROR] Lỗi trong quá trình xử lý ảnh hoặc dự đoán: {e}")
            return jsonify({"error": f"Lỗi server: {e}"}), 500 # Internal Server Error

    return jsonify({"error": "Lỗi không xác định"}), 500

# --- Chạy Server ---
if __name__ == '__main__':
    # Chạy server, lắng nghe trên tất cả các địa chỉ IP và port 5000
    # debug=True chỉ nên dùng khi phát triển, tắt khi deploy thực tế
    app.run(host='0.0.0.0', port=5000, debug=False)