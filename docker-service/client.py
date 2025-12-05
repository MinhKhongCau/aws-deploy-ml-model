import cv2
import requests  # <-- THAY THẾ boto3
import numpy as np
import json
import time

# --- CẤU HÌNH BẮT BUỘC CHO SERVER LOCAL ---
# 1. Thay thế bằng URL của server Flask của bạn
LOCAL_API_URL = "http://54.206.102.233:5000/predict" 

# --- KHỞI TẠO CAMERA ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không thể truy cập Camera. Vui lòng kiểm tra lại thiết bị hoặc quyền truy cập.")
    exit()

print(f"--- Đang kết nối tới Server Local: {LOCAL_API_URL} ---")
print("Nhấn 'S' để chụp ảnh và gửi đi, Nhấn 'Q' để thoát.")

# --- VÒNG LẶP CHỤP VÀ GỬI ẢNH ---
while True:
    ret, frame = cap.read()

    if not ret:
        print("Lỗi: Không thể nhận khung hình (stream end?).")
        break

    display_frame = frame.copy() 
    key = cv2.waitKey(1) & 0xFF
    
    # Nhấn 'S' (Save & Send)
    if key == ord('s'):
        print("\n--- [INFO] Đang chụp ảnh và gửi đến Server Local...")
        
        # Mã hóa khung hình thành bytes (dạng JPEG)
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            print("Lỗi: Không thể mã hóa ảnh.")
            continue
            
        image_bytes = buffer.tobytes()
        
        try:
            start_time = time.time()
            
            # --- [THAY ĐỔI] Gửi yêu cầu dự đoán bằng 'requests' ---
            # Server của bạn mong đợi 'multipart/form-data' với trường tên 'image'
            files_to_send = {
                'image': ('capture.jpg', image_bytes, 'image/jpeg')
            }
            
            response = requests.post(
                LOCAL_API_URL,
                files=files_to_send,
                timeout=10 # Thêm timeout 10 giây
            )
            # --- [KẾT THÚC THAY ĐỔI] ---

            end_time = time.time()
            latency = end_time - start_time
            
            # 3. Phân tích phản hồi
            if response.status_code == 200:
                # Đọc body phản hồi và giải mã (thư viện requests làm tự động)
                result = response.json() 
                predictions = result.get('predictions', [])

                if predictions:
                    # Duyệt qua các dự đoán (nếu có nhiều khuôn mặt)
                    # Server trả về: [{"label": "...", "bbox": [x1, y1, x2, y2]}, ...]
                    for pred in predictions:
                        label = pred.get('label', 'unknown')
                        bbox = pred.get('bbox', [0, 0, 10, 10]) 
                        x1, y1, x2, y2 = map(int, bbox) # Chuyển bbox thành int

                        # Vẽ khung và nhãn lên khung hình hiển thị
                        color = (0, 0, 255) if label == 'without_mask' or label == 'incorrect_mask' else (0, 255, 0)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    print(f"--- [SUCCESS] Dự đoán: {predictions} | Thời gian: {latency:.2f}s")
                    
                else:
                    print("--- [INFO] Không tìm thấy khuôn mặt nào trong ảnh.")
                    
            else:
                print(f"--- [ERROR] Lỗi API: Mã {response.status_code}, Phản hồi: {response.text}")

        # Bắt lỗi cụ thể nếu không thể kết nối đến server
        except requests.exceptions.ConnectionError:
            print(f"--- [ERROR] Lỗi kết nối: Không thể kết nối tới {LOCAL_API_URL}.")
            print("---       Vui lòng đảm bảo server của bạn đang chạy.")
        except Exception as e:
            print(f"--- [ERROR] Lỗi khi gọi Endpoint: {e}")
        
    # Nhấn 'Q' (Quit)
    elif key == ord('q'):
        break
    
    # Hiển thị khung hình (đã cập nhật sau dự đoán)
    cv2.imshow('Face Mask Detector - Client (Local Server)', display_frame)

# --- DỌN DẸP ---
cap.release()
cv2.destroyAllWindows()