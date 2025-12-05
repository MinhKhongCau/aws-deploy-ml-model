import cv2
import urllib.request
import numpy as np
import until

# URL ESP32-CAM - thay bằng địa chỉ IP của bạn
url = "http://192.168.1.21/capture"

yolo_model = until.load_yolo_model()
svm_model = until.load_svm_model()
resnet50_fe_model = until.load_resnet50_fe_model()

while True:
    try:
        print("Lấy ảnh từ ESP32-CAM...")
        # Lấy dữ liệu hình ảnh từ URL
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        print("Ảnh đã được lấy thành công.")

        # try :
        #     label, (x1, y1, x2, y2) = until.crop_face_and_predict(img, yolo_model, svm_model, resnet50_fe_model)
        #     if label is None:            
        #         print("No face detected.")
        #         print("Predicted Label:", label)

        #     # Vẽ bbox và nhãn
        #     if label == "with_mask" or label == 0:
        #         color = (0, 255, 0)
        #         text = "With Mask"
        #     elif label == "without_mask" or label == 1:
        #         color = (0, 0, 255)
        #         text = "Without mask"
        #     else:
        #         color = (0, 255, 255)
        #         text = "Incorrect Mask"

        #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        #     cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # except Exception as e:
        #     print("Lỗi trong quá trình nhận diện khuôn mặt và dự đoán:", e)
        # Hiển thị hình ảnh
        cv2.imshow("ESP32-CAM Stream", img)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Lỗi:", e)
        break

cv2.destroyAllWindows()
