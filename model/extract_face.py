from ultralytics import YOLO
import cv2
import until

yolo_model = until.load_yolo_model()
svm_model = until.load_finetuned_svm_model()
resnet50_fe_model = until.load_resnet50_fe_model()

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

label, (x1, y1, x2, y2) = until.crop_face_and_predict(frame, yolo_model, svm_model, resnet50_fe_model)
if label is None:
    print("No face detected.")
    exit()
print("Predicted Label:", label)

# Vẽ bbox và nhãn
if label == "with_mask" or label == 0:
    color = (0, 255, 0)
    text = "With Mask"
elif label == "without_mask" or label == 1:
    color = (0, 0, 255)
    text = "Without mask"
else:
    color = (0, 255, 255)
    text = "Incorrect Mask"

cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Hiển thị khuôn mặt được cắt
cv2.imshow("Detected Face", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
