import cv2
from ultralytics import YOLO
import until  

svm_model = until.load_finetuned_svm_model()
resnet50_fe_model = until.load_resnet50_fe_model()
yolo_model = until.load_yolo_model()

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    label, (x1, y1, x2, y2) = until.crop_face_and_predict(frame, yolo_model, svm_model, resnet50_fe_model)
    if label is None:
        continue

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
    cv2.putText(frame, text, (x1, y1-10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Hiển thị webcam
    cv2.imshow("Face Mask Detection YOLO + SVM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
