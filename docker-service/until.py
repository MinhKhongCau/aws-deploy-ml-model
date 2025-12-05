import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.svm import SVC


MODEL_SVM_PATH = "model/svm_model.pkl"
MODEL_RESNET50_FE_PATH = "model/resnet50_feature_extractor.h5"
MODEL_YOLO_PATH = "model/yolov8n-face.pt"

#
# Functions to load pre-trained models
#
def load_yolo_model():
    return YOLO(MODEL_YOLO_PATH) 

def load_svm_model():
    return joblib.load(MODEL_SVM_PATH)

def load_resnet50_fe_model():
    return load_model(MODEL_RESNET50_FE_PATH)

def load_finetuned_svm_model():
    return joblib.load("model/svm_ft_model.pkl")

#
# Function for training the model
#
def finetune_model(X_train, y_train, X_test, y_test):
    # Load pre-trained ResNet50 feature extractor
    resnet_model = load_resnet50_fe_model()

    # Extract features
    train_features = resnet_model.predict(np.array(X_train))
    test_features = resnet_model.predict(np.array(X_test))

    # Train SVM
    svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=42)
    svm.fit(train_features, y_train.ravel())

    # Save model
    joblib.dump(svm, "model/svm_ft_model.pkl")
    print("---> Đã lưu model SVM vào model/svm_ft_model.pkl")

    # Evaluate
    acc = svm.score(test_features, y_test)
    print("---> Độ chính xác SVM:", acc)

    return svm, resnet_model

# Predict 1 image function
def predict_image(img=None, model_svm=None, model_resnet50_fe=None):
    if img is None:
        return None

    # Load models
    resnet50_fe_model = model_resnet50_fe or load_resnet50_fe_model()
    svm = model_svm or load_svm_model()

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    
    # Extract features
    features = resnet50_fe_model.predict(img)
    
    # Predicts SVM
    pred = svm.predict(features)
    # {"with_mask", "without_mask", "incorrect_mask"}
    print(f"Predicted Label: {pred}")
    return pred[0]

def crop_face(img, model_yolo=None):
    yolo = model_yolo or load_yolo_model()
    # Detect faces
    results = yolo.predict(img, conf=0.5, verbose=False)  # conf=threshold
    if len(results[0].boxes) == 0:
        print("[⚠️] Không phát hiện khuôn mặt.")
        return None
    # for r in results[0].boxes:  # results[0].boxes: list of Box 
    #     x1, y1, x2, y2 = map(int, r.xyxy[0])  # bounding box
    #     face = img[y1:y2, x1:x2]
    #     return face, (x1, y1, x2, y2)
    return results[0].boxes

def crop_face_and_predict(img, model_yolo=None, svm_model=None, resnet50_fe_model=None):
    yolo = model_yolo or load_yolo_model()
    if yolo is None:
        print("Error: YOLO model is not loaded.")
        return None, None
    # Detect faces
    bboxes = crop_face(img, yolo)
    if bboxes is None:
        return None
    labels = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        face = img[y1:y2, x1:x2]
        # Dự đoán nhãn mask bằng SVM
        label = predict_image(face, svm_model, resnet50_fe_model)
        labels.append((label,(x1, y1, x2, y2)))
    return labels

def predict_batch(X_test, y_test):
    preds = []

    print(f"[🔍] Đang dự đoán {len(X_test)} ảnh test...")
    for i, img in enumerate(X_test):
        try:
            pred = predict_image(img)
            preds.append(pred)
        except Exception as e:
            print(f"[⚠️] Lỗi tại ảnh {i}: {e}")
            preds.append("unknown")

    # Tính accuracy
    preds = np.array(preds)
    y_test = np.array(y_test)

    # Lọc ảnh hợp lệ (không bị 'unknown')
    valid_idx = preds != "unknown"
    acc = accuracy_score(y_test[valid_idx], preds[valid_idx])

    print(f"\n✅ Độ chính xác (Accuracy): {acc * 100:.2f}%")
    return preds