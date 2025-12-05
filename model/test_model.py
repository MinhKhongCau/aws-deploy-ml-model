import numpy as np
import until
import cv2

# Test với ảnh mẫu
mask_img = cv2.imread("test_mask.jpg")
mask_img = cv2.resize(mask_img, (224, 224))
mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
no_mask_img = cv2.imread("test_no_mask.jpg")
no_mask_img = cv2.resize(no_mask_img, (224, 224))
no_mask_img = cv2.cvtColor(no_mask_img, cv2.COLOR_BGR2RGB)
incorrect_mask_img = cv2.imread("test_incorrect_mask.jpg")
incorrect_mask_img = cv2.resize(incorrect_mask_img, (224, 224))
incorrect_mask_img = cv2.cvtColor(incorrect_mask_img, cv2.COLOR_BGR2RGB)

print("Testing sample images...", mask_img.shape, no_mask_img.shape, incorrect_mask_img.shape)

# Example usage
classes = ["with_mask", "without_mask", "incorrect_mask"]
mask_label = until.predict_image(mask_img)
print("Predicted mask Label:", mask_label)
no_mask_label = until.predict_image(no_mask_img)
print("Predicted no mask Label:", no_mask_label)
incorrect_mask_label = until.predict_image(incorrect_mask_img)
print("Predicted incorrect mask Label:", incorrect_mask_label)


