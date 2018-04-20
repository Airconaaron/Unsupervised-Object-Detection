import numpy as np
import cv2

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

# from keras.applications.mobilenet import MobileNet
# from keras.applications.mobilenet import preprocess_input, decode_predictions

from keras.preprocessing import image
from keras import backend as K
K.set_image_dim_ordering('tf')

classifier = ResNet50(weights='imagenet')
# classifier = MobileNet(weights='imagenet')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def calculate_iou(img_mask, gt_mask, img):
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    preds = classifier.predict(x)
    return decode_predictions(preds, top=1)[0][0][2]
    
def calculate_real_iou(img_mask, gt_mask):
        gt_mask *= 1.0
        img_and = cv2.bitwise_and(img_mask, gt_mask)
        img_or = cv2.bitwise_or(img_mask, gt_mask)
        j = np.count_nonzero(img_and)
        i = np.count_nonzero(img_or)
        iou = float(float(j)/float(i))
        return iou
        


def calculate_overlapping(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap = float(float(j)/float(i))
    return overlap


def follow_iou(gt_masks, mask, array_classes_gt_objects, object_id, last_matrix, available_objects, img=None):
    results = np.zeros([np.size(array_classes_gt_objects), 1])
    for k in range(np.size(array_classes_gt_objects)):
        if array_classes_gt_objects[k] == object_id:
            if available_objects[k] == 1:
                gt_mask = gt_masks[:, :, k]
                iou = calculate_iou(mask, gt_mask, img)
                results[k] = iou
            else:
                results[k] = -1
    max_result = max(results)
    ind = np.argmax(results)
    iou = last_matrix[ind]
    new_iou = max_result
    return iou, new_iou, results, ind

def follow_real_iou(gt_masks, mask, array_classes_gt_objects, object_id, last_matrix, available_objects):
    results = np.zeros([np.size(array_classes_gt_objects), 1])
    for k in range(np.size(array_classes_gt_objects)):
        if array_classes_gt_objects[k] == object_id:
            if available_objects[k] == 1:
                gt_mask = gt_masks[:, :, k]
                iou = calculate_real_iou(mask, gt_mask)
                results[k] = iou
            else:
                results[k] = -1
    max_result = max(results)
    ind = np.argmax(results)
    iou = last_matrix[ind]
    new_iou = max_result
    return iou, new_iou, results, ind
