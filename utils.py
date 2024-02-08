import cv2
import numpy as np
import pickle
from skimage.transform import resize

with open('classifier_model/spot_model.p', 'rb') as file:
    MODEL = pickle.load(file)
EMPTY = True
NOT_EMPTY = False


def empty_or_not(spot_image):
    data_flat = []
    data_img = resize(spot_image, (15, 15, 3))
    data_flat.append(data_img.flatten())
    data_flat = np.asarray(data_flat)
    y = MODEL.predict(data_flat)

    if y == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    spots = []
    (totalLabels, label_id, values, centroids) = connected_components
    for i in range(1, totalLabels):
        x = int(values[i, cv2.CC_STAT_LEFT])
        y = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])
        (cx, cy) = centroids[i]

        spots.append([x, y, w, h, int(cx), int(cy)])
    return spots


def diff_img(img1, img2):
    return np.abs(np.mean(img1) - np.mean(img2))
