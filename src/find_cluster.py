import cv2
import numpy as np


def kmeans_clustering(input, K, iter):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iter, 1.0)
    ret, label, center = cv2.kmeans(np.float32(input), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return label
