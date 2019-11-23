######################################################
# Texture Enhancer and Feature Segmenter using optimal Gabor Kernels
# Author: Rodney Gracian Dsouza
######################################################

import cv2
import numpy as np

from src import feature_extractor as fext
from src import feature_selector as fsel
from src import find_cluster as fclt
from src import gabor_filterbank as gfb
from src import utils

color = np.float32([[1, 0, 0], [0, 1, 0]])
fmax = 1 / (2 + (2 * np.sqrt(np.log(2)) / np.pi))
f_n = 5
theta_n = 8
ksize = 31
sigma_x = sigma_y = 0.5
K = 2

if __name__ == "__main__":
    input_img = cv2.imread('../res/zebra.jpg')
    input_dim = input_img.shape

    print("Creating Gabor Filter Bank")
    freq, theta = gfb.createBankParams(fmax, f_n, theta_n)
    filter_bank = gfb.createFilterBank(ksize, freq, theta, sigma_x, sigma_y)
    img_bank, img_accum = gfb.filterImage(input_img, filter_bank)

    print("Applied filter bank on the image! Selecting features...")

    fsp_bank, fsp_img = fsel.select_features(img_bank)
    fsp_bank_m = np.asarray(fsp_bank)

    print("Extracting features")

    fsp_bank_m = utils.normalize_horizontal(fsp_bank_m)
    fsp_bank_m = fext.sigmoid_energy(fsp_bank_m, 1, 1)

    fsp_bank_m = utils.normalize_horizontal(fsp_bank_m)
    fsp_bank_m2 = fext.gaussian_smoothing(fsp_bank_m, input_dim, 5, 2)
    fsp_bank_m2 = utils.normalize_horizontal(fsp_bank_m2)

    print("Clustering selected features")

    cluster_img = fclt.kmeans_clustering(fsp_bank_m2.T, K, 10)

    # color = np.random.randn(K, 3)
    if (sum(cluster_img) / cluster_img.size) > 0.5:
        seg_color = color[cluster_img.flatten()]
    else:
        seg_color = color[~cluster_img.flatten()]

    output = seg_color.reshape(input_dim)

    cv2.imshow('Input', input_img)
    cv2.imshow('Texture Enhanced', img_accum)
    cv2.imshow('Selected Features', fsp_img)
    cv2.imshow('Segmented Features', output)

    print("Blue: Foreground Features; Green: Background")

    key = cv2.waitKey(0)
    if key & 0xFF == ord('k'):
        utils.plotGaborKernels(filter_bank, freq, theta)
    if key & 0xFF == ord('f'):
        utils.plotFilteredImages(img_bank, freq, theta)

    cv2.destroyAllWindows()
