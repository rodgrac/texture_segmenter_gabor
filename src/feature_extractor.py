import cv2
import numpy as np


def gaussian_smoothing(img_bank, shape, ksize, sigma):
    gauss = cv2.getGaussianKernel(ksize, sigma)
    for i in range(img_bank.shape[0]):
        img = np.reshape(img_bank[i], (-1, shape[1]))
        temp = cv2.filter2D(img, -1, gauss)
        img_bank[i] = temp.flatten()

    return img_bank


def sigmoid_energy(input, alpha, type):
    if type:
        output = 1 / (1 + np.exp(-input * alpha))
    else:
        output = np.tanh(input * alpha)

    return output
