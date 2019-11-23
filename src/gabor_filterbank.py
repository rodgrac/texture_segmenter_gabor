import cv2
import numpy as np


def createBankParams(f_max, f_n, theta_div):
    f = np.geomspace(f_max / 2 ** (f_n - 1), f_max, num=f_n, dtype=float)
    thet = np.arange(theta_div) * 2 * np.pi / theta_div

    return f, thet


def createGaborKernel(window_size, f, theta, sigma_x, sigma_y):
    mid_val = int((window_size - 1) / 2)
    [xx, yy] = np.meshgrid(range(-mid_val, mid_val + 1), range(-mid_val, mid_val + 1))

    xx_dot = xx * np.cos(theta) + yy * np.sin(theta)
    yy_dot = -xx * np.sin(theta) + yy * np.cos(theta)

    gaussian = np.exp(-f ** 2 * ((xx_dot ** 2 / sigma_x ** 2) + (yy_dot ** 2 / sigma_y ** 2)))
    gaussian = (f ** 2 / (np.pi * sigma_x * sigma_y)) * gaussian

    harmonic = np.cos(2 * np.pi * f * xx_dot)

    gabor = gaussian * harmonic

    return gabor


def createFilterBank(ksize, freq, theta, sigma_x, sigma_y):
    filter_bank = []
    for f in freq:
        for t in theta:
            g_kernel = createGaborKernel(ksize, f, t, sigma_x, sigma_y)
            g_kernel /= 2 * g_kernel.sum()
            filter_bank.append(g_kernel)
    return filter_bank


def filterImage(img, filter_bank):
    filtered_images = []
    img_acc = np.zeros_like(img)
    for fb in filter_bank:
        img_p = cv2.filter2D(img, cv2.CV_8UC3, fb)
        filtered_images.append(img_p)
        np.maximum(img_acc, img_p, img_acc)

    return filtered_images, img_acc
