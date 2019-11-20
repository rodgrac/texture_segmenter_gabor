######################################################
# Gabor Filter Implementation
# Author: Rodney Gracian Dsouza
######################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt


def createbankParams(f_max, f_n, theta_div):
    f = np.geomspace(f_max / 2 ** (f_n - 1), f_max, num=f_n, dtype=float)
    thet = np.arange(theta_div) * 2 * np.pi / theta_div

    return f, thet


# f: frequency; theta: orientation; sigma: gaussian envelope
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


if __name__ == "__main__":
    img_g = cv2.imread('../res/zebra.jpeg')
    cv2.imshow('input', img_g)
    # img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    freq, theta = createbankParams(1 / (2 + (2 * np.sqrt(np.log(2)) / np.pi)), 4, 8)

    fig1, axs1 = plt.subplots(np.size(freq), np.size(theta), sharex='col', sharey='row',
                              gridspec_kw={'hspace': 0, 'wspace': 0})
    fig1.suptitle("Gabor kernels - Orientation against Frequency")
    fig2, axs2 = plt.subplots(np.size(freq), np.size(theta), sharex='col', sharey='row',
                              gridspec_kw={'hspace': 0, 'wspace': 0})

    accum = np.zeros_like(img_g)
    for i in range(np.size(freq)):
        for j in range(np.size(theta)):
            g_kernel = createGaborKernel(31, freq[i], theta[j], 0.5, 0.5)
            # g_kernel = cv2.getGaborKernel((31, 31), 4.0, theta[j], 10.0, 0.5, 0, ktype=cv2.CV_32F)
            g_kernel /= 2 * g_kernel.sum()
            out = cv2.filter2D(img_g, cv2.CV_8UC3, g_kernel)
            np.maximum(accum, out, accum)
            kh, kw = g_kernel.shape[:2]
            ih, iw = img_g.shape[:2]
            g_kernel_resized = cv2.resize(g_kernel, (3 * kw, 3 * kh), interpolation=cv2.INTER_CUBIC)
            axs1[i, j].imshow(g_kernel_resized, cmap='gray')
            axs1[i, j].set(xlabel=(theta[j] / np.pi) * 180, ylabel=freq[i])
            axs1[i, j].set_xticklabels([])
            axs1[i, j].set_yticklabels([])
            axs1[i, j].label_outer()

            axs2[i, j].imshow(out, cmap='gray')
            axs2[i, j].set_xticklabels([])
            axs2[i, j].set_yticklabels([])
            axs2[i, j].label_outer()

    cv2.imshow('Out', accum)
    cv2.waitKey(0)

    plt.show()
    cv2.destroyAllWindows()
