######################################################
# Gabor Filter Implementation
# Author: Rodney Gracian Dsouza
######################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt

lam = np.arange(0.24, 0.12, -0.02)
theta_res = np.pi / 8
theta_count = 9

fig1, axs1 = plt.subplots(np.size(lam), theta_count + 1, sharex='col', sharey='row',
                          gridspec_kw={'hspace': 0, 'wspace': 0})
fig1.suptitle("Gabor kernels - Orientation against Frequency")
fig2, axs2 = plt.subplots(np.size(lam), theta_count + 1, sharex='col', sharey='row',
                          gridspec_kw={'hspace': 0, 'wspace': 0})


# f: frequency; theta: orientation; sigma: gaussian envelope
def createGaborKernel(window_size, f, theta, sigma):
    mid_val = int((window_size - 1) / 2)
    [xx, yy] = np.meshgrid(range(-mid_val, mid_val + 1), range(-mid_val, mid_val + 1))

    xx_dot = xx * np.cos(theta) + yy * np.sin(theta)
    yy_dot = -xx * np.sin(theta) + yy * np.cos(theta)

    gaussian = np.exp((-0.5 * f ** 2 * (xx_dot ** 2 + yy_dot ** 2)) / sigma ** 2)
    gaussian = (f ** 2 / (2 * np.pi * sigma ** 2)) * gaussian

    harmonic = np.cos(f * xx_dot) * np.exp(sigma ** 2 / 2)

    gabor = gaussian * harmonic

    return gabor


if __name__ == "__main__":
    img = cv2.imread('../res/zebra.jpeg')
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0, np.size(lam)):
        for j in range(0, theta_count + 1):
            g_kernel = createGaborKernel(31, 1 / lam[i], theta_res * j, 6.0)
            # g_kernel = cv2.getGaborKernel((31, 31), 2.0, theta_res, lam[i], 1, 0, ktype=cv2.CV_32F)
            out = cv2.filter2D(img_g, cv2.CV_8UC3, g_kernel)
            kh, kw = g_kernel.shape[:2]
            ih, iw = img_g.shape[:2]
            g_kernel_resized = cv2.resize(g_kernel, (3 * kw, 3 * kh), interpolation=cv2.INTER_CUBIC)
            axs1[i, j].imshow(g_kernel_resized, cmap='gray')
            axs1[i, j].set(xlabel=np.str(j) + "π/8", ylabel=lam[i])
            axs1[i, j].set_xticklabels([])
            axs1[i, j].set_yticklabels([])
            axs1[i, j].label_outer()

            axs2[i, j].imshow(out, cmap='gray')
            axs2[i, j].set_xticklabels([])
            axs2[i, j].set_yticklabels([])
            axs2[i, j].label_outer()

    plt.show()
