import cv2
import numpy as np
from matplotlib import pyplot as plt


def normalize_horizontal(input):
    output = ((input.T - np.min(input.T, axis=0)) / (
            (np.max(input.T, axis=0)) - (np.min(input.T, axis=0)))).T

    return output


def stat_normalize_horizontal(input):
    output = ((input.T - np.mean(input.T, axis=0)) / (np.std(input.T, axis=0))).T
    return output


def plotGaborKernels(kernels, freq, theta):
    f_n = np.size(freq)
    t_n = np.size(theta)

    fig1, axs1 = plt.subplots(np.size(freq), np.size(theta), sharex='col', sharey='row',
                              gridspec_kw={'hspace': 0, 'wspace': 0})
    fig1.suptitle("Gabor kernels - Orientation against Frequency")

    for i in range(f_n):
        for j in range(t_n):
            g_kernel = kernels[t_n * i + j]
            kh, kw = g_kernel.shape[:2]
            g_kernel_resized = cv2.resize(g_kernel, (3 * kw, 3 * kh), interpolation=cv2.INTER_CUBIC)
            axs1[i, j].imshow(g_kernel_resized, cmap='gray')
            axs1[i, j].set(xlabel=(theta[j] / np.pi) * 180, ylabel=np.around([freq[i]], decimals=2))
            axs1[i, j].set_xticklabels([])
            axs1[i, j].set_yticklabels([])
            axs1[i, j].label_outer()

    plt.show()


def plotFilteredImages(f_img, freq, theta):
    f_n = np.size(freq)
    t_n = np.size(theta)

    fig2, axs2 = plt.subplots(np.size(freq), np.size(theta), sharex='col', sharey='row',
                              gridspec_kw={'hspace': 0, 'wspace': 0})
    for i in range(f_n):
        for j in range(t_n):
            axs2[i, j].imshow(f_img[t_n * i + j])
            axs2[i, j].set_xticklabels([])
            axs2[i, j].set_yticklabels([])
            axs2[i, j].label_outer()

    plt.show()
