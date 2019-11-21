######################################################
# Texture Enhancer using Gabor Kernels
# Author: Rodney Gracian Dsouza
######################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt


def createbankParams(f_max, f_n, theta_div):
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

    return img_acc, filtered_images


def reduced_image(img_bank):
    img_bank_mat = np.asarray(img_bank)
    t_shape = img_bank_mat.shape
    img_tensor = np.reshape(img_bank_mat, (t_shape[0], t_shape[1] * t_shape[2], t_shape[3]))
    img_acc = np.zeros((t_shape[1], t_shape[2], t_shape[3]))

    for i in range(3):
        tensor_cov = np.cov(img_tensor[:, :, i])
        [U, D, V] = np.linalg.svd(tensor_cov)
        t = np.where(np.cumsum(D) / np.sum(D) > 0.99)[0][0]
        reduced_ker = U[:, :t + 1] / D[:t + 1]

        reduced_fsp = np.dot(reduced_ker.T, img_tensor[:, :, i])

        img_acc[:, :, i] = np.max(reduced_fsp, axis=0).reshape(-1, t_shape[2])
    return img_acc


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


if __name__ == "__main__":
    img = cv2.imread('../res/tiger.jpg')

    freq, theta = createbankParams(1 / (2 + (2 * np.sqrt(np.log(2)) / np.pi)), 4, 8)

    filter_bank = createFilterBank(31, freq, theta, 0.5, 0.5)
    out, img_bank = filterImage(img, filter_bank)

    output = reduced_image(img_bank)

    cv2.imshow('Input', img)
    cv2.imshow('Output1', out)
    cv2.imshow('Output2', output)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('k'):
        plotGaborKernels(filter_bank, freq, theta)
    if key & 0xFF == ord('f'):
        plotFilteredImages(img_bank, freq, theta)

    cv2.destroyAllWindows()
