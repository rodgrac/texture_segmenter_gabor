import cv2
import numpy as np
from matplotlib import pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (0, 0, 255)
thickness = 2


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


def plotOutputImages(output, n_row, n_col):
    fig2, axs2 = plt.subplots(n_col, n_row, sharex='col', sharey='row',
                              gridspec_kw={'hspace': 0, 'wspace': 0})
    for i in range(len(output) // n_col):
        for j in range(n_col):
            axs2[i, j].imshow(output[n_col * i + j])
            axs2[i, j].set_xticklabels([])
            axs2[i, j].set_yticklabels([])
            axs2[i, j].label_outer()

    plt.show()


def text_overlay(image, text, loc):
    out = cv2.putText(image, text, loc, font, fontScale, fontColor, thickness, cv2.LINE_AA)
    return out


def stitch_output(out1, out2, out3, out4):
    h1 = np.concatenate((out1, out2), axis=1)
    h2 = np.concatenate((out3, out4), axis=1)
    h = np.concatenate((h1, h2), axis=0)
    out = cv2.resize(h, (640, 480), interpolation=cv2.INTER_CUBIC)

    text_overlay(out, "Input", (int(0.03 * out.shape[1]), int(0.03 * out.shape[1])))
    text_overlay(out, "Enhanced Texture",
                 (int(0.03 * out.shape[1] + out.shape[1] / 2), int(0.03 * out.shape[1])))
    text_overlay(out, "Selected Features",
                 (int(0.03 * out.shape[1]), int(0.03 * out.shape[1] + out.shape[0] / 2)))
    text_overlay(out, "Segmented Features", (
        int(0.03 * out.shape[1] + out.shape[1] / 2), int(0.03 * out.shape[1] + out.shape[0] / 2)))

    return out
