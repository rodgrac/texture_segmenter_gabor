import numpy as np


def select_features(img_bank):
    reduced_fsp_bank = []
    img_bank_mat = np.asarray(img_bank)
    t_shape = img_bank_mat.shape
    img_tensor = np.reshape(img_bank_mat, (t_shape[0], t_shape[1] * t_shape[2], t_shape[3])).astype('float32')
    img_acc = np.zeros((t_shape[1], t_shape[2], t_shape[3]))

    for i in range(3):
        temp = img_tensor[:, :, i]
        tensor_cov = np.cov(temp)
        [U, D, V] = np.linalg.svd(tensor_cov)
        t = np.where(np.cumsum(D) / np.sum(D) > 0.99)[0][0]
        reduced_ker = U[:, :t + 1] / D[:t + 1]

        reduced_fsp = np.dot(reduced_ker.T, temp)

        reduced_fsp_bank.extend(reduced_fsp)

        img_acc[:, :, i] = np.sqrt(np.sum(reduced_fsp ** 2, axis=0)).reshape(-1, t_shape[2])

    return reduced_fsp_bank, img_acc
