import numpy as np


def erosion(img, window_height=3, window_width=3, type='binary'):
    se = np.ones((window_height, window_width), dtype=np.uint8)

    pad_height = window_height // 2
    pad_width = window_width // 2
    padded_img = np.pad(img, ((pad_height, pad_height),
                        (pad_width, pad_width)), mode='constant', constant_values=0)

    eroded_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neighborhood = padded_img[i:i + window_height, j:j + window_width]
            if type == 'grayscale':
                eroded_img[i, j] = np.min(neighborhood[se == 1])
            else:
                if np.all(neighborhood[se == 1] == 1):
                    eroded_img[i, j] = 1

    return eroded_img


def dilation(img, window_height=3, window_width=3, type='binary'):
    se = np.ones((window_height, window_width), dtype=np.uint8)

    pad_height = window_height // 2
    pad_width = window_width // 2
    padded_img = np.pad(img, ((pad_height, pad_height),
                        (pad_width, pad_width)), mode='constant', constant_values=0)

    dilated_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neighborhood = padded_img[i:i + window_height, j:j + window_width]
            if type == 'grayscale':
                dilated_img[i, j] = np.max(neighborhood[se == 1])
            else:
                if np.any(neighborhood[se == 1]):
                    dilated_img[i, j] = 1

    return dilated_img


def opening(img, window_height=3, window_width=3, type='binary'):
    img = erosion(img, window_height, window_width, type)
    img = dilation(img, window_height, window_width, type)
    return img


def closing(img, window_height=3, window_width=3, type='binary'):
    img = dilation(img, window_height, window_width, type)
    img = erosion(img, window_height, window_width, type)
    return img


def top_hat(img, window_height=3, window_width=3, type='binary'):
    img = img - opening(img, window_height, window_width, type)
    return img
