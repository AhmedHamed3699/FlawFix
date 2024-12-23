import numpy as np
import cv2

def gaussian_filter(img, kernel_size=(5, 5), sigma=0):
    img = np.array(img)
    if len(img.shape) == 2:
        smoothed_img = cv2.GaussianBlur(img, kernel_size, sigma)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        smoothed_img = cv2.GaussianBlur(img, kernel_size, sigma)
    else:
        raise ValueError("Input image must be either grayscale or RGB.")
    return smoothed_img

def median_filter_single_channel(channel, window_size):
    edgex = window_size[1] // 2
    edgey = window_size[0] // 2
    filtered_channel = np.zeros_like(channel)
    
    for x in range(edgex, channel.shape[1] - edgex):
        for y in range(edgey, channel.shape[0] - edgey):
            window = []
            for fx in range(window_size[1]):
                for fy in range(window_size[0]):
                    window.append(channel[y + fy - edgey, x + fx - edgex])
            
            window = sorted(window)
            filtered_channel[y, x] = window[len(window) // 2]
    return filtered_channel

def rgb_median_filter(img, window_size):
    img = np.array(img)
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be an RGB image.")
    filtered_img = np.zeros_like(img)
    for channel in range(3):
        filtered_img[:, :, channel] = median_filter_single_channel(img[:, :, channel], window_size)
    return filtered_img

def apply_noise_removal (img, window_size=(3,3), sigma=0):
    salt_pepper_noise_removed_img = rgb_median_filter(img,window_size)
    return gaussian_filter(salt_pepper_noise_removed_img,window_size,sigma)