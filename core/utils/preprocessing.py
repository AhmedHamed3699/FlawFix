import numpy as np
from PIL import ImageEnhance, Image
import matplotlib.pyplot as plt

from core.utils.common_functions import show_images

def analyze_brightness(image):
    image = np.array(image)
    show_images([image], ['original'])

    if len(image.shape) == 2:
        mean_brightness = image.mean()
    elif len(image.shape) == 3 and image.shape[2] == 3:
        brightness = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        mean_brightness = brightness.mean()
    else:
        raise ValueError("Unsupported image format. Image must be grayscale or RGB.")
    return mean_brightness

def adjust_brightness_to_range(image, brightness_range=(100, 150)):
    current_brightness = analyze_brightness(image)
    print(f"Current brightness: {current_brightness}")

    min_brightness, max_brightness = brightness_range
    if current_brightness < min_brightness:
        adjustment_factor = min_brightness / current_brightness
    elif current_brightness > max_brightness:
        adjustment_factor = max_brightness / current_brightness
    else:
        return image 
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(adjustment_factor)
    return adjusted_image

import numpy as np

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
