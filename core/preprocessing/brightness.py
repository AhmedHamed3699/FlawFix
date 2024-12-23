import numpy as np
from PIL import Image, ImageEnhance

def analyze_brightness(image):
    image = np.array(image)
    if len(image.shape) == 2:
        mean_brightness = image.mean()
    elif len(image.shape) == 3 and image.shape[2] == 3:
        brightness = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        mean_brightness = brightness.mean()
    else:
        raise ValueError("Unsupported image format. Image must be grayscale or RGB.")
    return mean_brightness

def adjust_brightness_to_range(image, brightness_range=(100, 150)):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

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
