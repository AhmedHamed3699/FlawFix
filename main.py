import argparse
import cv2

from core.preprocessing.brightness import adjust_brightness_to_range
from core.preprocessing.contrast import enhance_contrast
from core.preprocessing.noise_reduction import rgb_median_filter, gaussian_filter
from core.utils.common_functions import show_images


def preprocessing(img):

    brightness_adjusted_image = adjust_brightness_to_range(img)
    show_images([brightness_adjusted_image], ['brightness_adjusted_image'])

    salt_pepper_noise_removed_img = rgb_median_filter(img, (5, 5))
    show_images([salt_pepper_noise_removed_img], [
                'salt_pepper_noise_removed_img'])

    noise_removed_image = gaussian_filter(img)
    show_images([noise_removed_image], ['noise_removed_image'])

    contrast_enhanced_image = enhance_contrast(img)
    show_images([contrast_enhanced_image], ['contrast_enhanced_image'])

    return contrast_enhanced_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="📷 Path to the input image")
    parser.add_argument(
        "-o", "--output", help="💾 Path to save the adjusted image (optional)")

    args = parser.parse_args()

    print("\n✨ Welcome to the FlawFix! ✨")
    print(f"📂 Input Image: {args.input_image}")
    if args.output:
        print(f"💾 Output Image will be saved at: {args.output}")
    else:
        print("👀 The adjusted image will be displayed (not saved).")

    print("\n🛠️ Processing your image... Please wait...\n")

    image = cv2.imread(args.input_image)
    show_images([image], ['original_image'])

    preprocessing(image)
    ######################## Write Your Function Calls Here #######################

    ################################################################################
    print("\nFinished Processing.\n")


if __name__ == "__main__":
    main()
