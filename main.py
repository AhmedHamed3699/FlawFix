import argparse
import cv2

from core.preprocessing.brightness import adjust_brightness_to_range
from core.preprocessing.noise_reduction import apply_noise_removal, rgb_median_filter
from core.utils.common_functions import show_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="📷 Path to the input image")
    parser.add_argument("-o", "--output", help="💾 Path to save the adjusted image (optional)")

    args = parser.parse_args()

    print("\n✨ Welcome to the FlawFix! ✨")
    print(f"📂 Input Image: {args.input_image}")
    if args.output:
        print(f"💾 Output Image will be saved at: {args.output}")
    else:
        print("👀 The adjusted image will be displayed (not saved).")

    print("\n🛠️ Processing your image... Please wait...\n")

    image = cv2.imread(args.input_image)
    ################################ PREPROCESSING ################################
    brightness_adjusted_image = adjust_brightness_to_range(image)
    show_images([brightness_adjusted_image], ['brightness_adjusted_image'])

    noise_removed_image = apply_noise_removal(brightness_adjusted_image)
    show_images([noise_removed_image], ['noise_removed_image'])

    ######################## Write Your Function Calls Here #######################

    

    ################################################################################
    print("\nFinished Processing.\n")

if __name__ == "__main__":
    main()
