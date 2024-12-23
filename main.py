import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from core.utils.common_functions import show_images
from core.utils.preprocessing import adjust_brightness_to_range, rgb_median_filter

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

    image = Image.open(args.input_image)

    ################################ PREPROCESSING ################################
    brightness_adjusted_image = adjust_brightness_to_range(image)
    noise_removed_image = rgb_median_filter(brightness_adjusted_image,(3,3));
    
    show_images([noise_removed_image], ['noise_removed_image'])

    ######################## Write Your Function Calls Here #######################


    ################################################################################
    print("\nFinished Processing.\n")

if __name__ == "__main__":
    main()
