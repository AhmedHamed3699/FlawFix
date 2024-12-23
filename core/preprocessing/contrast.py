import cv2

def enhance_contrast(img):
    if len(img.shape) == 3:
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_img = img
    enhanced_img = cv2.equalizeHist(grayscale_img)

    if len(img.shape) == 3:
        channels = cv2.split(img)
        equalized_channels = [cv2.equalizeHist(c) for c in channels]
        enhanced_img = cv2.merge(equalized_channels)
    return enhanced_img;