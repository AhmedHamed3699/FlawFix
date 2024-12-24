import numpy as np
import cv2 as cv
import inpainting

from core.utils.morphology import dilation, erosion, opening, closing, top_hat


def scratch_detection(img):

    # Top Hat
    ridges = top_hat(img, 5, 5, 'grayscale')
    mask = cv.normalize(ridges, None, 0, 255, cv.NORM_MINMAX)

    # Thresholding
    mask = dilation(mask, 7, 7, 'grayscale')
    _, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)

    # Closing
    mask = closing(mask, 21, 21)

    # Final Dilation
    mask = dilation(mask, 5, 5)
    return mask


def scratch_detection_faster(img):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    ridges = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    mask = cv.normalize(ridges, None, 0, 255, cv.NORM_MINMAX)

    mask = cv.dilate(mask, np.ones((7, 7), np.uint8))
    _, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 21))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    mask = cv.dilate(mask, np.ones((5, 5), np.uint8))
    return mask


def scratch_removal(img):
    mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    mask = scratch_detection_faster(mask)

    # result = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
    result= inpainting.inpaint_region(img,mask,3)
    return result
