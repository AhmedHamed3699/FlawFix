import numpy as np
import cv2 as cv


def scratch_detection(img):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    ridges = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    mask = cv.normalize(ridges, None, 0, 255, cv.NORM_MINMAX)

    mask = cv.dilate(mask, np.ones((7, 7), np.uint8))
    _, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 21))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    mask = cv.dilate(mask, np.ones((5, 5), np.uint8))
    return mask


def scratch_removal(img, resize=False):
    if resize:
        img = cv.resize(img, (0, 0), fx=2, fy=2)
    mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('gray', mask)
    cv.waitKey(0)

    mask = scratch_detection(mask)

    result = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
    return result
