import numpy as np
import cv2 as cv


def scratch_detection_tmp(img):
    mask = cv.Laplacian(img, cv.CV_64F, ksize=3)
    mask = cv.convertScaleAbs(mask)

    cv.imshow('laplacian', mask)
    cv.waitKey(0)
    
    _, mask = cv.threshold(mask, 180, 255, cv.THRESH_BINARY)

    cv.imshow('threshold', mask)
    cv.waitKey(0)

    mask = cv.dilate(mask, np.ones((7, 7), np.uint8))
    mask = cv.erode(mask, np.ones((7, 7), np.uint8))

    cv.imshow('closing', mask)
    cv.waitKey(0)

    mask = cv.dilate(mask, np.ones((5, 5), np.uint8))

    cv.imshow('dilated - final mask', mask)
    cv.waitKey(0)

    cv.destroyAllWindows()
    return mask


def scratch_detection(img):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    ridges = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    mask = cv.normalize(ridges, None, 0, 255, cv.NORM_MINMAX)
    
    cv.imshow('enhanced ridges', mask)
    cv.waitKey(0)
    
    mask = cv.dilate(mask, np.ones((7, 7), np.uint8))
    _, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)

    cv.imshow('threshold', mask)
    cv.waitKey(0)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 21))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    cv.imshow('closing', mask)
    cv.waitKey(0)

    mask = cv.dilate(mask, np.ones((5, 5), np.uint8))

    cv.imshow('dilated - final mask', mask)
    cv.waitKey(0)

    cv.destroyAllWindows()
    return mask


def scratch_removal(img, img_name, resize=False):
    if resize:
        img = cv.resize(img, (0, 0), fx=2, fy=2)
    mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('gray', mask)
    cv.waitKey(0)

    mask = scratch_detection(mask)

    result = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
    # result = inpaint_with_gradients(img, mask)

    result = cv.medianBlur(result, 3)

    cv.imwrite(f'data/output/output_{img_name}', result)


# Needed Stuff
#   Laplacian
#   inpaint
