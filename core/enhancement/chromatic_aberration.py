import cv2
import numpy as np

def rmCA(bgr_list, threshold):
    height, width = bgr_list[0].shape

    for i in range(height):
        bptr = bgr_list[0][i]
        gptr = bgr_list[1][i]
        rptr = bgr_list[2][i]

        for j in range(1, width - 1):
            # Find the edge by checking if the green channel gradient exceeds the threshold
            if abs(int(gptr[j + 1]) - int(gptr[j - 1])) >= threshold:
                sign = 1 if gptr[j + 1] > gptr[j - 1] else -1 

                # Search for the boundary for the correction range
                lpos = j - 1
                rpos = j + 1

                while lpos > 0:
                    ggrad = (int(gptr[lpos + 1]) - int(gptr[lpos - 1])) * sign
                    bgrad = (int(bptr[lpos + 1]) - int(bptr[lpos - 1])) * sign
                    rgrad = (int(rptr[lpos + 1]) - int(rptr[lpos - 1])) * sign
                    if max(bgrad, ggrad, rgrad) < threshold:
                        break
                    lpos -= 1

                if (lpos > 0):
                    lpos -= 1

                while rpos < width - 1:
                    ggrad = (int(gptr[rpos + 1]) - int(gptr[rpos - 1])) * sign 
                    bgrad = (int(bptr[rpos + 1]) - int(bptr[rpos - 1])) * sign
                    rgrad = (int(rptr[rpos + 1]) - int(rptr[rpos - 1])) * sign
                    if max(bgrad, ggrad, rgrad) < threshold:
                        break
                    rpos += 1
                if (rpos < width - 1):
                    rpos += 1

                # Record the maximum and minimum color differences between R&G and B&G
                bgmax_val = max(int(bptr[lpos]) - int(gptr[lpos]), int(bptr[rpos]) - int(gptr[rpos]))
                bgmin_val = min(int(bptr[lpos]) - int(gptr[lpos]), int(bptr[rpos]) - int(gptr[rpos]))
                rgmax_val = max(int(rptr[lpos]) - int(gptr[lpos]), int(rptr[rpos]) - int(gptr[rpos]))
                rgmin_val = min(int(rptr[lpos]) - int(gptr[lpos]), int(rptr[rpos]) - int(gptr[rpos]))


                for k in range(lpos, rpos + 1):
                    bdiff = int(bptr[k]) - int(gptr[k])
                    rdiff = int(rptr[k]) - int(gptr[k])

                    bptr[k] = np.clip(
                        bgmax_val + gptr[k] if bdiff > bgmax_val else (gptr[k] if bdiff >= bgmin_val else bgmin_val + gptr[k]),
                        0, 255
                    ).astype(np.uint8)

                    rptr[k] = np.clip(
                        rgmax_val + gptr[k] if rdiff > rgmax_val else (rptr[k] if rdiff >= rgmin_val else rgmin_val + gptr[k]),
                        0, 255
                    ).astype(np.uint8)

                j = rpos - 2

def CACorrection(src):
    bgr_list = cv2.split(src)

    # Setting threshold to find the edge and correction range (in g channel)
    threshold = 30

    rmCA(bgr_list, threshold)

    # Transpose the R, G, B channels to correct chromatic aberration in the vertical direction
    bgr_list = [channel.T for channel in bgr_list]

    rmCA(bgr_list, threshold)

    # Merge channels back and rotate the image to the original position
    dst = cv2.merge(bgr_list)
    dst = cv2.transpose(dst)

    return dst


def main():
    # Load the source image
    src = cv2.imread("CA4.jpg", cv2.IMREAD_COLOR)
    if src is None:
        print("Error: Could not read the input image.")
        return

    # Perform chromatic aberration correction
    dst = CACorrection(src)

    # Display the original and corrected images
    cv2.imshow("Original", src)
    cv2.imshow("Result", dst)

    # Save the corrected image
    cv2.imwrite("./imgs/result.bmp", dst)

    # Wait for a key press and clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


