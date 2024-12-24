from math import sqrt as sqrt
import cv2
import heapq
import numpy as np

INF = 1e9 # instead of np.inf to avoid inf * 0
NIGATIVE_INF = 1e-9

# status_map
KNOWN = 0
BAND = 1
UNKNOWN = 2

#----------------------------------------------------------------------------------#
def solve_eikonal_equation(y1, x1, y2, x2, img_img_height, img_img_width, dists, status_map):
    """
    Solves the Eikonal equation to estimate the distance of a pixel to the nearest known pixel.
    This determines the propagation of the inpainting process.
    """

    # Check if the first point is outside the image boundaries
    if y1 < 0 or y1 >= img_img_height or x1 < 0 or x1 >= img_img_width:
        return INF

    # Check if the second point is outside the image boundaries
    if y2 < 0 or y2 >= img_img_height or x2 < 0 or x2 >= img_img_width:
        return INF

    status1 = status_map[y1, x1]
    status2 = status_map[y2, x2]

    # If both pixels are already known, calculate distance based on their values
    if status1 == KNOWN and status2 == KNOWN:
        dist1 = dists[y1, x1]
        dist2 = dists[y2, x2]

        # Solve the quadratic equation
        delta = 2.0 - (dist1 - dist2) ** 2
        if delta > 0:
            root = sqrt(delta)
            sol1 = (dist1 + dist2 - root) / 2.0
            sol2 = sol1 + root

            # Return the valid solution
            if sol1 >= dist1 and sol1 >= dist2:
                return sol1
            if sol2 >= dist1 and sol2 >= dist2:
                return sol2
            
        return INF  # Unsolvable case

    if status1 == KNOWN:
        return 1.0 + dists[y1, x1]
    if status2 == KNOWN:
        return 1.0 + dists[y2, x2]

    # If neither pixel is unknown, return infinity
    return INF

#----------------------------------------------------------------------------------#
def calculate_distances_map(img_img_height,img_img_width,dists,status_map,band_pixels_heap,radius):
    '''
    calculate distances between initial mask contour and pixels outside mask, using FMM (Fast Marching Method)
    propagate outward so the directions are inverted
    '''

    band_pixels = band_pixels_heap.copy()
    orignal_status_map = status_map
    status_map = orignal_status_map.copy()

    # swap KOWN <=> UNKOWN
    status_map[orignal_status_map == KNOWN] = UNKNOWN
    status_map[orignal_status_map == UNKNOWN] = KNOWN

    last_dist = 0.0
    diameter = radius * 2
    while band_pixels:

        #  reached limit, stop FMM
        if last_dist >= diameter:
            break

        # pixel closest to mask contour and update its stateh as KNOWN
        _, y, x = heapq.heappop(band_pixels)
        status_map[y, x] = KNOWN

        # process nighbors
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for n_y, n_x in neighbors:

            # skip out of frame
            if n_y < 0 or n_y >= img_img_height or n_x < 0 or n_x >= img_img_width:
                continue

            # neighbor already processed, nothing to do
            if status_map[n_y, n_x] != UNKNOWN:
                continue

            # compute neighbor distance to initial mask contour
            last_dist = min([
                solve_eikonal_equation(n_y - 1, n_x, n_y, n_x - 1, img_img_height, img_img_width, dists, status_map),
                solve_eikonal_equation(n_y + 1, n_x, n_y, n_x + 1, img_img_height, img_img_width, dists, status_map),
                solve_eikonal_equation(n_y - 1, n_x, n_y, n_x + 1, img_img_height, img_img_width, dists, status_map),
                solve_eikonal_equation(n_y + 1, n_x, n_y, n_x - 1, img_img_height, img_img_width, dists, status_map)
            ])
            dists[n_y, n_x] = last_dist

            # add neighbor to narrow band pixels
            status_map[n_y, n_x] = BAND
            heapq.heappush(band_pixels, (last_dist, n_y, n_x))

    # distances are opposite to actual FFM propagation direction
    dists *= -1.0
#----------------------------------------------------------------------------------#
def init(img_height, img_width, mask, radius):
    '''
    initialize contour, distances, and status map
    '''
    # init all distances to infinity
    dists = np.full((img_height, img_width), INF, dtype=float)

    # status of each pixel, ie KNOWN, BAND or UNKNOWN
    status_map = mask.astype(int) * UNKNOWN
    # narrow band, queue of contour pixels
    band = []

    mask_ys, mask_xs = mask.nonzero()
    for y in mask_ys:
        for x in mask_xs:
            # look for BAND pixels in neighbors (top/bottom/left/right)
            neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
            for n_y, n_x in neighbors:
                # neighbor out of frame
                if n_y < 0 or n_y >= img_height or n_x < 0 or n_x >= img_width:
                    continue

                # neighbor is already BAND
                if status_map[n_y, n_x] == BAND:
                    continue

                # neighbor out of mask => mask contour
                if mask[n_y, n_x] == 0:
                    status_map[n_y, n_x] = BAND
                    dists[n_y, n_x] = 0.0
                    heapq.heappush(band, (0.0, n_y, n_x))

    # compute distance to inital mask contour for KNOWN pixels
    # using FMM
    calculate_distances_map(img_height, img_width, dists, status_map, band, radius)

    return dists, status_map, band

#----------------------------------------------------------------------------------#
def calculate_pixel_gradient(y, x, img_height, img_width, vals, status_map):
    '''
    calculate the distance y gradient using finite diffrence method 
    '''
    val = vals[y, x]

    # compute grad_y
    prev_y = y - 1
    next_y = y + 1

    #check if the coordinate is not out of range 
    if prev_y < 0 or next_y >= img_height:
        grad_y = INF
    else:
        prev_y_status = status_map[prev_y, x]
        next_y_status = status_map[next_y, x]

        if prev_y_status != UNKNOWN and next_y_status != UNKNOWN:
            grad_y = (vals[next_y, x] - vals[prev_y, x]) / 2.0
        elif prev_y_status != UNKNOWN:
            grad_y = val - vals[prev_y, x]
        elif next_y_status != UNKNOWN:
            grad_y = vals[next_y, x] - val
        else:
            grad_y = 0.0

    # compute grad_x
    prev_x = x - 1
    next_x = x + 1

    #check if the coordinate is not out of range 
    if prev_x < 0 or next_x >= img_width:
        grad_x = INF
    else:
        prev_x_state = status_map[y, prev_x]
        flag_next_x = status_map[y, next_x]

        if prev_x_state != UNKNOWN and flag_next_x != UNKNOWN:
            grad_x = (vals[y, next_x] - vals[y, prev_x]) / 2.0
        elif prev_x_state != UNKNOWN:
            grad_x = val - vals[y, prev_x]
        elif flag_next_x != UNKNOWN:
            grad_x = vals[y, next_x] - val
        else:
            grad_x = 0.0

    return grad_y, grad_x

#----------------------------------------------------------------------------------#
def inpaint_pixel(y, x, img, img_height, img_width, dists, status_map, radius):
    '''
     returns the 3 channels red,green,blue values for the inpainted pixel 
     by processing telea's algorihtm for the surrounding KOWN values
    '''
    dist = dists[y, x]

    # normal to pixel, ie direction of propagation of the FFM
    dist_grad_y, dist_grad_x = calculate_pixel_gradient(y, x, img_height, img_width, dists, status_map)
    inpainted_value = np.zeros((3), dtype=float)
    weight_sum = 0.0

    for n_y in range(y - radius, y + radius + 1):
        #  pixel out of range
        if n_y < 0 or n_y >= img_height:
            continue

        for n_x in range(x - radius, x + radius + 1):
            # pixel out of range
            if n_x < 0 or n_x >= img_width:
                continue

            # skip unknown pixels
            if status_map[n_y, n_x] == UNKNOWN:
                continue

            # direction from point to neighbor
            y_direction = y - n_y
            x_direction = x - n_x
            dir_length_square =    y_direction ** 2 + x_direction ** 2
            dir_length = sqrt(dir_length_square)
            # pixel out of neighborhood
            if dir_length > radius:
                continue

            # weight calculation
            # neighbor has same direction gradient => contributes more
            dir = abs(y_direction * dist_grad_y + x_direction * dist_grad_x)
            if dir == 0.0:                                            
                dir = NIGATIVE_INF

            # neighbor has same contour distance => contributes more
            n_dist = dists[n_y, n_x]
            lev = 1.0 / (1.0 + abs(n_dist - dist))

            # neighbor is distant => contributes less
            dist_factor = 1.0 / (dir_length * dir_length_square)

            weight = abs(dir * dist_factor * lev)

            inpainted_value += weight * img[n_y, n_x]

            weight_sum += weight

    inpainted_value=inpainted_value / weight_sum
    return inpainted_value

#----------------------------------------------------------------------------------#
def inpaint_region(img, mask, radius=7):
    '''
    loop till there is no band and inpaint the UNKOWN neighbour pixels  
    update the neighbour pixel value using its neighbour distances
    '''

    img_height, img_width = img.shape[0:2]
    dists, status_map, band = init(img_height, img_width, mask, radius)

    while band:
        _, y, x = heapq.heappop(band)

        # process his neighbors
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for n_y, n_x in neighbors:

            # pixel out of frame
            if n_y < 0 or n_y >= img_height or n_x < 0 or n_x >= img_width:
                continue

            if status_map[n_y, n_x] != UNKNOWN:
                continue

            # compute neighbor distance to inital mask contour
            n_dist = min([
                solve_eikonal_equation(n_y - 1, n_x, n_y, n_x - 1, img_height, img_width, dists, status_map),
                solve_eikonal_equation(n_y + 1, n_x, n_y, n_x + 1, img_height, img_width, dists, status_map),
                solve_eikonal_equation(n_y - 1, n_x, n_y, n_x + 1, img_height, img_width, dists, status_map),
                solve_eikonal_equation(n_y + 1, n_x, n_y, n_x - 1, img_height, img_width, dists, status_map)
            ])
            dists[n_y, n_x] = n_dist

            pixel_vals = inpaint_pixel(n_y, n_x, img, img_height, img_width, dists, status_map, radius)

            img[n_y, n_x] = pixel_vals
            
            # mark it as KNOWN
            status_map[y, x] = KNOWN

            #add new pixel to band
            status_map[n_y, n_x] = BAND
            heapq.heappush(band, (n_dist, n_y, n_x))

# #test 
# in_img = cv2.imread('samples/cat_in.png')
# mask_img = cv2.imread('samples/cat_mask.png')
# mask = mask_img[:, :, 0].astype(bool, copy=False)
# out_img = in_img.copy()

# inpaint_region(out_img, mask, 2)
# cv2.imwrite('samples/cat_out.png', out_img)
