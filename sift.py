# There are always some difference between theory and implementations
# The paper provides detailed parameter settings
# The followong code provides easy implementation methods
# OpenCV has both parameter settings and implementation methods
# User can set it below

#####################
# User setting area #
#####################
# Use OpenCV lib?   #
#####################
lib = True
#####################
# Use OpenCV arg?   #
#####################
arg = True
#####################

###########################################
# Do not easily modify the following code #
###########################################

import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Use OpenCV to compare
import cv2 as cv

# Utils

def Gray(image, lib = False):
    if not lib:
        if len(image.shape) == 3: # RGB or RGBA
            return image.mean(axis = -1).astype(np.uint8)

        else: # Gray
            return image

    else:
        return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

def GaussianKernel(sigma):
    dimension = int(6 * sigma + 1) # `6` from 3 sigma principle
    if dimension % 2 == 0:
        dimension += 1

    start = -(dimension // 2)
    end = start + dimension

    x_square = np.array([[n for n in range(start, end)] for m in range(dimension)]) ** 2
    y_square = x_square.T
    denominator = 2 * sigma ** 2

    result = np.exp(-(x_square + y_square) / denominator)

    return result / result.sum() # Normalization

def Convolve(filter, image):
    filter_size = filter.shape[0]

    pad_size = filter_size // 2
    pad_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')

    image_row, image_col = image.shape

    result = np.zeros((image_row, image_col))

    for i in range(image_row): # Brute force
        for j in range(image_col):
            result[i, j] = (filter * pad_image[i: i + filter_size, j: j + filter_size]).sum()

    return result

def GaussianSmooth(sigma, image, lib = False):
    if not lib:
        filter = GaussianKernel(sigma)

        return Convolve(filter, image)

    else:
        return cv.GaussianBlur(image, (0, 0), sigmaX = sigma, sigmaY = sigma)

def UpSample(image, lib = False):
    if not lib:
        image_row, image_col = image.shape

        result = np.zeros((2 * image_row, 2 * image_col))

        for i in range(image_row - 1):
            for j in range(image_col - 1):
                result[2 * i, 2 * j] = image[i, j]
                result[2 * i + 1, 2 * j] = (image[i, j] + image[i + 1, j]) / 2
                result[2 * i, 2 * j + 1] = (image[i, j] + image[i, j + 1]) / 2
                result[2 * i + 1, 2 * j + 1] = (result[2 * i + 1, 2 * j] + result[2 * i, 2 * j + 1]) / 2

        for i in range(image_row - 1):
            result[2 * i, 2 * image_col - 2] = result[2 * i, 2 * image_col - 1] = image[i, image_col - 1]
            result[2 * i + 1, 2 * image_col - 2] = result[2 * i + 1, 2 * image_col - 1] = (image[i, image_col - 1] + image[i + 1, image_col - 1]) / 2

        for j in range(image_col - 1):
            result[2 * image_row - 2, 2 * j] = result[2 * image_row - 1, 2 * j] = image[image_row - 1, j]
            result[2 * image_row - 2, 2 * j + 1] = result[2 * image_row - 1, 2 * j + 1] = (image[image_row - 1, j] + image[image_row - 1, j + 1]) / 2

        result[2 * image_row - 1, 2 * image_col - 1] = (result[2 * image_row - 1, 2 * image_col - 2] + result[2 * image_row - 2, 2 * image_col - 1]) / 2

        return result

    else:
        return cv.resize(image, (image.shape[1] * 2, image.shape[0] * 2), fx = 0, fy = 0, interpolation = cv.INTER_LINEAR)

def DownSample(image, lib = False):
    if not lib:
        return image[: : 2, : : 2]

    else:
        return cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2), fx = 0, fy = 0, interpolation = cv.INTER_NEAREST)

# Procedures

def GaussianPyramid(O, S, sigma_0, image):
    result = [[None for temp_2 in range(S + 3)] for temp_1 in range(O)]

    K = 2 ** (1 / S)
    value = sigma_0

    sigma = [None for temp in range(S + 3)]
    sigma[0] = np.sqrt(sigma_0 ** 2 - (0.5 * 2) ** 2) # `0.5` from paper (original blur), `2` from paper (image expansion)

    for s in range(1, S + 3):
        sigma[s] = np.sqrt((K * value) ** 2 - value ** 2)

        value *= K

    new_image = UpSample(image, lib)
    new_image = GaussianSmooth(sigma[0], new_image, lib)

    for o in range(O):
        for s in range(S + 3):
            if o == 0 and s == 0:
                result[o][s] = new_image

            elif s == 0:
                result[o][s] = DownSample(result[o - 1][S], lib)

            else:
                result[o][s] = GaussianSmooth(sigma[s], result[o][s - 1], lib)

    return result

def DoGPyramid(O, S, sigma_0, image):
    pyramid = GaussianPyramid(O, S, sigma_0, image.astype(np.float))

    return pyramid, [[pyramid[temp_1][temp_2 + 1] - pyramid[temp_1][temp_2] for temp_2 in range(S + 2)] for temp_1 in range(O)]

def Keypoint(pyramid):
    O = len(pyramid)        # Number of octave
    S = len(pyramid[0]) - 2 # Number of scale

    result = [] # Keypoint

    if arg:
        threshold = np.floor(0.5 * 0.04 / S * 255) # `0.04` from OpenCV (contrast threshold)

    # Extrema detection
    for o in range(O):
        R, C = pyramid[o][0].shape # Number of row and column

        for s in range(1, S + 1):
            for r in range(1, R - 1):
                for c in range(1, C - 1):
                    center = pyramid[o][s][r, c] # Center of detection
                    offset = [-1, 0, 1]          # Offset of detection

                    flag = True # Flag of extrema

                    if arg:
                        if np.abs(center) <= threshold:
                            continue

                    if center >= pyramid[o][s - 1][r, c] and center >= pyramid[o][s + 1][r, c]:
                        if arg:
                            if center <= 0:
                                continue

                        for i in offset:
                            for j in offset:
                                if i != 0 or j != 0:
                                    if center < pyramid[o][s - 1][r + i, c + j] or center < pyramid[o][s][r + i, c + j] or center < pyramid[o][s + 1][r + i, c + j]:
                                        flag = False
                                        break

                            if not flag:
                                break

                        if not flag:
                            continue

                        result.append([o, s, r, c])

                    elif center <= pyramid[o][s - 1][r, c] and center <= pyramid[o][s + 1][r, c]:
                        if arg:
                            if center >= 0:
                                continue

                        for i in offset:
                            for j in offset:
                                if i != 0 or j != 0:
                                    if center > pyramid[o][s - 1][r + i, c + j] or center > pyramid[o][s][r + i, c + j] or center > pyramid[o][s + 1][r + i, c + j]:
                                        flag = False
                                        break

                            if not flag:
                                break

                        if not flag:
                            continue

                        result.append([o, s, r, c])

                    else:
                        continue

    index = 0 # Index of keypoint

    # Detailed fit
    while index < len(result):
        o, s, r, c = result[index]

        R, C = pyramid[o][0].shape # Number of row and column

        count = 0 # Count iteration

        flag = True # Flag of removing keypoint

        delete = [] # Item to be deleted
        
        while count < 5: # `5` from OpenCV (max iteration number)
            count += 1

            dr = (pyramid[o][s][r + 1, c] - pyramid[o][s][r - 1, c]) / (2 * 255)
            dc = (pyramid[o][s][r, c + 1] - pyramid[o][s][r, c - 1]) / (2 * 255)
            ds = (pyramid[o][s + 1][r, c] - pyramid[o][s - 1][r, c]) / (2 * 255)

            D = np.array([dr, dc, ds])

            center = pyramid[o][s][r, c]

            drr = (pyramid[o][s][r + 1, c] + pyramid[o][s][r - 1, c] - 2 * center) / 255
            dcc = (pyramid[o][s][r, c + 1] + pyramid[o][s][r, c - 1] - 2 * center) / 255
            dss = (pyramid[o][s + 1][r, c] + pyramid[o][s - 1][r, c] - 2 * center) / 255

            drc = (pyramid[o][s][r + 1, c + 1] - pyramid[o][s][r - 1, c + 1] - pyramid[o][s][r + 1, c - 1] + pyramid[o][s][r - 1, c - 1]) / (4 * 255)
            drs = (pyramid[o][s + 1][r + 1, c] - pyramid[o][s + 1][r - 1, c] - pyramid[o][s - 1][r + 1, c] + pyramid[o][s - 1][r - 1, c]) / (4 * 255)
            dcs = (pyramid[o][s + 1][r, c + 1] - pyramid[o][s + 1][r, c - 1] - pyramid[o][s - 1][r, c + 1] + pyramid[o][s - 1][r, c - 1]) / (4 * 255)

            H = np.array([[drr + 1e-8, drc, drs], [drc, dcc + 1e-8, dcs], [drs, dcs + 1e-8, dss]]) # Prevent singular

            x_hat = -np.linalg.solve(H, D)

            if (np.abs(x_hat) < 0.5).all():
                for item in delete:
                    result.remove(item)

                D_hat = pyramid[o][s][r, c] / 255 + 1 / 2 * np.matmul(D, x_hat)

                trace = drr + dcc
                determinant = drr * dcc - drc ** 2

                if arg:
                    if np.abs(D_hat) >= 0.04 / S and determinant > 0 and trace ** 2 / determinant < 8: # `0.04` from OpenCV (constrast threshold), `12.1` from paper ((r + 1) ^ 2 / r when r = 10 (r is edge threshold in OpenCV))
                        flag = False
                
                else:
                    if np.abs(D_hat) >= 0.03 and determinant >= 0 and trace ** 2 / determinant <= 12.1: # `0.03` from paper (extrema threshold), `12.1` from paper ((r + 1) ^ 2 / r when r = 10 (r is edge threshold in OpenCV))
                        flag = False

                break

            else:
                if arg:
                    if (np.abs(x_hat) > 2147483647 / 3).any():
                        break

                s += int(np.round(x_hat[2]))
                r += int(np.round(x_hat[0]))
                c += int(np.round(x_hat[1]))
                
                if s < 1 or s > S or r < 1 or r > R - 2 or c < 1 or c > C - 2:
                    break

                else:
                    item = [o, s, r, c]

                    if item in result:
                        position = result.index(item)

                        if position < index:
                            break

                        else:
                            delete.append(item)

                    result[index] = item

        if flag:
            del result[index]

        else:
            index += 1

    return result

def Orientation(pyramid, keypoint, sigma_0):
    S = len(pyramid[0]) - 3 # Number of scale
    K = 2 ** (1 / S)        # Factor of increment

    result = [] # Keypoint with direction

    for o, s, r, c in keypoint:
        R, C = pyramid[o][0].shape # Number of row and column

        sigma = K ** s * sigma_0
        radius = int(np.round(3 * sigma)) # `3` from 3 sigma principle

        histogram = [0 for temp in range(36)]

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                p = r + i # Row index of point with offset i
                q = c + j # Column index of point with offset j

                if p <= 0 or p >= R - 1 or q <= 0 or q >= C - 1:
                    continue

                dp = pyramid[o][s][p + 1, q] - pyramid[o][s][p - 1, q]
                dq = pyramid[o][s][p, q + 1] - pyramid[o][s][p, q - 1]

                weight = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                magnitude = np.sqrt(dp ** 2 + dq ** 2)
                direction = np.arctan2(dq, dp) / np.pi * 180 + 180

                index = int(np.round(direction / 10)) % 36

                histogram[index] += weight * magnitude

        smooth_histogram = []

        for middle in range(36):
            left_2 = middle - 2
            left_1 = middle - 1
            right_1 = (middle + 1) % 36
            right_2 = (middle + 2) % 36

            value = (histogram[left_2] + histogram[right_2]) * (1 / 16) + (histogram[left_1] + histogram[right_1]) * (4 / 16) + histogram[middle] * (6 / 16) # Gaussian smooth

            smooth_histogram.append(value)

        peak = max(smooth_histogram)

        for middle in range(36):
            left = middle - 1
            right = (middle + 1) % 36

            if smooth_histogram[left] < smooth_histogram[middle] > smooth_histogram[right] and smooth_histogram[middle] >= 0.8 * peak: # `0.8` from paper (peak threshold)
                index = middle + 1 / 2 * (smooth_histogram[left] - smooth_histogram[right]) / (smooth_histogram[left] - 2 * smooth_histogram[middle] + smooth_histogram[right]) # Parabolic interpolation

                if index < 0:
                    index += 36

                elif index >= 36:
                    index -= 36

                result.append([o, s, r, c, index * 45])

    return result

def Descriptor(pyramid, keypoint, sigma_0):
    S = len(pyramid[0]) # Number of scale
    K = 2 ** (1 / S)    # Factor of increment

    result = [] # Descriptor

    for o, s, r, c, d in keypoint:
        R, C = pyramid[o][0].shape # Number of row and column

        sigma = K ** s * sigma_0
        radius = int(np.round((4 + 1) * (3 * sigma) * np.sqrt(2) / 2)) # `4` from paper (region number), `3` from OpenCV (region size)

        d = (d - 180) / 180 * np.pi
        sin_d = np.sin(d)
        cos_d = np.cos(d)

        histogram = np.zeros((4, 4, 8)) # `4` from paper (region number)

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                p = r + i # Row index with offset i
                q = c + j # Column index with offset j

                i_rotated_normalized = (cos_d * i - sin_d * j) / (3 * sigma) # `3` from OpenCV (region size)
                j_rotated_normalized = (sin_d * i + cos_d * j) / (3 * sigma) # `3` from OpenCV (region size)

                p_bin = i_rotated_normalized + 4 / 2 - 0.5 # `4` from paper (region number), `0.5` from OpenCV
                q_bin = j_rotated_normalized + 4 / 2 - 0.5 # `4` from paper (region number), `0.5` from OpenCV

                if p <= 0 or p >= R - 1 or q <= 0 or q >= C - 1 or p_bin <= -1 or p_bin >= 4 or q_bin <= -1 or q_bin >= 4: # `4` from paper (region number)
                    continue

                dp = pyramid[o][s][p + 1, q] - pyramid[o][s][p - 1, q]
                dq = pyramid[o][s][p, q + 1] - pyramid[o][s][p, q - 1]

                weight = np.exp(-(i_rotated_normalized ** 2 + j_rotated_normalized ** 2) / (1 / 2 * (4 / 2) ** 2)) # `4` from paper (region number)
                magnitude = np.sqrt(dp ** 2 + dq ** 2)
                direction = (np.arctan2(dq, dp) + d) / np.pi * 180 + 180

                o_bin = direction / 45

                p_bin_index = int(np.floor(p_bin))
                q_bin_index = int(np.floor(q_bin))
                o_bin_index = int(np.floor(o_bin))

                p_bin_offset = p_bin - p_bin_index
                q_bin_offset = q_bin - q_bin_index
                o_bin_offset = o_bin - o_bin_index

                for x in range(2):
                    if p_bin_index + x <= -1 or p_bin_index + x >= 4: # `4` from paper (region number)
                        continue

                    value_1 = magnitude * (1 - p_bin_offset) if x == 0 else magnitude * p_bin_offset

                    for y in range(2):
                        if q_bin_index + y <= -1 or q_bin_index + y >= 4: # `4` from paper (region number)
                            continue

                        value_2 = value_1 * (1 - q_bin_offset) if y == 0 else value_1 * q_bin_offset

                        for z in range(2):
                            value_3 = value_2 * (1 - o_bin_offset) if z == 0 else value_2 * o_bin_offset

                            histogram[p_bin_index + x, q_bin_index + y, (o_bin_index + z) % 8] += value_3

        descriptor = histogram.reshape(-1)

        threshold = 0.2 * descriptor.sum() # `0.2` from paper (descriptor threshold)

        descriptor[(descriptor > threshold)] = threshold
        descriptor = descriptor / descriptor.sum()

        result.append(descriptor)

    return result

# Display

def DrawCircle(p, q, r, c):
    theta = np.arange(0, 2 * np.pi, 1e-3)

    x = p + r * np.cos(theta)
    y = q + r * np.sin(theta)

    plt.scatter(y, x, s = 1e-5, color = c)

def DrawLine(p, q, s, t, c):
    x = [p, s]
    y = [q, t]

    plt.plot(y, x, linewidth = 0.5, color = c)

# Main

if __name__ == '__main__':
    S = 3         # `3` from paper (scale)
    sigma_0 = 1.6 # `1.6` form paper (initial sigma)

    # First image
    image = plt.imread(sys.argv[1])
    image = Gray(image, lib)

    image_row, image_col = image.shape

    O = int(np.log2(min(2 * image_row, 2 * image_col))) - 2 + 1 # `2` from OpenCV

    gaussianPyramid, doGPyramid = DoGPyramid(O, S, sigma_0, image)
    keypoint = Keypoint(doGPyramid)
    keypoint = Orientation(gaussianPyramid, keypoint, sigma_0)
    descriptor = Descriptor(gaussianPyramid, keypoint, sigma_0)

    # Second image, target
    image_t = plt.imread(sys.argv[2])
    image_t = Gray(image_t, lib)

    image_t_row, image_t_col = image_t.shape

    O_t = int(np.log2(min(2 * image_t_row, 2 * image_t_col))) - 2 + 1 # `2` from OpenCV

    gaussianPyramid_t, doGPyramid_t = DoGPyramid(O_t, S, sigma_0, image_t)
    keypoint_t = Keypoint(doGPyramid_t)
    keypoint_t = Orientation(gaussianPyramid_t, keypoint_t, sigma_0)
    descriptor_t = Descriptor(gaussianPyramid_t, keypoint_t, sigma_0)

    # K nearest neighbour
    knc = KNeighborsClassifier(n_neighbors = 1)
    knc.fit(descriptor, [0 for temp in range(len(descriptor))])
    distance, index = knc.kneighbors(descriptor_t, n_neighbors = 1, return_distance = True)

    distance = distance.reshape(-1)
    index = index.reshape(-1)

    index_t = np.argsort(distance)

    index = index[index_t][: 10]
    index_t = index_t[: 10]

    print(distance[index_t])

    # Show match
    row = max(image_row, image_t_row)
    col = max(image_col, image_t_col)
    new_image = np.ones((row, 2 * col)) * 255

    new_image[: image_row, : image_col] = image
    new_image[: image_t_row, col: col + image_t_col] = image_t

    paint = []
    paint_t = []

    K = 2 ** (1 / S)

    for i in index:
        o, s, r, c, d = keypoint[i]
        factor = 2 ** (o - 1)
        sigma = K ** s * sigma_0
        d = (d - 180) / 180 * np.pi
        x, y, r = [int(factor * r), int(factor * c), int(factor * np.round(3 * sigma))]
        a = x + r * np.cos(d)
        b = y + r * np.sin(d)
        paint.append([x, y, r, a, b])

    for i in index_t:
        o, s, r, c, d = keypoint_t[i]
        factor = 2 ** (o - 1)
        sigma = K ** s * sigma_0
        d = (d - 180) / 180 * np.pi
        x, y, r = [int(factor * r), int(factor * c), int(factor * np.round(3 * sigma))]
        a = x + r * np.cos(d)
        b = y + r * np.sin(d)
        paint_t.append([x, y + col, r, a, b + col])

    color = ['r', 'g', 'b']

    count = 0
    for x, y, r, a, b in paint:
        DrawCircle(x, y, r, color[count % 3])
        DrawLine(x, y, a, b, color[count % 3])
        count += 1

    count = 0
    for x, y, r, a, b in paint_t:
        DrawCircle(x, y, r, color[count % 3])
        DrawLine(x, y, a, b, color[count % 3])
        count += 1

    count = 0
    for i in range(10):
        DrawLine(paint[i][0], paint[i][1], paint_t[i][0], paint_t[i][1], color[count % 3])
        count += 1

    plt.show()
