import cv2
import numpy as np
import glob
import time
import copy
import os
from statistics import mean
import Process_data_pupil_detection as Process

# constants
fast_eye_width = 50
threshold_gradient = 50
weight_blur_size = 5
post_process_threshold = 0.97

# path = "Videos/Video1/Calibration/"
# p2 = "LeftEyes/"


def init_image_corners(path: str, pos: int):
    # load left upper corner of the images
    upper_left_corners = np.loadtxt(path + "posEyes.csv", delimiter=';', skiprows=2)[:, [pos * 2, pos * 2 + 1]]
    upper_left_corners_x = upper_left_corners[:, 0]
    upper_left_corners_y = upper_left_corners[:, 1]
    return upper_left_corners_x, upper_left_corners_y


def init_inner_eye_corners(path: str, pos: int):
    # load coordinates inner eye corner
    inner_eye_corners = np.loadtxt(path + "eyeCorners.csv", delimiter=';', skiprows=2)[:, [pos * 4 + 2, pos * 4 + 2 + 1]]
    inner_eye_corners_x = inner_eye_corners[:, 0]
    inner_eye_corners_y = inner_eye_corners[:, 1]
    return inner_eye_corners_x, inner_eye_corners_y


class Rect:

    def __init__(self, upper_left_corner_x: int, upper_left_corner_y: int, width: int, height: int):
        self.x = upper_left_corner_x
        self.y = upper_left_corner_y
        self.w = width
        self.h = height

    def get_x(self) -> int:
        return self.x

    def get_y(self) -> int:
        return self.y

    def get_width(self) -> int:
        return self.w

    def get_height(self) -> int:
        return self.h


def detect_pupil_gradient_descent(path1: str, path2: str, modus: str, eye: str, pos: int, video: int) -> (list, list):
    # define list to save the vectors between the eye corner and the pupil
    data_x = list()
    data_y = list()
    vectors = list()
    vectors.append("X;Y")

    # init lists to save images
    global imgs_gr
    imgs_gr = list()
    global imgs_weights
    imgs_weights = list()
    global imgs_products
    imgs_products = list()
    global imgs_thresh_pr
    imgs_thresh_pr = list()
    global imgs_mask
    imgs_mask = list()
    global imgs_pupil
    imgs_pupil = list()

    path = path1 + path2 + modus + "/"
    eye2 = eye + "/"
    inner_eye_corners_x, inner_eye_corners_y = init_inner_eye_corners(path, pos)
    upper_left_corners_x, upper_left_corners_y = init_image_corners(path, pos)

    img = glob.glob(path + eye2 + "*.jpg")
    img.sort(key=len)
    img = [cv2.imread(i, 0) for i in img]

    duration = list()

    index = 0
    for im in img:
        start_time = time.time()

        center = find_eye_center(im)
        e = copy.deepcopy(im)
        if center is not None:
            cv2.circle(e, center, 1, (255, 255, 255))

            if inner_eye_corners_x[index] == 0 and inner_eye_corners_y[index] == 0:
                # blinking eye
                vx = 0
                vy = 0
            else:
                vx = center[0] + upper_left_corners_x[index] - inner_eye_corners_x[index]
                vy = center[1] + upper_left_corners_y[index] - inner_eye_corners_y[index]

            vectors.append(str(vx) + ";" + str(vy))
            data_x.append(vx)
            data_y.append(vy)
        else:
            vectors.append(str(0) + ";" + str(0))
            data_x.append(0)
            data_y.append(0)

        cv2.imshow("eye", e)
        imgs_pupil.append(e)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
        index += 1

    end_time = time.time()
    duration.append(end_time - start_time)

    data_v = (data_x, data_y)

    create_dir(path + eye2, "Method_3")
    s = path + eye2 + "Method_3/"
    create_dir(s, "Gradients")
    create_dir(s, "Weights")
    create_dir(s, "Dotproducts")
    create_dir(s, "Threshold_dotproducts")
    create_dir(s, "Mask")
    create_dir(s, "Pupil")

    # write the vectors to a file
    f = open(path + eye2 + "Method_3/vectors.csv", "w+")
    i = -1
    for v in vectors:
        f.write(str(v) + "\n")
        if i > 0:
            cv2.imwrite(path + eye2 + "Method_3/Gradients/im_" + str(i) + ".jpg", imgs_gr.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_3/Weights/im_" + str(i) + ".jpg", imgs_weights.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_3/Dotproducts/im_" + str(i) + ".jpg", imgs_products.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_3/Threshold_dotproducts/im_" + str(i) + ".jpg", imgs_thresh_pr.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_3/Mask/im_" + str(i) + ".jpg", imgs_mask.__getitem__(i))
            cv2.imwrite(path + eye2 + "Method_3/Pupil/im_" + str(i) + ".jpg", imgs_pupil.__getitem__(i))
        i += 1
    f.close()
    cv2.destroyAllWindows()

    return Process.process_data(data_v, path, "pos.csv", eye, "METHOD3", modus, video, mean(duration))


def find_eye_center(eye: np.ndarray) -> (int, int):
    eye_unscaled = eye
    original_size = Rect(0, 0, eye_unscaled.shape[1], eye_unscaled.shape[0])
    eye_roi = scale2fast_size(eye_unscaled)

    # Debug
    # cv2.imshow(name_debug_window, eye_roi)

    # Find the gradients
    gradients_x = compute_mat_x_gradient(eye_roi)
    gradients_y = compute_mat_x_gradient(eye_roi.transpose()).transpose()

    # Normalize and threshold the gradients
    # Compute all magnitudes
    mags = matrix_magnitude(gradients_x, gradients_y)
    # print(mags.dtype)
    # Compute the threshold
    gradient_threshold = compute_dynamic_threshold(mags, threshold_gradient)
    # Normalize
    for i in range(eye_roi.shape[0]):
        for j in range(eye_roi.shape[1]):
            g_x = gradients_x[i, j]
            g_y = gradients_y[i, j]
            magnitude = mags[i, j]
            if magnitude > gradient_threshold:
                gradients_x[i, j] = g_x/magnitude
                gradients_y[i, j] = g_y/magnitude
            else:
                gradients_x[i, j] = 0
                gradients_y[i, j] = 0

    imgs_gr.append(gradients_x)

    # Create a blurred and inverted image for weighting
    weights = cv2.GaussianBlur(eye_roi, ksize=(weight_blur_size, weight_blur_size), sigmaX=0, sigmaY=0)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = 255 - weights[i, j]
    imgs_weights.append(weights)

    # t1 = time.time()
    # Run the algorithm
    # evaluate every possible gradient location for every center
    dx, dy = (np.zeros(weights.shape), np.zeros(weights.shape))
    for i in range(dx.shape[1]):
        dx[:, i] = -i
    for j in range(dy.shape[0]):
        dy[j, :] = -j

    out_sum = np.zeros(weights.shape)
    for y in range(weights.shape[0]):
        dy_n = dy + y
        for x in range(weights.shape[1]):
            dx_n = dx + x
            m = matrix_magnitude(dx_n, dy_n)
            dx_n = np.where(m > 0, np.divide(dx_n, m), 0)
            dy_n = np.where(m > 0, np.divide(dy_n, m), 0)

            dot_products = weights[y, x]*((dx_n * gradients_x + dy_n * gradients_y) * (dx_n * gradients_x + dy_n * gradients_y))

            out_sum[y, x] = np.sum(dot_products)


    # # Scale all the values down
    # num_gradients = np.size(weights)
    # if out_sum is not None:
    #     out_sum = np.float32(out_sum)/num_gradients
    # else:
    #     # print("No eye")
    #     return None
    imgs_products.append(out_sum)

    # Find the maximum point
    _, max_value, _, max_loc = cv2.minMaxLoc(out_sum)

    # Flood fill the edges
    flood_threshold = max_value * post_process_threshold
    _, flood_clone = cv2.threshold(out_sum, flood_threshold, 0.0, cv2.THRESH_TOZERO)
    imgs_thresh_pr.append(flood_clone)

    mask = flood_kill_edges(flood_clone)
    imgs_mask.append(mask)

    _, max_value, _, max_loc = cv2.minMaxLoc(out_sum, mask)

    return unscale_point(max_loc, original_size)


def scale2fast_size(unscaled_eye: np.ndarray) -> np.ndarray:
    rows, cols = unscaled_eye.shape
    scaled = cv2.resize(unscaled_eye, dsize=(fast_eye_width, round((fast_eye_width/cols) * rows)))
    return scaled


def unscale_point(loc: tuple, size: Rect) -> tuple:
    ratio = fast_eye_width/size.get_width()
    x = round(loc[0]/ratio)
    y = round(loc[1]/ratio)
    return x, y


def compute_mat_x_gradient(mat: np.ndarray) -> np.ndarray:
    rows = mat.shape[0]
    cols = mat.shape[1]
    mat = np.float64(mat)
    gradient_mat = np.float64(np.zeros((rows, cols)))

    for i in range(rows):
        gradient_mat[i, 0] = mat[i, 1] - mat[i, 0]
        for j in range(1, cols - 1):
            gradient_mat[i, j] = (mat[i, j+1] - mat[i, j-1])/2
        gradient_mat[i, cols - 1] = mat[i, cols - 1] - mat[i, cols - 2]
    return gradient_mat


def matrix_magnitude(x_gradients: np.ndarray, y_gradients: np.ndarray) -> np.ndarray:
    m = np.sqrt((x_gradients * x_gradients) + (y_gradients * y_gradients))
    return m


def compute_dynamic_threshold(magnitudes: np.ndarray, threshold: int):
    mean_magnitude, std_magnitude = cv2.meanStdDev(magnitudes)
    std_dev = std_magnitude[0]/(np.sqrt(magnitudes.shape[0]*magnitudes.shape[1]))
    thresh = threshold * std_dev + mean_magnitude[0]
    return thresh


# def test_possible_centers(x: int, y: int, weights: np.ndarray, gradient_x: np.float64, gradient_y: np.float64) \
#         -> np.ndarray:
#     out = np.float64(np.zeros((weights.shape[0], weights.shape[1])))
#     # For all possible centers
#     for cy in range(out.shape[0]):
#         for cx in range(out.shape[1]):
#             if cy == y and cx == x:
#                 continue
#             else:
#                 # Create vector between possible center and gradient origin
#                 dx = x - cx
#                 dy = y - cy
#
#                 # Normalize the vector
#                 magnitude = np.sqrt(np.square(dx) + np.square(dy))
#                 dx = dx/magnitude
#                 dy = dy/magnitude
#
#                 dot_product = dx * gradient_x + dy * gradient_y
#                 dot_product = max(0.0, dot_product)
#
#                 # Square and multiply by hte weight
#                 out[cy, cx] = out[cy, cx] + np.square(dot_product) * weights[cy, cx]
#     return out


def flood_kill_edges(mat: np.ndarray) -> np.ndarray:
    cv2.rectangle(mat, (0, 0), (mat.shape[1], mat.shape[0]), 255)
    mask = np.uint8(np.ones((mat.shape[0], mat.shape[1]))) * 255
    to_do = list()
    to_do.append((0, 0))
    while to_do.__len__() > 0:
        p = to_do.__getitem__(0)
        to_do.__delitem__(0)
        if mat[p[1], p[0]] == 0.0:
            continue
        # Add in every direction
        # Right
        new_p = (p[0] + 1, p[1])
        if in_bound(new_p, mat):
            to_do.append(new_p)
        # Left
        new_p = (p[0] - 1, p[1])
        if in_bound(new_p, mat):
            to_do.append(new_p)
        # Down
        new_p = (p[0], p[1] + 1)
        if in_bound(new_p, mat):
            to_do.append(new_p)
        # Up
        new_p = (p[0], p[1] - 1)
        if in_bound(new_p, mat):
            to_do.append(new_p)

        # Kill it
        mat[p[1], p[0]] = 0.0
        mask[p[1], p[0]] = 0
    return mask


def in_bound(p: tuple, mat: np.ndarray) -> bool:
    if p[0] < 0 or p[0] >= mat.shape[1]:
        return False
    elif p[1] < 0 or p[1] >= mat.shape[0]:
        return False
    else:
        return True


def create_dir(path1: str, d1: str):
    p2 = path1 + d1
    d2 = os.path.join(p2)
    if not os.path.exists(d2):
        os.mkdir(d2)
