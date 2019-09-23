import cv2
import numpy as np
import glob
import time

# constants
fast_eye_width = 28
threshold_gradient = 50
weight_blur_size = 5
post_process_threshold = 0.97


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


def main():
    # eye = cv2.imread("Videos/Video1/Calibration/LeftEyes/0.jpg", cv2.IMREAD_GRAYSCALE)
    img = glob.glob("Videos/Video1/Calibration/LeftEyes/*.jpg")
    img.sort(key=len)
    img = [cv2.imread(i, 0) for i in img]
    for eye in img:
        center = find_eye_center(eye, "debug")
        if center is not None:
            cv2.circle(eye, center, 1, (255, 255, 255))
        cv2.imshow("eye", eye)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    return 0


def find_eye_center(eye: np.ndarray, name_debug_window: str) -> (int, int):
    eye_unscaled = eye
    original_size = Rect(0, 0, eye_unscaled.shape[1], eye_unscaled.shape[0])
    eye_roi = scale2fast_size(eye_unscaled)

    # Debug
    #cv2.imshow(name_debug_window, eye_roi)

    # Find the gradients
    gradients_x = compute_mat_x_gradient(eye_roi)
    gradients_y = compute_mat_x_gradient(eye_roi.transpose()).transpose()
    # print(gradients_x.dtype)

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

    # debug
    # cv2.imshow(name_debug_window, gradients_x)

    # Create a blurred and inverted image for weighting
    weights = cv2.GaussianBlur(eye_roi, ksize=(weight_blur_size, weight_blur_size), sigmaX=0, sigmaY=0)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = 255 - weights[i, j]

    t1 = time.time()
    # Run the algorithm
    # evaluate every possible gradient location for every center
    out_sum = None
    for y in range(weights.shape[0]):
        for x in range(weights.shape[1]):
            if gradients_x[y, x] == 0.0 and gradients_y[y, x] == 0.0:
                continue
            else:
                out_sum = test_possible_centers(x, y, weights, gradients_x[y, x], gradients_y[y, x])

    t2 = time.time()
    print(t2-t1)
    # Scale all the values down
    num_gradients = np.size(weights)
    if out_sum is not None:
        out_sum = np.float32(out_sum)/num_gradients
    else:
        print("No eye")
        return None
    # debug
    # cv2.imshow(name_debug_window, out_sum)

    # Find the maximum point
    _, max_value, _, max_loc = cv2.minMaxLoc(out_sum)

    # Flood fill the edges
    flood_threshold = max_value * post_process_threshold
    _, flood_clone = cv2.threshold(out_sum, flood_threshold, 0.0, cv2.THRESH_TOZERO)
    mask = flood_kill_edges(flood_clone)
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
    rows = x_gradients.shape[0]
    cols = x_gradients.shape[1]
    magnitudes = np.float64(np.zeros((rows, cols)))
    for i in range(rows):
        for j in range(cols):
            g_x = x_gradients[i, j]
            g_y = y_gradients[i, j]
            magnitude = np.sqrt(np.square(g_x) + np.square(g_y))
            magnitudes[i, j] = magnitude
    return magnitudes


def compute_dynamic_threshold(magnitudes: np.ndarray, threshold: int):
    mean_magnitude, std_magnitude = cv2.meanStdDev(magnitudes)
    std_dev = std_magnitude[0]/(np.sqrt(magnitudes.shape[0]*magnitudes.shape[1]))
    thresh = threshold * std_dev + mean_magnitude[0]
    return thresh


def test_possible_centers(x: int, y: int, weights: np.ndarray, gradient_x: np.float64, gradient_y: np.float64) \
        -> np.ndarray:
    out = np.float64(np.zeros((weights.shape[0], weights.shape[1])))
    # For all possible centers
    for cy in range(out.shape[0]):
        for cx in range(out.shape[1]):
            if cy == y and cx == x:
                continue
            else:
                # Create vector between possible center and gradient origin
                dx = x - cx
                dy = y - cy

                # Normalize the vector
                magnitude = np.sqrt(np.square(dx) + np.square(dy))
                dx = dx/magnitude
                dy = dy/magnitude

                dot_product = dx * gradient_x + dy * gradient_y
                dot_product = max(0.0, dot_product)

                # Square and multiply by hte weight
                out[cy, cx] = out[cy, cx] + np.square(dot_product) * weights[cy, cx]
    return out


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


if __name__ == "__main__":
    main()
