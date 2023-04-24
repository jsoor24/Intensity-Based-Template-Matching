import numpy as np
import os


def white_to_black(img):
    for m in range(0, img.shape[0]):
        for n in range(0, img.shape[1]):
            if img[m, n] >= 240:
                img[m, n] = 0
    return img


def get_object_name(file):
    return file[4:len(file) - 4]


def check_templates(rot_file, sca_file, templates, training, rotations, octaves):
    if not os.path.exists(rot_file) or not os.path.exists(sca_file):
        return False

    if len(os.listdir(templates)) - 2 != len(os.listdir(training)):
        return False

    if rotations == np.load(rot_file, allow_pickle=True) \
            and octaves == np.load(sca_file, allow_pickle=True):
        return True

    return False


def get_scale_percentage(depth, n=100.0):
    if depth == 0:
        return n

    return get_scale_percentage(depth - 1, n / 2)


def create_gaussian_pyramid_image(img, result):
    comp_rows = max(img.shape[0], sum(p.shape[0] for _, p in result))
    comp_cols = img.shape[0] + result[0][1].shape[1]
    comp_img = np.full((comp_rows, comp_cols), 255, dtype=np.int64)

    comp_img[:img.shape[0], :img.shape[0]] = img

    i_row = 0

    for _, p in result:
        n_rows, n_cols = p.shape
        comp_img[i_row:i_row + n_rows, img.shape[0]:img.shape[0] + n_cols] = p
        i_row += n_rows

    return comp_img


# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
# Used StackOverflow post to understand out how to compute intersection over union
def calculate_iou(tlA, brA, tlB, brB):
    """
    Calculates the intersection over union of two bounding boxes
    One of the bounding boxes is the result from template matching
    The other is the ground truth from the annotations
    :param tlA: Top left of BB A
    :param brA: Bottom right of BB A
    :param tlB: Top left of BB B
    :param brB: Bottom right of BB B
    :return: The IoU score
    """

    # Calculate the coordinates of the intersection box
    x_left = max(tlA[0], tlB[0])
    y_top = max(tlA[1], tlB[1])
    x_right = min(brA[0], brB[0])
    y_bottom = min(brA[1], brB[1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    areaA = (brA[0] - tlA[0]) * (brA[1] - tlA[1])
    areaB = (brB[0] - tlB[0]) * (brB[1] - tlB[1])

    iou = intersection_area / float(areaA + areaB - intersection_area)
    return iou


def to_int(var):
    """
    Takes the string "(123, 456)"
    and returns the tuple (123, 456)
    :param var: String to parse to integer tuple
    :return: The integer tuple
    """
    var = var.replace("(", "")
    var = var.replace(")", "")
    return tuple([int(n) for n in var.split(", ")])
