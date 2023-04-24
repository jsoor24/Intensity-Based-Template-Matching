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
