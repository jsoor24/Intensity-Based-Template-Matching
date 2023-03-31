import pickle
import cv2 as cv
import matplotlib
import numpy as np
import imutils
import matplotlib.pyplot as plt
from helpers import *

TRAINING_FOLDER = "Task2Dataset/Training/png/"
TEST_IMAGES_FOLDER = "Task2Dataset/TestWithoutRotations/images/"
TEST_ANNOTATIONS_FOLDER = "Task2Dataset/TestWithoutRotations/annotations/"

OCTAVES = [1, 2, 3, 4]


def white_to_black(img):
    for m in range(0, img.shape[0]):
        for n in range(0, img.shape[1]):
            if img[m, n] == 255:
                img[m, n] = 0
    return img


def create_gaussian_pyramid(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = white_to_black(img)

    result = []
    previous = img
    maximum = max(OCTAVES)

    for o in range(0, maximum + 1):
        # Apply Gaussian filter
        blurred = cv.GaussianBlur(previous, [5, 5], 1)

        # Calculate dimensions of sub-sampled image
        h = blurred.shape[0]
        w = blurred.shape[1]
        scaled = np.zeros((int(h / 2), int(w / 2)), dtype=np.uint8)

        # Sub-sample
        i = 0
        for m in range(0, h, 2):
            j = 0
            for n in range(0, w, 2):
                scaled[i, j] = blurred[m, n]
                j += 1
            i += 1

        if o in OCTAVES:
            result.append((get_scale_percentage(o), scaled))
        previous = scaled

    return result


def rotations(images):
    rots = []
    for img in images:
        for angle in range(0, 360, 90):
            rots.append(imutils.rotate(img, angle))
    return rots


scaled_pyramid = create_gaussian_pyramid(TRAINING_FOLDER + "029-theater.png")

for percentage, img in scaled_pyramid:
    file = open("{}test.dat".format(percentage), 'wb')
    pickle.dump(img, file)
    file.close()

# rotations = rotations(scaled_pyramid)
test = cv.imread(TEST_IMAGES_FOLDER + "test_image_1.png", cv.IMREAD_GRAYSCALE)
test = cv.GaussianBlur(test, [5, 5], 1)

method = eval('cv.TM_CCOEFF_NORMED')

best_val = 0
best_loc = 0
b_w = 0
b_h = 0

for percentage, r in scaled_pyramid:
    template = np.load("{}test.dat".format(percentage), mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
    w, h = template.shape[::-1]

    result = cv.matchTemplate(test, template, method)
    # print(result)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    print("{}%: {}".format(percentage, max_val))

    if percentage == 6.25:
        max_val = max_val * 0.5
    if percentage == 12.5:
        max_val = max_val * 0.6
    if percentage == 25:
        max_val = max_val * 0.7
    if percentage == 50:
        max_val = max_val * 0.7

    print("{}%: {}".format(percentage, max_val))

    if max_val > best_val:
        best_val = max_val
        best_loc = max_loc
        b_w = w
        b_h = h

top_left = best_loc
bottom_right = (top_left[0] + b_w, top_left[1] + b_h)
print(best_val)

cv.rectangle(test, top_left, bottom_right, 0, 2)
plt.imshow(test, cmap='gray')

plt.show()

