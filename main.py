import pickle
import cv2 as cv
import matplotlib
import numpy as np
import imutils
import matplotlib.pyplot as plt
import os
from helpers import *

TRAINING_FOLDER = "Task2Dataset/Training/png/"
TEST_IMAGES_FOLDER = "Task2Dataset/TestWithoutRotations/images/"
TEST_ANNOTATIONS_FOLDER = "Task2Dataset/TestWithoutRotations/annotations/"
ROT_FILE = "templates/rotations.pkl"
SCA_FILE = "templates/scales.pkl"

OCTAVES = [1, 2, 3, 4]
ROTATIONS = [0, 90, 180, 270]


def create_gaussian_pyramid(img):
    result = []
    previous = img
    maximum = max(OCTAVES)

    for o in range(0, maximum):
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


def generate_templates():
    if os.path.exists(ROT_FILE) and os.path.exists(SCA_FILE):
        if ROTATIONS == np.load(ROT_FILE, allow_pickle=True):
            if OCTAVES == np.load(SCA_FILE, allow_pickle=True):
                print("Already have templates")
                return

    for file in os.listdir(TRAINING_FOLDER):
        # Read the image, grayscale the image then fill the background with black
        image = cv.imread(TRAINING_FOLDER + file)
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = white_to_black(image)

        object_name = get_object_name(file)

        # Create the gaussian pyramid and rotations for each of the scaled images
        pyramid = create_gaussian_pyramid(image)

        # Check if there is already a folder for templates for this object
        if not os.path.exists("templates/"):
            os.makedirs("templates/")

        dictionary = {"object_name": object_name}

        for scale, scaled in pyramid:
            rotations = dict()
            for r in ROTATIONS:
                rotations[r] = imutils.rotate(scaled, r)
            dictionary[scale] = rotations

        if object_name == "gas-station":
            print(dictionary)

        with open("templates/{}.pkl".format(object_name), 'wb') as f:
            pickle.dump(dictionary, f)

        # Write all the scaled/rotated templates to files
        # for scale, scaled in pyramid:
        #     for r in ROTATIONS:
        #         rotated = imutils.rotate(scaled, r)
        #         output = open("templates/{}/r{}-s{}.dat".format(object_name, r, scale), 'wb')
        #         pickle.dump(rotated, output)
        #         output.close()

    output = open(ROT_FILE, 'wb')
    pickle.dump(ROTATIONS, output)
    output.close()

    output = open(SCA_FILE, 'wb')
    pickle.dump(OCTAVES, output)
    output.close()


generate_templates()
