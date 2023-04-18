import pickle
import cv2 as cv
import matplotlib
import imutils
import matplotlib.pyplot as plt
from helpers import *

TRAINING_FOLDER = "Task2Dataset/Training/png/"
TEST_IMAGES_FOLDER = "Task2Dataset/TestWithoutRotations/images/"
TEST_ANNOTATIONS_FOLDER = "Task2Dataset/TestWithoutRotations/annotations/"
TEMPLATES_FOLDER = "templates/"
ROT_FILE = TEMPLATES_FOLDER + "rotations.pkl"
SCA_FILE = TEMPLATES_FOLDER + "scales.pkl"

OCTAVES = [1, 2, 3, 4]
ROTATIONS = [0, 90, 180, 270]


def create_gaussian_pyramid(img):
    result = []
    previous = img
    maximum = max(OCTAVES)

    for o in range(0, maximum + 1):
        # Apply Gaussian filter
        blurred = cv.GaussianBlur(previous, [5, 5], 0.5)

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
    if check_templates(ROT_FILE, SCA_FILE, TEMPLATES_FOLDER, TRAINING_FOLDER, ROTATIONS, OCTAVES):
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
        if not os.path.exists(TEMPLATES_FOLDER):
            os.makedirs(TEMPLATES_FOLDER)

        dictionary = {"object_name": object_name}

        for scale, scaled in pyramid:
            rotations = dict()
            for r in ROTATIONS:
                rotations[r] = imutils.rotate(scaled, r)
            dictionary[scale] = rotations

        with open("{}{}.pkl".format(TEMPLATES_FOLDER, object_name), 'wb') as f:
            pickle.dump(dictionary, f)

        # Write all the scaled/rotated templates to files
        # for scale, scaled in pyramid:
        #     for r in ROTATIONS:
        #         rotated = imutils.rotate(scaled, r)
        #         output = open("{}{}/r{}-s{}.dat".format(TEMPLATES_FOLDER, object_name, r, scale), 'wb')
        #         pickle.dump(rotated, output)
        #         output.close()

    output = open(ROT_FILE, 'wb')
    pickle.dump(ROTATIONS, output)
    output.close()

    output = open(SCA_FILE, 'wb')
    pickle.dump(OCTAVES, output)
    output.close()


def template_matching(path="test_image_1.png"):
    test = cv.imread(TEST_IMAGES_FOLDER + path, cv.IMREAD_GRAYSCALE)
    test = cv.GaussianBlur(test, [9, 9], 1)
    test = white_to_black(test)

    for file in os.listdir(TEMPLATES_FOLDER):
        if file in ROT_FILE or file in SCA_FILE:
            continue

        with open("{}{}".format(TEMPLATES_FOLDER, file), 'rb') as f:
            dictionary = pickle.load(f)

        best_val = 0
        best_loc = 0
        b_h = 0
        b_w = 0

        for o in OCTAVES:
            scale = get_scale_percentage(o)

            rotations = dictionary[scale]

            for r, template in rotations.items():
                w, h = template.shape[::-1]

                # cv.TM_SQDIFF_NORMED
                # cv.TM_CCORR_NORMED
                result = cv.matchTemplate(test, template, cv.TM_CCORR_NORMED)

                _, max_val, _, max_loc = cv.minMaxLoc(result)

                if scale == 6.25:
                    max_val = max_val * 0.5
                elif scale == 12.5:
                    max_val = max_val * 0.6
                elif scale == 25:
                    max_val = max_val * 0.7
                elif scale == 50:
                    max_val = max_val * 0.7
                else:
                    max_val = max_val * 0.4

                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    b_h = h
                    b_w = w

                # if dictionary["object_name"] == "windmill":
                #     plt.imshow(template, cmap='gray')
                #     plt.show()

        if best_val > 0.64:
            print("{}:\t\t\t{}".format(dictionary["object_name"], best_val))

        # if dictionary["object_name"] == "gas-station":
        #     cv.rectangle(test, top_left, bottom_right, 255, 2)
        #     plt.imshow(test, cmap='gray')
        #     plt.show()

        if best_val > 0.64:
            top_left = best_loc
            bottom_right = (top_left[0] + b_w, top_left[1] + b_h)
            cv.rectangle(test, top_left, bottom_right, 255, 2)

    plt.imshow(test, cmap='gray')
    plt.show()


generate_templates()

for i in range(1, 21):
    template_matching("test_image_{}.png".format(i))