import pickle
import re
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
from helpers import *
import time

# Where to find the training images
TRAINING_FOLDER = "Task2Dataset/Training/png/"

# Where to find the test images
# TEST_IMAGES_FOLDER = "Task2Dataset/TestWithoutRotations/images/"
TEST_IMAGES_FOLDER = "Task3Dataset/images/"

# Where to find the answers
# TEST_ANNOTATIONS_FOLDER = "Task2Dataset/TestWithoutRotations/annotations/"
TEST_ANNOTATIONS_FOLDER = "Task3Dataset/annotations/"
# ANNOTATIONS_FILE_EXTENSION = ".txt"
ANNOTATIONS_FILE_EXTENSION = ".csv"

# Where to write the templates
TEMPLATES_FOLDER = "templates/"
ROT_FILE = TEMPLATES_FOLDER + "rotations.pkl"
SCA_FILE = TEMPLATES_FOLDER + "scales.pkl"

OCTAVES = [1, 2, 3]
ROTATIONS = [0]

metrics = {}


def main():
    t = generate_templates()
    test_template_matching(t)


def create_gaussian_pyramid(img):
    """
    Takes an image and creates a Gaussian pyramid by blurring the
    image and then down-sampling by a factor of 2. Number of octaves
    determined by OCTAVES
    :param img: The image to create the Gaussian pyramid from
    :return: The Gaussian pyramid as a list of tuples (scale %, image)
    """

    result = []
    previous = img
    maximum = max(OCTAVES)

    for o in range(1, maximum + 1):
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

    # pyramid = create_gaussian_pyramid_image(img, result)
    # cv.imwrite("pyramidImage.jpg", pyramid)
    # plt.imshow(pyramid, cmap='gray')
    # plt.show()
    # plt.close()
    return result


def generate_templates():
    """
    Goes through all the provided template images and checks if we've already
    trained with the current parameters (OCTAVES/ROTATIONS).
    If not, for each file:
     - create Gaussian pyramid
     - write all rotations for each scale to a dictionary
     - write dictionary to file
    And write two metadata files holding OCTAVES/ROTATIONS
    :return: ---
    """

    if check_templates(ROT_FILE, SCA_FILE, TEMPLATES_FOLDER, TRAINING_FOLDER, ROTATIONS, OCTAVES):
        print("Already have templates")
        return
    print("Generating templates...")
    start_time = time.time()

    for file in os.listdir(TRAINING_FOLDER):
        # Read the image, grayscale the image then fill the background with black
        image = cv.imread(TRAINING_FOLDER + file, cv.IMREAD_GRAYSCALE)
        # image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
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

    output = open(ROT_FILE, 'wb')
    pickle.dump(ROTATIONS, output)
    output.close()

    output = open(SCA_FILE, 'wb')
    pickle.dump(OCTAVES, output)
    output.close()

    total_time = time.time() - start_time
    print("Done in {:.2f}s".format(total_time))
    return total_time


def template_matching(path="test_image_1.png", method='cv.TM_CCORR_NORMED', cutoff=0.64):
    """
    For each scaled/rotated template perform library template matching
    Scale the score if the image is smaller (to give lower score
    to smaller templates)
    Record the best template and add it to a list
    :param path: The test image to perform template matching on
    :param method: Method to use in library template matching
    :param cutoff: The lower-bound of the value for a match
    :return: List of templates that matched as tuples:
    (object name, matching score, top left, bottom right)
    """

    test = cv.imread(TEST_IMAGES_FOLDER + path, cv.IMREAD_GRAYSCALE)
    # test = cv.GaussianBlur(test, [9, 9], 1)
    test = white_to_black(test)

    matches = []

    for file in os.listdir(TEMPLATES_FOLDER):
        if file in ROT_FILE or file in SCA_FILE:
            continue

        with open("{}{}".format(TEMPLATES_FOLDER, file), 'rb') as f:
            dictionary = pickle.load(f)

        best_val = 0
        best_loc = 0
        b_h = 0
        b_w = 0
        b_octave = 0
        b_rotation = 0

        for o in OCTAVES:
            scale = get_scale_percentage(o)

            rotations = dictionary[scale]

            for r, template in rotations.items():
                w, h = template.shape[::-1]

                # Perform library template matching
                result = cv.matchTemplate(test, template, eval(method))

                # Gets information about the best match
                _, max_val, _, max_loc = cv.minMaxLoc(result)

                # Scales the score so smaller images are less likely to be chosen
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

                # Store the match if it's better than previous matches
                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    b_h = h
                    b_w = w
                    b_octave = o
                    b_rotation = r

                # if dictionary["object_name"] == "windmill":
                #     plt.imshow(template, cmap='gray')
                #     plt.show()

        # Filter out low scoring matches
        if best_val > cutoff:
            top_left = best_loc
            bottom_right = (top_left[0] + b_w, top_left[1] + b_h)
            similarity = histogram_matching(test[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]],
                                            dictionary[get_scale_percentage(b_octave)][b_rotation])
            if similarity > 0.9:
                matches.append((dictionary["object_name"], best_val, top_left, bottom_right))

    return matches


def histogram_matching(test_region, template):
    # plt.subplot(121), plt.imshow(test_img, cmap='gray')
    # plt.subplot(122), plt.imshow(template, cmap='gray')
    # plt.show()
    # plt.close()

    template_hist = cv.calcHist([template], [0], None, [256], [0, 256])
    img_hist = cv.calcHist([test_region], [0], None, [256], [0, 256])
    similarity = cv.compareHist(template_hist, img_hist, cv.HISTCMP_CORREL)

    return similarity


def test_template_matching(t):
    """
    Performs template matching on all the available test images
    Note: Assumes test files are named "test_image_i.png"
    :return: ---
    """
    # methods = [('cv.TM_CCOEFF_NORMED', 0.51), ('cv.TM_CCORR_NORMED', 0.625)]
    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    # methods = [('cv.TM_CCORR_NORMED', 0.625)]
    # methods = [('cv.TM_CCORR_NORMED', 0.55)]

    methods = [('cv.TM_CCOEFF_NORMED', 0.48)]

    print("Testing template matching...")

    for m, c in methods:
        final_results = {}
        answers = {}

        total_time = total_icons = total_iou = incorrect_val = correct_val = incorrect = correct = 0

        # Loop through all test images
        # for i in [10]:
        for i in range(1, 21):
            start_time = time.time()
            test_img_path = "test_image_{}.png".format(i)
            annotation = "{}test_image_{}{}".format(TEST_ANNOTATIONS_FOLDER, i, ANNOTATIONS_FILE_EXTENSION)
            final_results[test_img_path] = template_matching(test_img_path, m, c)
            test_img = cv.imread(TEST_IMAGES_FOLDER + test_img_path)

            # print("\n" + test_img + "\n\tAnswers:")

            # Read in the ground truth answers
            with open(annotation, 'r') as reader:
                answers[test_img_path] = []
                for line in sorted(reader.readlines()):
                    total_icons += 1
                    name, tl, br = re.split(r", (?=\()", line.rstrip())
                    answers[test_img_path].append((name, tl, br))
                    # print("\t\t{} ->\t{}, {}".format(name, tl, br))

            # print("\tResults:")

            # Parse the results from the template matching
            for name, val, top_l, bot_r in final_results[test_img_path]:
                # See if the result is in the answers
                match = [item for item in answers[test_img_path] if item[0] == name]

                # No match
                if len(match) == 0:
                    incorrect += 1
                    incorrect_val += val
                    cv.rectangle(test_img, top_l, bot_r, [0, 0, 255], 2)
                    cv.putText(test_img, name, (bot_r[0] - 10, bot_r[1] + 10), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                               [0, 0, 255])
                    continue

                # Match
                match = match[0]
                iou = calculate_iou(top_l, bot_r, to_int(match[1]), to_int(match[2]))

                # Incorrect
                if iou < 0.5:
                    incorrect += 1
                    incorrect_val += val
                    cv.rectangle(test_img, top_l, bot_r, [0, 0, 255], 2)
                    cv.putText(test_img, name, (bot_r[0] - 10, bot_r[1] + 10), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                               [0, 0, 255])
                    continue

                # Correct
                correct += 1
                correct_val += val
                total_iou += iou
                cv.rectangle(test_img, top_l, bot_r, [0, 255, 0], 2)
                cv.putText(test_img, "{}, {:.2f}".format(name, iou), (top_l[0], top_l[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                           0.4, [0, 255, 0])

            total_time += time.time() - start_time
            plt.imshow(cv.cvtColor(test_img, cv.COLOR_BGR2RGB))
            # plt.show()
            plt.close()

            # print("\t\t{} ->\t{}, {}\t({})".format(name, top_l, bot_r, val))

        print("\n----------------------\n")
        print(m)
        print("{} total icons\n{} correct template matches".format(total_icons, correct))
        print("{:.2f}% accuracy".format(correct * 100 / total_icons))
        print("{} false positives".format(incorrect))
        print("{} missed".format(total_icons - correct))
        if correct != 0:
            print("{:.3f} average IoU".format(total_iou / correct))
        if correct != 0:
            print("{:.2f} average value when correct".format(correct_val / correct))
        if incorrect != 0:
            print("{:.2f} average value when incorrect".format(incorrect_val / incorrect))
        print("{} cut-off".format(c))
        print("{:.2f}s total time\n{:.2f}s average time (assuming 20 tests)".format(total_time, total_time / 20))

        # correct, accuracy, false positives, total_time, training_time
        metrics[max(OCTAVES)] = [correct, correct * 100 / total_icons, incorrect, total_time, t]

    print("Done\n\n")

    output = open("results.pkl", 'wb')
    pickle.dump(metrics, output)
    output.close()


if __name__ == "__main__":
    main()
