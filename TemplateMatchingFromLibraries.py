import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

test = cv.imread('Task2Dataset/TestWithoutRotations/images/test_image_1.png', cv.IMREAD_GRAYSCALE)
assert test is not None, "file could not be read, check with os.path.exists()"
test2 = test.copy()
template = cv.imread('Task2Dataset/Training/png/011-trash.png', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"

w, h = template.shape[::-1]

result = cv.matchTemplate(test, template, cv.TM_CCOEFF_NORMED)
print(result)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv.rectangle(test, top_left, bottom_right, 255, 2)

plt.subplot(121), plt.imshow(result, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(test, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

plt.show()
