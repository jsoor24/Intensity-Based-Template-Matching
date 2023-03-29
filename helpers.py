from os import listdir


def white_to_black(img):
    for m in range(0, img.shape[0]):
        for n in range(0, img.shape[1]):
            if img[m, n] == 255:
                img[m, n] = 0
    return img


def get_object_name(file):
    object_name = file[4:len(file) - 4]
