import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np


ORIGINAL_PATH = 'F:\edges2cars\cars_generator\cars512nobg\\'
GEN_PATH = 'F:\edges2cars\cars_generator\\'
RESIZE = 512
LOWER_THRESH = 255/3
HIGHER_THRESH = 255



def get_contour(img_gray):
    thresh = cv2.Canny(img_gray, LOWER_THRESH, HIGHER_THRESH)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE , method=cv2.CHAIN_APPROX_NONE)
    blank_image = np.ones((RESIZE,RESIZE,3), np.uint8) * 255
    cv2.drawContours(image=blank_image, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return blank_image

input_images_path = f'output{RESIZE}nobg'
output_images_path = f'input{RESIZE}nobg'


if __name__ == "__main__":
    onlyfiles = [f for f in listdir(ORIGINAL_PATH) if isfile(join(ORIGINAL_PATH, f))]
    os.chdir(GEN_PATH)
    try:
        os.makedirs(input_images_path)
        os.makedirs(f'input{RESIZE}nobg')
    except FileExistsError:
        pass
    for file in onlyfiles:
        print(file)
        image = cv2.imread(ORIGINAL_PATH + file, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (RESIZE,RESIZE))
        print(cv2.imwrite(output_images_path + '\\' + file, image))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contour = get_contour(gray_image)
        cv2.imwrite(input_images_path + '\\' + file, contour)