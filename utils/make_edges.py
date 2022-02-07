import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np


path = 'F:\edges2cars\cars_generator\cars_train\cars_train\\'
GEN_PATH = 'F:\edges2cars\cars_generator\\'

RESIZE = 256
LOWER_THRESH = 130
HIGHER_THRESH = 260


def get_contour(img_gray):
    thresh = cv2.Canny(img_gray, LOWER_THRESH, HIGHER_THRESH)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE , method=cv2.CHAIN_APPROX_NONE)
    blank_image = np.ones((RESIZE,RESIZE,3), np.uint8) * 255
    cv2.drawContours(image=blank_image, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return blank_image


onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
os.chdir(GEN_PATH)
try:
    os.makedirs(f'output{RESIZE}')
    os.makedirs(f'input{RESIZE}')
except FileExistsError:
    pass
for file in onlyfiles:
    print(file)
    image = cv2.imread(path + file, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (RESIZE,RESIZE))
    print(cv2.imwrite(f'output{RESIZE}' + '\\' + file, image))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = get_contour(gray_image)
    cv2.imwrite(f'input{RESIZE}' + '\\' + file, contour)