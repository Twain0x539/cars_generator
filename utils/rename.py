import cv2
import os
from os import listdir
from os.path import isfile, join

path = 'C:\CarsDataset\cars_test\cars_test\\'


onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

for file in onlyfiles:
    new_name = file[1:]
    new_name = '1' + new_name
    print(new_name)
    os.rename(path + file, path + new_name)

print(onlyfiles)