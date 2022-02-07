# 1.Remove bad generated edges from /input***
# 2. Run this script for removing original from /output (for hard processing images)


import os
from os import listdir, remove
from os.path import isfile, join, exists

input_path = "F:\edges2cars\cars_generator\input512"
output_path = "F:\edges2cars\cars_generator\output512"

input_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
output_files = [f for f in listdir(output_path) if isfile(join(output_path, f))]

for file in output_files:
    if not (exists(join(input_path, file))):
        os.remove(join(output_path, file))