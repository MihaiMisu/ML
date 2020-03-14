# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:55:56 2020

@author: Mihai
"""

# %%     IMPORTS
# %%

import numpy as np

from os import listdir, chdir, getcwd, remove as f_remove
from os.path import (join, dirname, abspath, isfile)

from PIL import Image
from typing import NewType
from datetime import datetime

chdir(dirname(abspath(__file__)))

# %%     CONSTANTS
# %%

# Source paths
IMG_PATH = join(getcwd(), "Receipts", "Masks")
IMG_TYPES = [".jpg", ".jpeg", ".png"]

PILImage = NewType("PILimage", Image)
FilePath = NewType("File Path", str)
DirPath = NewType("Directory Path", str)
DirStruct = NewType("Directory structure", dict)

# %%     FUNCTIONS
# %%


def open_img(src: FilePath) -> PILImage:
    """
    Converts an image file from whatever format is has to a PNG format
    """

    # check if source and destination are properly set
    if not isfile(src):
        print(F"{datetime.now().isoformat()} | Source is not a file: {src}")
        return

    # read image with PIL and convert it to RBG just to be safe
    img = Image.open(src)
    img = img.convert("L")

    return img


# %%     CLASSES
# %%


# %%     MAIN
# %%

if __name__ == "__main__":
    images = sorted(listdir(IMG_PATH))

    for file in images:
        img_path = join(IMG_PATH, file)
        img_obj = open_img(img_path)

        bw = np.asarray(img_obj).copy()

        # Pixel range is 0...255, 256/2 = 128
        bw[bw != 0] = 1  # White

        # Now we put it back in Pillow/PIL land
        imfile = Image.fromarray(bw)

        f_remove(img_path)

        imfile.save(img_path)
        imfile.close()
