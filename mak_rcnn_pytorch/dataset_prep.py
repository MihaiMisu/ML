# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:21:21 2019

@author: Mihai
"""

# %%     IMPORTS
# %%
from os import listdir, chdir, getcwd, rename
from os.path import (join, dirname, abspath, isfile, isdir, splitext)

from PIL import Image
from copy import deepcopy
from typing import NewType
from datetime import datetime

chdir(dirname(abspath(__file__)))

# %%     CONSTANTS
# %%

# Source paths
IMG_SRC_PATH = join(getcwd(), "Receipts", "raw_img_dataset")
IMG_TYPES = [".jpg", ".jpeg", ".png"]

IMG_NAME_TEMPLATE = "im_{}.PNG"
MASK_NAME_TEMPLATE = "im_{}_mask.PNG"
ANNOT_NAME_TEMPLATE = "im_{}.txt"
ANNOT_CONTENT_TEMAPLATE = join(getcwd(), "Receipts", "template.txt")

# Destination paths
IMG_DST_PATH = join(getcwd(), "Receipts", "Images")
MASK_DST_PATH = join(getcwd(), "Receipts", "Masks")
ANNOT_DST_PATH = join(getcwd(), "Receipts", "Annotation")
DATASET_DST_PATH = {
    "img_dst": IMG_DST_PATH,
    "mask_dst": MASK_DST_PATH,
    "annot_dst": ANNOT_DST_PATH
    }

COPIED_FLAG = "copied"

DOWN_SIZE_RATIO = 0.3

STATS_FILE_PATH = join(getcwd(), "Receipts", "statistics.txt")

PILImage = NewType("PILimage", Image)
FilePath = NewType("File Path", str)
DirPath = NewType("Directory Path", str)
DirStruct = NewType("Directory structure", dict)

# %%     FUNCTIONS
# %%


def random_err(*args, **kwargs) -> None:
    print(F"{datetime.now().isoformat()} | Something went wrong here")
    return None


def timeit(func):
    def handler(*args, **kwargs):
        start = datetime.now()
        print(F"{str(start)[:-3]} | Entering: {func.__name__}")
        fn_results = func(*args, **kwargs)
        stp = datetime.now()
        time = (stp - start).total_seconds()
        print(F"{str(stp)[:-3]} | Exiting: {func.__name__}. ({time})")
        return fn_results
    return handler


@timeit
def get_images(path: str) -> list:
    """
    Returns a list of image files from the specified path. Each image
    is considered to be a valid one if its extenios is between the predefined
    ones. Sorting method is applied to find the last added file.
    """
    if not isdir(path):
        raise Exception(F"{get_images.__name__}: Invalid path {path}")
    images = [item for item in listdir(path)
                if splitext(item)[-1].lower() in IMG_TYPES]
    return images


@timeit
def get_annotation_files(path: str) -> list:
    """
    Returns a sorted list of txt files from specified path. There shall be a
    1-to-1 correspondance between each of these files and each picture. Using
    this function, it can be known what is the last file created. Files shall
    have a name which will make sorting very easy giving the last added file
    on the last position.
    """
    if not isdir(path):
        raise NotADirectoryError(F"{get_images.__name__}: Invalid path {path}")
    files = sorted([item for item in listdir(path)
                    if item.split(".")[-1].lower() == "txt"])
    if not files:
        return []
    return files


def generate_blk_img(size: tuple) -> PILImage:
    """
    Generates a black image of given width and height in black and white.
    """
    img = Image.new("L", size)
    return img


@timeit
def image_resize(img: PILImage, ratio: float):
    """
    Will return an image object of the resized picture of its original
    formatting standard. The new shape of the image is obtained from the
    initial width and height, weighted with a coefficient.
    :param img: PIL image object

    :param ratio: float
        Number with which to multipli width and height of the imag to get the
        new shape.
    """
    width, height = img.size
    width, height = int(width*DOWN_SIZE_RATIO), int(height*DOWN_SIZE_RATIO)
    img = img.resize((width, height), Image.ANTIALIAS)
    return img


@timeit
def jpg_to_png(src: FilePath) -> PILImage:
    """
    Converts an image file from whatever format is has to a PNG format
    """

    # check if source and destination are properly set
    if not isfile(src):
        print(F"{datetime.now().isoformat()} | Source is not a file: {src}")
        return

    # read image with PIL and convert it to RBG just to be safe
    img = Image.open(src)
    img = img.convert("RGB")

    return img


@timeit
def filter_copied_imgs(img_list: list, str_flag: str) -> list:
    return [img for img in img_list if str_flag not in img]


@timeit
def generate_ds_files(src: DirPath, dst: DirStruct) -> None:
    """
    On a specific folder structure, by giving source directory with images,
    it fills up all the folders with proper data: new formated images, a mask
    for each image and a text file. The mask will be written as balck and
    white image and will have no actually mask - THIS HAVE TO BE DONE BY HAND
    BY THE USER and each text file will be empty - ALSO SHALL BE WRITTEN BY
    HAND BY THE USER.
    The expected structure can be seen by looking at MANDATORY_FIELDS, each
    representing a directory inside the whole dataset folder.
    """

    MANDATORY_FIELDS = ["img_dst", "mask_dst", "annot_dst"]
    if not all([i in dst.keys() for i in MANDATORY_FIELDS]):
        raise Exception(F"Dataset directory does not have the required"
                        F"structure. Mandatory folders: {MANDATORY_FIELDS}")

    img_to_png = {
        "jpeg": jpg_to_png,
        "jpg": jpg_to_png
    }

    # open statistics file to write info about each ptocessing
    stats_fd = open(STATS_FILE_PATH, "a")
    stats_fd.write(F"\n\n\nNEW PROCESS ON: {datetime.now().isoformat()}")

    # ----- Get all images from the desired folder, which will be later
    # filtered by checking if the name contain a specific word/sign (for
    # example 'copied')
    imgs = get_images(src)
    imgs = filter_copied_imgs(imgs, COPIED_FLAG)

    if not imgs:
        print(F"No images to process from path {src}")
        stats_fd.write(F"\nNo images to process from path {src}")
        stats_fd.write(F"\n" + "-"*60)
        stats_fd.close()
        return

    # ----- Get the last image number so the counting can be continued
    img_cnt = 1
    existing_imgs = get_images(dst["img_dst"])
    if existing_imgs:
        existing_imgs.sort(key=lambda x: int(x[3: -4]))
        img_cnt = int(splitext(existing_imgs[-1])[0].split("_")[-1]) + 1

    annot_file_template_fd = open(ANNOT_CONTENT_TEMAPLATE, "r")
    annot_file_cntnt = annot_file_template_fd.readlines()
    annot_file_template_fd.close()

    stats_fd.write(F"\nImages number to process: {len(imgs)}\n")
    for img in imgs:
        src_img_path = join(src, img)

        stats_fd.write(F"\n\tProcessing image from src: {src_img_path}")

        img_type = splitext(img)[-1].replace(".", "").lower()
        png_image = img_to_png.get(img_type, random_err)(src_img_path)
        if not png_image:
            print(F"{datetime.now().isoformat()} | No conversion fn for image"
                  F"type {img_type}")
            stats_fd.write(F"\n{datetime.now().isoformat()} | No conversion fn"
                           F"for image type {img_type}")
            continue
        png_image = image_resize(png_image, DOWN_SIZE_RATIO)

        # ----- Save new PNG image
        img_name, extension = splitext(img)
        # PAY ATTENTION - extension already contains the .(dot)
        img = F"{img_name}_{COPIED_FLAG}{extension}"

        img_dst = join(dst["img_dst"], IMG_NAME_TEMPLATE.format(img_cnt))
        png_image.save(img_dst, "PNG")

        stats_fd.write(F"\nPNG image saved at: {img_dst}")

        # ----- Create black mask image
        blk_img = generate_blk_img(png_image.size)
        blk_img_dst = join(dst["mask_dst"], MASK_NAME_TEMPLATE.format(img_cnt))
        blk_img.save(blk_img_dst)

        stats_fd.write(F"\nMask image saved at: {blk_img_dst}")

        # ----- Create annotation empty file
        annot_file_dst = join(dst["annot_dst"],
                              ANNOT_NAME_TEMPLATE.format(img_cnt))
        new_content = deepcopy(annot_file_cntnt)

        new_content[1] = new_content[1].replace(
                "img_name",
                IMG_NAME_TEMPLATE.format(img_cnt))

        new_content[2] = new_content[2].replace(
                "width",
                str(png_image.size[0]))
        new_content[2] = new_content[2].replace(
                "height",
                str(png_image.size[1]))

        new_content[11] = new_content[11].replace(
                "mask_file",
                MASK_NAME_TEMPLATE.format(img_cnt))

        with open(annot_file_dst, "w") as fd:
            for line in new_content:
                fd.write(F"{line}")

        stats_fd.write(F"\nAnnotation file saved at: {annot_file_dst}")

        # ----- Increment cnt and rename original file so it won't be copied
        # again on a future run
        img_cnt += 1
        rename(src_img_path, join(src, img))

        stats_fd.write(F"\n" + "-"*30)

    stats_fd.write(F"\n" + "-"*60)
    stats_fd.close()

# %%     CLASSES
# %%


# %%     MAIN
# %%

generate_ds_files(IMG_SRC_PATH, DATASET_DST_PATH)
