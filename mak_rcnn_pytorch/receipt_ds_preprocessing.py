# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:21:21 2019

@author: Mihai
"""

# %%     IMPORTS
# %%
import PIL

from os import listdir, chdir
from os.path import dirname, abspath, isfile, join, isdir
chdir(dirname(abspath(__file__)))


# %%     CONSTANTS
# %%

ROOT_FILE_NAME = 'img'
TARGET_DIR = 'Receipts'
TARGET_DIR_SUBTREE = ["Annotation", "Images", "Masks"]

IMG_SRC = "Images"

# %%     FUNCTIONS
# %%

# %%     CLASSES
# %%


class DatasetPreprocess:
    """
    TO BE ADDED
    """

    MANDATORY_SUB_FOLDERS = sorted(["Images", "Masks", "Annotation"])

    def __init__(self, target_dir: str, subtree_struct: list) -> None:
        if not isdir(target_dir):
            raise Exception(F"Base directory {target_dir} does not exists")
        if sorted(subtree_struct) != self.MANDATORY_SUB_FOLDERS:
            raise Exception(F"Missing mandatory folders/Bad structure. "
                            "Expected: ['Images', 'Masks', 'Annotation']")
        self.__directory = target_dir
        self.__dir_structure = {key: None
                                for key, val in zip(self.MANDATORY_SUB_FOLDERS,
                                                    subtree_struct)}

        img_src = join(self.__directory, "Images")
        self.__images = sorted([im for im in listdir(img_src) if isfile(join(img_src, im))])

    def __len__(self) -> int:
        return len(self.__images)

    # TODO: return a tuple of image, mask and annotation
    def __getitem__(self, idx: int):
        try:
            return self.__images[idx]
        except ValueError as ex:
            print(F"\nItem at index {idx} does not exists")
            print(F"ERROR: {str(ex)}\n")
            return None

    # TODO: add functionality
    def check_dataset_integrity(self):
        return None


# %%     MAIN
# %%

if __name__ == "__main__":
    dataset = DatasetPreprocess(TARGET_DIR, TARGET_DIR_SUBTREE)
