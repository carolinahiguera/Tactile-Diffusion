import os
from os import path as osp
import numpy as np
import cv2
import random
import glob
from typing import List


main_root = os.path.abspath(os.path.join("..", os.pardir))
root = os.path.abspath(os.path.join(".", os.pardir))


# FOR YCB_SLIDE DATASET
DIRS_YCB = {
    "root": root,
    "data": osp.join(root, "datasets", "YCB-Slide", "dataset"),
    "bgs": osp.join(root, "datasets", "YCB-Slide", "bgs"),
    "project": osp.join(root, "tacto_diffusion"),
}

# background images for each object
bgs_objects = {
    "004_sugar_box": 0,
    "005_tomato_soup_can": 1,
    "006_mustard_bottle": 2,
    "021_bleach_cleanser": 3,
    "025_mug": 4,
    "035_power_drill": 0,
    "037_scissors": 5,
    "042_adjustable_wrench": 6,
    "048_hammer": 8,
    "055_baseball": 8,
}


def get_train_test_idx(obj_name: str, dataset_id: int, test_ratio: float) -> None:
    path_dataset = "{0}/{1}/dataset_{2}/".format(DIRS_YCB["data"], obj_name, dataset_id)
    dir_rgb = "{0}/{1}/dataset_{2}/frames/".format(
        DIRS_YCB["data"], obj_name, dataset_id
    )
    files_train = [f for f in os.listdir(dir_rgb) if f.endswith(".jpg")]
    files_test = random.sample(files_train, int(len(files_train) * test_ratio))
    for f in files_test:
        files_train.remove(f)
    np.savez(f"{path_dataset}/files_train_test.npz", train=files_train, test=files_test)


def split_ycb_dataset(test_ratio: float = 0.2):
    dataset_names = sorted(glob.glob("{0}/*".format(DIRS_YCB["data"])))
    for dataset_name in dataset_names:
        obj_name = dataset_name.split("/")[-1]
        if not (".zip" in obj_name or ".sh" in obj_name):
            subdirs = sorted(glob.glob("{0}/{1}/*".format(DIRS_YCB["data"], obj_name)))
            for folder in subdirs:
                f_name = folder.split("/")[-1]
                if "dataset" in f_name:
                    dataset_id = f_name.split("_")[-1]
                    get_train_test_idx(obj_name, dataset_id, test_ratio)


# FOR BRAILLE DATASET
DIRS_BRAILLE = {
    "root": root,
    "data": osp.join(root, "datasets", "braille", "real_fine_tuning"),
    "data_color": osp.join(root, "datasets", "braille", "real_fine_tuning", "color"),
    "data_mask": osp.join(root, "datasets", "braille", "real_fine_tuning", "mask"),
    "bgs": osp.join(root, "datasets", "braille", "bgs"),
    "project": osp.join(root, "tacto_diffusion")
}

bgs_letters = {
    "a": 11,
    "b": 11,
    "c": 12,
    "d": 12,
    "e": 11,
    "f": 12,
    "g": 12,
    "h": 12,
    "i": 12,
    "j": 12,
    "k": 11,
    "l": 11,
    "m": 12,
    "n": 11,
    "o": 12,
    "p": 12,
    "q": 11,
    "r": 11,
    "s": 11,
    "t": 11,
    "u": 12,
    "v": 12,
    "w": 12,
    "x": 11,
    "y": 12,
    "z": 12,
    "#": 11,
}


def split_braille_dataset(dataset_ratio=1.0, test_ratio=0.2):
    dataset_letters = {}
    for letter in bgs_letters.keys():
        files = glob.glob("{0}/{1}/*".format(DIRS_BRAILLE["data_mask"], letter))
        random.shuffle(files)
        n_files = len(files)

        # control amount if data
        idx_files = list(range(n_files))
        idx_files_subset = list(
            np.random.choice(
                idx_files, size=int(n_files * dataset_ratio), replace=False
            )
        )
        files_subset = [files[i] for i in idx_files_subset]
        files = files_subset
        n_files = len(files)

        # train
        train_ratio = 1 - test_ratio
        idx_files = list(range(n_files))
        idx_train = list(
            np.random.choice(idx_files, size=int(n_files * train_ratio), replace=False)
        )
        # test
        idx_test = list(set(idx_files) - set(idx_train))
        # save idx
        files_train = [files[i].split("/")[-1] for i in idx_train]
        files_test = [files[i].split("/")[-1] for i in idx_test]
        dataset_letters[letter] = {"train": files_train, "test": files_test}
    np.save("{0}/files_train_test.npy".format(DIRS_BRAILLE["data"]), dataset_letters)


# MISC
def save_image(img: np.ndarray, i: int, save_path: str, filename: str) -> None:
    """
    Save heightmap as .jpg file
    """
    img = cv2.resize(img, (240, 320), interpolation=cv2.INTER_AREA)
    cv2.imwrite(
        "{path}/{name}".format(path=save_path, name=filename), img.astype("uint8")
    )


def save_images(
    images: List[np.ndarray], offset: int, save_path: str, filename: List[str]
) -> None:
    """
    Save tactile images as .jpg files
    """
    for i, heightmapsImage in enumerate(images):
        save_image(heightmapsImage, i + offset, save_path, filename[i])


def create_folder(folder: str) -> None:
    """
    Create folder if it does not exist
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
