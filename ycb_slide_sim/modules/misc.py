import os
from os import path as osp
import shutil
import GPUtil
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple

from scipy.spatial.transform import Rotation as R

root = os.path.abspath(os.path.join('.', os.pardir))
DIRS = {
    "root": root,
    "data": osp.join(root, "datasets", "YCB-Slide", "real"),
    "obj_models": osp.join(root, "datasets", "YCB-Slide", "obj_models"),
}


def save_image(tactileImage: np.ndarray, i: int, save_path: str) -> None:
    """
    Save tactile image as .jpg file
    """
    tactileImage = Image.fromarray(tactileImage.astype("uint8"), "RGB")
    tactileImage.save("{path}/frame_{p_i:07d}.jpg".format(path=save_path, p_i=i))

def save_images(tactileImages: List[np.ndarray], save_path: str) -> None:
    """
    Save tactile images as .jpg files
    """
    for i, tactileImage in enumerate(tactileImages):
        save_image(tactileImage, i, save_path)


def save_heightmap(heightmap: np.ndarray, i: int, save_path: str) -> None:
    """
    Save heightmap as .jpg file
    """
    cv2.imwrite(
        "{path}/{p_i}.jpg".format(path=save_path, p_i=i), heightmap.astype("float32")
    )


def save_heightmaps(heightmaps: List[np.ndarray], save_path: str) -> None:
    """
    Save heightmaps as .jpg files
    """
    for i, heightmap in enumerate(heightmaps):
        save_heightmap(heightmap, i, save_path)


def save_contactmask(contactMask: np.ndarray, i: int, save_path: str) -> None:
    """
    Save contact mask as .jpg file
    """
    cv2.imwrite(
        "{path}/{p_i}.jpg".format(path=save_path, p_i=i),
        255 * contactMask.astype("uint8"),
    )


def save_contactmasks(contactMasks: List[np.ndarray], save_path: str) -> None:
    """
    Save contact masks as .jpg files
    """
    for i, contactMask in enumerate(contactMasks):
        save_contactmask(contactMask, i, save_path)


def remove_and_mkdir(results_path: str) -> None:
    """
    Remove directory (if exists) and create
    """
    if osp.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

def get_device(cpu: bool = False, verbose: bool = True) -> str:
    """
    Check GPU utilization and return device for torch
    """
    if cpu:
        device = "cpu"
        if verbose:
            print("Override, using device:", device)
    else:
        try:
            deviceID = GPUtil.getFirstAvailable(
                order="first",
                maxLoad=0.8,
                maxMemory=0.8,
                attempts=5,
                interval=1,
                verbose=False,
            )
            device = torch.device(
                "cuda:" + str(deviceID[0]) if torch.cuda.is_available() else "cpu"
            )
            if verbose:
                print("Using device:", torch.cuda.get_device_name(deviceID[0]))
        except:
            device = "cpu"
            if verbose:
                print("Using device:", device)
    return device

# CUSTOM
def save_images_heightmaps(heightmapsImages: List[np.ndarray], save_path: str) -> None:
    """
    Save tactile images as .jpg files
    """
    for i, heightmapsImage in enumerate(heightmapsImages):
        save_hm(heightmapsImage, i, save_path)

def save_hm(heightmap: np.ndarray, i: int, save_path: str) -> None:
    """
    Save heightmap as .jpg file
    """
    heightmap = cv2.flip(heightmap, 0)
    heightmap = cv2.flip(heightmap, 1)
    cv2.imwrite(
        "{path}/frame_{p_i:07d}.jpg".format(path=save_path, p_i=i), heightmap.astype("float32")
    )