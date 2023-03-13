import numpy as np
import random
import glob
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.misc import *
import matplotlib.pyplot as plt


class Diffusion_Dataset(Dataset):
    def __init__(
        self,
        image_size: int,
        transform=None,
        remove_bg: bool = True,
        mode: str = "train",
    ):
        self.image_size = image_size
        self.transform = transform
        self.transform2tensor = transforms.ToTensor()
        self.remove_bg = remove_bg

        self.bg = cv2.imread("{0}/bg_{1}.png".format(DIRS_BRAILLE["bgs"], 10044))

        self.split_dataset = dict(
            np.load(
                "{0}/files_train_test.npy".format(DIRS_BRAILLE["data"]),
                allow_pickle=True,
            ).item()
        )

        self.files_dataset_color = []
        self.files_dataset_mask = []

        for letter in bgs_letters.keys():
            files = self.split_dataset[letter][mode]
            for f in files:
                path_color = "{0}/{1}/{2}".format(DIRS_BRAILLE["data_color"], letter, f)
                path_mask = "{0}/{1}/{2}".format(DIRS_BRAILLE["data_mask"], letter, f)
                self.files_dataset_color.append(path_color)
                self.files_dataset_mask.append(path_mask)

        self.num_data = len(self.files_dataset_color)
        print(f"[INFO DATAIO] Loaded {self.num_data} images")

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        return self.get_item(index)

    def compute_diff(
        self, img1: np.ndarray, img2: np.ndarray, offset: float = 0.0
    ) -> np.ndarray:
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        return img

    def get_item(self, index):
        try:
            f_color = self.files_dataset_color[index]
            f_mask = self.files_dataset_mask[index]
            letter = f_color.split("/")[-2]

            data_color = cv2.imread(f_color)
            data_mask = cv2.imread(f_mask)
            data_mask = cv2.cvtColor(data_mask, cv2.COLOR_RGB2GRAY)

            bg = self.bg

            if self.remove_bg:
                gt_img = self.compute_diff(img1=data_color, img2=bg, offset=0.5)
            else:
                gt_img = data_color

            gt_img = self.resize_image(gt_img)
            data_mask = self.resize_image(data_mask)
            data_color = self.resize_image(data_color)
            bg = self.resize_image(bg)

            sample = {
                "gt_tactile": self.transform2tensor(gt_img).to(torch.float32),
                "cond_depth": self.transform(data_mask),
                "bg": bg if self.remove_bg else None,
            }
            return sample

        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__()))
            # return None
