import numpy as np
import random
import glob
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.misc import *


class Diffusion_Dataset(Dataset):
    def __init__(
        self,
        dirs: dict,
        obj_name: str,
        image_size: int,
        transform=None,
        remove_bg: bool = False,
        dataset_train: bool = True,
    ):
        self.dirs = dirs
        self.obj_name = obj_name
        self.image_size = image_size
        self.transform = transform
        self.transform2tensor = transforms.ToTensor()
        self.remove_bg = remove_bg

        self.subdirs_dataset = sorted(
            glob.glob("{0}/{1}/*".format(dirs["data"], obj_name))
        )
        self.data_tactile = []
        self.data_depth = []

        for f in self.subdirs_dataset:
            if "dataset" in f.split("/")[-1]:
                files_dataset = np.load(f"{f}/files_train_test.npz")
                path_color = "{0}/frames/".format(f)
                path_depth = "{0}/frames_depth/".format(f)
                if dataset_train:
                    files_train = files_dataset["train"]
                    l_rgb = [path_color + x for x in files_train]
                    l_depth = [path_depth + x for x in files_train]
                else:
                    files_test = files_dataset["test"]
                    l_rgb = [path_color + x for x in files_test]
                    l_depth = [path_depth + x for x in files_test]

                self.data_tactile.extend(l_rgb)
                self.data_depth.extend(l_depth)

        self.num_data = len(self.data_tactile)
        print(f"[INFO DATAIO] Loaded {self.num_data} images for object {obj_name}")

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        return self.get_item(index)

    def compute_diff(self, img1, img2, offset=0.0):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff

    def resize_image(self, img):
        img = cv2.resize(
            img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        return img

    def get_item(self, index):
        try:
            data_tactile = cv2.imread(self.data_tactile[index])
            data_depth = cv2.imread(self.data_depth[index])
            data_depth = cv2.cvtColor(data_depth, cv2.COLOR_BGR2GRAY)

            data_tactile = self.resize_image(data_tactile)
            data_depth = self.resize_image(data_depth)

            if self.remove_bg:
                bg = cv2.imread(
                    "{0}/bg_{1}.jpg".format(
                        self.dirs["bgs"], bgs_objects[self.obj_name]
                    )
                )
                bg = self.resize_image(bg)
                gt_img = self.compute_diff(img1=data_tactile, img2=bg, offset=0.5)
            else:
                gt_img = data_tactile

            sample = {
                "gt_tactile": self.transform2tensor(gt_img).to(torch.float32),
                "cond_depth": self.transform(data_depth),
                "org_tactile": data_tactile,
                "bg": bg if self.remove_bg else None,
            }
            return sample

        except Exception as e:
            # print(e)
            return self.get_item(index=random.randint(0, self.__len__()))
