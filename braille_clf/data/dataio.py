import numpy as np
import glob
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class Braille_Dataset(Dataset):
    def __init__(self, cfg, mode: str = "train"):
        self.transform2tensor = transforms.ToTensor()
        self.files_dataset = []
        self.letters = []
        self.img_size = cfg.img_size
        self.remove_bg = True

        if mode == "train":
            self.remove_bg = cfg.remove_bg_train
            bg_id = 11 if not cfg.fine_tune else 10044
            self.bg = cv2.imread(
                "{0}/{1}/bg_{2}.png".format(cfg.root_dir, cfg.dir_bgs, bg_id)
            )
            subdirs = glob.glob("{0}/{1}/*".format(cfg.root_dir, cfg.dir_tactile_imgs))
        else:
            self.bg = cv2.imread(
                "{0}/{1}/bg_{2}.png".format(cfg.root_dir, cfg.dir_bgs, 10044)
            )
            subdirs = glob.glob("{0}/{1}/*".format(cfg.root_dir, cfg.dir_tactile_real))

        n_letters = len(subdirs)
        for i in range(n_letters):
            letter = subdirs[i].split("/")[-1]
            self.letters.append(letter)
            files = glob.glob("{0}/*".format(subdirs[i]))
            self.files_dataset.extend(files)

        if cfg.fine_tune and mode == "train":
            self.files_dataset = np.random.choice(
                self.files_dataset, int(cfg.p_data_fine_tune * len(self.files_dataset))
            )

        self.letters = sorted(self.letters)
        self.n_dataset = len(self.files_dataset)
        self.classes = list(range(len(self.letters)))

    def __len__(self):
        return self.n_dataset

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
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )
        return img

    def get_item(self, index):
        try:
            f = self.files_dataset[index]
            letter = f.split("/")[-2]
            img = cv2.imread(f)
            if self.remove_bg:
                fg_img = self.compute_diff(img1=img, img2=self.bg, offset=0.5)
                fg_img = self.resize_image(fg_img)
            else:
                fg_img = img
            fg_img = self.transform2tensor(fg_img).float()
            label = self.letters.index(letter)
            return (fg_img, label, letter)

        except Exception as e:
            print(e)
            return None
