import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure


from data.dataio_braille import Diffusion_Dataset
from model.diffusion_model import DiffusionModel
from utils.misc import *


USE_CUDA = True
DIRS = DIRS_BRAILLE


def load_data(image_size: int, remove_bg: bool, batch_sz: int):

    test_datasets = []

    transform = Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    dataset_names_test = sorted(glob.glob("{0}/*".format(DIRS["data"])))
    obj2dataset = {}

    for i, dataset_name in enumerate(dataset_names_test):
        obj_name = dataset_name.split("/")[-1]
        if not (".zip" in obj_name or ".sh" in obj_name):
            obj2dataset[obj_name] = i
            test_datasets.append(
                Diffusion_Dataset(
                    transform=transform,
                    image_size=image_size,
                    remove_bg=remove_bg,
                    mode="test",
                )
            )

    # test_dataset = test_datasets[obj2dataset["025_mug"]]
    test_dataset = ConcatDataset(test_datasets)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_sz,
        drop_last=False,
        shuffle=True,
        num_workers=8,
    )
    return test_dataloader


def compute_diff(img1: np.ndarray, img2: np.ndarray, offset: float = 0.0) -> np.ndarray:
    img1 = np.int32(img1)
    img2 = np.int32(img2)
    diff = img1 - img2
    diff = diff / 255.0 + offset
    return diff


def add_bg(x_hat: torch.Tensor, x_gt: torch.Tensor, bg: torch.Tensor, ssim):
    bg = bg.permute(0, 3, 1, 2) / 255.0
    x_gt = x_gt.permute(0, 3, 1, 2) / 255.0
    x_hat_bg = (x_hat - 0.5).cpu()
    x_hat_bg = torch.clip(bg + x_hat_bg, 0.0, 1.0)
    ssim_metric_bg, _ = ssim(x_hat_bg, x_gt)
    ssim_metric_bg = ssim_metric_bg.cpu().numpy()
    return x_hat_bg, ssim_metric_bg


def resize_img(img: np.ndarray):
    img = cv2.resize(img, (240, 320), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def main(conf):
    model_name = conf["name"]
    timesteps = conf["model"]["beta_schedule"]["test"]["n_timestep"]
    batch_sz = conf["dataset"]["batch_size"]
    image_size = conf["dataset"]["image_size"]
    channels = conf["dataset"]["channels"]
    remove_bg = conf["dataset"]["remove_bg"]

    test_dataloader = load_data(image_size, remove_bg, batch_sz)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssim = StructuralSimilarityIndexMeasure(
        data_range=None, reduction="none", return_full_image=True
    ).to(device)

    model = DiffusionModel(
        image_size=image_size,
        channels=channels,
        device=device,
        lr=conf_params["model"]["lr"],
        timesteps=timesteps,
        beta_schedule=conf_params["model"]["beta_schedule"]["test"]["schedule"],
        beta_start=conf_params["model"]["beta_schedule"]["test"]["linear_start"],
        beta_end=conf_params["model"]["beta_schedule"]["test"]["linear_end"],
        unet_channels=conf_params["model"]["unet"]["channel_mults"],
        unet_res_blocks=conf_params["model"]["unet"]["res_blocks"],
        unet_att_heads=conf_params["model"]["unet"]["num_head_channels"],
        unet_att_res=conf_params["model"]["unet"]["attn_res"],
    )

    dataset = conf_params["dataset"]["name"]
    path_chck = "{0}/outputs/{1}/checkpoints/diffusion_{2}.pt".format(
        DIRS["project"], dataset, dataset
    )
    checkpoint = torch.load(path_chck, map_location=device)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.model.eval()

    path_results = "{0}/outputs/results/{1}/".format(DIRS["project"], model_name)
    create_folder(path_results)

    data_ssim_fg = []
    data_ssim_bg = []
    img_i = 0
    max_num_batches = 10

    for k in range(max_num_batches):
        batch = next(iter(test_dataloader))
        batch_size = batch["gt_tactile"].shape[0]
        batch["gt_tactile"] = batch["gt_tactile"].to(device)
        batch["cond_depth"] = batch["cond_depth"].to(device)

        x_target = batch["gt_tactile"]
        x_cond = batch["cond_depth"]
        x_color_target = batch["org_tactile"]
        bg = batch["bg"]

        x_hat, _ = model.sample(
            x_cond, image_size=image_size, batch_size=batch_size, channels=channels
        )
        x_hat_t = torch.stack(x_hat)
        x_hat = x_hat_t[-1]

        # ssim only foreground
        ssim_metric_fg, ssim_img_fg = ssim(x_hat, x_target)
        ssim_metric_fg = ssim_metric_fg.cpu().numpy()
        data_ssim_fg.extend(list(ssim_metric_fg))

        # ssim with background
        x_hat_bg, ssim_metric_bg = add_bg(x_hat, x_color_target, bg, ssim)
        data_ssim_bg.extend(list(ssim_metric_bg))

        # only foreground
        x_hat_bg = x_hat_bg.permute(0, 2, 3, 1).cpu().numpy()
        x_hat_bg = (x_hat_bg * 255).astype(np.uint8)

        # with background
        x_hat_fg = x_hat.permute(0, 2, 3, 1).cpu().numpy()
        x_hat_fg = (x_hat_fg * 255).astype(np.uint8)

        # target background
        x_target_fg = x_target.permute(0, 2, 3, 1).cpu().numpy()
        x_target_fg = (x_target_fg * 255).astype(np.uint8)
        x_target_bg = x_color_target.cpu().numpy()

        # ssim img
        ssim_img_fg = ssim_img_fg.permute(0, 2, 3, 1).cpu().numpy()
        ssim_img_fg = (ssim_img_fg * 255).astype(np.uint8)

        # depth
        depth = x_cond.permute(0, 2, 3, 1).cpu().numpy()
        # depth = (depth*255).astype(np.uint8)

        for i in range(batch_size):

            fig, ax = plt.subplots(1, 6, figsize=(17, 4))
            img = cv2.cvtColor(x_target_bg[i], cv2.COLOR_BGR2RGB)
            ax[0].imshow(resize_img(img))
            ax[0].set_title("[Real]")

            img = cv2.cvtColor(x_hat_bg[i], cv2.COLOR_BGR2RGB)
            ax[1].imshow(resize_img(img))
            ax[1].set_title(f"[Diffusion] ssim={ssim_metric_bg[i]:.2f}")

            img = x_target_fg[i]
            ax[2].imshow(resize_img(img))
            ax[2].set_title("[Real]")

            img = x_hat_fg[i]
            ax[3].imshow(resize_img(img))
            ax[3].set_title(f"[Diffusion] ssim={ssim_metric_fg[i]:.2f}")

            img = ssim_img_fg[i]
            ax[4].imshow(resize_img(img))
            ax[4].set_title("[Difference fg]")

            img = depth[i]
            ax[5].imshow(resize_img(img), cmap="gray", vmin=-1.0, vmax=1.0)
            ax[5].set_title("[Sim depth]")

            ax[0].set_axis_off()
            ax[1].set_axis_off()
            ax[2].set_axis_off()
            ax[3].set_axis_off()
            ax[4].set_axis_off()
            ax[5].set_axis_off()

            fig.tight_layout()
            # plt.savefig("{0}/{1}.png".format(path_results, img_i), dpi=75)
            plt.show()
            plt.close()
            img_i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        # default="config/config.json",
        default="config/config_braille.json",
        help="JSON file for configuration",
    )

    args = parser.parse_args()
    f = open(args.config)
    conf_params = json.load(f)

    main(conf_params)

    print("")
