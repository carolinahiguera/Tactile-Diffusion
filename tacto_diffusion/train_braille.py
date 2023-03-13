import os
import glob
import json
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError

from data.dataio_braille import Diffusion_Dataset
from model.diffusion_model import DiffusionModel
from utils.misc import *

import matplotlib.pyplot as plt


def load_data(image_size: int, remove_bg: bool, batch_sz: int):
    train_datasets = []
    test_datasets = []

    transform = Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    train_datasets.append(
        Diffusion_Dataset(
            transform=transform,
            image_size=image_size,
            remove_bg=remove_bg,
            mode="train",
        )
    )

    test_datasets.append(
        Diffusion_Dataset(
            transform=transform,
            image_size=image_size,
            remove_bg=remove_bg,
            mode="test",
        )
    )

    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_sz,
        drop_last=False,
        shuffle=True,
        num_workers=8,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_sz,
        drop_last=False,
        shuffle=False,
        num_workers=8,
    )
    return (train_dataloader, test_dataloader)


def main(conf) -> None:
    model_name = conf["name"]
    timesteps = conf["model"]["beta_schedule"]["train"]["n_timestep"]
    batch_sz = conf["dataset"]["batch_size"]
    image_size = conf["dataset"]["image_size"]
    channels = conf["dataset"]["channels"]
    remove_bg = conf["dataset"]["remove_bg"]
    dataset_name = conf["dataset"]["name"]

    # create train dataloader
    train_dataloader, _ = load_data(image_size, remove_bg, batch_sz)
    # selecte device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create model
    model = DiffusionModel(
        image_size=image_size,
        channels=channels,
        device=device,
        lr=conf_params["model"]["lr"],
        timesteps=timesteps,
        beta_schedule=conf_params["model"]["beta_schedule"]["train"]["schedule"],
        beta_start=conf_params["model"]["beta_schedule"]["train"]["linear_start"],
        beta_end=conf_params["model"]["beta_schedule"]["train"]["linear_end"],
        unet_channels=conf_params["model"]["unet"]["channel_mults"],
        unet_res_blocks=conf_params["model"]["unet"]["res_blocks"],
        unet_att_heads=conf_params["model"]["unet"]["num_head_channels"],
        unet_att_res=conf_params["model"]["unet"]["attn_res"],
    )

    # load midastouch diffusion
    filename_chck = "./outputs/ycb_slide/checkpoints/diffusion_ycb_slide.pt"
    checkpoint = torch.load(filename_chck, map_location=device)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("YCB-Slide checkpoint loaded")

    epochs = conf_params["model"]["epochs"]
    # set path for saving tensorboard logs and checkpoints
    path_save = "{0}/outputs/{1}/checkpoints/{2}/".format(
        DIRS["project"], dataset_name, conf_params["name"]
    )
    writer = SummaryWriter(
        log_dir="{0}/outputs/{1}/tb_logs/{2}/".format(
            DIRS["project"], dataset_name, conf_params["name"]
        )
    )

    n_iter = 0
    ssim = StructuralSimilarityIndexMeasure(
        data_range=None, reduction="elementwise_mean"
    ).to(device)
    mse = MeanSquaredError().to(device)

    frq_log_loss = conf_params["model"]["freq_loss"]
    frq_log_ssim = conf_params["model"]["freq_metrics"]
    frq_save_chk = conf_params["model"]["freq_checkpoint"]

    # train diffusion model
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            model.optimizer.zero_grad()

            batch_size = batch["gt_tactile"].shape[0]
            batch["gt_tactile"] = batch["gt_tactile"].to(device)
            batch["cond_depth"] = batch["cond_depth"].to(device)

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            x_target = batch["gt_tactile"]
            x_cond = batch["cond_depth"]
            loss = model.p_losses(x_target, x_cond, t, loss_type="huber")

            if step % frq_log_loss == 0:
                writer.add_scalar("Loss/train", loss.item(), n_iter)
                print(f"Model {model_name} \t Epoch {epoch} \t Loss:{loss.item():.4f}")

            if step % frq_log_ssim == 0:
                # compute ssim
                model.model.eval()
                x_hat, x_cond = model.sample(
                    x_cond,
                    image_size=image_size,
                    batch_size=batch_size,
                    channels=channels,
                )
                x_hat_t = torch.stack(x_hat)
                x_hat = x_hat_t[-1]
                ssim_metric = ssim(x_hat, x_target)
                mse_metric = mse(x_hat, x_target)
                model.model.train()

                writer.add_scalar("Loss/train", loss.item(), n_iter)
                writer.add_scalar("SSIM/train", ssim_metric.item(), n_iter)
                writer.add_scalar("MSE/train", mse_metric.item(), n_iter)
                print(
                    f"Model {model_name} \t Epoch {epoch} \t Loss:{loss.item():.4f} \t ssim={ssim_metric.item():.4f} \t mse={mse_metric.item():.4f}"
                )

            loss.backward()
            model.optimizer.step()
            n_iter += 1

        if epoch % frq_save_chk == 0:
            filename = f"{path_save}/epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "loss": loss,
                },
                filename,
            )

    # save final model
    filename = f"{path_save}/diffusion_{dataset_name}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
            "loss": loss,
        },
        filename,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config_braille.json",
        help="JSON file for configuration",
    )

    args = parser.parse_args()
    f = open(args.config)
    conf_params = json.load(f)

    dataset = conf_params["dataset"]["name"]
    DIRS = DIRS_BRAILLE
    split_braille_dataset(
        dataset_ratio=conf_params["dataset"]["dataset_ratio"],
        test_ratio=conf_params["dataset"]["test_ratio"],
    )

    # create folder to save checkpoints
    create_folder(
        "{0}/outputs/{1}/checkpoints/{2}/".format(
            DIRS["project"], dataset, conf_params["name"]
        )
    )
    create_folder(
        "{0}/outputs/{1}/tb_logs/{2}/".format(
            DIRS["project"], dataset, conf_params["name"]
        )
    )

    main(conf_params)

    print("END TRAINING")
