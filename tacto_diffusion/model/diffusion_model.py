""" Diffusion model for tactile image generation.
Arquitecture based on The Annotated Diffusion Model blog
https://huggingface.co/blog/annotated-diffusion
"""

import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from model.nn_blocks import *


class DiffusionModel:
    def __init__(
        self,
        image_size: int,
        channels: int,
        device,
        lr: float = 1e-4,
        timesteps: int = 300,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        unet_channels: list = [1, 2, 4, 8],
        unet_att_res: int = 32,
        unet_att_heads: int = 4,
        unet_res_blocks: int = 4,
    ):
        self.timesteps = timesteps
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.lr = lr

        # define beta schedule
        if beta_schedule == "linear":
            self.betas = linear_beta_schedule(
                timesteps=self.timesteps, beta_start=beta_start, beta_end=beta_end
            )
        elif beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif beta_schedule == "quadratic":
            self.betas = quadratic_beta_schedule(
                timesteps=self.timesteps, beta_start=beta_start, beta_end=beta_end
            )
        elif beta_schedule == "sigmoid":
            self.betas = sigmoid_beta_schedule(
                timesteps=self.timesteps, beta_start=beta_start, beta_end=beta_end
            )

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # define model
        self.model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=unet_channels,
            self_condition=True,
            resnet_block_groups=unet_res_blocks,
            att_heads=unet_att_heads,
            att_res=unet_att_res,
        )
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def p_losses(self, x_target, x_cond, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_target)

        x_noisy = self.q_sample(x_start=x_target, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, x_cond)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def get_noisy_image(self, x_start, t, reverse_transform):
        # add noise
        x_noisy = self.q_sample(x_start, t=t)
        # turn back into PIL image
        noisy_image = reverse_transform(x_noisy.squeeze())
        return noisy_image

    @torch.no_grad()
    def sample(self, x_cond, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(
            x_cond, shape=(batch_size, channels, image_size, image_size)
        )

    @torch.no_grad()
    def p_sample(self, x, x_cond, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # recover noise added to x
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t, x_cond) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # reverse diffusion process
    @torch.no_grad()
    def p_sample_loop(self, x_cond, shape):
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=self.device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):
            img = self.p_sample(
                img,
                x_cond,
                torch.full((b,), i, device=self.device, dtype=torch.long),
                i,
            )
            imgs.append(img)
        return imgs, x_cond
