"""
등고선 -> 배수 이미지 변환 모델 래퍼.

mode == "unet": U-Net(오토인코더) 하나만 두고 L1 재구성 손실로만 학습.
mode == "gan" : 위 U-Net을 생성자로, PatchGAN 판별자를 추가해 pix2pix 방식으로 학습
                (L1 + adversarial loss).
두 모드 모두 생성자 구조(U-Net)는 동일하므로 나중에 GAN으로 바꾸고 싶으면
config의 mode만 바꾸면 된다.
"""

import torch
import torch.nn as nn

from src.models.unet_generator import UnetGenerator
from src.models.discriminator import NLayerDiscriminator


class TranslationModel:
    def __init__(self, mode="unet", in_channels=3, out_channels=3, image_size=256,
                 lr=0.0002, beta1=0.5, lambda_l1=100.0, gan_loss="vanilla", device="cpu",
                 use_film=False, cond_dim=1, ngf=64):
        self.mode = mode
        self.device = device
        self.lambda_l1 = lambda_l1
        self.use_film = use_film

        self.netG = UnetGenerator(in_channels=in_channels, out_channels=out_channels,
                                   image_size=image_size, ngf=ngf, use_film=use_film,
                                   cond_dim=cond_dim).to(device)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion_l1 = nn.L1Loss()

        self.netD = None
        self.optimizer_D = None
        if mode == "gan":
            self.netD = NLayerDiscriminator(in_channels=in_channels + out_channels).to(device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
            self.criterion_gan = nn.MSELoss() if gan_loss == "lsgan" else nn.BCEWithLogitsLoss()

    def set_input(self, batch):
        self.real_input = batch["input"].to(self.device)
        self.real_target = batch["target"].to(self.device)
        self.real_cond = batch["cond"].to(self.device) if "cond" in batch else None

    def forward(self):
        self.fake_target = self.netG(self.real_input, self.real_cond)

    def _gan_target(self, pred, is_real):
        label = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return label

    def optimize_parameters(self):
        self.forward()

        losses = {}
        if self.mode == "gan":
            # --- 판별자 ---
            self.optimizer_D.zero_grad()
            fake_pair = torch.cat([self.real_input, self.fake_target.detach()], dim=1)
            pred_fake = self.netD(fake_pair)
            loss_d_fake = self.criterion_gan(pred_fake, self._gan_target(pred_fake, False))

            real_pair = torch.cat([self.real_input, self.real_target], dim=1)
            pred_real = self.netD(real_pair)
            loss_d_real = self.criterion_gan(pred_real, self._gan_target(pred_real, True))

            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            self.optimizer_D.step()
            losses["loss_D"] = loss_d.item()

            # --- 생성자 ---
            self.optimizer_G.zero_grad()
            fake_pair = torch.cat([self.real_input, self.fake_target], dim=1)
            pred_fake = self.netD(fake_pair)
            loss_g_gan = self.criterion_gan(pred_fake, self._gan_target(pred_fake, True))
            loss_g_l1 = self.criterion_l1(self.fake_target, self.real_target) * self.lambda_l1
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            self.optimizer_G.step()
            losses["loss_G_gan"] = loss_g_gan.item()
            losses["loss_G_l1"] = loss_g_l1.item()
        else:
            # 순수 오토인코더(U-Net) 모드: L1 재구성 손실만 사용
            self.optimizer_G.zero_grad()
            loss_g_l1 = self.criterion_l1(self.fake_target, self.real_target)
            loss_g_l1.backward()
            self.optimizer_G.step()
            losses["loss_G_l1"] = loss_g_l1.item()

        return losses

    def save_checkpoint(self, path, extra=None):
        state = {
            "mode": self.mode,
            "netG": self.netG.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
        }
        if self.netD is not None:
            state["netD"] = self.netD.state_dict()
            state["optimizer_D"] = self.optimizer_D.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path, map_location=None, load_optimizer=False):
        state = torch.load(path, map_location=map_location or self.device)
        self.netG.load_state_dict(state["netG"])
        if self.netD is not None and "netD" in state:
            self.netD.load_state_dict(state["netD"])
        if load_optimizer:
            if "optimizer_G" in state:
                self.optimizer_G.load_state_dict(state["optimizer_G"])
            if self.netD is not None and "optimizer_D" in state:
                self.optimizer_D.load_state_dict(state["optimizer_D"])
        return state
