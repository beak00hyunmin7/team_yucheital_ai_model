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
                 use_film=False, cond_dim=1, ngf=64, wet_weight=1.0):
        self.mode = mode
        self.device = device
        self.lambda_l1 = lambda_l1
        self.use_film = use_film
        self.wet_weight = wet_weight

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

    def _l1(self, fake, real):
        """젖은 픽셀에 가중치를 준 L1.

        타깃(수심 RGB)에서 물이 있는 영역은 전체 픽셀의 소수라, 균등 L1은 얇은
        배수채널을 뭉개고 배경을 맞추는 쪽이 이득이다. 실제로 v5는 학습에 쓴
        지형에서조차 최대 수심을 절반으로 예측한다(IMPROVEMENT_GUIDE.md Phase 0-B).

        Blues 컬러맵은 마름=흰색(R≈0.97) -> 깊음=진파랑(R≈0.03)으로 R채널이 단조
        감소하므로 `wetness = 1 - R` 을 "얼마나 젖었나"의 대리값으로 쓴다.

        가중치 평균으로 나눠서 손실의 절대 크기를 균등 L1과 같은 스케일로 유지한다.
        이렇게 해야 lambda_l1 을 건드리지 않고 baseline 과 공정하게 비교할 수 있다
        (가중치가 크기를 키우는 게 아니라 페널티를 재배분하기만 한다).
        """
        if self.wet_weight <= 1.0:
            return self.criterion_l1(fake, real)
        wetness = 1.0 - (real[:, 0:1] + 1.0) * 0.5          # [-1,1] -> [0,1], 흰색=0
        w = 1.0 + (self.wet_weight - 1.0) * wetness.clamp(0.0, 1.0)
        return ((fake - real).abs() * w).mean() / w.mean()

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
            loss_g_l1 = self._l1(self.fake_target, self.real_target) * self.lambda_l1
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            self.optimizer_G.step()
            losses["loss_G_gan"] = loss_g_gan.item()
            losses["loss_G_l1"] = loss_g_l1.item()
        else:
            # 순수 오토인코더(U-Net) 모드: L1 재구성 손실만 사용
            self.optimizer_G.zero_grad()
            loss_g_l1 = self._l1(self.fake_target, self.real_target)
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
