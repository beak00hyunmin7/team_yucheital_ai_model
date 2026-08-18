"""
PatchGAN 판별자. GAN 모드(mode: "gan")에서만 쓰인다.
(등고선 입력, 배수 이미지) 쌍을 채널 방향으로 이어붙여 진짜/가짜를 판별한다.
"""

import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=6, ndf=64, n_layers=3):
        """in_channels = 입력 채널수 + 타깃 채널수 (기본 RGB+RGB = 6)."""
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  # 패치별 real/fake logit

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
