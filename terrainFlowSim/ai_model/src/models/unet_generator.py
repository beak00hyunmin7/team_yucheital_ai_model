"""
pix2pix 스타일 U-Net 생성자 (skip-connection 재귀 블록 구조) + FiLM 강우 조건화.

입력: (N, in_channels, H, W) 정규화된 등고선
      + (선택) cond: (N, cond_dim) 강우량 등 스칼라 조건
출력: (N, out_channels, H, W) 정규화된 수심맵, tanh로 [-1,1] 범위

use_film=False(기본)면 down+submodule+up을 하나의 nn.Sequential로 묶는 원래
구조를 그대로 쓴다 - 예전에 저장된 체크포인트(checkpoints/, checkpoints_gan*/,
checkpoints_raincond/ 등)와 state_dict 키 이름이 100% 호환된다.

use_film=True면 cond를 병목(bottleneck)까지 통과시켜야 해서 nn.Sequential로
묶을 수 없고(중간에 인자를 못 끼워넣음), down/submodule/up을 분리해 forward에서
직접 순서대로 호출하는 별도 블록(_FiLMUnetSkipConnectionBlock)을 쓴다. cond를
작은 MLP로 임베딩해서 병목 특징맵에 채널별 scale/shift로 곱하고 더한다
(FiLM: Feature-wise Linear Modulation) - 강우량을 등고선과 같은 입력 채널에
얹는 것보다 조건을 모든 위치에 명시적으로 강제 반영시킨다.
"""

import math

import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    """원본 구조 그대로 (state_dict 키 호환성 유지용). cond는 받지 않는다."""

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class FiLM(nn.Module):
    """스칼라 조건 -> 채널별 (gamma, beta) -> x * (1+gamma) + beta."""

    def __init__(self, num_features, cond_dim=1, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, num_features * 2),
        )

    def forward(self, x, cond):
        gamma, beta = self.net(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class _FiLMUnetSkipConnectionBlock(nn.Module):
    """FiLM 조건화용: cond를 forward에 명시적으로 받아 innermost까지 전달해야 해서
    down/submodule/up을 분리해 직접 호출한다 (nn.Sequential은 단일 인자만 전달 가능)."""

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, use_dropout=False,
                 use_film_here=False, cond_dim=1):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.submodule = submodule
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        self.film = FiLM(inner_nc, cond_dim=cond_dim) if use_film_here else None

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            self.down = nn.Sequential(downconv)
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            self.down = nn.Sequential(downrelu, downconv)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            self.down = nn.Sequential(downrelu, downconv, downnorm)
            up_layers = [uprelu, upconv, upnorm]
            if use_dropout:
                up_layers.append(nn.Dropout(0.5))
            self.up = nn.Sequential(*up_layers)

    def forward(self, x, cond=None):
        d = self.down(x)
        if self.innermost:
            if self.film is not None and cond is not None:
                d = self.film(d, cond)
            u = self.up(d)
        else:
            u = self.up(self.submodule(d, cond))

        if self.outermost:
            return u
        return torch.cat([x, u], dim=1)


class UnetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, image_size=256, ngf=64,
                 use_dropout=True, use_film=False, cond_dim=1):
        """image_size는 2의 거듭제곱이어야 함 (64/128/256 권장). num_downs = log2(image_size).
        use_film=True면 forward(x, cond)의 cond(예: 정규화된 강우량)를 병목에 주입한다
        (이 경우 예전 체크포인트와 state_dict 키 이름이 달라짐 - 새로 학습한 FiLM
        체크포인트 전용)."""
        super().__init__()
        num_downs = int(round(math.log2(image_size)))
        assert 2 ** num_downs == image_size, "image_size는 2의 거듭제곱이어야 합니다 (예: 64, 128, 256)"

        self.use_film = use_film
        Block = _FiLMUnetSkipConnectionBlock if use_film else UnetSkipConnectionBlock

        def make_block(outer_nc, inner_nc, input_nc=None, submodule=None,
                       outermost=False, innermost=False, use_dropout=False, film_here=False):
            if use_film:
                return Block(outer_nc, inner_nc, input_nc=input_nc, submodule=submodule,
                              outermost=outermost, innermost=innermost, use_dropout=use_dropout,
                              use_film_here=film_here, cond_dim=cond_dim)
            return Block(outer_nc, inner_nc, input_nc=input_nc, submodule=submodule,
                         outermost=outermost, innermost=innermost, use_dropout=use_dropout)

        unet_block = make_block(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                 innermost=True, film_here=use_film)
        for _ in range(num_downs - 5):
            unet_block = make_block(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                     use_dropout=use_dropout)
        unet_block = make_block(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = make_block(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = make_block(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        self.model = make_block(out_channels, ngf, input_nc=in_channels,
                                 submodule=unet_block, outermost=True)

    def forward(self, x, cond=None):
        if self.use_film:
            return self.model(x, cond)
        return self.model(x)
