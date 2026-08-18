"""
pix2pix 스타일 U-Net 생성자 (skip-connection 재귀 블록 구조).
입력: (N, in_channels, H, W) 정규화된 등고선(+옵션 강우 채널)
출력: (N, out_channels, H, W) 정규화된 수심맵, tanh로 [-1,1] 범위
"""

import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
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


class UnetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, image_size=256, ngf=64, use_dropout=True):
        """image_size는 2의 거듭제곱이어야 함 (64/128/256 권장). num_downs = log2(image_size)."""
        super().__init__()
        num_downs = int(round(__import__("math").log2(image_size)))
        assert 2 ** num_downs == image_size, "image_size는 2의 거듭제곱이어야 합니다 (예: 64, 128, 256)"

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                  use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(out_channels, ngf, input_nc=in_channels,
                                              submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)
