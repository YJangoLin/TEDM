# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 15:43
# @Author  : ZL.Liang
# @FileName: InConv.py
# @Software: PyCharm
# @Blog    ：https://github.com/YJangoLin
import math

import torch
import torch.nn as nn


class DWconv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0)):
        super(DWconv, self).__init__()
        self.ch_in = in_channels
        self.ch_out = out_channels
        self.depth_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                          groups=in_channels, bias=False)
        self.point_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class ITPLConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=False, groups=1, device='cuda'):
        super(ITPLConv, self).__init__()
        if type(padding) == int:
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        if type(kernel_size) == int: kernel_size = (kernel_size, kernel_size)
        if type(stride) == int: stride = (stride, stride)
        self.padding = nn.ZeroPad2d(padding)
        self.stride = stride
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.kernel_size = kernel_size
        self.conv = DWconv(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], requires_grad=True).to(device)
        self.weight = nn.Parameter(self.weight)
        self.bias = None
        if bias:
            self.bias = torch.randn(out_channels).to(device)
            self.bias = nn.Parameter(self.bias)

    def substract(self, x, device='cuda'):
        b, c, h, w = x.shape
        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
        output_h, output_w = math.ceil((h - kernel_h + 1) / self.stride[0]), math.ceil(
            (w - kernel_w + 1) / self.stride[1])
        result_x = torch.zeros((b, c, output_h, output_w)).to(device)
        b, c, h, w = x.shape
        ind1 = 0
        for i in range(0, h - kernel_h + 1, self.stride[0]):
            ind2 = 0
            for j in range(0, w - kernel_w + 1, self.stride[1]):
                result_x[:, :, ind1, ind2] = abs(
                    x[:, :, i: i + kernel_h, j: j + kernel_w].reshape(b, c, -1) - x[:, :, kernel_h, kernel_w].unsqueeze(
                        -1)).sum(dim=-1)
                ind2 += 1
            ind1 += 1
        return self.norm(result_x)

    def substract2(self, x, device='cuda'):
        n, c, h, w = x.shape
        d, c, k, j = self.weight.shape
        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
        # output_h, output_w = math.ceil((h - kernel_h + 1) / self.stride[0]), math.ceil(
        #     (w - kernel_w + 1) / self.stride[1])
        # result_x = torch.zeros((n, c, output_h, output_w)).to(device)
        unflod_x = x.unfold(2, kernel_h, self.stride[0]).unfold(3, kernel_w, self.stride[1])
        _, _, uh, uw, _, _ = unflod_x.shape
        result_x = unflod_x - x[:, :, :uh, :uw].unsqueeze(4).unsqueeze(5)
        result_x = torch.einsum('nchwkj,dckj->ndhw', result_x, self.weight)
        if self.bias:
            result_x = result_x + self.bias.view(1, -1, 1, 1)
        return result_x

    def forward(self, x, device='cuda'):
        x = self.padding(x) # 左右上下
        if self.kernel_size != (1, 1):
            substract_out = self.substract2(x, device)
        else:
            substract_out = x
        result_x = self.conv(substract_out)
        return result_x


class EFEM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False, isAdd=True):
        super(EFEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias, groups=groups)
        self.ITPLConv = ITPLConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=bias, groups=groups)
        self.isAdd = isAdd
        # self.bs = nn.Sequential(
        #     nn.BatchNorm2d(out_channels),
        #     nn.SiLU()
        # )

    def forward(self, x):
        if self.isAdd:
            out1 = self.conv1(x)
            out2 = self.ITPLConv(x, device=x.device)
            # print(f'ITPLConv:{out2.shape}')
            out = out1 + out2
        else:
            out = self.ITPLConv(x, device=x.device)
        # print(f'conv2d:{out1.shape}')
        # out2 = self.ITPLConv(x, device=x.device)
        # # print(f'ITPLConv:{out2.shape}')
        # out = out1 + out2
        return out


if __name__ == '__main__':
    # param: kernel_size = 3, stride=2, pading = 1
    # x: (13, 64, 32, 32)
    # conv2d: (13, 64, 16, 16)
    # INConv: (13, 64, 17, 17)
    x = torch.randn(2, 3, 64, 64)
    iplLayer = EFEM(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    print(iplLayer(x).shape)
