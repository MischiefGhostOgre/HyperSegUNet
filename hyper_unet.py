import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmcv.runner import BaseModule
import math
import numpy as np
from einops import rearrange
from functools import partial

class HyperModel(nn.Module):
    def __init__(self, hyper_num=1, hyper_layer_num=6, hyper_layer_size=256):
        super(HyperModel, self).__init__()

        hyper_layer_unit = [hyper_layer_size] * hyper_layer_num
        self.hyper_dense_List = nn.ModuleList()

        pre = hyper_num
        for ii in range(hyper_layer_num):

            self.hyper_dense_List.append(nn.Linear(pre, hyper_layer_unit[ii]))
            self.hyper_dense_List.append(nn.ReLU(inplace=True))

            pre = hyper_layer_unit[ii]

    def forward(self, hyper):
        for layer in self.hyper_dense_List:
            hyper = layer(hyper)

        return hyper

class SuperConv2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=16, kernel_size=3, padding=1, stride=1, use_batchnorm=True,hyper_size=256):
        super(SuperConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.hyper_size = hyper_size
        self.padding = padding
        self.stride = stride
        self.use_batchnorm = use_batchnorm

        self.dense_weight = nn.Linear(hyper_size, self.in_channels * self.out_channels * self.kernel_size * self.kernel_size)
        self.dense_bias = nn.Linear(hyper_size, self.out_channels)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=not (use_batchnorm))

        self.conv_weight_shape = self.conv.weight.shape
        # self.conv_bias_shape = self.conv.bias.shape


    def forward(self, hyper, x):
        weight = self.dense_weight(hyper)
        weight = weight.reshape(self.conv_weight_shape)


        if not self.use_batchnorm:
            bias = self.dense_bias(hyper)
            bias = bias.squeeze()
            x = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)
        else:
            x = F.conv2d(x, weight, stride=self.stride, padding=self.padding)

        return x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class SuperConvBlock(nn.Module):
    def __init__(self, in_channels,mid_channels , out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = SuperConv2D(mid_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = SuperConv2D(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, hyper, x):
        out = self.conv0(x)
        resual = out.clone()

        out = self.conv1(hyper, out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(hyper, out)
        out = self.bn2(out)
        out = self.relu(out)

        return out + resual


class Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=9, hyper_layer_num=6, hyper_layer_size=512, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = SuperConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = SuperConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = SuperConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = SuperConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.hyper_model = HyperModel(512, hyper_layer_num, hyper_layer_size)


    def forward(self, hyper, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        hyper = self.hyper_model(hyper)
        x3_1 = self.conv3_1(hyper, torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(hyper, torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(hyper, torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(hyper, torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


