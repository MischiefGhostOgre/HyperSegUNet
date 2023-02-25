import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HyperModel(nn.Module):
    def __init__(self, hyper_num=1, hyper_layer_num=6, hyper_layer_size=256):
        super(HyperModel, self).__init__()

        hyper_layer_unit = [hyper_layer_size] * hyper_layer_num
        self.hyper_dense_List = nn.ModuleList()

        pre = hyper_num
        for ii in range(hyper_layer_num):
            self.hyper_dense_List.append(HyperLinear(pre, hyper_layer_unit[ii]))
            if hyper_layer_num % 1 == 0:
                self.hyper_dense_List.append(nn.ReLU(inplace=True))
            self.hyper_dense_List.append(nn.Dropout(0.10))

            pre = hyper_layer_unit[ii]


    def forward(self, hyper):
        out = hyper.clone()
        for layer in self.hyper_dense_List:
            out = layer(out)

        return out

class HyperLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(HyperLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.elem_weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.elem_bias = nn.Parameter(torch.Tensor(self.out_features))

        self.init_params()
        
    def init_params(self):
        stdv = 1. / math.sqrt(self.in_features) 
        
        self.elem_weight.data.uniform_(-stdv, stdv)
        if self.elem_bias is not None:
           self.elem_bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        out = F.linear(x, self.elem_weight, self.elem_bias)
        return out


class HyperConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True,
                 hyper_size=256):
        super(HyperConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.use_batchnorm = use_batchnorm
        self.hyper_size = hyper_size

        self.dense_weight = nn.Linear(self.hyper_size,
                                      self.in_channels * self.out_channels * self.kernel_size * self.kernel_size)
        self.dense_bias = nn.Linear(self.hyper_size, self.out_channels)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                              bias=not use_batchnorm)
        self.conv_weight_shape = self.conv.weight.shape
        self.init_params()

    def init_params(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.dense_weight.weight.data.uniform_(-stdv, stdv)
        if self.dense_bias is not None:
            self.dense_bias.bias.data.uniform_(-stdv, stdv)
            self.dense_bias.bias.data.uniform_(-stdv, stdv)


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


class HyperConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = HyperConv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = HyperConv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(0.2) 

    def forward(self, hyper, x):
        out = self.conv0(x)
        out = self.bn0(out)
        resual = out.clone()
        
        out = self.conv1(hyper, out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(hyper, out)
        out = self.bn2(out)
        out = out + resual
        out = self.relu(out)
        out = self.dropout(out)

        return out


class Unet(nn.Module):
    def __init__(self, input_channels=3, num_classes=9, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = HyperConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = HyperConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = HyperConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = HyperConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

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

