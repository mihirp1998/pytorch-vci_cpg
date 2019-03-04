#!/usr/bin/python

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.batch = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        for param in self.batch.parameters():
            param.requires_grad = False
        
        for param in self.conv.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        for param in self.batch1.parameters():
            param.requires_grad = False

        for param in self.conv1.parameters():
            param.requires_grad = False
        

    def forward(self, x,conv_ws,conv_bs):
        conv_w = self.conv.weight
        conv_b = self.conv.bias
        x = self.conv(x)
        # x= F.conv2d(x,conv_w,conv_b,padding=1)

        x = self.batch(x)
        x = self.relu(x)

        conv1_w =  self.conv1.weight
        conv1_b =  self.conv1.bias

        # x= F.conv2d(x,conv1_w,conv1_b,padding=1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x, conv_w, conv_b):
        x = self.conv(x,conv_w,conv_b)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.mpconv =  double_conv(in_ch, out_ch)


    def forward(self, x, conv_w, conv_b):
        x = self.maxpool(x)
        x = self.mpconv(x, conv_w, conv_b)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # else:
        #     self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2,conv_w, conv_b):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, conv_w, conv_b)
        return x

