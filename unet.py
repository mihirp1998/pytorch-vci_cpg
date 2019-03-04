#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from unet_parts import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def name_parameters(model):
    return [name for name,p in model.named_parameters() ]
    

class UNet(nn.Module):
    def __init__(self, n_channels, shrink):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64 // shrink)
        self.down1 = down(64 // shrink, 128 // shrink)
        self.down2 = down(128 // shrink, 256 // shrink)
        self.down3 = down(256 // shrink, 512 // shrink)
        self.down4 = down(512 // shrink, 512 // shrink)
        self.up1 = up(1024 // shrink, 256 // shrink)
        self.up2 = up(512 // shrink, 128 // shrink)
        self.up3 = up(256 // shrink, 64 // shrink)
        # self.up4 = up(128 // shrink, 64 // shrink)

    def forward(self, x, unet_kernels,unet_bias):
        x1 = self.inc(x, unet_kernels[0:2],unet_bias[0:2])
        x2 = self.down1(x1, unet_kernels[2:4],unet_bias[2:4])
        x3 = self.down2(x2, unet_kernels[4:6],unet_bias[4:6])
        x4 = self.down3(x3, unet_kernels[6:8],unet_bias[6:8])
        x5 = self.down4(x4, unet_kernels[8:10],unet_bias[8:10])
        out1 = self.up1(x5, x4, unet_kernels[10:12],unet_bias[10:12])
        out2 = self.up2(out1, x3, unet_kernels[12:14],unet_bias[12:14])
        out3 = self.up3(out2, x2, unet_kernels[14:16],unet_bias[14:16])
        return [out1, out2, out3]

if __name__ == "__main__":
    u = UNet(3,2)
    u.train()
    # uname =name_parameters(u)
    a = u.state_dict()
    # print("params",count_parameters(u))
    # print("num",[i for i in a.keys() if ("up" in i and "tracked" not in i) ]) 
    print(a['up1.conv.conv.weight'].shape,a['up1.conv.conv.bias'].shape,"shapes for bias")
    print("num",sum([a[i].numel() for i in a.keys() if ("tracked" not in i) ]) )
    print("num",[a[i] for i in a.keys() if ("tracked" in i) ]) 

    # print("num",[(a[i].numel(),i) for i in a.keys() if "inc" in i])
    # print("kes",a.keys(),len(a.keys()))