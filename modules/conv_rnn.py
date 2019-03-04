import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.utils import _pair


class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy



class ConvLSTMCellTemp(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvLSTMCellTemp, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.bias=bias
        self.hidden_kernel_size = _pair(hidden_kernel_size)

        self.hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels
     
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        for param in self.conv_ih.parameters():
            param.requires_grad = False

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=self.hidden_kernel_size,
            stride=1,
            padding=self.hidden_padding,
            dilation=1,
            bias=bias)

        for param in self.conv_hh.parameters():
            param.requires_grad = False
          

    def forward(self, input , hidden):
        hx, cx = hidden

        conv_w_i = self.conv_ih.weight 
        conv_w_h = self.conv_hh.weight 
        #hx =hx.view(1,-1,hx.shape[2],hx.shape[3])
        #print(conv_w_i.shape,"weird")
        gates = self.conv_ih(input) + self.conv_hh(hx)
        # print("hx",input.shape,hx.shape)
        #gate_input =  F.conv2d(input, conv_w_i, groups=self.batchsize,stride=self.stride,padding=self.padding)
        #gate_hidden =  F.conv2d(hx, conv_w_h, groups=self.batchsize,stride=1,padding=self.hidden_padding)
        #print("gate",gate_input.shape,gate_hidden.shape)
        #gate_input = gate_input.view(self.batchsize,512,gate_input.shape[2],gate_input.shape[3])
        #gate_hidden = gate_hidden.view(self.batchsize,512,gate_hidden.shape[2],gate_hidden.shape[3])
        #print(gate_input.shape,gate_hidden.shape)
        # gate_input =  F.conv2d(input,conv_w_i,stride=self.stride,padding=self.padding)
        # gate_hidden = F.conv2d(hx,conv_w_h,stride=1,padding=self.hidden_padding)
        #gate_input= batchConv2d(input,conv_w_i,batchsize,stride=self.stride,padding=self.padding,dilation=self.dilation,bias=self.bias)
        #gate_hidden= batchConv2d(hx,conv_w_h,batchsize,stride=1,padding=self.hidden_padding,dilation=1,bias=self.bias)

        # gates = gate_input + gate_hidden

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy

