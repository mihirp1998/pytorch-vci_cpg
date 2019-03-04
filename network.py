import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, ParameterList
from modules import ConvLSTMCellTemp as ConvLSTMCell, Sign


class EncoderCell(nn.Module):
    def __init__(self, v_compress, stack, fuse_encoder, fuse_level):
        super(EncoderCell, self).__init__()

        # Init.
        self.v_compress = v_compress
        self.fuse_encoder = fuse_encoder
        self.fuse_level = fuse_level
        if fuse_encoder:
            print('\tEncoder fuse level: {}'.format(self.fuse_level))

        # Layers.
        self.conv = nn.Conv2d(
            9 if stack else 3, 
            64, 
            kernel_size=3, stride=2, padding=1, bias=False)

        for param in self.conv.parameters():
            param.requires_grad = False

        self.rnn1 = ConvLSTMCell(
            128 if fuse_encoder and v_compress else 64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        # print(fuse_encoder,"fuse_encoder",v_compress,"v_compress")
        self.rnn2 = ConvLSTMCell(
            ((384 if fuse_encoder and v_compress else 256) 
             if self.fuse_level >= 2 else 256),
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvLSTMCell(
            ((768 if fuse_encoder and v_compress else 512) 
             if self.fuse_level >= 3 else 512),
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)


    def forward(self, input, hidden1, hidden2, hidden3,
                unet_output1, unet_output2,wenc):
        init_conv,rnn1_i,rnn1_h,rnn2_i,rnn2_h,rnn3_i,rnn3_h = wenc
        init_conv=  self.conv.weight
        x = self.conv(input)
        # x= F.conv2d(input,init_conv,stride=2,padding=1)
        # Fuse
        if self.v_compress and self.fuse_encoder:
            x = torch.cat([x, unet_output1[2], unet_output2[2]], dim=1)

        hidden1 = self.rnn1(x,rnn1_i,rnn1_h, hidden1)
        x = hidden1[0]
        # Fuse.
        if self.v_compress and self.fuse_encoder and self.fuse_level >= 2:
            x = torch.cat([x, unet_output1[1], unet_output2[1]], dim=1)

        hidden2 = self.rnn2(x,rnn2_i,rnn2_h,hidden2)
        x = hidden2[0]
        # Fuse.
        if self.v_compress and self.fuse_encoder and self.fuse_level >= 3:
            x = torch.cat([x, unet_output1[0], unet_output2[0]], dim=1)

        hidden3 = self.rnn3(x,rnn3_i,rnn3_h,hidden3)
        x = hidden3[0]
        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self, bits):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, bits, kernel_size=1, bias=False)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.sign = Sign()

    def forward(self, input,init_conv):
        init_conv =  self.conv.weight
        feat = self.conv(input)
        # feat = F.conv2d(input,init_conv,stride=1,padding=0)
        x = F.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self, v_compress, shrink, bits, fuse_level):

        super(DecoderCell, self).__init__()

        # Init.
        self.v_compress = v_compress
        self.fuse_level = fuse_level
        print('\tDecoder fuse level: {}'.format(self.fuse_level))

        # Layers.
        self.conv1 = nn.Conv2d(
            bits, 512, kernel_size=1, stride=1, padding=0, bias=False)

        for param in self.conv1.parameters():
            param.requires_grad = False

        self.rnn1 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = ConvLSTMCell(
            (((128 + 256 // shrink * 2) if v_compress else 128) 
             if self.fuse_level >= 3 else 128), #out1=256
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvLSTMCell(
            (((128 + 128//shrink*2) if v_compress else 128) 
             if self.fuse_level >= 2 else 128), #out2=128
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.rnn4 = ConvLSTMCell(
            (64 + 64//shrink*2) if v_compress else 64, #out3=64
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.conv2 = nn.Conv2d(
            32,
            3, 
            kernel_size=1, stride=1, padding=0, bias=False)

        for param in self.conv2.parameters():
            param.requires_grad = False

    def forward(self, input, hidden1, hidden2, hidden3, hidden4,
                unet_output1, unet_output2,wdec):
        init_conv,rnn1_i,rnn1_h,rnn2_i,rnn2_h,rnn3_i,rnn3_h,rnn4_i,rnn4_h,final_conv = wdec

        init_conv = self.conv1.weight

        # x= F.conv2d(input,init_conv,stride=1,padding=0)

        x = self.conv1(input)
        hidden1 = self.rnn1(x,rnn1_i,rnn1_h,hidden1)

        # rnn 2
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        if self.v_compress and self.fuse_level >= 3:
            x = torch.cat([x, unet_output1[0], unet_output2[0]], dim=1)

        hidden2 = self.rnn2(x,rnn2_i,rnn2_h, hidden2)

        # rnn 3
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        if self.v_compress and self.fuse_level >= 2:
            x = torch.cat([x, unet_output1[1], unet_output2[1]], dim=1)

        hidden3 = self.rnn3(x,rnn3_i,rnn3_h,hidden3)

        # rnn 4
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

        if self.v_compress:
            x = torch.cat([x, unet_output1[2], unet_output2[2]], dim=1)

        hidden4 = self.rnn4(x, rnn4_i, rnn4_h, hidden4)

        # final
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)

        final_conv = self.conv2.weight
        # x= F.conv2d(x,final_conv,stride=1,padding=0)
        x = self.conv2(x)

        x = F.tanh(x) / 2

        return x, hidden1, hidden2, hidden3, hidden4

class HyperNetwork(nn.Module):

    def __init__(self,num_vids):
        super(HyperNetwork, self).__init__()
        emb_size = num_vids
        emb_dimension= 16

        self.context_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        initrange = 0.5 / emb_dimension
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.z_dim = z_dim
        self.enclayer =   [[64,9,3,3]]+[[1024,128,3,3]]+[[1024,256,1,1]]+[[2048,384,3,3]]+[[2048,512,1,1]]+[[2048,512,3,3]]+[[2048,512,1,1]]
        self.declayer = [[512,8,1,1]]+[[2048,512,3,3]]+[[2048,512,1,1]]+[[2048,128,3,3]]+[[2048,512,1,1]]+[[1024,128,3,3]]+[[1024,256,3,3]]+[[512,128,3,3]]+[[512,128,3,3]]+[[3,32,1,1]]
        self.binlayer=   [[8,512,1,1]]
        self.unetconvlayer = [[32,3,3,3],[32,32,3,3],[64,32,3,3],[64,64,3,3],[128,64,3,3],[128,128,3,3],[256,128,3,3],[256,256,3,3],[256,256,3,3],[256,256,3,3],[128,512,3,3],[128,128,3,3],[64,256,3,3],[64,64,3,3],[32,128,3,3],[32,32,3,3]]

        # layer = np.array(declayer+enclayer+binlayer)

        self.encoderWeights = nn.ParameterList([Parameter(torch.zeros(emb_dimension, self.total(i)) ) for i in self.enclayer])
        self.decoderWeights = nn.ParameterList([Parameter(torch.zeros(emb_dimension, self.total(i)) ) for i in self.declayer])
        self.binWeights = nn.ParameterList([Parameter(torch.zeros(emb_dimension, self.total(i)) ) for i in self.binlayer])
        self.unetConvW_weights = nn.ParameterList([Parameter(torch.zeros(emb_dimension, self.total(i)) ) for i in self.unetconvlayer])
        self.unetConvB_weights = nn.ParameterList([Parameter(torch.zeros(emb_dimension, i[0]) ) for i in self.unetconvlayer])

        #self.w1 =torch.nn.init.xavier_normal(self.w1)
        self.encoderBias = nn.ParameterList([Parameter(torch.fmod(torch.zeros((self.total(i))),2)) for i in self.enclayer])
        self.decoderBias = nn.ParameterList([Parameter(torch.fmod(torch.zeros((self.total(i))),2)) for i in self.declayer])
        self.binBias = nn.ParameterList([Parameter(torch.fmod(torch.zeros((self.total(i))),2)) for i in self.binlayer])
        self.unetConvW_bias = nn.ParameterList([Parameter(torch.zeros(self.total(i)) ) for i in self.unetconvlayer])
        self.unetConvB_bias = nn.ParameterList([Parameter(torch.zeros(i[0]) ) for i in self.unetconvlayer])
        #self.b1 =torch.nn.init.xavier_normal(self.b1)

        #self.w2 = Parameter(torch.fmod(torch.randn((h,f)),2))
        #self.b2 = Parameter(torch.fmod(torch.randn((f)),2))
    def total(self,tensor_shape):
        return tensor_shape[0]*tensor_shape[1]*tensor_shape[2]*tensor_shape[3]

    def forward(self,id_num):
        contextEmbed = self.context_embeddings(id_num)
        #h_final= self.linear(contextEmbed)
        #h_final = self.linear1(h_final)
        enc_kernels = [(torch.matmul(contextEmbed,self.encoderWeights[i])  + self.encoderBias[i]).view(self.enclayer[i]) for i in range(len(self.encoderWeights))]
        dec_kernels = [(torch.matmul(contextEmbed,self.decoderWeights[i])  + self.decoderBias[i]).view(self.declayer[i]) for i in range(len(self.decoderWeights))]
        bin_kernels = [(torch.matmul(contextEmbed,self.binWeights[i])  + self.binBias[i]).view(self.binlayer[i]) for i in range(len(self.binWeights))]
        unet_kernels = [(torch.matmul(contextEmbed,self.unetConvW_weights[i])  + self.unetConvW_bias[i]).view(self.unetconvlayer[i]) for i in range(len(self.unetconvlayer))]
        unet_bias = [(torch.matmul(contextEmbed,self.unetConvB_weights[i])  + self.unetConvB_bias[i]).view(self.unetconvlayer[i][0]) for i in range(len(self.unetconvlayer))]

        # h_final = self.linear(contextEmbed
        # print(enc_kernels[0].shape)

        return enc_kernels,dec_kernels,bin_kernels[0],unet_kernels,unet_bias


if __name__ == "__main__":
    hn = HyperNetwork(10)
    a,b,c,d,e = hn(torch.tensor(2))
    print([i.shape for i in a],[i.shape for i in b],c.shape)
    print([i.shape for i in d  ],[i.shape for i in e])
    print(a[0][0],"val")

