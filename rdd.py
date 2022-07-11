import os
import torch.nn as nn
import time
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *
import torch.nn.functional as F
import cv2
import numpy as np
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor):
        super(Net, self).__init__()
        # base_filter=256
        # feat=64
        self.nFrames = nFrames

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(6, base_filter, 3, 1, 1, activation='prelu', norm=None)

        # Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        # modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        modules_body1.append(ConvBlock(base_filter, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # # Res-Block4
        modules_body4 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        # modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        modules_body4.append(ConvBlock(base_filter, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat4 = nn.Sequential(*modules_body4)

        # Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        # Res-Block3
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        # modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        modules_body3.append(ConvBlock(feat, base_filter, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)

        # # # �����
        # self.decouping1 = ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None)
        # self.decouping2 = ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None)
        # self.decouping3 = ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None)

        # # # # �����2
        self.decouping1 = ConvBlock(feat, 3, 3, 1, 1, activation='prelu', norm=None)
        self.decouping2 = ConvBlock(feat + 3, 2, 3, 1, 1, activation='prelu', norm=None)
        self.decouping3 = ConvBlock(feat + 3 + 2, 3, 3, 1, 1, activation='prelu', norm=None)
        self.feat4 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        #
        # # # attn
        self.attnConv1 = AttentionBlock((nFrames - 1) * feat, nFrames - 1)
        # self.attnConv1 = AttentionBlock(nFrames * feat, nFrames)

        # Reconstruction
        self.output = ConvBlock((nFrames - 1) * feat, num_channels, 3, 1, 1, activation=None, norm=None)

        self.outRain = ConvBlock((nFrames - 1) * 3, num_channels, 3, 1, 1, activation=None, norm=None)
        self.outMotion = ConvBlock((nFrames - 1) * 2, 2, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    # <editor-fold desc='backbone-de-di-attn-work'>
    def forward(self, x, neigbor, flow):  # old
        feat_input = self.feat0(x)
        feat_frame = []
        for j in range(len(neigbor)):
            # feat_frame.append(self.feat1(torch.cat((x, neigbor[j], flow[j]), 1)))  # 8  ws__flow
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j]), 1)))  # 6
            # feat_frame.append(self.feat1(flow[j]))  # 2
        ####Projection
        Ht = []
        R = []
        M = []
        tmp = []
        for j in range(len(neigbor)):
            h0 = self.res_feat4(feat_input)
            # # #### �����2 ��ǰ
            h1 = self.res_feat1(feat_frame[j])
            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            # # #### h �����2
            r = self.decouping1(h)
            h = torch.cat((h, r), 1)
            R.append(r)
            m = self.decouping2(h)
            h = torch.cat((h, m), 1)
            M.append(m)
            b_F = self.decouping3(h)
            if len(tmp) == 0:
                b = x - b_F
            else:
                b = tmp[-1] - b_F
            tmp.append(b)
            h = self.feat4(b)
            # # ####
            Ht.append(h)
            feat_input = self.res_feat3(h)

        ####Reconstruction
        # print(len(Ht))
        # print(Ht[0].size())
        out = torch.cat(Ht, 1)
        # print(out.shape)
        # exit()
        out_attn = self.attnConv1(out)
        output_tmp = []
        # print(Ht[0].size())
        # print(out_attn.size())
        for o in range(len(Ht)):
            output_tmp.append(Ht[o] * out_attn[:, o, :, :])
        output_cat = torch.cat(output_tmp, 1)
        # print(out_attn.shape)
        # print(output_tmp[0].shape)

        Rain = torch.cat(R, 1)
        Motion = torch.cat(M, 1)
        # output = self.output(out)
        # print(output.shape)
        # exit()
        output = self.output(output_cat)
        out_rain = self.outRain(Rain)
        out_motion = self.outMotion(Motion)

        return output, out_rain, out_motion
