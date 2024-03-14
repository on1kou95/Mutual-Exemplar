# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:08:11 2022

@author: loua2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s, res2net152_v1b_26w_4s
import math
import torchvision.models as models
import os

import torch.nn as nn

class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(CONV_Block, self).__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道不同，则使用1x1卷积来改变维度
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 加入残差连接
        out += identity

        out = self.relu(out)

        return out



class preUnet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv_up_1 = CONV_Block(1024, 1024, 512)
        self.conv_up_2 = CONV_Block(1024, 512, 512)
        self.conv_up_3 = CONV_Block(512, 512, 256)
        self.conv_up_4 = CONV_Block(512, 256, 256)
        self.conv_up_5 = CONV_Block(256, 256, 64)
        self.conv_up_6 = CONV_Block(128, 64, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)



    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_k = self.resnet.maxpool(x)      # bs, 64, 88, 88
        
        # ----------- low-level features -------------
        
        x1 = self.resnet.layer1(x_k)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        
        x_up_1 = self.conv_up_1(self.up(x3)) # 512,44,44
        x_up_1 = self.conv_up_2(torch.cat([x2, x_up_1], 1)) #512 ,44,44
        
        x_up_2 = self.conv_up_3(self.up(x_up_1)) # 256,88,88
        x_up_2 = self.conv_up_4(torch.cat([x1, x_up_2], 1)) # 256,88,88

        x_up_3 = self.conv_up_5(self.up(x_up_2)) # 64,176,176
        x_up_3 = self.conv_up_6(torch.cat([x, x_up_3], 1)) # 64,88,88
        
        x_up_4 = self.up(x_up_3)
        output = self.final(x_up_4)
        return output

class preUnet_l1(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, pretrained_layers=[ "conv1", "bn1", "relu", "maxpool",
            "layer1", "layer2", "layer3", "layer4",
            "avgpool", "fc"], **kwargs):
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.pretrained_layers = pretrained_layers
        self._reset_weights_except_pretrained()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # Define convolutional blocks
        self.conv_up_1 = CONV_Block(1024, 1024, 512)
        self.conv_up_2 = CONV_Block(1024, 512, 512)
        self.conv_up_3 = CONV_Block(512, 512, 256)
        self.conv_up_4 = CONV_Block(512, 256, 256)
        self.conv_up_5 = CONV_Block(256, 256, 64)
        self.conv_up_6 = CONV_Block(128, 64, 64)

        # Final convolutional layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_b = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_8 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.final_16 = nn.Conv2d(1024, num_classes, kernel_size=1)

        # Print pretrained and non-pretrained parts
        self._print_pretrained_parts()

    def _reset_weights_except_pretrained(self):
        for name, child in self.resnet.named_children():
            if name not in self.pretrained_layers:
                if isinstance(child, nn.Conv2d):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    print(f"Weights reset for resnet.{name}")

    def _print_pretrained_parts(self):
        print("Pretrained Parts:")
        for name in self.pretrained_layers:
            print(f"\t- resnet.{name}")
        print("Non-Pretrained Parts:")
        for name, _ in self.resnet.named_children():
            if name not in self.pretrained_layers:
                print(f"\t- resnet.{name}")
        print("\t- All upsampling and convolutional blocks")

    def forward(self, x):
        # Initial layers of Res2Net
        x_dir = self.resnet.conv1(x)

        x = self.resnet.bn1(x_dir)
        x = self.resnet.relu(x)
        x_k = self.resnet.maxpool(x)  # bs, 64, 88, 88

        # Res2Net layer1 and layer2
        x1 = self.resnet.layer1(x_k)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        # Res2Net layer3
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22

        # Upsampling and concatenation
        x_up_1 = self.conv_up_1(self.up(x3))  # 512, 44, 44
        x_up_1 = self.conv_up_2(torch.cat([x2, x_up_1], 1))  # 512, 44, 44

        x_up_2 = self.conv_up_3(self.up(x_up_1))  # 256, 88, 88
        x_up_2 = self.conv_up_4(torch.cat([x1, x_up_2], 1))  # 256, 88, 88

        x_up_3 = self.conv_up_5(self.up(x_up_2))  # 64, 176, 176
        x_up_3 = self.conv_up_6(torch.cat([x, x_up_3], 1))  # 64, 176, 176

        # Final upsampling and output layer
        x_up_4 = self.up(x_up_3)
        output = self.final(x_up_4)

        # Final upsampling and output layer2
        x_up_dirc_2 = self.up_2(x_dir)
        dirc_output_1 = self.final_b(x_up_dirc_2)
        # Final upsampling and output layer8
        x_up_dirc_8 = self.up_8(x2)
        dirc_output_8 = self.final_8(x_up_dirc_8)
        # Final upsampling and output layer16
        x_up_dirc_16 = self.up_16(x3)
        dirc_output_16 = self.final_16(x_up_dirc_16)
        return output, dirc_output_16, dirc_output_1

class preUnet_l2(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, pretrained_layers=['conv1','layer3'], **kwargs):
        super().__init__()
        self.resnet = res2net101_v1b_26w_4s(pretrained=True)
        self.pretrained_layers = pretrained_layers
        self._reset_weights_except_pretrained()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # Define convolutional blocks
        self.conv_up_1 = CONV_Block(1024, 1024, 512)
        self.conv_up_2 = CONV_Block(1024, 512, 512)
        self.conv_up_3 = CONV_Block(512, 512, 256)
        self.conv_up_4 = CONV_Block(512, 256, 256)
        self.conv_up_5 = CONV_Block(256, 256, 64)
        self.conv_up_6 = CONV_Block(128, 64, 64)

        # Final convolutional layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(1024, num_classes, kernel_size=1)

        # Print pretrained and non-pretrained parts
        self._print_pretrained_parts()

    def _reset_weights_except_pretrained(self):
        for name, child in self.resnet.named_children():
            if name not in self.pretrained_layers:
                if isinstance(child, nn.Conv2d):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    print(f"Weights reset for resnet.{name}")

    def _print_pretrained_parts(self):
        print("Pretrained Parts:")
        for name in self.pretrained_layers:
            print(f"\t- resnet.{name}")
        print("Non-Pretrained Parts:")
        for name, _ in self.resnet.named_children():
            if name not in self.pretrained_layers:
                print(f"\t- resnet.{name}")
        print("\t- All upsampling and convolutional blocks")

    def forward(self, x):
        # Initial layers of Res2Net
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_k = self.resnet.maxpool(x)  # bs, 64, 88, 88

        # Res2Net layer1 and layer2
        x1 = self.resnet.layer1(x_k)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        # Res2Net layer3
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22

        # Upsampling and concatenation
        x_up_1 = self.conv_up_1(self.up(x3))  # 512, 44, 44
        x_up_1 = self.conv_up_2(torch.cat([x2, x_up_1], 1))  # 512, 44, 44

        x_up_2 = self.conv_up_3(self.up(x_up_1))  # 256, 88, 88
        x_up_2 = self.conv_up_4(torch.cat([x1, x_up_2], 1))  # 256, 88, 88

        x_up_3 = self.conv_up_5(self.up(x_up_2))  # 64, 176, 176
        x_up_3 = self.conv_up_6(torch.cat([x, x_up_3], 1))  # 64, 176, 176

        # Final upsampling and output layer
        x_up_4 = self.up(x_up_3)
        output = self.final(x_up_4)

        # Final upsampling and output layer2
        x_up_dirc = self.up2(x3)
        dirc_output = self.final2(x_up_dirc)
        return output, dirc_output

class preUnet_l3(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, pretrained_layers=[ "conv1", "bn1", "relu", "maxpool",
            "layer1", "layer2", "layer3", "layer4",
            "avgpool", "fc"], **kwargs):
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.pretrained_layers = pretrained_layers
        self._reset_weights_except_pretrained()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # Define convolutional blocks
        self.conv_up_1 = CONV_Block(1024, 1024, 512)
        self.conv_up_2 = CONV_Block(1024, 512, 512)
        self.conv_up_3 = CONV_Block(512, 512, 256)
        self.conv_up_4 = CONV_Block(512, 256, 256)
        self.conv_up_5 = CONV_Block(256, 256, 64)
        self.conv_up_6 = CONV_Block(128, 64, 64)

        # Final convolutional layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(1024, num_classes, kernel_size=1)

        # Print pretrained and non-pretrained parts
        self._print_pretrained_parts()

    def _reset_weights_except_pretrained(self):
        for name, child in self.resnet.named_children():
            if name not in self.pretrained_layers:
                if isinstance(child, nn.Conv2d):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    print(f"Weights reset for resnet.{name}")

    def _print_pretrained_parts(self):
        print("Pretrained Parts:")
        for name in self.pretrained_layers:
            print(f"\t- resnet.{name}")
        print("Non-Pretrained Parts:")
        for name, _ in self.resnet.named_children():
            if name not in self.pretrained_layers:
                print(f"\t- resnet.{name}")
        print("\t- All upsampling and convolutional blocks")

    def forward(self, x):
        # Initial layers of Res2Net
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_k = self.resnet.maxpool(x)  # bs, 64, 88, 88

        # Res2Net layer1 and layer2
        x1 = self.resnet.layer1(x_k)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        # Res2Net layer3
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22

        # Upsampling and concatenation
        x_up_1 = self.conv_up_1(self.up(x3))  # 512, 44, 44
        x_up_1 = self.conv_up_2(torch.cat([x2, x_up_1], 1))  # 512, 44, 44

        x_up_2 = self.conv_up_3(self.up(x_up_1))  # 256, 88, 88
        x_up_2 = self.conv_up_4(torch.cat([x1, x_up_2], 1))  # 256, 88, 88

        x_up_3 = self.conv_up_5(self.up(x_up_2))  # 64, 176, 176
        x_up_3 = self.conv_up_6(torch.cat([x, x_up_3], 1))  # 64, 176, 176

        # Final upsampling and output layer
        x_up_4 = self.up(x_up_3)
        output = self.final(x_up_4)

        # Final upsampling and output layer2
        x_up_dirc = self.up2(x3)
        dirc_output = self.final2(x_up_dirc)
        return output, dirc_output

class preUnet_a(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, pretrained_layers=[ "conv1", "bn1", "relu", "maxpool",
            "layer1", "layer2", "layer3", "layer4",
            "avgpool", "fc"], **kwargs):
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.pretrained_layers = pretrained_layers
        self._reset_weights_except_pretrained()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # Define convolutional blocks
        self.conv_up_1 = CONV_Block(1024, 1024, 512)
        self.conv_up_2 = CONV_Block(1024, 512, 512)
        self.conv_up_3 = CONV_Block(512, 512, 256)
        self.conv_up_4 = CONV_Block(512, 256, 256)
        self.conv_up_5 = CONV_Block(256, 256, 64)
        self.conv_up_6 = CONV_Block(128, 64, 64)

        # Final convolutional layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_4 = nn.Conv2d(256, num_classes, kernel_size=1)

        # Print pretrained and non-pretrained parts
        self._print_pretrained_parts()

    def _reset_weights_except_pretrained(self):
        for name, child in self.resnet.named_children():
            if name not in self.pretrained_layers:
                if isinstance(child, nn.Conv2d):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    print(f"Weights reset for resnet.{name}")

    def _print_pretrained_parts(self):
        print("Pretrained Parts:")
        for name in self.pretrained_layers:
            print(f"\t- resnet.{name}")
        print("Non-Pretrained Parts:")
        for name, _ in self.resnet.named_children():
            if name not in self.pretrained_layers:
                print(f"\t- resnet.{name}")
        print("\t- All upsampling and convolutional blocks")

    def forward(self, x):
        # Initial layers of Res2Net
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x_in = self.resnet.relu(x)
        x_k = self.resnet.maxpool(x_in)  # bs, 64, 88, 88

        # Res2Net layer1 and layer2
        x1 = self.resnet.layer1(x_k)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        # Res2Net layer3
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22

        # Upsampling and concatenation
        x_up_1 = self.conv_up_1(self.up(x3))  # 512, 44, 44
        x_up_1 = self.conv_up_2(torch.cat([x2, x_up_1], 1))  # 512, 44, 44

        x_up_2 = self.conv_up_3(self.up(x_up_1))  # 256, 88, 88
        x_up_2 = self.conv_up_4(torch.cat([x1, x_up_2], 1))  # 256, 88, 88

        x_up_3 = self.conv_up_5(self.up(x_up_2))  # 64, 176, 176
        x_up_3 = self.conv_up_6(torch.cat([x, x_up_3], 1))  # 64, 176, 176

        # Final upsampling and output layer
        x_up_4 = self.up(x_up_3)
        output = self.final(x_up_4)

        # # Final upsampling and output layer2
        # dirc_output_2 = self.final_2(x_in)

        # Final upsampling and output layer2
        x_up_dirc_4 = self.up_4(x1)
        dirc_output_4 = self.final_4(x_up_dirc_4)
        return output, dirc_output_4


class preUnet_b(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, **kwargs):
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # Define convolutional blocks
        self.conv_up_1 = CONV_Block(1024, 1024, 512)
        self.conv_up_2 = CONV_Block(1024, 512, 512)
        self.conv_up_3 = CONV_Block(512, 512, 256)
        self.conv_up_4 = CONV_Block(512, 256, 256)
        self.conv_up_5 = CONV_Block(256, 256, 64)
        self.conv_up_6 = CONV_Block(128, 64, 64)

        # Final convolutional layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_k = self.resnet.maxpool(x)  # bs, 64, 88, 88

        # ----------- low-level features -------------

        x1 = self.resnet.layer1(x_k)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22

        x_up_1 = self.conv_up_1(self.up(x3))  # 512,44,44
        x_up_1 = self.conv_up_2(torch.cat([x2, x_up_1], 1))  # 512 ,44,44

        x_up_2 = self.conv_up_3(self.up(x_up_1))  # 256,88,88
        x_up_2 = self.conv_up_4(torch.cat([x1, x_up_2], 1))  # 256,88,88

        x_up_3 = self.conv_up_5(self.up(x_up_2))  # 64,176,176
        x_up_3 = self.conv_up_6(torch.cat([x, x_up_3], 1))  # 64,88,88

        x_up_4 = self.up(x_up_3)
        output = self.final(x_up_4)

        # Final upsampling and output layer2
        x_up_dirc_4 = self.up_4(x1)
        dirc_output_4 = self.final_4(x_up_dirc_4)
        return output, dirc_output_4