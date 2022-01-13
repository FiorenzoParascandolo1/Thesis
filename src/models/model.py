from typing import Optional
import torch
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


class LocallyConnected2d(nn.Module):
    """
    Implementation of Locally Connected 2d conv layer.
    https://prateekvjoshi.com/2016/04/12/understanding-locally-connected-layers-in-convolutional-neural-networks/
    """

    def __init__(self,
                 input_channels: int,
                 num_channels: int,
                 input_size: tuple,
                 kernel_size: tuple,
                 strides: tuple) -> None:
        """
        :param input_channels: number of input channels.
        :param num_channels: number of output channels.
        :param input_size: input image size (W, H)
        :param kernel_size:
        :param strides:
        :return:
        """
        super().__init__()
        # Compute height output size
        self.H_out = int((input_size[0] - (kernel_size[0] - 1) - 1) / strides[0]) + 1
        # Compute width output size
        self.W_out = int((input_size[1] - (kernel_size[1] - 1) - 1) / strides[1]) + 1
        self.stride = strides
        self.kernel_size = kernel_size
        # Matrix of convolutional layer W_out X H_out
        self.convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(in_channels=input_channels,
                                                             out_channels=num_channels,
                                                             kernel_size=kernel_size,
                                                             stride=(1, 1)) for _ in range(self.W_out)]) for _ in
                                    range(self.H_out)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        :return: activation
        """

        # Apply convolutional layer for each spatial location
        y = [[self.convs[i][j](x[:, :, i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1])])
              for j in range(self.W_out)]
             for i in range(self.H_out)]

        # Concatenate the activations produced by the different convolutional layers
        y = torch.cat([torch.cat(y[i], dim=3) for i in range(self.H_out)], dim=2)

        return y


class CoordConv(nn.Module):
    """
    Implementation of CoordConv layer.
    https://arxiv.org/pdf/1807.03247.pdf
    """

    def __init__(self,
                 input_channels: int,
                 num_channels: int,
                 input_size: tuple,
                 kernel_size: tuple,
                 strides: tuple) -> None:
        """
        :param input_channels: number of input channels.
        :param num_channels: number of output channels.
        :param input_size: input image size (W, H)
        :param kernel_size:
        :param strides:
        :return:
        """
        super().__init__()
        self.input_size = input_size
        self.conv1 = LocallyConnected2d(input_channels=7,
                                        num_channels=14,
                                        input_size=(60, 60),
                                        kernel_size=(1, 1), strides=(1, 1))
        self.conv = nn.Conv2d(in_channels=(input_channels + 2) * 2,
                              out_channels=num_channels,
                              kernel_size=kernel_size,
                              stride=strides)

        self.xx_channel = (torch.arange(input_size[0]).repeat(1, input_size[1], 1).float() /
                           (input_size[0] - 1)) * 2 - 1
        self.yy_channel = (torch.arange(input_size[1]).repeat(1, input_size[0], 1).transpose(1, 2).float() /
                           (input_size[1] - 1)) * 2 - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx_channel = self.xx_channel.repeat(x.shape[0], 1, 1, 1).transpose(2, 3) \
            if x.shape[0] != 1 else self.xx_channel.unsqueeze(dim=1)
        yy_channel = self.yy_channel.repeat(x.shape[0], 1, 1, 1).transpose(2, 3) \
            if x.shape[0] != 1 else self.yy_channel.unsqueeze(dim=1)
        x = torch.cat([x,
                       xx_channel,
                       yy_channel], dim=1)
        x = self.conv1(x)
        return self.conv(x)


class LocallyConnectedNetwork(nn.Module):

    def __init__(self,
                 pixels,
                 actor=True):
        super().__init__()

        self.actor = actor

        self.stage_1 = nn.Sequential(LocallyConnected2d(input_channels=5,
                                                        num_channels=16,
                                                        input_size=(20, 20),
                                                        kernel_size=(5, 5), strides=(1, 1)),
                                     nn.Tanh())

        self.stage_2 = nn.Sequential(LocallyConnected2d(input_channels=16,
                                                        num_channels=16,
                                                        input_size=(16, 16),
                                                        kernel_size=(5, 5), strides=(1, 1)),
                                     nn.Tanh())

        self.stage_3 = nn.Sequential(LocallyConnected2d(input_channels=16,
                                                        num_channels=32,
                                                        input_size=(12, 12),
                                                        kernel_size=(3, 3), strides=(1, 1)),
                                     nn.Tanh())

        self.stage_4 = nn.Sequential(LocallyConnected2d(input_channels=32,
                                                        num_channels=32,
                                                        input_size=(10, 10),
                                                        kernel_size=(3, 3), strides=(1, 1)),
                                     nn.Tanh())

        self.stage_5 = nn.Sequential(LocallyConnected2d(input_channels=32,
                                                        num_channels=32,
                                                        input_size=(8, 8),
                                                        kernel_size=(3, 3), strides=(1, 1)),
                                     nn.Tanh())
        self.stage_6 = nn.Sequential(LocallyConnected2d(input_channels=32,
                                                        num_channels=64,
                                                        input_size=(6, 6),
                                                        kernel_size=(3, 3), strides=(1, 1)),
                                     nn.Tanh())
        self.stage_7 = nn.Sequential(LocallyConnected2d(input_channels=64,
                                                        num_channels=64,
                                                        input_size=(4, 4),
                                                        kernel_size=(3, 3), strides=(1, 1)),
                                     nn.Tanh())

        with torch.no_grad():
            x = torch.randn(1, 5, pixels * 2, pixels * 2)
            x = self.stage_1(x)
            x = self.stage_2(x)
            x = self.stage_3(x)
            x = self.stage_4(x)
            x = self.stage_5(x)
            x = self.stage_6(x)
            x = self.stage_7(x)

            x = torch.flatten(x, start_dim=1)

        if actor:
            self.head = nn.Linear(x.shape[1] + 8, 2)
        else:
            self.head = nn.Linear(x.shape[1] + 8, 1)

    def forward(self, x, info):

        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        if len(info.shape) == 1:
            info = info.unsqueeze(dim=1)

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.stage_5(x)
        x = self.stage_6(x)
        x = self.stage_7(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, info), dim=1)

        if self.actor:
            return F.softmax(self.head(x), dim=1)
        else:
            return self.head(x)


class Vgg(nn.Module):

    def __init__(self,
                 pixels,
                 actor=True):
        super().__init__()

        self.actor = actor

        self.stage_1 = nn.Sequential(nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.Tanh(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Dropout(p=0.25))
        self.stage_2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
                                     nn.Tanh(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
                                     nn.Tanh(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Dropout(p=0.25))
        self.stage_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
                                     nn.Tanh(),
                                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3)),
                                     nn.Tanh())

        with torch.no_grad():
            x = torch.randn(1, 5, pixels * 2, pixels * 2)
            x = self.stage_1(x)
            x = self.stage_2(x)
            x = self.stage_3(x)

            x = torch.flatten(x, start_dim=1)

        if actor:
            self.head = nn.Linear(x.shape[1] + 8, 2)
        else:
            self.head = nn.Linear(x.shape[1] + 8, 1)

    def forward(self, x, info):

        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        if len(info.shape) == 1:
            info = info.unsqueeze(dim=1)

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, info), dim=1)

        if self.actor:
            return F.softmax(self.head(x), dim=1)
        else:
            return self.head(x)


class DeepFace(nn.Module):

    def __init__(self,
                 pixels,
                 actor=True):
        super().__init__()

        self.actor = actor

        self.stage_1 = nn.Sequential(nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.Tanh())
        self.stage_2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
                                     nn.Tanh(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.Tanh())

        self.stage_3 = nn.Sequential(LocallyConnected2d(input_channels=64,
                                                        num_channels=64,
                                                        input_size=(13, 13),
                                                        kernel_size=(5, 5), strides=(1, 1)),
                                     nn.Tanh())

        self.stage_4 = nn.Sequential(LocallyConnected2d(input_channels=64,
                                                        num_channels=64,
                                                        input_size=(9, 9),
                                                        kernel_size=(5, 5), strides=(1, 1)),
                                     nn.Tanh())

        self.stage_5 = nn.Sequential(LocallyConnected2d(input_channels=64,
                                                        num_channels=64,
                                                        input_size=(5, 5),
                                                        kernel_size=(3, 3), strides=(1, 1)),
                                     nn.Tanh())

        with torch.no_grad():
            x = torch.randn(1, 5, pixels * 2, pixels * 2)
            x = self.stage_1(x)
            x = self.stage_2(x)
            x = self.stage_3(x)
            x = self.stage_4(x)
            x = self.stage_5(x)
            x = torch.flatten(x, start_dim=1)

        if actor:
            self.head = nn.Linear(x.shape[1] + 8, 2)
        else:
            self.head = nn.Linear(x.shape[1] + 8, 1)

    def forward(self, x, info):

        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        if len(info.shape) == 1:
            info = info.unsqueeze(dim=1)

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.stage_5(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, info), dim=1)

        if self.actor:
            return F.softmax(self.head(x), dim=1)
        else:
            return self.head(x)
