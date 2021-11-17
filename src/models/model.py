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
    TODO: the class actually works iff kernel_size = stride. Extend for the case kernel_size != stride
    TODO: evaluate to add skip connection
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
        y = [[self.convs[i][j](x[:, :,
                               (i * self.kernel_size[0]):(i * self.kernel_size[0] + self.kernel_size[0]),
                               (j * self.kernel_size[1]):(j * self.kernel_size[1] + self.kernel_size[1])])
              for j in range(self.W_out)]
             for i in range(self.H_out)]

        # Concatenate the activations produced by the different convolutional layers
        y = torch.cat([torch.cat(y[i], dim=3) for i in range(self.H_out)], dim=2)

        return y


class NonLocalBlock(nn.Module):
    """
    Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick.
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf
    """

    def __init__(self,
                 in_channels: int,
                 inter_channels: Optional[int] = None,
                 mode: str = 'gaussian',
                 dimension: int = 2,
                 bn_layer: bool = False) -> None:
        """
        :param in_channels: original channel size (1024 in the paper)
        :param inter_channels: channel size inside the block if not specified reduced to half (512 in the paper)
        :param mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
        :param dimension: can be 1 (temporal), 2 (spatial), 3 (spatio-temporal)
        :param bn_layer: whether to add batch norm
        """
        super(NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        kernel_size_dimension = {1: (1,),
                                 2: (1, 1),
                                 3: (1, 1, 1)}

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # The channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # Assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        # Function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=kernel_size_dimension[dimension])

        # Add BatchNorm layer after the last conv layer
        if bn_layer:
            self.w_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=kernel_size_dimension[dimension]),
                bn(self.in_channels)
            )
            # From section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local
            # Block is identity mapping
            nn.init.constant_(self.w_z[1].weight, 0)
            nn.init.constant_(self.w_z[1].bias, 0)
        else:
            self.w_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=kernel_size_dimension[dimension])

            # From section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing
            # architecture
            nn.init.constant_(self.w_z.weight, 0)
            nn.init.constant_(self.w_z.bias, 0)

        # Define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=kernel_size_dimension[dimension])
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=kernel_size_dimension[dimension])

        if self.mode == "concatenate":
            self.w_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1,
                          kernel_size=kernel_size_dimension[dimension]),
                nn.ReLU()
            )

    def forward(self, x):
        """
        :param x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        global f, f_dic_c
        batch_size = x.size(0)

        # (N, C, THW)
        # This reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.w_f(concat)
            f = f.view(f.shape[0], f.shape[2], f.shape[3])

        if self.mode == "gaussian" or self.mode == "embedded":
            f_dic_c = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            # Number of position in x
            n = f.shape[-1]
            f_dic_c = f / n

        y = torch.matmul(f_dic_c, g_x)

        # Contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        w_y = self.w_z(y)
        # Residual connection
        z = w_y + x

        return z


class LocallyConnectedNetwork(nn.Module):
    """
    Neural network architecture used for actor/critic
    """
    def __init__(self,
                 actor=True) -> None:
        """
        :param actor: actor influences the output size: if actor -> output_size = |action_space|, 1 otherwise.
        :return:
        """
        super().__init__()

        # TODO: change name; add hyper parameters for kernel_size, strides, number of layers, action space cardinality
        self.actor = actor
        self.b0 = LocallyConnected2d(input_channels=5,
                                     num_channels=32,
                                     input_size=(60, 60),
                                     kernel_size=(3, 3), strides=(3, 3))
        self.b1 = LocallyConnected2d(input_channels=32,
                                     num_channels=64,
                                     input_size=(20, 20),
                                     kernel_size=(2, 2), strides=(2, 2))
        self.b2 = LocallyConnected2d(input_channels=64,
                                     num_channels=128,
                                     input_size=(10, 10),
                                     kernel_size=(5, 5), strides=(5, 5))

        """
        self.b0 = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(3, 3))
        self.b1 = nn.Conv2d(32, 64, kernel_size=(2, 2), stride=(2, 2))
        self.b2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(5, 5))
        """

        # The actor neural network returns the probability for each action
        if self.actor:
            self.classifier = nn.Linear(521, 2)
        # The critic neural network return the state-value
        else:
            self.classifier = nn.Linear(521, 1)

    def forward(self,
                x: torch.Tensor,
                info: torch. Tensor) -> torch.Tensor:
        """
        :param x: GAF images tensor
        :param info: info tensor
        :return: probability for each action if actor; state_value otherwise
        TODO: encapsulate 'info' in 'x' somehow
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        if len(info.shape) == 1:
            info = info.unsqueeze(dim=1)
        # plt.imshow(x[0, :3, :, :].detach().permute(1, 2, 0).numpy())
        # plt.show()
        x = torch.tanh(self.b0(x))
        x = torch.tanh(self.b1(x))
        x = torch.tanh(self.b2(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, info), dim=1)
        x = self.classifier(x)

        if self.actor:
            x = F.softmax(x, dim=1)
        return x