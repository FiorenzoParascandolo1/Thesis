from typing import Optional
import torch
from scipy.stats import entropy
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch import nn


class LocallyConnected2d(nn.Module):
    def __init__(self, input_channels, num_channels, input_size, kernel_size=(3, 3), strides=(1, 1)):
        super().__init__()
        self.H_out = int((input_size[0] - (kernel_size[0] - 1) - 1) / strides[0]) + 1
        self.W_out = int((input_size[1] - (kernel_size[1] - 1) - 1) / strides[1]) + 1
        self.stride = strides
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(in_channels=input_channels,
                                                             out_channels=num_channels,
                                                             kernel_size=kernel_size,
                                                             stride=(1, 1)) for _ in range(self.W_out)]) for _ in
                                    range(self.H_out)])

    def forward(self, x):
        y = [[self.convs[i][j](x[:, :,
                               (i * self.kernel_size[0]):(i * self.kernel_size[0] + self.kernel_size[0]),
                               (j * self.kernel_size[1]):(j * self.kernel_size[1] + self.kernel_size[1])])
              for j in range(self.W_out)]
             for i in range(self.H_out)]

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
                 bn_layer: bool = False):
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

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=kernel_size_dimension[dimension])

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.w_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=kernel_size_dimension[dimension]),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local
            # block is identity mapping
            # nn.init.constant_(self.w_z[1].weight, 0)
            # nn.init.constant_(self.w_z[1].bias, 0)
        else:
            self.w_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=kernel_size_dimension[dimension])

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing
            # architecture
            # nn.init.constant_(self.w_z.weight, 0)
            # nn.init.constant_(self.w_z.bias, 0)

        # define theta and phi for all operations except gaussian
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
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
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
            # number of position in x
            n = f.shape[-1]
            f_dic_c = f / n

        y = torch.matmul(f_dic_c, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        w_y = self.w_z(y)
        # residual connection
        z = w_y + x

        return z


class Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, input_shape, input_channels, num_channels, strides=(1, 1), use_1x1conv=True):
        super().__init__()

        self.conv1 = LocallyConnected2d(input_shape, input_channels, num_channels,
                                        kernel_size=(3, 3), stride=strides, padding=1)
        self.conv2 = LocallyConnected2d([int(input_shape / 2), int(input_shape / 2)], input_channels, num_channels,
                                        kernel_size=(3, 3), stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=(1, 1), stride=strides)
        else:
            self.conv3 = None
        # self.non_local = NonLocalBlock(num_channels)

    def forward(self, X):
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)

        if self.conv3:
            X = self.conv3(X)
        Y += F.relu(X)
        # Y = self.non_local(Y)

        return Y


class resCNN(nn.Module):
    def __init__(self, actor=True):
        super().__init__()
        self.actor = actor

        # self.nl = NonLocalBlock(6)
        self.b0 = LocallyConnected2d(input_channels=6,
                                     num_channels=32,
                                     input_size=(60, 60),
                                     kernel_size=(3, 3), strides=(3, 3))
        self.b1 = LocallyConnected2d(input_channels=32,
                                     num_channels=64,
                                     input_size=(20, 20),
                                     kernel_size=(2, 2), strides=(2, 2))
        # self.bn1 = nn.BatchNorm2d(32)
        self.b2 = LocallyConnected2d(input_channels=64,
                                     num_channels=128,
                                     input_size=(10, 10),
                                     kernel_size=(5, 5), strides=(5, 5))
        # self.bn2 = nn.BatchNorm2d(64)
        """
        self.b3 = LocallyConnected2d(input_channels=128,
                                     num_channels=256,
                                     input_size=(10, 10),
                                     kernel_size=(5, 5), strides=(5, 5))
        """
        # self.bn3 = nn.BatchNorm2d(128)

        # self.linear_1 = nn.Linear(512, 1)
        if self.actor:
            self.classifier = nn.Linear(514, 2)
        else:
            self.classifier = nn.Linear(514, 1)

    def forward(self, x, info):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        if len(info.shape) == 1:
            info = info.unsqueeze(dim=1)
        """

        x = torch.cat([x[:, :, 0:30, 0:30],
                       x[:, :, 0:30, 30:60],
                       x[:, :, 30:60, 0:30],
                       x[:, :, 30:60, 30:60]], dim=3)
        plt.imshow(x[0, 0:3, :, :].permute(1, 2, 0).detach().numpy())
        plt.show()
        x = torch.tanh(self.nl(x))
        x = torch.cat([torch.cat([x[:, :, 0:30, 0:30],
                                  x[:, :, 0:30, 30:60]], dim=3),
                       torch.cat([x[:, :, 0:30, 60:90],
                                  x[:, :, 0:30, 90:120]], dim=3)
                       ], dim=2)
        plt.imshow(x[0, 0:3, :, :].permute(1, 2, 0).detach().numpy())
        plt.show()
        """
        x = torch.tanh(self.b0(x))
        x = torch.tanh(self.b1(x))
        x = torch.tanh(self.b2(x))
        # x = F.leaky_relu(self.b3(x))
        x = torch.flatten(x, start_dim=1)
        # x = F.leaky_relu(F.dropout(self.linear_1(x), p=0.5))
        x = torch.cat((x, info), dim=1)
        x = self.classifier(x)

        if self.actor:
            x = F.softmax(x, dim=1)
        return x
