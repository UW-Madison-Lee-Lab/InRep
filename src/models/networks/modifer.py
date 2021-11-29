"""
    GAN Reprogram class
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
from torch.nn import Parameter
from .condbatchnorm import CondBatchNorm2d, CondBatchNorm1d


class ActNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :]

    def shift(self):
        return self._shift[None, :]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() 
        return x * torch.exp(log_scale) + self.shift(), logdet





class ModBlock_v1(nn.Module):
    def __init__(self, in_dim, activation_fn):
        super(ModBlock, self).__init__()
        nonlin = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "leaky": nn.LeakyReLU(0.1),
            "softplus": nn.Softplus(),
        }[activation_fn]

        self.block = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, in_dim)),
            nonlin,
            spectral_norm(nn.Linear(in_dim, in_dim)),
            nonlin,
            # spectral_norm(nn.Linear(in_dim, in_dim)),
        )


    def forward(self, x):
        # out, _ = self.actnorm(x)
        out = x
        out = self.block(out)
        out = out + x
        return out



class ModBlock(nn.Module):
    def __init__(self, in_dim, activation_fn):
        super(ModBlock, self).__init__()
        nonlin = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "leaky": nn.LeakyReLU(negative_slope=0.1, inplace=True),
            "softplus": nn.Softplus(),
        }[activation_fn]

        self.fc1 = spectral_norm(nn.Linear(in_dim, in_dim))
        self.fc2 = spectral_norm(nn.Linear(in_dim, in_dim))
        self.nonlin = nonlin


    def forward(self, x):
        # out, _ = self.actnorm(x)
        out = x
        # out = self.block(out)
        out = self.fc1(out)
        out = self.nonlin(out)
        out = self.fc2(out)
        out = self.nonlin(out)
        out = out + x
        return out



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, input_nc, inner_nc, num_classes):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.nonlin = nn.LeakyReLU(0.1)
        self.downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=2,
                                stride=2, padding=1, bias=False)
        self.upconv = nn.ConvTranspose2d(inner_nc, input_nc,
                                    kernel_size=2, stride=2,
                                    padding=1, bias=False)
        self.norm = CondBatchNorm2d(inner_nc, num_classes)

    def forward(self, x, labels):
        out = self.downconv(x)
        out = self.norm(out, labels)
        out = self.nonlin(out)
        out = self.upconv(out)
        return out
        


class Adapter(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Adapter, self).__init__()
        self.nonlin = nn.LeakyReLU(0.2)
        self.downconv = nn.Linear(in_features, 128, bias=True)
        self.upconv = nn.Linear(128, in_features, bias=True)
        self.norm = CondBatchNorm1d(128, num_classes)

    def forward(self, x, labels):
        out = self.downconv(x)
        out = self.norm(out, labels)
        out = self.nonlin(out)
        out = self.upconv(out)
        return out
        


class ResNetModifier(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, nclasses=10):
        super(ResNetModifier, self).__init__()
        activation_fn = "leaky"
        self.blocks = []
        self.nclasses = nclasses
        self.embed = nn.Embedding(num_embeddings=self.nclasses, embedding_dim=10)
        for _ in range(n_layers):
            self.blocks += [[ModBlock(in_dim=output_dim, activation_fn=activation_fn)]]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # if self.nclasses > input_dim:
            # self.fc = spectral_norm(nn.Linear(input_dim + min(nclasses, 50), output_dim))
        # else:
        # #     self.fc = None
        # self.adapter = None
        # self.adapter = Adapter(channels * (image_size // 8) *image_size // 8), nclasses)
        # self.adapter = nn.Sequential(
        #     UnetSkipConnectionBlock(channels, channels//2, nclasses)
        # )
        
            
    def forward(self, u, label):
        act = torch.cat((u, self.embed(label)), dim=1)
        # if self.fc is not None:
            # act = self.fc(act)
        for _, blocklist in enumerate(self.blocks):
            for block in blocklist:
                act = block(act)
        return act