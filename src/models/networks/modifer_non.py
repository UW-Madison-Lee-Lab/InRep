

"""
    GAN Reprogram class
    @author 
    @editor 
    @date 08/01/2020
"""
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod, ABC
from models.model_ops import init_net, onehot, get_norm
from torch.nn.utils import spectral_norm
import constant


class ModBlock(nn.Module):
    def __init__(self, in_dim, activation_fn):
        super(ModBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_dim)
        self.bn2 = nn.BatchNorm1d(num_features=in_dim)
        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)

    def forward(self, x, label=None):
        x0 = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.fc2(x)
        out = x + x0
        return out


class ResNetModifier(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, n_neurons):
        super(ResNetModifier, self).__init__()
        activation_fn = "Leaky_ReLU"
        self.output_dim = output_dim
        n_neurons = input_dim
        self.blocks = []
        for index in range(n_layers):
            self.blocks += [[ModBlock(in_dim=n_neurons, activation_fn=activation_fn)]]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = nn.BatchNorm1d(n_neurons)
        self.activation = nn.ELU(alpha=1.0, inplace=True)
        self.fc3 = spectral_norm(nn.Linear(n_neurons, output_dim))
        self.tanh = nn.Tanh()
        # Weight init
        init_net(self, init_type='orthogonal')
    
    def project(self, x):
        # Projection to the surface of unit d-dim hypersphere : E = 0
        norms = torch.norm(x, dim=1, keepdim=True).expand(x.shape)
        return x / norms * np.sqrt(self.output_dim)
        # polar interpolation: sqrt(p) x + sqrt(1-p) y

    def forward(self, z, label=None, evaluation=False):
        act = z
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                act = block(act, label)
        act = self.bn4(act)
        act = self.activation(act)
        act = self.fc3(act)
        # out = self.tanh(act)
        out = act + z
        # out = self.project(out)
        return out
