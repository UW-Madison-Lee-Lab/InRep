"""
    GAN Reprogram class
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
from torch.nn import Parameter
from .condbatchnorm import CondBatchNorm2d, CondBatchNorm1d


class ModBlock(nn.Module):
    def __init__(self, in_dim, activation_fn):
        super(ModBlock, self).__init__()
        nonlin = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leaky": nn.LeakyReLU(0.1),
            "softplus": nn.Softplus(),
        }[activation_fn]

        self.block = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, in_dim)),
            nonlin,
            # nn.BatchNorm1d(in_dim),
            spectral_norm(nn.Linear(in_dim, in_dim)),
            nonlin,
        )

    def forward(self, x):
        out = self.block(x)
        out = out + x
        return out




class ResNetModifier_v0(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, nclasses=10):
        super(ResNetModifier, self).__init__()
        activation_fn = "relu"
        self.blocks = []
        self.nclasses = nclasses
        self.embed = nn.Embedding(num_embeddings=self.nclasses, embedding_dim=output_dim-input_dim)
        for _ in range(n_layers):
            self.blocks += [[ModBlock(in_dim=output_dim, activation_fn=activation_fn)]]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
            
    def forward(self, u, label):
        act = torch.cat((u, self.embed(label)), dim=1)
        for _, blocklist in enumerate(self.blocks):
            for block in blocklist:
                act = block(act)
        return act


class ResNetModifier(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, nclasses=10):
        super(ResNetModifier, self).__init__()
        activation_fn = "leaky"
        self.blocks = []
        self.nclasses = nclasses
        self.embed = nn.Embedding(num_embeddings=self.nclasses, embedding_dim=output_dim-input_dim)
        for _ in range(n_layers):
            self.blocks += [[ModBlock(in_dim=output_dim, activation_fn=activation_fn)]]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # self.norm = CondBatchNorm1d(input_dim, nclasses)
            
    def forward(self, u, label):
        act = torch.cat((u, self.embed(label)), dim=1)
        # act = self.norm(u, label)
        for _, blocklist in enumerate(self.blocks):
            for block in blocklist:
                act = block(act)
        return act