"""
    GAN Reprogram class
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
from torch.nn import Parameter


class ActNorm_v0(nn.Module):
    def __init__(self, channels):
        super(ActNorm, self).__init__()
        size       = (1,channels)
        self.logs  = torch.nn.Parameter(torch.zeros(size,dtype=torch.float,requires_grad=True))
        self.b     = torch.nn.Parameter(torch.zeros(size,dtype=torch.float,requires_grad=True))
        self.initialized = False
        
    def initialize(self, x):
        with torch.no_grad():
            b_    = x.clone().mean(dim=0,keepdim=True)
            s_    = ((x.clone() - b_)**2).mean(dim=0,keepdim=True)
            b_    = -1 * b_
            logs_ = -1 * torch.log(torch.sqrt(s_)) 
            self.logs.data.copy_(logs_.data)
            self.b.data.copy_(b_.data)
            self.initialized = True

    def apply_bias(self, x):
        x = x + self.b
        if np.isnan(x.mean().item()):
            from IPython import embed; embed()
        assert not np.isnan(x.mean().item()), "nan after apply_bias in forward: x=%0.3f, b=%0.3f"%(x.mean().item(), self.b.mean().item())
        assert not np.isinf(x.mean().item()), "inf after apply_bias in forward: x=%0.3f, b=%0.3f"%(x.mean().item(), self.b.mean().item())
        return x
        
    def apply_scale(self, x):
        x = x * torch.exp(self.logs)
        return x
        
    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        x = self.apply_bias(x)
        x = self.apply_scale(x)
        loss_mean   = x.mean(dim=0,keepdim=True).mean()
        loss_std    = ((x - loss_mean)**2).mean(dim=0,keepdim=True).mean()
        actnormloss = torch.abs(loss_mean) + torch.abs(1 - loss_std)
            
        return x, actnormloss
    

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



class ModBlock(nn.Module):
    def __init__(self, in_dim, activation_fn):
        super(ModBlock, self).__init__()
        # if activation_fn == "ReLU":
        #     self.activation = nn.ReLU(inplace=True)
        # elif activation_fn == "Leaky_ReLU":
        #     self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # elif activation_fn == "ELU":
        #     self.activation = nn.ELU(alpha=1.0, inplace=True)
        # elif activation_fn == "GELU":
        #     self.activation = nn.GELU()
        # elif activation_fn == "PReLU":
        #     self.activation = nn.PReLU()
        # else:
        #     raise NotImplementedError
        
        nonlin = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "leaky": nn.LeakyReLU(0.2), #nn.LeakyReLU(negative_slope=0.1, inplace=True)
            "softplus": nn.Softplus(),
        }[activation_fn]

        self.block = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, in_dim)),
            nonlin,
            spectral_norm(nn.Linear(in_dim, in_dim)),
            nonlin
        )
        # self.actnorm = ActNorm(in_dim)

    def forward(self, x):
        # out, _ = self.actnorm(x)
        out = x
        out = self.block(out)
        out = out + x
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
        #     self.fc = spectral_norm(nn.Linear(input_dim + self.nclasses, output_dim))
        # else:
        #     self.fc = None
            
    def forward(self, u, label):
        act = torch.cat((u, self.embed(label)), dim=1)
        # if self.fc is not None:
            # act = self.fc(act)
        for _, blocklist in enumerate(self.blocks):
            for block in blocklist:
                act = block(act)
        return act