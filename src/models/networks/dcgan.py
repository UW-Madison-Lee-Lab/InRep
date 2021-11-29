

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
import constant

class BaseGenerator(nn.Module):
    def __init__(self, condition_strategy, noise_dim, embed_dim, num_class):
        super().__init__()
        self.condition_strategy, self.noise_dim, self.embed_dim, self.num_class = condition_strategy, noise_dim, embed_dim, num_class
        if self.condition_strategy:
            self.noise_dim += self.num_class
        # self.embed_dim = ?

    @abstractmethod
    def make_embed_net(self, nz, ngf, nc):
        pass
        return deconv

    def cat_label_noises(self, noises, labels):
        labels = onehot(labels, self.num_class)  # B -> B x ny
        noises = torch.cat([noises, labels], 1)
        return noises

    def make_net(self, ngf, nc):
        self.embed_noises = nn.Linear(self.noise_dim, self.embed_dim)
        self.deconv = self.make_embed_net(self.embed_dim, ngf, nc)
        init_net(self, init_type='orthogonal')

    def forward(self, noises, labels=None):
        if self.condition_strategy == 'no':
            x = noises
        else:
            noises = self.cat_label_noises(noises, labels)
            x = self.embed_noises(noises)
        x = x.view(-1, self.embed_dim, 1, 1)
        x = self.deconv(x)
        return x




class BaseDiscriminator(nn.Module):
    def __init__(self, condition_strategy, num_class):
        super().__init__()
        self.condition_strategy, self.num_class = condition_strategy, num_class

    @abstractmethod
    def make_embed_net(self, nc, ndf):
        pass

    def make_net(self, nc, ndf):
        self.embed_net, self.out_d = self.make_embed_net(nc, ndf)
        # self.out_d = nn.Linear(self.embed_dim, 1)
        if self.condition_strategy == 'acgan':
            self.out_c = nn.Linear(self.embed_dim * 16, self.num_class)
        elif self.condition_strategy == 'projgan':
            self.linear_y = norm(nn.Embedding(self.num_class, self.embed_dim))
        init_net(self, init_type='orthogonal')

    def embed(self, input):
        x = self.embed_net(input)
        return x

    def forward(self, input, labels=None):
        # label is None
        x = self.embed_net(input)
        authen_output = self.out_d(x)
        if labels is None or self.condition_strategy == 'no':
            return authen_output
        else:
            c_x = torch.flatten(x, start_dim=1)
            if self.condition_strategy == 'acgan':
                cls_output = self.out_c(c_x)
                return authen_output, cls_output 
            elif self.condition_strategy == 'projgan':
                proj_output = torch.sum(self.linear_y(labels) * c_x, dim=1, keepdim=True)
                return authen_output + proj_output