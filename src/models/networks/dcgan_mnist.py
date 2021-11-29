

"""
    GAN Reprogram class
    @editor 
    @date 08/01/2020
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod, ABC
from models.model_ops import init_net, onehot, get_norm
import constant



class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)

class Generator(nn.Module):
    def __init__(self, noise_dim, image_size, num_class):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_size = image_size
        self.num_class = num_class
        ngf = 64
        nz = noise_dim
        nc=1
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )
        init_net(self, init_type='orthogonal')

    def forward(self, noises, labels=None):
        x = noises
        x = x.view(-1, self.noise_dim, 1, 1)
        x = self.main(x)
        return x

# class Generator(nn.Module):
#     def __init__(self, nz):
#         super(Generator, self).__init__()
#         self.nz = nz
#         self.main = nn.Sequential(
#             nn.Linear(self.nz, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, 784),
#             nn.Tanh(),
#         )
#     def forward(self, x):
#         return self.main(x).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self, condition_strategy, num_class):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.authen = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.embed = nn.utils.spectral_norm(nn.Embedding(num_class, 256))
    def forward(self, x, labels=None):
        x = x.view(-1, 784)
        output = self.main(x)
        authen_output = self.authen(output)
        if labels is not None:
            out_y = torch.sum(self.embed(labels) * output, dim=1, keepdim=True) # class 
            authen_output += out_y
        return authen_output, None

# class Discriminator(nn.Module):
#     def __init__(self, condition_strategy, num_class):
#         super().__init__()
#         self.condition_strategy, self.num_class = condition_strategy, num_class
#         nc, ndf = 1, 16
#         self.embed_dim = 36 * ndf
#         self.main = nn.Sequential(
#              # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
#         )
        # self.linear_y = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.num_class), 
        #     nn.utils.spectral_norm(nn.Embedding(self.num_class, self.embed_dim)),
        #     nn.Linear(self.embed_dim, self.num_class), 
        #     )
#         init_net(self, init_type='orthogonal')

#     def embed(self, inputs):
#         x = self.main[:-1](inputs)
#         x = torch.flatten(x, start_dim=1)
#         return x
    
#     def projection(self, inputs, labels):
#         x = self.embed(inputs)
#         x_auth = self.linear_y[2](x)
#         x_proj = torch.sum(self.linear_y[1](labels) * x, dim=1, keepdim=True)
#         x_aux = self.linear_y[0](x)
#         encoded_labels = onehot(labels, self.num_class).view(inputs.shape[0], self.num_class, 1)
#         # trick
#         x_auth = torch.bmm(x_auth.view(inputs.shape[0], 1, self.num_class), encoded_labels).view(inputs.shape[0])
#         return x_auth, x_aux, x_proj

#     def forward(self, inputs, labels=None):
#         # x = self.main[:-1](inputs)
#         # x = torch.flatten(x, start_dim=1)
#         # authen_output = self.out_d(x)
#         authen_output = self.main(inputs)
#         return authen_output, None



# class Discriminator(nn.Module):
#     def __init__(self, condition_strategy, num_class):
#         super().__init__()
#         self.condition_strategy, self.num_class = condition_strategy, num_class
#         ndf = 4
#         self.embed_dim = 512
#         self.embed_net = self.make_embed_net(1, ndf)
#         self.out_d = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#         norm = get_norm(True)
#         if self.condition_strategy == constant.ACGAN:
#             self.out_c = nn.Linear(self.embed_dim * 16, self.num_class)
#         elif self.condition_strategy == constant.PROJGAN:
#             self.linear_y = norm(nn.Embedding(self.num_class, self.embed_dim))
#         init_net(self, init_type='orthogonal')

#     def make_embed_net(self, nc, ndf):
#         norm = get_norm(use_sn=True)
#         embed_net = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
            
#         )
#         out_d = nn.Sequential(
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#         )
#         return embed_net


#     def embed(self, input):
#         x = self.embed_net(input)
#         x = torch.flatten(x, start_dim=1)
#         # x = x.view(x.shape[0], -1)
#         return x

#     def forward(self, input, labels=None):
#         # label is None
#         x = self.embed_net(input)
#         x = torch.flatten(x, start_dim=1)
#         authen_output = self.out_d(x)
#         if labels is None or self.condition_strategy == 'no':
#             return authen_output
#         else:
#             c_x = torch.flatten(x, start_dim=1)
#             if self.condition_strategy == constant.ACGAN:
#                 cls_output = self.out_c(c_x)
#                 return authen_output, cls_output 
#             elif self.condition_strategy == constant.PROJGAN:
#                 proj_output = torch.sum(self.linear_y(labels) * c_x, dim=1, keepdim=True)
#                 return authen_output + proj_output


if __name__ == '__main__':
    D = Discriminator('no', 10).eval()
    G = Generator('no', 100, 100, 10).eval()

    # load weights
    D.load_state_dict(torch.load('../../../results/checkpoints/repgan/mnist-10/uncond/net_D.pth'))
    G.load_state_dict(torch.load('../../../results/checkpoints/repgan/mnist-10/uncond/net_G.pth'))