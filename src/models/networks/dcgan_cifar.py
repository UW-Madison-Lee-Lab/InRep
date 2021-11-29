

"""
	GAN Reprogram class
	@editor 
	@date 08/01/2020
"""
import torch
import torch.nn as nn
from .dcgan import BaseGenerator, BaseDiscriminator
from models.model_ops import get_norm


class Generator(BaseGenerator):
	def __init__(self, gan_type, noise_dim, embed_dim, num_class, ngf, nc):
		BaseGenerator.__init__(self, gan_type, noise_dim, embed_dim, num_class)
		self.make_net(ngf, nc)

	def make_embed_net(self, nz, ngf, nc):
		deconv = nn.Sequential(
			# tconv1
			# input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 32 x 32
		)
		return deconv



class Discriminator(BaseDiscriminator):
	def __init__(self, gan_type, num_class, nc, ndf):
		BaseDiscriminator.__init__(self, gan_type, num_class)
		self.make_net(nc, ndf)

	def make_embed_net(self, nc, ndf):
		norm = get_norm(use_sn=True)
		embed_net = nn.Sequential(
            norm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            norm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 1 x 32
            norm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            norm(nn.Conv2d(ndf*2, ndf * 2, 4, 2, 1, bias=True)),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4
            norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True)),
            # nn.LeakyReLU(0.1, inplace=True),
            # norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
		)
		out_d = nn.Sequential(
			nn.LeakyReLU(0.1, inplace=True),
			norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
		)
		self.embed_dim = 8 * ndf# * 4 * 4
		return embed_net, out_d
