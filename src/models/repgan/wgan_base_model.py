
"""
	WGAN_GP class
	@author 
	@editor 
	@date 08/01/2020
"""


import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import BaseModel
from models.model_ops import init_weights, gradient_penalty
from models.networks.ops import define_G, define_D
from utils.helper import Helper
import constant

class WGANGPBaseModel(BaseModel):
	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		# specify the training losses
		self.loss_names = ['D', 'G', 'W_dist', 'grad_penalty', 'D_real', 'D_fake']
		self.visual_names = ['real', 'fake']
		self.lamb = opt.wgan_lambda
		self.image_dataset = opt.data_type != constant.TOY
		# generator
		self.netG = define_G(opt.data_type, opt.gan_type, opt.z_dim, opt.num_classes).to(self.device)

		if self.is_training:
			# discriminator
			self.netD = define_D(opt.data_type, opt.gan_type, opt.num_classes).to(self.device)
			self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.d_lr , betas=(opt.beta1, opt.beta2)) # Optimize over "w"
			self.optimizers.append(self.optimizer_D)

		self.setup(opt)

	def backward_D(self, real):
		"""Calculate GAN loss for the discriminator"""
		# adding some instant noises
		self.netD.train()
		real_ = real.to(self.device) #+ torch.randn(real.shape).to(self.device) * self.eps
		fake_ = self.fake #+ torch.randn(self.fake.shape).to(self.device) * self.eps

		# Fake; stop backprop to the generator by detaching fake_B
		D_fake = self.netD(fake_.detach())
		loss_D_fake = torch.mean(D_fake)
		# Real
		D_real = self.netD(real_)
		loss_D_real = torch.mean(D_real)
		# gradient penalty
		self.loss_grad_penalty = gradient_penalty(self.netD, real_, fake_, self.device, True)

		# combine loss and calculate gradients
		self.loss_D = -loss_D_real + loss_D_fake + self.lamb * self.loss_grad_penalty
		self.loss_W_dist = loss_D_real - loss_D_fake
		self.loss_D_real = loss_D_real
		self.loss_D_fake = loss_D_fake
		self.loss_D.backward()

	def backward_G(self):
		fake_ = self.fake #+ torch.randn(self.fake.shape).to(self.device) * self.eps
		D_fake = self.netD(fake_)
		self.loss_G = -torch.mean(D_fake)
		self.loss_G.backward()

	def optimize_D(self, real):
		# update D
		self.optimizer_D.zero_grad()     # set D's gradients to zero
		self.backward_D(real)                # calculate gradients for D
		self.optimizer_D.step()          # update D's weights

	def optimize_G(self):
		self.optimizer_G.zero_grad()        # set G's gradients to zero
		self.backward_G()                   # calculate graidents for G
		self.optimizer_G.step()             # udpate G's weights
