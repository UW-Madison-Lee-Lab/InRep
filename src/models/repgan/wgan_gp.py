
"""
	WGAN_GP class
	@author 
	@editor 
	@date 08/01/2020
"""


import torch
import torch.optim as optim

from .wgan_base_model import WGANGPBaseModel
from models.model_ops import init_weights
from utils.helper import Helper

class WGANGP(WGANGPBaseModel):
	def __init__(self, opt):
		WGANGPBaseModel.__init__(self, opt)
		# specify the training losses
		if self.is_training:
			self.model_names = ['G', 'D']
			# discriminator
			self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
			self.optimizers.append(self.optimizer_G)
		else:
			self.model_names = ['G']

		self.setup(opt)

	def forward(self, noise):
		self.fake = self.netG(noise.to(self.device))
