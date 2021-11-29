
"""
	GAN Reprogram class
	@author 
	@editor 
	@date 08/01/2020
"""


import torch
import torch.optim as optim

from .wgan_base_model import WGANGPBaseModel
from models.model_ops import init_weights
from models.networks.ops import define_M
from utils.helper import Helper


class ReprogramGAN(WGANGPBaseModel):
	def __init__(self, opt):
		WGANGPBaseModel.__init__(self, opt)
		# specify the training losses
		self.netG.load_state_dict(torch.load(opt.trained_netG_path))
		self.netG.eval()
		# modifer
		self.netM = define_M(opt.data_type, opt.u_dim, opt.z_dim).to(self.device)
		if self.is_training:
			self.model_names = ['M', 'D']
			# for netM
			self.optimizer_G = optim.Adam(self.netM.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
			self.optimizers.append(self.optimizer_G)
		else:
			self.model_names = ['M']

		self.setup(opt)

	def forward(self, z):
		z = z.to(self.device)
		self.fake = self.netG(self.netM(z))
