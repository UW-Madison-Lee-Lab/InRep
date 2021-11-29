
"""
	WGAN_GP class
	@editor 
	@date 08/01/2020
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

from models.base_model import BaseModel
from models.model_ops import GANLoss
from models.nets import define_D, define_G
from utils.helper import Helper
from utils.losses import loss_hinge_dis, loss_hinge_gen


class ContraGAN(BaseModel):
	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		# specify the training losses
		self.loss_names = ['D', 'G']
		self.visual_names = ['real', 'fake']
		self.z_dim, self.nclasses = opt.z_dim, opt.num_classes
		# generator
		self.netG = define_G(opt).to(self.device)
		self.model_names = ['G']
		self.setup(opt)

	def forward(self, noise, label):
		self.fake = self.netG(noise.to(self.device), label.to(self.device))
		return self.fake
	
	def sample(self, data_size, labels=None, target_class=None):
		self.netG.eval()
		with torch.no_grad():
			noises = Helper.make_z_normal_(data_size, self.z_dim).to(self.device)
			if labels is None:
				if target_class is None:
					labels = Helper.make_y(data_size, self.nclasses)
				else:
					labels = Helper.make_y(data_size, self.nclasses, target_class)
			labels = labels.long().to(self.device)
			fake_images = self.netG(noises, labels)
		return fake_images