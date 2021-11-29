
"""
	WGAN_GP class
	@editor 
	@date 08/01/2020
"""


import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import BaseModel
from models.model_ops import GANLoss
from models.networks.ops import define_D, define_G
from utils.helper import Helper


class VGAN(BaseModel):
	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		# specify the training losses
		self.loss_names = ['D', 'G', 'dis_errD_real', 'dis_errD_fake']
		self.visual_names = ['real', 'fake']
		# generator
		self.latent_dim = opt.z_dim
		self.netG = define_G(opt.data_type, False, opt.z_dim, opt.num_classes).to(self.device)
		if self.is_training:
			self.model_names = ['G', 'D']
			# discriminator
			self.netD = define_D(opt.data_type, False, opt.num_classes).to(self.device)
			# loss
			self.dis_criterion = GANLoss().to(self.device)
			# optimizers
			self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
			self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "w"
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)
		else:
			self.model_names = ['G']
			self.netM.eval()

		self.setup(opt)

	def forward(self, z):
		noise = z.to(self.device)
		self.fake = self.netG(noise)

	def backward_D(self, real):
		"""Calculate GAN loss for the discriminator"""
		self.netD.train()
		# Fake; stop backprop to the generator by detaching fake_B
		dis_output_fake = self.netD(self.fake.detach())
		dis_errD_fake = self.dis_criterion(dis_output_fake, False)

		real = real.to(self.device)
		# unlabeld  loss
		dis_output_real = self.netD(real)
		dis_errD_real = self.dis_criterion(dis_output_real, True)

		self.loss_D = dis_errD_real + dis_errD_fake
		self.loss_dis_errD_real = dis_errD_real
		self.loss_dis_errD_fake = dis_errD_fake
		self.loss_D.backward()

	def backward_G(self):
		# self.netD.eval()
		dis_output_fake = self.netD(self.fake)
		dis_errD_fake = self.dis_criterion(dis_output_fake, True) # real labels here
		self.loss_G = dis_errD_fake
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

	def sample(self, data_size):
		self.netG.eval()
		with torch.no_grad():
			noises = Helper.make_z_normal_(data_size, self.latent_dim).to(self.device)
			fake_images = self.netG(noises)
		self.netG.train()
		return fake_images

	def train_iter(self, epoch, dataloader):
		for _, (images, labels) in enumerate(dataloader):
			noises = Helper.get_noises(True, images.shape[0], self.latent_dim)
			self.set_requires_grad(self.netD, True)  # enable backprop for D
			self.forward(noises)  # compute fake images: G(A)
			self.optimize_D(images)
			self.set_requires_grad(self.netD, False)
			self.optimize_G()