
"""
	WGAN_GP class
	@editor 
	@date 08/01/2020
"""


import torch
import torch.nn as nn
import torch.optim as optim
import random, math
import numpy as np

from models.base_model import BaseModel
from models.model_ops import GANLoss
from models.networks.ops import define_D, define_G, define_M, define_F
from utils.helper import Helper


class ClassRepGAN(BaseModel):
	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		# specify the training losses
		self.loss_names = ['D', 'G', 'dis_errD_real', 'dis_errD_fake']
		self.visual_names = ['real', 'fake']
		# generator
		self.netG = define_G(opt.data_type, False, opt.z_dim, opt.num_classes).to(self.device)
		self.netG.load_state_dict(torch.load(opt.trained_netG_path))
		self.netG.eval()
		self.netD_uncond = define_D(opt.data_type, False, opt.num_classes).to(self.device)
		self.netD_uncond.load_state_dict(torch.load(opt.trained_netD_path))
		self.netD_uncond.eval()

		self.netM = define_M(opt.data_type, opt.u_dim, opt.z_dim).to(self.device)
		self.nclasses = opt.repgan_num_classes
		self.gan_class = opt.gan_class
		self.u_dim = opt.u_dim
		self.mode_lambda = 1.0
		self.is_pu = opt.repgan_pu
		if self.is_training:
			self.model_names = ['M', 'D']
			self.dis_criterion = GANLoss().to(self.device)
			# discriminator
			self.netD = define_F(opt.data_type, self.netD_uncond.embed_dim).to(self.device)
			# self.netD = define_D(opt.data_type, opt.gan_type, opt.num_classes).to(self.device)
			# loss
			self.optimizer_G = optim.Adam(self.netM.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
			self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "w"
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)
		else:
			self.model_names = ['M']
			self.netM.eval()
		self.setup(opt)

	def hinge_loss(self, x, real=True):
		# 	DX_score = torch.abs(DX_score - 1/self.nclasses)
		if real:
			DX_score = x
		else:
			DX_score = -x
		score = torch.mean(torch.relu(1 - x))
		return score

	def diverse(self):
		# mode_penalty
		tau = 0.1
		n = z.shape[0]
		indices = np.arange(n)
		random.shuffle(indices)
		noise_1 = self.noise.view(n, -1)
		fake_1 = self.fake.view(n, -1)
		noise_2 = noise_1[indices, ...]
		fake_2 = fake_1[indices, ...]
		norm_g = torch.norm(fake_2 - fake_1, dim=1)
		norm_z = torch.norm(noise_2 - noise_1, dim=1)
		v = norm_g/(norm_z + 1e-8)
		self.loss_penalty = torch.mean(tau - torch.relu(tau - v))

	def forward(self, z):
		z = z.to(self.device)
		self.noise = self.netM(z)
		# fake_labels = Helper.make_y(z.shape[0], self.nclasses, self.gan_class).to(self.device)
		self.fake = self.netG(self.noise)
		return self.fake

	def pu_loss(self, alpha, dis_errD_real, dis_errD_real_g, dis_errD_fake):
		coeff = (1 - alpha*(1-1/self.nclasses)) * 0.5
		# loss = coeff * dis_errD_real + torch.relu(dis_errD_fake - coeff*dis_errD_real_g)
		loss = coeff * dis_errD_real + torch.relu(dis_errD_fake - coeff*dis_errD_real_g)
		return loss

	def backward_D(self, real, alpha):
		"""Calculate GAN loss for the discriminator"""
		self.netD.train()
		# Fake; stop backprop to the generator by detaching fake_B
		# dis_output_fake = self.netD(self.fake.detach())
		dis_output_fake = self.netD(self.netD_uncond.embed(self.fake.detach()))
		dis_errD_fake = self.dis_criterion(dis_output_fake, False)

		real = real.to(self.device)
		# dis_output_real = self.netD(real)
		dis_output_real = self.netD(self.netD_uncond.embed(real))
		dis_errD_real = self.dis_criterion(dis_output_real, True)

		if self.is_pu:
			dis_errD_real_g = self.dis_criterion(dis_output_real, False)
			self.loss_D = self.pu_loss(alpha, dis_errD_real, dis_errD_real_g, dis_errD_fake)
		else:
			self.loss_D = dis_errD_fake + dis_errD_real

		self.loss_dis_errD_real = dis_errD_real
		self.loss_dis_errD_fake = dis_errD_fake
		self.loss_D.backward()

	def backward_G(self, k=-1):
		# self.netD.eval()
		dis_output_fake = self.netD(self.netD_uncond.embed(self.fake))
		# dis_output_fake = self.netD(self.fake)
		self.loss_G = self.dis_criterion(dis_output_fake, True)
		self.loss_G.backward()

	def optimize_D(self, real, alpha):
		# update D
		self.optimizer_D.zero_grad()     # set D's gradients to zero
		self.backward_D(real, alpha)                # calculate gradients for D
		self.optimizer_D.step()          # update D's weights

	def optimize_G(self, k=-1):
		self.optimizer_G.zero_grad()        # set G's gradients to zero
		self.backward_G(k)                   # calculate graidents for G
		self.optimizer_G.step()             # udpate G's weights

	def sample(self, data_size):
		self.netG.eval()
		self.netM.eval()
		with torch.no_grad():
			noises = Helper.get_noises(False, data_size, self.u_dim).to(self.device)
			fake_images = self.netG(self.netM(noises))
		self.netG.train()
		self.netM.train()
		return fake_images

	def train_iter(self, epoch, dataloader, init=1.0):
		alpha = init * math.pow(0.99, epoch)
		for batch_idx, (images, labels) in enumerate(dataloader):
			noises = Helper.get_noises(False, images.shape[0], self.u_dim)
			self.set_requires_grad(self.netD, True)  # enable backprop for D
			self.forward(noises)  # compute fake images: G(A)
			self.optimize_D(images, alpha)
			self.set_requires_grad(self.netD, False)
			self.optimize_G()
