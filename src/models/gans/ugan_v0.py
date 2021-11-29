import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

from models.base_model import BaseModel
from models.nets import define_D, define_G
from models.model_ops import GANLoss
from utils.helper import Helper
from utils.losses import loss_hinge_dis, loss_hinge_gen


class UGAN(BaseModel):
	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		# specify the training losses
		self.loss_names = ['D', 'G']
		self.visual_names = ['real', 'fake']
		self.z_dim = opt.z_dim
		self.nclasses = opt.num_classes
		# generator
		self.netG = define_G(opt).to(self.device)

		if self.is_training:
			self.model_names = ['G', 'D']
			# discriminator
			self.netD = define_D(opt).to(self.device)
			self.criterion = GANLoss().to(self.device)
			# optimizers
			self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
			self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "w"
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)
		else:
			self.model_names = ['G']

		self.setup(opt)

	def forward(self, noise, label=None):
		# random labels
		# fake images
		self.fake = self.netG(noise.to(self.device), None)
		return self.fake

	def backward_D(self, real, label):
		"""Calculate GAN loss for the discriminator"""
		self.netD.train()
		# Fake; stop backprop to the generator by detaching fake_B
		# dis_output_fake, aux_output_fake = self.netD(self.fake.detach())
		dis_output_fake = self.netD(self.fake.detach(), None)
		dis_output_real = self.netD(real.to(self.device), None)
		self.loss_D = loss_hinge_dis(dis_output_real, dis_output_fake)
		# self.loss_D = self.criterion(dis_output_real, True) + self.criterion(dis_output_fake, False)
		self.loss_D.backward()

	def backward_G(self):
		# self.netD.eval()
		dis_output_fake = self.netD(self.fake, None)
		self.loss_G = loss_hinge_gen(dis_output_fake)
		# self.loss_G = self.criterion(dis_output_fake, True)
		self.loss_G.backward()

	def optimize_D(self, real, label):
		# update D
		self.optimizer_D.zero_grad()     # set D's gradients to zero
		self.backward_D(real, label)                # calculate gradients for D
		self.optimizer_D.step()          # update D's weights

	def optimize_G(self):
		self.optimizer_G.zero_grad()        # set G's gradients to zero
		self.backward_G()                   # calculate graidents for G
		self.optimizer_G.step()             # udpate G's weights
	
	def sample(self, data_size, labels=None, target_class=None):
		self.netG.eval()
		with torch.no_grad():
			noises = Helper.make_z_normal_(data_size, self.z_dim).to(self.device)
			fake_images = self.netG(noises, None)
		return fake_images
	
	def train_iter(self, epoch, dataloader):
		self.netG.train()
		self.netD.train()
		for _, (images, labels) in enumerate(dataloader):
			batch_size = images.shape[0]
			noises = Helper.get_noises(True, batch_size, self.z_dim)
			self.set_requires_grad(self.netD, True)
			self.forward(noises, None)
			self.optimize_D(images, None)
			self.set_requires_grad(self.netD, False)
			self.optimize_G()