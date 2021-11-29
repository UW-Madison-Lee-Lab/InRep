
"""
	@editor 
	@date 08/01/2020
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, math
import numpy as np

from models.base_model import BaseModel
from models.model_ops import GANLoss
from models.nets import define_D, define_G, define_M, define_F
from utils.helper import Helper
from utils.losses import loss_hinge_dis, loss_hinge_gen

def load_pretrained_net(net, load_path, device):
	net = net.to(device)
	check_point = torch.load(load_path, map_location=str(device))
	if isinstance(check_point, dict)  and 'state_dict' in check_point:
		state_dict = check_point['state_dict']
	else:
		state_dict = check_point
	net.load_state_dict(state_dict)
	return net

class ClassRepGAN(BaseModel):
	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		# specify the training losses
		self.loss_names = ['D', 'G']
		self.visual_names = ['real', 'fake']
		self.nclasses, self.gan_class, self.u_dim = opt.num_classes, opt.gan_class, opt.u_dim
		# generator
		self.pre_netG = load_pretrained_net(define_G(opt), opt.trained_netG_path, self.device)
		self.pre_netG.eval()
		# self.netD = load_pretrained_net(define_D(opt), opt.trained_netD_path, self.device)
		self.netD = define_D(opt).to(self.device)
		# self.pre_netD = load_pretrained_net(define_D(opt), opt.trained_netD_path, self.device)

		self.netM = define_M(opt).to(self.device)
		
		if self.is_training:
			self.model_names = ['M', 'D']
			self.dis_criterion = GANLoss().to(self.device)
			# self.netD = define_F(opt.data_type, self.pre_netD.embed_dim).to(self.device)
			self.optimG = optim.Adam(filter(lambda p: p.requires_grad, self.netM.parameters()), opt.g_lr, (opt.beta1, opt.beta2), eps=1e-6) # Optimize over "theta"
			self.optimD = optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()), opt.d_lr, (opt.beta1, opt.beta2), eps=1e-6) # Optimize over "w"
			self.optimizers.append(self.optimG)
			self.optimizers.append(self.optimD)
		else:
			self.model_names = ['M']
			self.netM.eval()
		self.setup(opt)

	def forward(self, z, label=None):
		z = z.to(self.device)
		self.noise = self.netM(z)
		self.fake = self.pre_netG(self.noise, None)
		return self.fake

	def pu_loss(self, alpha, dis_errD_real, dis_errD_real_g, dis_errD_fake):
		coeff = (1 - alpha*(1-1/self.nclasses)) * 0.5
		# loss = coeff * dis_errD_real + torch.relu(dis_errD_fake - coeff*dis_errD_real_g)
		loss = (1 + coeff) * dis_errD_real + torch.relu(dis_errD_fake - coeff*dis_errD_real_g)
		return loss

	def backward_D(self, real, alpha):
		"""Calculate GAN loss for the discriminator"""
		self.netD.train()
		# Fake; stop backprop to the generator by detaching fake_B
		# dis_output_fake = self.netD(self.pre_netD.embed(self.fake.detach()))
		dis_output_fake = self.netD(self.fake.detach(), None)
		dis_errD_fake = self.dis_criterion(dis_output_fake, False)

		real = real.to(self.device)
		# dis_output_real = self.netD(self.pre_netD.embed(real)) # real
		dis_output_real = self.netD(real, None)
		dis_errD_real = self.dis_criterion(dis_output_real, True)
		# dis_errD_real_g = self.dis_criterion(dis_output_real, False)

		# dis_errD_real = torch.mean(F.relu(1. - dis_output_real)) 
		# dis_errD_fake = torch.mean(F.relu(1. + dis_output_fake))
		# dis_errD_real_g = torch.mean(F.relu(1. + dis_output_real))
		# self.loss_D = self.pu_loss(alpha, dis_errD_real, dis_errD_real_g, dis_errD_fake)
		self.loss_D = dis_errD_real + dis_errD_fake

		self.loss_dis_errD_real = dis_errD_real
		self.loss_dis_errD_fake = dis_errD_fake
		self.loss_D.backward()

	def backward_G(self):
		# self.netD.eval()
		# gen_output_fake = self.netD(self.pre_netD.embed(self.fake))
		gen_output_fake = self.netD(self.fake, None)
		self.loss_G = self.dis_criterion(gen_output_fake, True)
		# self.loss_G = -torch.mean(gen_output_fake)
		self.loss_G.backward()

	def optimize_D(self, real, alpha):
		# update D
		self.optimD.zero_grad()     # set D's gradients to zero
		self.backward_D(real, alpha)                # calculate gradients for D
		self.optimD.step()          # update D's weights

	def optimize_G(self):
		self.optimG.zero_grad()        # set G's gradients to zero
		self.backward_G()                   # calculate graidents for G
		self.optimG.step()             # udpate G's weights

	def sample(self, data_size, target_class=None):
		self.pre_netG.eval()
		self.netM.eval()
		with torch.no_grad():
			noises = Helper.get_noises(False, data_size, self.u_dim).to(self.device)
			fake_images = self.pre_netG(self.netM(noises), None)
		return fake_images

	def train_iter(self, epoch, dataloader, init=1.0):
		self.netM.train()
		self.netD.train()
		alpha = init * math.pow(0.99, epoch)
		for batch_idx, (images, labels) in enumerate(dataloader):
			noises = Helper.get_noises(False, images.shape[0], self.u_dim)
			self.set_requires_grad(self.netD, True)  # enable backprop for D
			self.forward(noises)  # compute fake images: G(A)
			self.optimize_D(images, alpha)
			self.set_requires_grad(self.netD, False)
			self.optimize_G()