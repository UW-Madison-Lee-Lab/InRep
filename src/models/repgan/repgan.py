
"""
	GAN Reprogram class
	@author 
	@editor 
	@date 08/01/2020
"""


import os
import torch
import torch.nn as nn
import numpy as np
from models.nets import define_G, define_M
from utils.helper import Helper

def load_pretrained_net(net, load_path, device):
	net = net.to(device)
	check_point = torch.load(load_path, map_location=str(device))
	if isinstance(check_point, dict)  and 'state_dict' in check_point:
		state_dict = check_point['state_dict']
	else:
		state_dict = check_point
	net.load_state_dict(state_dict)
	net.eval()
	return net

class RepGAN():
	def __init__(self, opt):
		self.device, self.num_classes, self.u_dim = opt.device, opt.num_classes, opt.u_dim
		self.netG = load_pretrained_net(define_G(opt), opt.trained_netG_path, opt.device)
		self.netMs = {}
		for k in range(self.num_classes):
			load_path = os.path.join(opt.checkpoint_dir, 'c-{}/net_M.pth'.format(k))
			self.netMs[k] = load_pretrained_net(define_M(opt), load_path, opt.device)
	
	def load_networks(self, epoch=-1):
		pass

	def sample(self, data_size, target_class):
		with torch.no_grad():
			noises = Helper.get_noises(False, data_size, self.u_dim).to(self.device)
			fake_images = self.netG(self.netMs[target_class](noises), None)
		return fake_images

	def forward(self, noise, c):
		assert c < self.num_classes
		noise_ = self.netMs[c](noise.to(self.device))
		return self.netG(noise_, None)
