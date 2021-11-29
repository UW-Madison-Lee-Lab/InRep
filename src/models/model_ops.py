import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.autograd import grad
from torch.utils.data import Dataset
from utils.helper import Helper
import functools, os
import numpy as np
# from  importlib import import_module
import constant


###############################################################################
# Helper Functions
###############################################################################

class GANDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, data, labels):
		self.data = data
		self.labels = labels # to(dtype=torch.long)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		images = self.data[idx, ...]
		labels = self.labels[idx]
		sample = (images, labels)
		return sample



class GANLoss(nn.Module):
	def __init__(self, target_real_label=1.0, target_fake_label=0.0, reduction='mean'):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

	def get_target_tensor(self, prediction, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(prediction)

	def __call__(self, prediction, target_is_real):
		target_tensor = self.get_target_tensor(prediction, target_is_real)
		loss = self.loss(prediction, target_tensor)
		return loss


def gradient_penalty(netD, real, fake, device, image_dataset=True):
	alpha = torch.rand((real.shape[0], 1, 1, 1)).to(device) if image_dataset else torch.rand((real.shape[0], 1)).to(device)
	alpha = alpha.expand(real.shape)
	# interpolated point
	x_hat = alpha * real.data + (1 - alpha) * fake.data
	x_hat.requires_grad = True
	D_hat = netD(x_hat)

	torch_ones = torch.ones(D_hat.size()).to(device)
	gradients = grad(outputs=D_hat, inputs=x_hat, grad_outputs=torch_ones,
				 create_graph=True, retain_graph=True, only_inputs=True)[0]

	grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return grad_penalty

def get_scheduler(optimizer, opt):
	"""Return a learning rate scheduler

	Parameters:
		optimizer          -- the optimizer of the network
		opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
							  opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

	For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
	and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
	For other schedulers (step, plateau, and cosine), we use the deinit_netfault PyTorch schedulers.
	See https://pytorch.org/docs/stable/optim.html for more details.
	"""
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.2)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nepochs,
			eta_min=opt.lr_min_rate * optimizer.param_groups[0]['lr'])
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
	"""Initialize network weights.

	Parameters:
		net (network)   -- network to be initialized
		init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

	We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
	work better for some applications. Feel free to try yourself.
	"""
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		#if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
		if hasattr(m, 'weight') and (classname.find('Conv') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		# elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
		# 	init.normal_(m.weight.data, 1.0, init_gain)
		# 	init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	"""Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
	Parameters:
		net (network)      -- the network to be initialized
		init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		gain (float)       -- scaling factor for normal, xavier and orthogonal.
		gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

	Return an initialized network.
	"""
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	init_weights(net, init_type, init_gain=init_gain)
	return net

def get_norm(use_sn):
	if use_sn:  # spectral normalization
		return nn.utils.spectral_norm
	else:  # identity mapping
		return lambda x: x

def onehot(y, class_num):
	eye = torch.eye(class_num).type_as(y)  # ny x ny
	onehot = eye[y.view(-1)].float()  # B -> B x ny
	return onehot
