import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import model_ops as ops
import constant
from torch.nn.parameter import Parameter

class BaseModel(ABC):

	def __init__(self, opt):

		self.opt = opt
		self.gpu_ids = opt.gpu_ids
		self.is_training = opt.is_train
		self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
		self.save_dir = opt.checkpoint_dir  # save all the checkpoints to save_dir

		self.loss_names = []
		self.model_names = []
		self.visual_names = []
		self.optimizers = []
		self.image_paths = []
		self.metric = 0  # used for learning rate policy 'plateau'


	# @abstractmethod
	# def set_input(self, input):
	# 	"""Unpack input data from the dataloader and perform necessary pre-processing steps.
	#
	# 	Parameters:
	# 		input (dict): includes the data itself and its metadata information.
	# 	"""
	# 	pass

	@abstractmethod
	def forward(self):
		"""Run forward pass; called by both functions <optimize_parameters> and <test>."""
		pass

	def setup(self, opt):
		"""Load and print networks; create schedulers

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		if self.is_training:
			self.schedulers = [ops.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
		# if not self.is_training or opt.continue_train:
		# 	load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
		# 	self.load_networks(load_suffix)
		self.print_networks(opt.verbose)

	def eval(self):
		"""Make models eval mode during test time"""
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, 'net' + name)
				net.eval()

	def test(self):
		"""Forward function used in test time.

		This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
		It also calls <compute_visuals> to produce additional visualization results
		"""
		with torch.no_grad():
			self.forward()
			self.compute_visuals()

	def compute_visuals(self):
		"""Calculate additional output images for visdom and HTML visualization"""
		pass

	def get_image_paths(self):
		""" Return image paths that are used to load current data"""
		return self.image_paths

	def update_learning_rate(self):
		"""Update learning rates for all the networks; called at the end of every epoch"""
		for scheduler in self.schedulers:
			if self.opt.lr_policy == 'plateau':
				scheduler.step(self.metric)
			else:
				scheduler.step()
		lr = self.optimizers[0].param_groups[0]['lr']
		print('learning rate = %.7f' % lr)

	def get_current_visuals(self):
		"""Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
		visual_ret = OrderedDict()
		for name in self.visual_names:
			if isinstance(name, str):
				visual_ret[name] = getattr(self, name)
		return visual_ret

	def get_current_losses(self):
		"""Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
		errors_ret = OrderedDict()
		for name in self.loss_names:
			if isinstance(name, str):
				errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
		return errors_ret

	def save_networks(self, epoch=0):
		"""Save all the networks to the disk.
		Parameters:
			epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, 'net' + name)
				net_path = os.path.join(self.save_dir, 'net_%s.pth' % (name))
				state = {
					'state_dict': net.state_dict(),
					'epoch': epoch
				}
				torch.save(state, net_path)


	def load_networks_v0(self, epoch=-1):
		"""Load all the networks from the disk.
		Parameters:
			epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		for name in self.model_names:
			if isinstance(name, str):
				# if epoch > 0 and epoch < 100:
				# 	load_filename = '%s_net_%s.pth' % (epoch, name)
				# else:
				load_filename = 'net_%s.pth' % (name)
				load_path = os.path.join(self.save_dir, load_filename)
				print('loading the model from %s' % load_path)

				net = getattr(self, 'net' + name)
				check_point = torch.load(load_path, map_location=str(self.device))
				if isinstance(check_point, dict) and 'state_dict' in check_point:
					state_dict = check_point['state_dict']
				else:
					state_dict = check_point
				net.load_state_dict(state_dict)

	

	def load_networks(self, epoch=-1):
		"""Load all the networks from the disk.
		Parameters:
			epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		def load_my_state_dict(net, state_dict):
			own_state = net.state_dict()
			for name, param in state_dict.items():
				if name not in own_state or 'authen' in name:
					continue
				if isinstance(param, Parameter):
					# backwards compatibility for serialized parameters
					param = param.data
				own_state[name].copy_(param)

		for name in self.model_names:
			if isinstance(name, str):
				load_filename = 'net_%s.pth' % (name)
				load_path = os.path.join(self.save_dir, load_filename)
				if not os.path.isfile(load_path):
					load_filename = 'net_%s.pt' % (name)
					load_path = os.path.join(self.save_dir, load_filename)
				print('loading the model from %s' % load_path)

				net = getattr(self, 'net' + name)
				check_point = torch.load(load_path, map_location=str(self.device))
				if isinstance(check_point, dict): 
					if 'state_dict' in check_point:
						state_dict = check_point['state_dict']
						net.load_state_dict(state_dict)
						# load_my_state_dict(net, state_dict)
					elif 'g_ema' in check_point:
						state_dict = check_point['g_ema']
						net.load_state_dict(state_dict, strict=False)
						# load_my_state_dict(net, state_dict)
					else:
						state_dict = check_point
						net.load_state_dict(state_dict)
						# load_my_state_dict(net, state_dict)
						print('Done loading 1')
				else:
					state_dict = check_point
					net.load_state_dict(state_dict)
					# load_my_state_dict(net, state_dict)
					print('Done loading 2')

	def print_networks(self, verbose):
		"""Print the total number of parameters in the network and (if verbose) network architecture

		Parameters:
			verbose (bool) -- if verbose: print the network architecture
		"""
		print('---------- Networks initialized -------------')
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, 'net' + name)
				num_params = 0
				for param in net.parameters():
					num_params += param.numel()
				if verbose:
					print(net)
				print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
		print('-----------------------------------------------')

	def set_requires_grad(self, nets, requires_grad=False):
		"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
		Parameters:
			nets (network list)   -- a list of networks
			requires_grad (bool)  -- whether the networks require gradients or not
		"""
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad
