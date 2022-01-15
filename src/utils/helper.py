import torch.nn as nn
import os, gzip, torch
import numpy as np
import scipy.misc
# import imageio
import torchvision
import math
from scipy.stats import truncnorm


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class Helper:

	@staticmethod
	def accuracy(output, target, topk=(1,)):
		"""Computes the precision@k for the specified values of k"""
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res
		
	###======================== Utilities ====================== ####
	@staticmethod
	def try_make_dir(d):
		if not os.path.isdir(d):
			# os.mkdir(d)
			os.makedirs(d) # nested is allowed

	@staticmethod
	def get_hms(seconds):
		m, s = divmod(seconds, 60)
		h, m = divmod(m, 60)
		return h, m, s

	@staticmethod
	def log(logf, msg, console_print=True):
		logf.write(msg + '\n')
		if console_print:
			print(msg)

	@staticmethod
	def make_z_normal_(size, nz):
		"""Return B x nz noise vector"""
		return torch.randn(size, nz) # B x nz
		# return Helper.make_z_normal_truncated(size, nz)

	@staticmethod
	def make_z_normal_truncated(size, nz, truncation=0.5, seed=None):
	  state = None if seed is None else np.random.RandomState(seed)
	  values = truncnorm.rvs(-2, 2, size=(size, nz), random_state=state)
	  return torch.Tensor(truncation * values)

	@staticmethod
	def make_z_uniform_(size, nz):
		"""Return B x nz noise vector"""
		return torch.rand(size, nz)  # B x nz

	@staticmethod
	def make_y(size, ny, value=None):
		"""Return B condition vector"""
		if value is None:
			return torch.randint(ny, [size]).long()  # B (random value)
		else:
			return torch.LongTensor(size).fill_(value)  # B (given value)
	
	@staticmethod
	def get_init_batch(dataloader, batch_size):
		"""
		gets a batch to use for init
		"""
		batches = []
		seen = 0
		for x, y in dataloader:
			batches.append(x)
			seen += x.size(0)
			if seen >= batch_size:
				break
		batch = torch.cat(batches)
		return batch

	###======================== Networks ====================== ####
	@staticmethod
	def print_network(net):
		num_params = 0
		for param in net.parameters():
			num_params += param.numel()
		print(net)
		print('Total number of parameters: %d' % num_params)

	@staticmethod
	def print_grad_norm(model):
		for p in model.parameters():
			print(p.grad.norm())

	@staticmethod
	def deactivate(model):
		for p in model.parameters():
			p.requires_grad = False

	@staticmethod
	def activate(model):
		for p in model.parameters():
			p.requires_grad = True


	@staticmethod
	def save_checkpoint(model, test_objective, args, checkpoint_path, model_name, epoch):
		net_path = os.path.join(checkpoint_path, '{}_e{}.pth'.format(model_name, epoch))
		print('Save checkpoint at ', net_path)
		state = {
			'model': model.state_dict(),
			'objective': test_objective,
			'opt': args
		}
		torch.save(state, net_path)

	@staticmethod
	def save_networks(net, save_dir, epoch, train_loss, train_acc1, val_loss, val_acc1, is_best):
		save_filename = '%s_net.pth' % (epoch)
		save_path = os.path.join(save_dir, save_filename)
		state = {
			'model': net.state_dict(),
			'loss': [val_loss, train_loss],
			'accuracy': [val_acc1, train_acc1],
			'epoch': epoch
		}
		torch.save(state, save_path)
		print('\nSave model ', save_filename)

		torch.save(state, os.path.join(save_dir, 'current_net.pth'))
		if is_best:
			best_path = os.path.join(save_dir, 'best_net.pth')
			torch.save(state, best_path)

	@staticmethod
	def learning_rate(init, epoch, factor=60):
		optim_factor = 0
		if epoch > factor * 3:
			optim_factor = 3
		elif epoch > factor * 2:
			optim_factor = 2
		elif epoch > factor:
			optim_factor = 1
		return init*math.pow(0.2, optim_factor)

	@staticmethod
	def update_lr(optimizer, lr):
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	@staticmethod
	# update learning rate (called once every epoch)
	def update_learning_rate(scheduler, optimizer):
		scheduler.step()
		lr = optimizer.param_groups[0]['lr']
		print('learning rate = %.7f' % lr)

	@staticmethod
	def adjust_learning_rate(optimizer, epoch, base_lr, lr_decay_period=20, lr_decay_rate=0.1):
		lr = base_lr * (lr_decay_rate ** (epoch // lr_decay_period))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


	###======================== Visualize ====================== ####
	@staticmethod
	def save_images(samples, sample_dir, sample_name, offset=0, nrows=0):
		if nrows == 0:
			bs = samples.shape[0]
			nrows = int(bs**.5)
		if offset > 0:
			sample_name += '_' + str(offset)
		save_path = os.path.join(sample_dir, sample_name + '.png')
		torchvision.utils.save_image(samples.cpu(), save_path, nrow=nrows, normalize=True)

	@staticmethod
	def save_toys(samples, sample_dir, sample_name, offset):
		data = samples[0].data.cpu().numpy()
		labels = samples[1]
		plot_toy(data, labels, os.path.join(sample_dir, "{}_{}.png".format(sample_name, offset)))

	@staticmethod
	def imsave(images, size, path):
		image = np.squeeze(merge(images, size))
		return scipy.misc.imsave(path, image)

	@staticmethod
	def log_tb_losses(writer, loss_dict, epochs, type_=' '):
		for loss_name, loss_value in loss_dict.items():
				writer.add_scalar(f'{type_}/{loss_name}', loss_value, epochs)

	@staticmethod
	def draw_in_tb(writer, images, epochs, type_):
		grid_image = torchvision.utils.make_grid(images.clone().cpu().data, 19, normalize=True)
		writer.add_image(f'{type_}/generated_images', grid_image, epochs)
