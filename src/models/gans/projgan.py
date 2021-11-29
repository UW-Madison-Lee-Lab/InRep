
"""
    WGAN_GP class
    @editor 
    @date 08/01/2020
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

from models.base_model import BaseModel
from models.nets import define_D, define_G
from models.networks.common import weights_init
from utils.helper import Helper
from utils.losses import loss_hinge_dis, loss_hinge_gen
from models.gans.gan_ops import generate_samples



class ProjGAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['D', 'G']
        self.visual_names = ['real', 'fake']
        self.z_dim = opt.z_dim
        self.nclasses = opt.num_classes
        # generator
        self.netG = define_G(opt).to(self.device)
        self.iteration = 0

        if self.is_training:
            self.model_names = ['G', 'D']
            # discriminator
            self.netD = define_D(opt).to(self.device)
            self.netG.apply(weights_init)
            self.netD.apply(weights_init)
            # optimizers
            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
            self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "w"
            def lr_lambda(iteration):
                lr = 1.0
                if opt.num_iterations_decay > 0:
                    lr = 1.0 - max(0,
                                    (iteration + 1 -
                                    (opt.num_iterations - opt.num_iterations_decay)
                                    )) / float(opt.num_iterations_decay)
                return lr
            self.lr_schedulers = (optim.lr_scheduler.LambdaLR(self.optimizer_G,
                                                lr_lambda=lr_lambda), 
                                optim.lr_scheduler.LambdaLR(self.optimizer_D,
                                                        lr_lambda=lr_lambda))

            self.batch_size = opt.batch_size
            self.d_steps_per_iter = opt.d_steps_per_iter
            self.iterator = opt.iterator
        else:
            self.model_names = ['G']

        self.setup(opt)

    def forward(self, noises, labels):
        return self.netG(noises.to(self.device), labels.to(self.device))

    def backward_semi_D(self, real, label):
        """Calculate GAN loss for the discriminator"""
        self.netD.train()
        # Fake; stop backprop to the generator by detaching fake_B
        # dis_output_fake, aux_output_fake = self.netD(self.fake.detach())
        dis_output_fake = self.netD(self.fake.detach(), self.fake_labels)

        real = real.to(self.device)
        label = label.to(self.device)
        idx_l = [i for i in range(len(label)) if label[i] > -1]
        idx_u = [i for i in range(len(label)) if label[i] == -1]
        dis_output_real = None 
        if len(idx_l) > 0:
            label_real = label[idx_l, ...]
            labeled_samples = real[idx_l, ...]
            dis_output_real = self.netD(labeled_samples, label_real)
        if len(idx_u) > 0:
            unlabeled_samples = real[idx_u, ...]
            authen_output = self.netD(unlabeled_samples, None)
            if dis_output_real is None:
                dis_output_real = authen_output
            else:
                dis_output_real = torch.cat([dis_output_real, authen_output], dim=0)
        self.loss_D = loss_hinge_dis(dis_output_real, dis_output_fake)
        self.loss_D.backward()

    def backward_D(self, reals, fakes, labels):
        # real samples 
        dis_out_real, _ = self.netD(reals, labels)
        dis_out_fake, _ = self.netD(fakes, labels) #
        self.loss_D = loss_hinge_dis(dis_out_real, dis_out_fake)

    def backward_G(self, fakes, y):
        dis_out_fake, _ = self.netD(fakes, y)
        self.loss_G = loss_hinge_gen(dis_out_fake)

    # new function
    def sample(self, data_size=None, labels=None, target_class=None):
        self.netG.eval()
        noises, labels = generate_samples(self.z_dim, self.nclasses, data_size, labels, target_class, self.device)
        with torch.no_grad():
            fake_images = self.forward(noises, labels)
        return fake_images
    
    def train_iter(self):
        self.netG.train()
        self.netD.train()
        if self.iteration > 0:
            # samples noises 
            z = Helper.make_z_normal_(2 * self.batch_size, self.z_dim)
            y = torch.cat([self.labels] * 2, dim=0)
            # forward 
            z = z.to(self.device)
            fakes = self.netG(z, y)
            self.backward_G(fakes, y)             # calculate graidents for G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.loss_G.backward()
            self.optimizer_G.step()             # udpate G's weights  
        for _ in range(self.d_steps_per_iter):
            reals, labels = next(self.iterator)
            reals = reals.to(self.device)
            labels = labels.to(self.device)
            z = Helper.make_z_normal_(self.batch_size, self.z_dim)
            z = z.to(self.device)
            with torch.no_grad():
                fakes = self.netG(z, labels)
            self.backward_D(reals, fakes, labels)      # calculate gradients for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.loss_D.backward()
            self.optimizer_D.step()          # update D's weights

        self.labels = labels
        self.iteration += 1
        if self.iteration > 1:
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()
            