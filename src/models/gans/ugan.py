import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

from models.base_model import BaseModel
from models.model_ops import GANLoss
from models.nets import define_D, define_G
from models.networks.common import weights_init
from utils.helper import Helper
from utils.losses import loss_hinge_dis, loss_hinge_gen
import constant

class UGAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['D', 'G']
        self.visual_names = ['real', 'fake']
        self.z_dim = opt.z_dim
        # generator
        self.decoder = opt.decoder_type
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

    def forward(self, noises):
        # random labels
        return self.netG(noises)

    def backward_D(self, reals, fakes):
        reals = reals.to(self.device)
        dis_out_real, _ = self.netD(reals)
        dis_out_fake, _ = self.netD(fakes)
        self.loss_D = loss_hinge_dis(dis_out_real, dis_out_fake)

    def backward_G(self, fakes):
        dis_out_fake, _ = self.netD(fakes)
        dis_fake = loss_hinge_gen(dis_out_fake)
        self.loss_G = dis_fake

    def sample(self, data_size=None, labels=None, target_class=None):
        if data_size is None:
            data_size = self.batch_size
        self.netG.eval()
        with torch.no_grad():
            noises = Helper.make_z_normal_(data_size, self.z_dim).to(self.device)
            fake_images = self.netG(noises)
        return fake_images
    
    def train_iter(self):
        self.netG.train()
        self.netD.train()
        if self.iteration > 0:
            # samples noises 
            z = Helper.make_z_normal_(2*self.batch_size, self.z_dim)
            z = z.to(self.device)
            # forward 
            fakes = self.forward(z)
            self.backward_G(fakes)                   # calculate graidents for G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.loss_G.backward()
            self.optimizer_G.step()             # udpate G's weights  
        for _ in range(self.d_steps_per_iter):
            reals, _ = next(self.iterator)
            reals = reals.to(self.device)
            z = Helper.make_z_normal_(self.batch_size, self.z_dim)
            z = z.to(self.device)
            with torch.no_grad():
                fakes = self.forward(z)
            self.backward_D(reals, fakes)       # calculate gradients for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.loss_D.backward()
            self.optimizer_D.step()          # update D's weights

        self.iteration += 1
        # update here
        if self.iteration > 1:
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()