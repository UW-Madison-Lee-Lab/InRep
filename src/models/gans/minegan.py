
"""
    @editor 
    @date 08/01/2020
"""

import random, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.base_model import BaseModel
from models.nets import define_D, define_G, Miner
from models.model_ops import GANLoss, onehot, get_norm, init_net
from models.gans.gan_ops import load_pretrained_net
from utils.helper import Helper
from utils.losses import loss_hinge_dis, loss_hinge_gen

class MineGAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['D', 'G']
        self.visual_names = ['real', 'fake']
        self.nclasses, self.u_dim = opt.num_classes, opt.u_dim
        self.phase = opt.phase
        self.lambda_cls_g = 0.1
        self.lambda_cls_d = 1.0

        # generator
        self.netG = define_G(opt).to(self.device)
        self.netG = load_pretrained_net(self.netG, opt.trained_netG_path, self.device)

        self.netM = Miner(opt.u_dim, opt.z_dim, opt.num_classes).to(self.device)

        if self.is_training:
            # self.netG = load_pretrained_net(self.netG, opt.trained_netG_path, self.device)
            self.netD = define_D(opt).to(self.device)
            self.dis_criterion = GANLoss().to(self.device)
            self.aux_criterion = nn.CrossEntropyLoss().to(self.device)
            self.model_names = ['M', 'D']
            if self.phase == 1:
                self.netG.eval()
                params = self.netM.parameters()
            else:
                self.model_names = ['M', 'D']
                params = list(self.netM.parameters()) + list(self.netG.parameters())
                opt.g_lr = opt.g_lr * 0.1
                opt.d_lr = opt.d_lr * 0.1
                # load previous networks
                self.load_networks()
                self.model_names.append('G')
            # optimizers
            self.optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, params), opt.g_lr, (opt.beta1, opt.beta2), eps=1e-6) # Optimize over "theta"
            self.optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()), opt.d_lr, (opt.beta1, opt.beta2), eps=1e-6) # Optimize over "w"
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
            self.iteration = 0
        else:
            # load model
            self.model_names = ['M']
            self.netM.eval()
            self.netG.eval()
        self.setup(opt)

    def forward(self, u, labels):
        z = self.netM(u.to(self.device), labels.to(self.device))
        return self.netG(z)

    def sample(self, data_size = None, labels=None, target_class=None):
        self.netM.eval()
        if data_size is None:
            c = self.nclasses if self.nclasses == 10 else 2
            data_size = c * self.nclasses
            labels = np.repeat(np.arange(self.nclasses), c).flatten()
            labels = torch.tensor(labels, dtype=torch.long)
        elif labels is None:
            labels = Helper.make_y(data_size, self.nclasses, target_class)

        with torch.no_grad():
            noises = Helper.make_z_normal_(data_size, self.u_dim).to(self.device)
            labels = labels.long().to(self.device)
            fake_images = self.forward(noises, labels)
        return fake_images

    def backward_D(self, reals, fakes, labels):
        # real samples 
        dis_out_real = self.netD(reals)
        aux_real = self.aux_criterion(dis_out_real[1], labels)
        dis_out_fake= self.netD(fakes)
        # loss
        self.loss_D = loss_hinge_dis(dis_out_real[0], dis_out_fake[0]) + self.lambda_cls_d * aux_real

    def backward_G(self, fakes, labels):
        dis_out_fake = self.netD(fakes)
        aux_fake = self.aux_criterion(dis_out_fake[1], labels)
        dis_fake = loss_hinge_gen(dis_out_fake[0])
        # loss
        self.loss_G = dis_fake + self.lambda_cls_g * aux_fake

    def train_iter(self):
        self.netM.train()
        self.netD.train()
        if self.iteration > 0:
            # samples noises 
            u = Helper.make_z_normal_(2*self.batch_size, self.u_dim)
            u = u.to(self.device)
            y = torch.cat([self.labels] * 2, dim=0)
            # forward 
            fakes = self.forward(u, y)
            self.backward_G(fakes, y)                   # calculate graidents for G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.loss_G.backward()
            self.optimizer_G.step()             # udpate G's weights  
        for _ in range(self.d_steps_per_iter):
            reals, labels = next(self.iterator)
            reals = reals.to(self.device)
            labels = labels.to(self.device)
            u = Helper.make_z_normal_(self.batch_size, self.u_dim)
            # fake samples 
            u = u.to(self.device)
            with torch.no_grad():
                fakes = self.forward(u, labels)
            self.backward_D(reals, fakes, labels)       # calculate gradients for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.loss_D.backward()
            self.optimizer_D.step()          # update D's weights

        self.labels = labels
        self.iteration += 1
        # update here
        if self.iteration > 1:
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()