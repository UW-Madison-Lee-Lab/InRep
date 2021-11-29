
"""
    WGAN_GP class
    @editor 
    @date 08/01/2020
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import BaseModel
from models.nets import define_D, define_G
from models.networks.common import weights_init
from utils.helper import Helper
from utils.losses import loss_hinge_dis, loss_hinge_gen


class ACGAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['D', 'G']
        self.visual_names = ['real', 'fake']
        self.lambda_cls_g = 0.1
        self.lambda_cls_d = 1.0
        self.z_dim, self.nclasses = opt.z_dim, opt.num_classes
        # generator
        self.netG = define_G(opt).to(self.device)
        self.iteration = 0

        if self.is_training:
            self.model_names = ['G', 'D']
            # discriminator
            self.netD = define_D(opt).to(self.device)
            self.netG.apply(weights_init)
            self.netD.apply(weights_init)
            self.aux_criterion = nn.CrossEntropyLoss().to(self.device)
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
        # random labels
        return self.netG(noises.to(self.device), labels.to(self.device))

    def backward_semi_D(self, reals, fakes, labels):
        # real samples 
        dis_out_real, aux_out_real, _ = self.netD(reals)
        aux_real = self.aux_criterion(aux_out_real, labels)
        dis_out_fake, _, _ = self.netD(fakes)
        # loss
        self.loss_D = loss_hinge_dis(dis_out_real, dis_out_fake) + self.lambda_cls_d * aux_real

        """Calculate GAN loss for the discriminator"""
        dis_output_fake, aux_output_fake = self.netD(self.fake.detach(), None)
        # dis_errD_fake = self.dis_criterion(dis_output_fake, False)
        aux_errD_fake = self.aux_criterion(aux_output_fake, self.aux_label_fake)
        # errD_fake = dis_errD_fake + self.lamda_aux * aux_errD_fake
        # Real: dislabel: real/fake
        # aux_label: c
        real = real.to(self.device)
        # unlabeld  loss
        dis_output_real, aux_output_real = self.netD(real, None)
        # dis_errD_real = self.dis_criterion(dis_output_real, True)
        
        # labeled loss
        idx_l = [i for i in range(len(label)) if label[i] != -1]
        if len(idx_l) > 0:
            aux_label_real = label[idx_l].to(self.device)
            aux_output_real = aux_output_real[idx_l, ...]
            aux_errD_real = self.aux_criterion(aux_output_real, aux_label_real)
        else:
            aux_errD_real = 0
        # errD_real = dis_errD_real + self.lamda_aux * aux_errD_real

        # combine loss and calculate gradients
        # self.loss_D = errD_real + errD_fake
        dis_errD = loss_hinge_dis(dis_output_real, dis_output_fake)
        self.loss_D = dis_errD + self.lamda_aux * (aux_errD_real + aux_errD_fake)
        # self.loss_D = dis_errD_fake + dis_errD_real
        self.loss_dis_errD_real = dis_errD
        self.loss_dis_errD_fake = dis_errD
        self.loss_aux_errD_real = aux_errD_real
        self.loss_aux_errD_fake = aux_errD_fake

    def backward_D(self, reals, fakes, labels):
        # real samples 
        dis_out_real, aux_out_real, _ = self.netD(reals)
        aux_real = self.aux_criterion(aux_out_real, labels)
        dis_out_fake, _, _ = self.netD(fakes)
        # loss
        self.loss_D = loss_hinge_dis(dis_out_real, dis_out_fake) + self.lambda_cls_d * aux_real

    def backward_G(self, fakes, labels):
        dis_out_fake, aux_out_fake, _ = self.netD(fakes)
        aux_fake = self.aux_criterion(aux_out_fake, labels)
        dis_fake = loss_hinge_gen(dis_out_fake)
        self.loss_G = dis_fake + self.lambda_cls_g * aux_fake

    def sample(self, data_size = None, labels=None, target_class=None):
        self.netG.eval()
        if data_size is None:
            c = self.nclasses if self.nclasses == 10 else 2
            data_size = c * self.nclasses
            labels = np.repeat(np.arange(self.nclasses), c).flatten()
            labels = torch.tensor(labels, dtype=torch.long)
        elif labels is None:
            labels = Helper.make_y(data_size, self.nclasses, target_class)

        with torch.no_grad():
            noises = Helper.make_z_normal_(data_size, self.z_dim).to(self.device)
            labels = labels.long().to(self.device)
            fake_images = self.forward(noises, labels)
        return fake_images
    
    def train_iter(self):
        self.netG.train()
        self.netD.train()
        if self.iteration > 0:
            # samples noises 
            z = Helper.make_z_normal_(self.batch_size * 2, self.z_dim)
            z = z.to(self.device)
            y = torch.cat([self.labels] * 2, dim=0)
            # forward 
            fakes = self.netG(z, y)
            self.backward_G(fakes, y)                   # calculate graidents for G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.loss_G.backward()
            self.optimizer_G.step()             # udpate G's weights  
        for _ in range(self.d_steps_per_iter):
            reals, labels = next(self.iterator)
            reals = reals.to(self.device)
            labels = labels.to(self.device)
            if labels is None:
                from IPython import embed; embed()
            z = Helper.make_z_normal_(self.batch_size, self.z_dim)
            # fake samples 
            z = z.to(self.device)
            with torch.no_grad():
                fakes = self.netG(z, labels)
            self.backward_D(reals, fakes, labels)  # calculate gradients for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.loss_D.backward()
            self.optimizer_D.step()          # update D's weights

        self.labels = labels
        self.iteration += 1
        # update here
        if self.iteration > 1:
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()