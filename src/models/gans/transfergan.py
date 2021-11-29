
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
from models.model_ops import GANLoss, onehot, get_norm, init_net, gradient_penalty
from models.nets import define_D, define_G
from utils.helper import Helper
from models.gans.gan_ops import load_pretrained_net
from utils.losses import loss_wgan_dis, loss_wgan_gen



class TransferGAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['D', 'G']
        self.visual_names = ['real', 'fake']
        self.nclasses, self.u_dim = opt.num_classes, opt.u_dim
        # generator
        self.netG = load_pretrained_net(define_G(opt), opt.trained_netG_path, self.device)
        self.netD = load_pretrained_net(define_D(opt), opt.trained_netD_path, self.device)
        
        if self.is_training:
            self.model_names = ['G', 'D']
            self.dis_criterion = GANLoss().to(self.device)
            self.aux_criterion = nn.CrossEntropyLoss().to(self.device)
           # optims
            self.optimG = optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), opt.g_lr, (opt.beta1, opt.beta2), eps=1e-6) # Optimize over "theta"
            self.optimD = optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()), opt.d_lr, (opt.beta1, opt.beta2), eps=1e-6) # Optimize over "w"
            # self.optimD = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, self.netD.blocks.parameters())}, {'params': filter(lambda p: p.requires_grad, self.netD.linear_y.parameters()), 'lr': opt.d_lr * 10}], lr=opt.d_lr, weight_decay=0.001)
            self.optimizers.append(self.optimG)
            self.optimizers.append(self.optimD)
        else:
            self.model_names = ['G', 'D']
            self.netG.eval()
            self.netD.eval()
        self.setup(opt)

    def forward(self, u, labels):
        encoded_labels = onehot(labels, self.nclasses)
        z = torch.cat([u, encoded_labels], dim=1).to(self.device)
        self.fake = self.netG(z, None)
        return self.fake


    def backward_D(self, real, labels):
        """Calculate GAN loss for the discriminator"""
        dis_out_fake, aux_out_fake, _ = self.netD.projection(self.fake.detach(), labels)
        dis_out_real, aux_out_real, _ = self.netD.projection(real, labels)
        # dis_errD_real = self.dis_criterion(dis_out_real, True)
        # dis_errD_fake = self.dis_criterion(dis_out_fake, False)
        #### Auxiliary loss
        aux_errD_real = self.aux_criterion(aux_out_real, labels)
        aux_errD_fake = self.aux_criterion(aux_out_fake, labels)

        # loss
        self.loss_D = loss_wgan_dis(dis_out_real, dis_out_fake) + aux_errD_real + aux_errD_fake
        self.loss_D.backward()

    def backward_G(self, labels):
        gen_out_fake, aux_out_fake, _ = self.netD.projection(self.fake, labels)
        ####  auxiliary loss
        # gen_errD_fake = self.dis_criterion(gen_out_fake, True)
        aux_errD_fake = self.aux_criterion(aux_out_fake, labels)
        self.loss_G = loss_wgan_gen(gen_out_fake) + aux_errD_fake
        self.loss_G.backward()

    def sample(self, data_size, target_class=None):
        self.netG.eval()
        # visulize
        if target_class is None:
            c = self.nclasses if self.nclasses == 10 else 2
            with torch.no_grad():
                u = Helper.get_noises(True, self.nclasses*c, self.u_dim)
                labels = np.repeat(np.arange(self.nclasses), c).flatten()
                labels = torch.tensor(labels, dtype=torch.long)
                fake_images = self.forward(u, labels)
        else:
            with torch.no_grad():
                u = Helper.get_noises(True, data_size, self.u_dim)
                labels = Helper.make_y(data_size, self.nclasses, target_class)
                labels = labels.long()
                fake_images = self.forward(u, labels)
        return fake_images

    def train_iter(self, epoch, dataloader, init=1.0):
        self.netG.train()
        self.netD.train()
        for _, (images, labels) in enumerate(dataloader):
            u = Helper.get_noises(True, images.shape[0], self.u_dim)
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            # self.set_requires_grad(self.netD.blocks, False)
            self.forward(u, labels)  # compute fake images: G(A)
            labels = labels.to(self.device)
            images = images.to(self.device)
            # self.optimize_D(images, labels, alpha)
            # update D
            self.optimD.zero_grad()     # set D's gradients to zero
            self.backward_D(images, labels)                # calculate gradients for D
            self.optimD.step()          # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)
            self.optimG.zero_grad()        # set G's gradients to zero
            self.backward_G(labels)                   # calculate graidents for G
            self.optimG.step()             # udpate G's weights