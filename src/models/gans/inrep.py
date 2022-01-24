
import math
from re import I
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.base_model import BaseModel
from models.nets import define_D, define_G, define_M
from utils.helper import Helper
from models.networks.common import weights_init, weights_init_zeros
from utils.losses import loss_hinge_dis, loss_hinge_gen
from models.gans.gan_ops import load_pretrained_net, generate_samples
from models.model_ops import GANLoss
import constant
torch.backends.cudnn.enabled = True

class InRep(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['D', 'G']
        self.visual_names = ['real', 'fake']
        self.nclasses, self.u_dim = opt.num_classes, opt.u_dim
        self.mode = opt.mode
        # generator
        self.pre_netG = load_pretrained_net(define_G(opt), opt.trained_netG_path, self.device)
        self.pre_netG.eval()
        self.netM = define_M(opt).to(self.device)

        if self.is_training:
            self.model_names = ['M', 'D']
            self.gan_loss = GANLoss().to(self.device)
            self.netD = define_D(opt).to(self.device)
            self.netD.apply(weights_init)
            self.netM.apply(weights_init)
            self.optimizer_G = optim.Adam(self.netM.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
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
            self.iteration = 0
            self.total_step = opt.nsteps
            self.iterator = opt.iterator
            self.imbalance = opt.exp_mode == constant.EXP_IMBALANCE
        else:
            self.model_names = ['M']
            self.netM.eval()
        self.setup(opt)

    def forward(self, u, labels):
        z = self.netM(u, labels)
        output = self.pre_netG(z)
        return output

    def pu_loss(self, dis_errD_real, dis_errD_real_g, dis_errD_fake):
        warmup_iters = 30
        # alpha = 1.0 * math.pow(0.99, self.iteration /self.total_step)
        n = self.nclasses
        if self.imbalance:
            n -= 2
        if self.iteration > warmup_iters:
            t = self.iteration - warmup_iters
            pi = 1/n + t * (1 - 1/n)/(self.total_step - 1)
            loss = (1 + pi) * dis_errD_real + torch.relu(dis_errD_fake - pi*dis_errD_real_g)
        else:
            loss = dis_errD_real + dis_errD_fake
        return loss

    def backward_D(self, reals, fakes, labels, hinge=True):
        dis_out_real, _ = self.netD(reals, labels) # out_quality, out_y, out_class
        dis_out_fake, _ = self.netD(fakes, labels)
        if hinge:
            dis_errD_real = torch.mean(F.relu(1. - dis_out_real))
            dis_errD_fake = torch.mean(F.relu(1. + dis_out_fake))
            dis_errD_real_g = torch.mean(F.relu(1. + dis_out_real))
        else:
            dis_errD_real = self.gan_loss(dis_out_real, True)
            dis_errD_fake = self.gan_loss(dis_out_fake, False)
            dis_errD_real_g = self.gan_loss(dis_out_real, False)
        self.loss_D = self.pu_loss(dis_errD_real, dis_errD_real_g, dis_errD_fake)

    def backward_G(self, fakes, labels):
        gen_out_fake, _ = self.netD(fakes, labels)
        self.loss_G = loss_hinge_gen(gen_out_fake)

    def sample(self, data_size=None, labels=None, target_class=None):
        self.netM.eval()
        noises, labels = generate_samples(self.u_dim, self.nclasses, data_size, labels, target_class, self.device)
        with torch.no_grad():
            fake_images = self.forward(noises, labels)
        return fake_images

    def train_iter(self, init=1.0):
        self.netM.train()
        self.netD.train()
        if self.iteration > 0:
            u = Helper.make_z_normal_(2*self.labels.shape[0], self.u_dim)
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