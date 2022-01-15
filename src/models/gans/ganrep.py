import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.base_model import BaseModel
from models.nets import define_D, define_G, Modifier
from utils.helper import Helper
from models.networks.common import weights_init
from utils.losses import loss_hinge_dis, loss_hinge_gen
from models.gans.gan_ops import load_pretrained_net


class GANRep(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['D', 'G']
        self.visual_names = ['real', 'fake']
        self.nclasses, self.u_dim, self.z_dim = opt.num_classes, opt.u_dim, opt.z_dim
        # generator
        self.pre_netG = load_pretrained_net(define_G(opt), opt.trained_netG_path, self.device)
        self.pre_netG.eval()

        self.netF = {}
        self.gan_class = opt.gan_class
        
        if self.is_training:
            self.model_names = ['F' + str(self.gan_class), 'D' + str(self.gan_class)]
            self.netD = define_D(opt).to(self.device)
            self.netF[self.gan_class] = Modifier(self.u_dim, self.z_dim).to(self.device)
            self.netD.apply(weights_init)
            self.netF[self.gan_class].apply(weights_init)

            self.optimizer_G = optim.Adam(self.netF[self.gan_class].parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
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
            self.iteration = 0
        else:
            self.model_names = []
            for i in range(self.nclasses):
                self.netF[i] = Modifier(self.u_dim, self.z_dim).to(self.device)
                self.model_names.append('F' + str(i))
        self.setup(opt)

    def forward(self, u, c):
        z = self.netF[c](u.to(self.device))
        return self.pre_netG(z)

    def backward_D(self, reals, fakes):
        # real samples 
        dis_out_real = self.netD(reals)
        dis_out_fake = self.netD(fakes)
        self.loss_D = loss_hinge_dis(dis_out_real[0], dis_out_fake[0])

    def backward_G(self, fakes):
        gen_out_fake = self.netD(fakes)
        self.loss_G = -torch.mean(gen_out_fake[0])

    def sample(self, data_size=None, labels=None, target_class=None):
        # visulize
        if target_class is None:
            target_class = self.gan_class
        self.netF[target_class].eval()
        if data_size is None:
            data_size = self.batch_size 
        with torch.no_grad():
            u = Helper.make_z_normal_(data_size, self.u_dim)
            fake_images = self.forward(u, target_class)
        return fake_images

    def train_iter(self):
        self.netF[self.gan_class].train()
        self.netD.train()
        if self.iteration > 0:
            # samples noises 
            u = Helper.make_z_normal_(2*self.batch_size, self.u_dim)
            u = u.to(self.device)
            # forward 
            fakes = self.forward(u, self.gan_class)
            self.backward_G(fakes)                   # calculate graidents for G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.loss_G.backward()
            self.optimizer_G.step()             # udpate G's weights  
        for _ in range(self.d_steps_per_iter):
            reals, labels = next(self.iterator)
            assert labels[0] == self.gan_class
            reals = reals.to(self.device)
            u = Helper.make_z_normal_(self.batch_size, self.u_dim)
            u = u.to(self.device)
            with torch.no_grad():
                fakes = self.forward(u, self.gan_class)
            self.backward_D(reals, fakes)       # calculate gradients for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.loss_D.backward()
            self.optimizer_D.step()          # update D's weights

        self.iteration += 1
        # update here
        if self.iteration > 1:
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step()
        

    def save_networks(self, epoch=0):
        # save 
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        def save_(net, name):
            net_path = os.path.join(self.save_dir, 'net_%s.pth' % (name))
            state = {
                    'state_dict': net.state_dict(),
                    'epoch': epoch
                }
            torch.save(state, net_path)
        # save F, D
        save_(self.netF[self.gan_class], 'F' + str(self.gan_class))
        save_(self.netD, 'D' + str(self.gan_class))

    def load_networks(self, epoch=-1):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        def load_(net, name):
            load_filename = 'net_%s.pth' % (name)
            load_path = os.path.join(self.save_dir, load_filename)
            print('loading the model from %s' % load_path)
            # net = getattr(self, 'net' + name)
            check_point = torch.load(load_path, map_location=str(self.device))
            if isinstance(check_point, dict) and 'state_dict' in check_point:
                state_dict = check_point['state_dict']
            else:
                state_dict = check_point
            net.load_state_dict(state_dict)

        if self.is_training:
            # load 2 models 
            load_(self.netF[self.gan_class], 'F' + str(self.gan_class))
            load_(self.netD, 'D' + str(self.gan_class))
        else:
            # load all F
            for i in range(self.nclasses):
                load_(self.netF[i], 'F' + str(i))


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        def print_net(net, name):
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        if self.is_training:
            # load 2 models 
            print_net(self.netF[self.gan_class], 'F' + str(self.gan_class))
            print_net(self.netD, 'D' + str(self.gan_class))
        else:
            # load all F
            print_net(self.netF[0], 'F0')
        print('-----------------------------------------------')