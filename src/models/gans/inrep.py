
"""
    @editor 
    @date 08/01/2020
"""

import math
from re import I
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from models.base_model import BaseModel
from models.nets import define_G, define_M, define_classifier
from utils.helper import Helper, AverageMeter
from models.networks.common import weights_init, weights_init_zeros
from utils.losses import loss_hinge_dis, loss_hinge_gen
from models.gans.gan_ops import load_pretrained_net, generate_samples
from models.model_ops import GANLoss, onehot
import constant
torch.backends.cudnn.enabled = True

# class LabelSmoothingLoss(nn.Module):
    # """
    # With label smoothing,
    # KL-divergence between q_{smoothed ground truth prob.}(w)
    # and p_{prob. computed by model}(w) is minimized.
    # """
    # def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
    #     assert 0.0 < label_smoothing <= 1.0
    #     self.ignore_index = ignore_index
    #     super(LabelSmoothingLoss, self).__init__()

    #     smoothing_value = label_smoothing / (tgt_vocab_size - 2)
    #     one_hot = torch.full((tgt_vocab_size,), smoothing_value)
    #     if self.ignore_index > -100:
    #         one_hot[self.ignore_index] = 0
    #     self.register_buffer('one_hot', one_hot.unsqueeze(0))

    #     self.confidence = 1.0 - label_smoothing

    # def forward(self, output, target):
    #     """
    #     output (FloatTensor): batch_size x n_classes
    #     target (LongTensor): batch_size
    #     """
    #     model_prob = self.one_hot.repeat(target.size(0), 1)
    #     model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
    #     if self.ignore_index > -100:
    #         model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

    #     return F.kl_div(output, model_prob, reduction='sum')


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class InRep(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses
        self.loss_names = ['classifier', 'accuracy']
        self.visual_names = ['real', 'fake']
        self.nclasses, self.u_dim = opt.num_classes, opt.u_dim
        self.mode = opt.mode
        # generator
        self.pre_netG = load_pretrained_net(define_G(opt), opt.trained_netG_path, self.device)
        self.classifier = define_classifier(opt.data_type).to(self.device)
        self.pre_netG.eval()
        self.classifier.eval()

        self.netM = define_M(opt).to(self.device)
        self.temperature = torch.ones(1).to(self.device) * 1.5
        if self.is_training:
            self.model_names = ['M']
            # self.gan_loss = GANLoss().to(self.device)
            # self.criterionCE = nn.CrossEntropyLoss().to(self.device)
            self.classifier_loss = LabelSmoothingLoss(self.nclasses, smoothing=0.1).to(self.device)

            self.netM.apply(weights_init_zeros) # set to be 0
            self.optimizer_G = optim.Adam(self.netM.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2)) # Optimize over "theta"
            def lr_lambda(iteration):
                lr = 1.0
                if opt.num_iterations_decay > 0:
                    lr = 1.0 - max(0,
                                    (iteration + 1 -
                                    (opt.num_iterations - opt.num_iterations_decay)
                                    )) / float(opt.num_iterations_decay)
                return lr
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer_G,
                                                lr_lambda=lr_lambda)
            self.batch_size = opt.batch_size
            self.iterator = opt.iterator
            self.iteration = 0
            self.total_step = opt.nsteps
        else:
            self.model_names = ['M']
            self.netM.eval()
        self.setup(opt)


    def forward(self, u, labels):
        z = self.netM(u, labels)
        return self.pre_netG(z)

    def sample(self, data_size=None, labels=None, target_class=None):
        self.netM.eval()
        noises, labels = generate_samples(self.u_dim, self.nclasses, data_size, labels, target_class, self.device)
        with torch.no_grad():
            fake_images = self.forward(noises, labels)
        return fake_images

    def normalize(self, x):
        #  = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        x = x * 0.5 + 0.5 # [0, 1]
        mean = constant.means[self.opt.data_type]
        std = constant.stds[self.opt.data_type]
        for c in range(3):
            x[:, c, ...] = (x[:, c, ...] - mean[c])/std[c]
        return x

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def train_iter(self):
        self.netM.train()
        reals, labels = next(self.iterator)
        reals = reals.to(self.device)
        labels = labels.to(self.device)
        u = Helper.make_z_normal_(self.batch_size, self.u_dim)
        u = u.to(self.device)
        fakes = self.forward(u, labels)

        fakes = self.normalize(fakes)
        
        logits = self.classifier(fakes)
        # logits = self.temperature_scale(logits)
        # loss = self.criterionCE(logits, labels)
        loss = self.classifier_loss(logits, labels)


        precs = Helper.accuracy(logits, labels, topk=(1,))
        prec1 = precs[0]
        self.loss_classifier = loss.item()
        self.loss_accuracy = prec1.item()
        
        self.optimizer_G.zero_grad()     # set D's gradients to zero
        loss.backward()
        self.optimizer_G.step()          # update D's weights

        # update here
        self.lr_scheduler.step()

    def evaluate_classifier(self, dataloader):
        self.classifier.eval()
        criterion = nn.CrossEntropyLoss()
        losses, top1 = AverageMeter(), AverageMeter()
        for _, (inputs, targets) in enumerate(dataloader):
            # print(batch_idx)
            targets = targets.to(self.device)
            inputs = inputs.to(self.device)
            inputs = self.normalize(inputs)
            with torch.no_grad():
                logits = self.classifier(inputs)
                loss = criterion(logits, targets)
            precs = Helper.accuracy(logits, targets, topk=(1, ))
            prec1 = precs[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

        print('Test loss: {:.4f} accuracy: {:.4f}'.format(losses.avg, top1.avg))