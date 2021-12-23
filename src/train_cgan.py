"""
    Helper class
    @author 
    @editor 
    @date 08/01/2020
"""

import os, csv, shutil, math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import torch.nn.functional as F
from utils.helper import Helper
from utils.provider import LoaderProvider
from evals.scorers import PrecisionScorer, FIDScorer, ClassFIDScorer
import constant
from models.gans.gan_ops import get_gan
# from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


class InfDataLoader():
    def __init__(self, data_loader, **kwargs):
        self.dataloader = data_loader
        def inf_dataloader():
            while True:
                for data in self.dataloader:
                    image, label = data
                    yield image, label
        self.inf_dataloader = inf_dataloader()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.inf_dataloader)

    def __del__(self):
        del self.dataloader

def get_scorer(opt):
    if opt.data_type in [constant.MNIST, constant.FASHION]:
        scorer = None if opt.data_type == constant.DECODER else PrecisionScorer(opt)
    elif opt.gan_type == constant.GANREP:
        scorer = ClassFIDScorer(opt, opt.gan_class)
    else:
        scorer = FIDScorer(opt)
    return scorer 

def print_message(logf, iteration, losses, loss_names, val_score=None):
    message = "Iter: [{:4d}] - ".format(iteration)
    for name in loss_names:
        message += "{}: {:.4f}  ".format(name, losses[name])
    if val_score is not None:
        message += " val: {:.2f}".format(val_score)
    Helper.log(logf, message)


def train(opt):
    imbalance_suffix = 'skewed' if opt.imbalance else ''
    logf = open('../logs/{}_{}{}_{}_{}_{}_{}.out'.format(opt.exp_mode, opt.data_type, imbalance_suffix, opt.gan_type, opt.gan_class, opt.phase, opt.mode), 'w')
    message = 'Train !!!!!\n=== Exp {}: {} labels, {} noises \n=== Data {} - {} {}\n=== GAN {} class {}'.format(
            opt.exp_mode,
            opt.label_ratio,
            opt.noise_ratio,
            opt.data_type,
            opt.num_classes,
            imbalance_suffix,
            opt.gan_type,
            opt.gan_class)
    Helper.log(logf, message)
    if opt.train_encoder and opt.gan_type == constant.DECODER:
        ugan, _ = get_gan(opt)
        ugan.train_encoder(opt.sample_dir)
    else:
        train_cgan(opt, logf)
    

def train_cgan(opt, logf):
    log_dir = '../results/runs/{}-{}'.format(opt.gan_type, opt.mode)     
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    # writer = SummaryWriter(log_dir)
    dataloader = LoaderProvider(opt).get_data_loader(True)
    opt.iterator = InfDataLoader(dataloader)
    if opt.nepochs == 0:
        opt.nsteps = 200
    else:
        opt.nsteps = opt.nepochs * (len(dataloader.dataset) // opt.batch_size)
    opt.num_iterations = opt.nsteps
    opt.num_iterations_decay = opt.nsteps
    gan, loss_names = get_gan(opt)
    
    if opt.gan_type == constant.GANREP:
        scorer = ClassFIDScorer(opt, opt.gan_class)
        min_score = 1000 # min score for precision 
        test_scorer = None
    else:
        if opt.eval_mode == constant.FID:
            min_score = 1000 # max score
            scorer = FIDScorer(opt)
            # scorer = ClassFIDScorer(opt, 0)
            test_scorer = None
            # test_scorer = PrecisionScorer(opt)
        elif opt.eval_mode == constant.INTRA_FID:
            min_score = 1000 # min score for precision 
            testclasses = [opt.gan_class] if opt.gan_class > -1 else [0, 3, 8]
            scorer = ClassFIDScorer(opt, testclasses)
            test_scorer = None #PrecisionScorer(opt)
        else:
            min_score = 0 # min score for precision 
            scorer = PrecisionScorer(opt)
            test_scorer = FIDScorer(opt)

    best_iteration = -1
    # out = scorer.evaluate(dataloader, is_transform=False)
    # Helper.log(logf, "=== Sanity test scorer -- loss {:.4f} prec {:.4f}".format(out[0], out[1]))
    if opt.resume > 0 and os.path.isfile(opt.checkpoint_dir + "/net_D.pth"):
        Helper.log(logf, 'Load models')
        gan.load_networks()
    if scorer is not None:
        min_score = scorer.validate(gan, logf)
        Helper.log(logf, "Score on initial point: {:.4f}".format(min_score))
        # min_score = float('inf')
        # test_scorer.validate(gan, logf)
    
    Helper.save_images(next(opt.iterator)[0], opt.sample_dir, 'real')
    Helper.log(logf, 'Total-step:' + str(opt.nsteps) + ' Batch: ' + str(opt.batch_size))
    for iteration in range(opt.nsteps):
        gan.train_iter()
        if iteration % opt.nsteps_save == 0 and iteration > 0:
            losses = gan.get_current_losses()
            if scorer is not None:
                val_score = scorer.validate(gan, logf)
                print_message(logf, iteration, losses, loss_names, val_score)
                if scorer.sign * (val_score - min_score) < 0:
                    min_score = val_score
                    gan.save_networks(iteration)
                    Helper.log(logf, 'Saving the best model (iter {})'.format(iteration))
                    best_iteration = iteration
                    # save images
                    fake_images = gan.sample()
                    Helper.save_images(fake_images, opt.sample_dir, 'fake',nrows=10)
                    if test_scorer is not None:
                        test_scorer.validate(gan, logf)
                        # print('FID = {:.2f}'.format(fid_score))s
                torch.cuda.empty_cache()
            else:
                print_message(logf, iteration, losses, loss_names)
                gan.save_networks(iteration)
                Helper.log(logf, 'Saving the best model (iter {})'.format(iteration))
                torch.cuda.empty_cache()
             # add to log
            # writer.add_scalar('fid/train', val_score, iteration)
            # writer.add_scalar('fid/loss_G', losses['G'], iteration)
            # writer.add_scalar('fid/loss_D', losses['D'], iteration)

    # end of training
    Helper.log(logf, 'Best validation: {:.4f} @ iteration {}'.format(min_score, best_iteration))
    # writer.close()
