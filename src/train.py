import os, shutil
import torch
import torch.nn.functional as F

from models.gans.gan_ops import get_gan
from utils.helper import Helper
from datasets.provider import LoaderProvider, InfDataLoader
from evals.scorers import PrecisionScorer, FIDScorer, ClassFIDScorer
import constant

import warnings
warnings.filterwarnings("ignore")
# import wandb


def get_scorer(opt):
    if opt.data_type in [constant.MNIST]:
        scorer = None if opt.data_type == constant.UGAN else PrecisionScorer(opt)
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


def train(opt, logf):
    ##### ========= TRAIN ============ ######
    Helper.log(logf, '\n======== START TRAINING !!!!! ======== \n ')
    dataloader = LoaderProvider(opt).get_data_loader(True)
    opt.iterator = InfDataLoader(dataloader)
    opt.nsteps = 200 if opt.nepochs == 0 else opt.nepochs * (len(dataloader.dataset) // opt.batch_size)
    opt.num_iterations = opt.nsteps
    opt.num_iterations_decay = opt.nsteps
    gan, loss_names = get_gan(opt)
    
    if opt.gan_type == constant.GANREP:
        scorer = ClassFIDScorer(opt, opt.gan_class)
        min_score = 1000 # min score for precision 
    else:
        if opt.eval_mode == constant.FID:
            min_score = 1000 # max score
            scorer = FIDScorer(opt)
        elif opt.eval_mode == constant.INTRA_FID:
            min_score = 1000 # min score for precision 
            testclasses = [opt.gan_class] if opt.gan_class > -1 else [0, 3, 8]
            scorer = ClassFIDScorer(opt, testclasses)

    best_iteration = -1
    if opt.resume > 0 and os.path.isfile(opt.checkpoint_dir + "/net_D.pth"):
        Helper.log(logf, 'Load models')
        gan.load_networks()
    if scorer is not None:
        min_score = scorer.validate(gan, logf)
        Helper.log(logf, "Score on initial point: {:.4f}".format(min_score))
    
    Helper.save_images(next(opt.iterator)[0], opt.sample_dir, 'real')
    Helper.log(logf, 'Total-step:' + str(opt.nsteps) + ' Batch: ' + str(opt.batch_size))
    for iteration in range(opt.nsteps):
        gan.train_iter()
        if iteration % opt.nsteps_save == 0 and iteration > 0:
            losses = gan.get_current_losses()
            if scorer is not None:
                val_score = scorer.validate(gan, logf)
                print_message(logf, iteration, losses, loss_names, val_score)
                if opt.use_wandb:
                    wandb.log({'Intra-FID': val_score})
                if scorer.sign * (val_score - min_score) < 0:
                    min_score = val_score
                    gan.save_networks(iteration)
                    Helper.log(logf, 'Saving the best model (iter {})'.format(iteration))
                    best_iteration = iteration
                    # save images
                    fake_images = gan.sample()
                    Helper.save_images(fake_images, opt.sample_dir, 'fake',nrows=10)
            else:
                print_message(logf, iteration, losses, loss_names)
                gan.save_networks(iteration)
                Helper.log(logf, 'Saving the best model (iter {})'.format(iteration))
            torch.cuda.empty_cache()
        # flush
        logf.flush()
    # end of training
    Helper.log(logf, 'Best validation: {:.4f} @ iteration {}'.format(min_score, best_iteration))
