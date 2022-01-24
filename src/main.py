
import os
import json
import argparse
import torch
import numpy as np
from utils.helper import Helper
from utils.misc import dict2clsattr
from evals.evaluate import Tester
from train import train
import constant
# import wandb

import warnings
warnings.filterwarnings("ignore")
def warn(*args, **kwargs):
    pass
warnings.warn = warn


# parsing and configuration
PARSER = argparse.ArgumentParser(description="ICML-22 submission")
PARSER.add_argument('-d', '--data_type', type=str, default='cifar10', help='Type of dataset')
PARSER.add_argument('--data_noise_type', type=str, default='symmetric')
PARSER.add_argument('--data_seed', type=int, default=3407)
# GANs
PARSER.add_argument('-g', '--gan_type', type=str, default='ugan', help='Unconditional Decoder')
PARSER.add_argument('-p', '--decoder_type', type=str, default='gan', help='The type of generator')
# inrep
PARSER.add_argument('-f', '--phase', type=int, default=1,  help='[uncond, cond]')
PARSER.add_argument('-c', '--gan_class', type=int, default=-1, help='The current class of inrep')
PARSER.add_argument('-m', '--mode', type=int, default=0, help='0: normal, 1:no pu, 2: no inv, 3: no')
PARSER.add_argument('-a', '--num_attrs', type=int, default=2)


# Experiments
PARSER.add_argument('-e', '--exp_mode', type=str, default='complexity', help='Type of experiment')
PARSER.add_argument('-l', '--label_ratio', type=float, default=1.0)
PARSER.add_argument('-s', '--noise_ratio', type=float, default=0.4)
# Evaluation
PARSER.add_argument('-t', '--eval_mode', type=str, default='fid', help='The type of experiment')

# paths
PARSER.add_argument('--data_dir', default='../data')
PARSER.add_argument('--save_dir', default='../results',
                    type=str, help='directory to save results')

# training
PARSER.add_argument('--is_train', action='store_true')
PARSER.add_argument('--tune', action='store_true')
PARSER.add_argument('--train_encoder', action='store_true')


PARSER.add_argument('--verbose', action='store_true')
PARSER.add_argument('-r', '--resume', default=0, type=int, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
PARSER.add_argument('-n', '--nepochs', type=int, default=100, help='The number of epochs to run')
PARSER.add_argument('--nsteps_log', type=int, default=1000)
PARSER.add_argument('--nsteps_save', type=int, default=100)

# optimization
PARSER.add_argument('--lr_g', type=float, default=2e-4)
PARSER.add_argument('--lr_d', type=float, default=2e-4)
PARSER.add_argument('--c_lr', type=float, default=0.001)
PARSER.add_argument('--lr_policy', type=str, default='step', \
    help='learning rate policy: lambda|step|plateau|cosine')
PARSER.add_argument('--lr_decay_iters', type=int, default=50,
                    help='multiply by a gamma every lr_decay_iters iterations')
PARSER.add_argument('--lr_min_rate', type=float, default=0,
                    help='minimum rate of learning rate for cosine scheduler')
PARSER.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
PARSER.add_argument('--warmup_epochs', default=10, type=int, help='epochs for warmup')


# others
PARSER.add_argument('--gpu_ids', type=str, default='0,', \
    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
PARSER.add_argument('--benchmark_mode', type=bool, default=True)

PARSER.add_argument("--use_wandb", type=int, default=0,
        help="Use WandDB?")

params = PARSER.parse_args()

params.use_wandb = params.use_wandb == 1
if params.use_wandb:
    wandb.init(project='InRep')
    params.is_train = True

# config path
suffix = "_tuned" if params.tune else ""
if params.gan_type == constant.UGAN:
    config_path = f"configs/ugan/{params.data_type}.json"
else:
    config_path = f"configs/{params.exp_mode}/{params.data_type}/{params.gan_type}{suffix}.json"
with open(config_path) as f:
    model_config = json.load(f)
train_config = vars(params)
cfgs = dict2clsattr(train_config, model_config)


# name
if cfgs.gan_type == constant.UGAN:
    num = 0 if cfgs.decoder_type == constant.STYLEGAN else cfgs.num_classes
    working_folder = f'{cfgs.gan_type}/{cfgs.decoder_type}/{cfgs.data_type}-{num}'
else:
    cgan_folder = f'{cfgs.data_type}-{cfgs.num_classes}/{cfgs.gan_type}'
    working_folder = f'{cfgs.exp_mode}/{cgan_folder}'
    if cfgs.exp_mode == constant.EXP_COMPLEXITY:
        working_folder += '/s-' + str(cfgs.label_ratio)
    # pretrained models
    extension =  'pth'
    pretrained_dir = os.path.join(cfgs.save_dir, f'checkpoints/ugan/{cfgs.decoder_type}/{cfgs.data_type}-{cfgs.num_classes}')
    cfgs.trained_netG_path = os.path.join(pretrained_dir, 'net_G.' + extension)
    cfgs.trained_netD_path = os.path.join(pretrained_dir, 'net_D.' + extension)

##### dir
cfgs.sample_dir = os.path.join(cfgs.save_dir, 'samples/' + working_folder)
cfgs.checkpoint_dir = os.path.join(cfgs.save_dir, 'checkpoints/' + working_folder)
cfgs.eval_dir = os.path.join(cfgs.save_dir, 'evals/' + cfgs.eval_mode)
cfgs.real_classifier_dir = os.path.join(cfgs.save_dir, f'checkpoints/real/{cfgs.data_type}-{cfgs.num_classes}')
cfgs.eval_path = os.path.join(cfgs.eval_dir, f"{cfgs.exp_mode}_{cfgs.eval_mode}-{cfgs.data_type}_{cfgs.num_classes}_{cfgs.gan_type}")
Helper.try_make_dir(cfgs.save_dir)
Helper.try_make_dir(cfgs.sample_dir)
Helper.try_make_dir(cfgs.checkpoint_dir)
Helper.try_make_dir(cfgs.eval_dir)
Helper.try_make_dir(cfgs.real_classifier_dir)

# gpu
cfgs.gpu_ids = [int(e) for e in cfgs.gpu_ids.split(',') if not e == '']
if len(cfgs.gpu_ids) > 0:
    cfgs.device = torch.device('cuda:{}'.format(cfgs.gpu_ids[0]))
else:
    cfgs.device = torch.device('cpu')


### ================================

if cfgs.benchmark_mode:
    torch.backends.cudnn.benchmark = True

if cfgs.use_wandb:
    cfgs.d_lr = cfgs.lr_d
    cfgs.g_lr = cfgs.lr_g


def get_configs(opt):
    config_name = f"{opt.data_type}_{opt.gan_type}_{opt.exp_mode}_"
    message = f"========Setting======== \n Data: {opt.data_type} \n GAN: {opt.gan_type} \n Experiment: {opt.exp_mode}"
    if opt.exp_mode == constant.EXP_COMPLEXITY:
        config_name += str(opt.label_ratio)
        message += f" -- label ratio: {opt.label_ratio}\n"
    elif opt.exp_mode == constant.EXP_ASYM_NOISE:
        config_name += str(opt.noise_ratio)
        message += f" -- noise ratio: {opt.noise_ratio}\n"

    if opt.gan_type == constant.GANREP:
        config_name += f"_{opt.gan_class}"
    elif opt.gan_type == constant.MINEGAN:
        config_name += f"_{opt.phase}"

    message += f" Evaluation: {opt.eval_mode} -- class: {opt.gan_class}"

    return config_name, message

config_name, message = get_configs(cfgs)
if cfgs.is_train:
    # Set the random seeds.
    seed = cfgs.data_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    logf = open(f'../logs/{config_name}.out', 'w')
    Helper.log(logf, message)
    train(cfgs, logf)
else:
    print("====Evaluating ====\n", message)
    tester = Tester(cfgs, load_data=False)
    tester.evaluate()
