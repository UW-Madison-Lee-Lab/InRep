
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
PARSER.add_argument('-g', '--gan_type', type=str, default='udecoder', help='Unconditional Decoder')
PARSER.add_argument('-p', '--decoder_type', type=str, default='gan', help='The type of generator')
# inrep
PARSER.add_argument('-f', '--phase', type=int, default=1,  help='[uncond, cond]')
PARSER.add_argument('-c', '--gan_class', type=int, default=-1, help='The current class of inrep')
PARSER.add_argument('-m', '--mode', type=int, default=0, help='0: normal, 1:no pu, 2: no inv, 3: no')
PARSER.add_argument('-a', '--num_attrs', type=int, default=2)


# Experiments
PARSER.add_argument('-e', '--exp_mode', type=str, default='complexity', help='Type of experiment')
PARSER.add_argument('-l', '--label_ratio', type=float, default=1.0)
PARSER.add_argument('-s', '--noise_ratio', type=float, default=0)
# Evaluation
PARSER.add_argument('-t', '--eval_mode', type=str, default='fid', help='The type of experiment')

# paths
PARSER.add_argument('--data_dir', default='../data')
PARSER.add_argument('--save_dir', default='../results',
                    type=str, help='directory to save results')

# training
PARSER.add_argument('--is_train', action='store_true')
PARSER.add_argument('--imbalance', action='store_true')
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

MY_ARGS = PARSER.parse_args()

MY_ARGS.use_wandb = MY_ARGS.use_wandb == 1
if MY_ARGS.use_wandb:
    wandb.init(project='InRep')
    MY_ARGS.is_train = True

# config path
config_path = "configs/{}/{}.json".format(MY_ARGS.data_type, MY_ARGS.gan_type)
with open(config_path) as f:
    model_config = json.load(f)
train_config = vars(MY_ARGS)

if train_config['data_type'] == constant.CELEBA:
    num_classes = 2**train_config['num_attrs']
    model_config['data_processing']['num_classes'] = num_classes
else:
    num_classes = model_config['data_processing']['num_classes']

imbalance_suffix = 'skewed' if train_config['imbalance'] else ''
# name
if train_config["gan_type"] == constant.DECODER:
    num = 0 if train_config['decoder_type'] == constant.STYLEGAN else num_classes
    working_folder = '{}/{}/{}-{}'.format(train_config['gan_type'], train_config['decoder_type'], train_config['data_type'], num)
else:
    cgan_folder = '{}-{}{}/{}'.format(train_config['data_type'], num_classes, imbalance_suffix, train_config['gan_type'])
    working_folder = '{}/{}'.format(train_config['exp_mode'], cgan_folder)
    if train_config['exp_mode'] == constant.EXP_COMPLEXITY:
        offset = train_config["label_ratio"]  
    else:
        offset = train_config['noise_ratio']
    working_folder += '/s-' + str(offset)
    # pretrained models
    if train_config['decoder_type'] in [constant.STYLEGAN]:
        num = 0
        extension = 'pt'
    else:
        num = num_classes
        extension =  'pth'
    for m in ['G', 'D']:
        train_config["trained_net" + m + "_path"] = os.path.join(train_config["save_dir"], 'checkpoints/udecoder/{}/{}-{}/net_{}.{}'.format(train_config['decoder_type'],train_config['data_type'], num, m, extension))

##### dir
Helper.try_make_dir(train_config["save_dir"])
for d in ["sample", "checkpoint", "eval"]:
    suffix = "" if d == 'eval' else working_folder
    train_config[d + "_dir"] = os.path.join(train_config["save_dir"], d + 's/' + suffix)
    Helper.try_make_dir(train_config[d + "_dir"])

train_config['eval_path'] = os.path.join(train_config["eval_dir"], \
    "{}/{}_{}-{}_{}{}_{}".format(
        train_config["eval_mode"],
        train_config["exp_mode"], train_config["eval_mode"], train_config["data_type"], num_classes, imbalance_suffix, train_config["gan_type"]))
train_config['real_classifier_dir'] = os.path.join(train_config["save_dir"], \
    'checkpoints/real/{}-{}'.format(train_config["data_type"], num_classes))
Helper.try_make_dir(train_config['real_classifier_dir'])

# gpu
train_config['gpu_ids'] = [int(e) for e in train_config['gpu_ids'].split(',') if not e == '']
if len(train_config['gpu_ids']) > 0:
    train_config['device'] = torch.device('cuda:{}'.format(train_config['gpu_ids'][0]))
else:
    train_config['device'] = torch.device('cpu')


### ================================
cfgs = dict2clsattr(train_config, model_config)
if cfgs.benchmark_mode:
    torch.backends.cudnn.benchmark = True

if cfgs.use_wandb:
    cfgs.d_lr = cfgs.lr_d
    cfgs.g_lr = cfgs.lr_g

if cfgs.is_train:
    # Set the random seeds.
    seed = cfgs.data_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    train(cfgs)
else:
    tester = Tester(cfgs, load_data=False)
    tester.evaluate()
