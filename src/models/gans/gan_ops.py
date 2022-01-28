import torch
from torch.nn.parameter import Parameter
import constant
from utils.helper import Helper
import numpy as np

def load_my_state_dict(net, state_dict):
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state or 'authen' in name:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def load_pretrained_net(net, load_path, device):
    net = net.to(device)
    check_point = torch.load(load_path, map_location=str(device))
    # if isinstance(check_point, dict):
    if 'state_dict' in check_point:
        state_dict = check_point['state_dict']
    elif 'g_ema' in check_point:
        state_dict = check_point['g_ema']
        net.load_state_dict(state_dict, strict=False)
        return net
    else:
        state_dict = check_point
    # net.load_state_dict(state_dict)
    load_my_state_dict(net, state_dict)
    return net


def get_gan(opt):
    loss_names = ['G', 'D']
    if opt.gan_type == constant.UGAN:
        from models.gans.ugan import UGAN as GAN
    elif opt.gan_type == constant.INREP:
        from models.gans.inrep import InRep as GAN
    elif opt.gan_type == constant.INREP_AB:
        from models.gans.inrep_ab import InRep as GAN
    elif opt.gan_type == constant.GANREP:
        from models.gans.ganrep import GANRep as GAN
    elif opt.gan_type == constant.ACGAN:
        from models.gans.acgan import ACGAN as GAN
    elif opt.gan_type == constant.PROJGAN:
        from models.gans.projgan import ProjGAN as GAN
    elif opt.gan_type == constant.CONTRAGAN:
        from models.gans.contragan import ContraGAN as GAN
    elif opt.gan_type == constant.MINEGAN:
        from models.gans.minegan import MineGAN as GAN
    else:
        print('Not implemented yet')
    # model
    gan = GAN(opt)
    return gan, loss_names

def generate_samples(z_dim, nclasses, data_size, labels, target_class, device):
    nclasses = min(10, nclasses)
    if data_size is None:
        c = 10 if nclasses <= 10 else 2
        data_size = c * nclasses
        labels = np.repeat(np.arange(nclasses), c).flatten()
        labels = torch.tensor(labels, dtype=torch.long)
    elif labels is None:
        labels = Helper.make_y(data_size, nclasses, target_class)

    noises = Helper.make_z_normal_(data_size, z_dim).to(device)
    labels = labels.long().to(device)
    return noises, labels

def lr_lambda(iteration, num_iters, num_iter_decay):
    lr = 1.0
    if num_iter_decay > 0:
        lr = 1.0 - max(0,
                        (iteration + 1 -
                        (num_iters - num_iter_decay)
                        )) / float(num_iter_decay)
    return lr