import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torchvision import transforms, datasets

import sys
sys.path.append('../../')
# from models.networks.glow import Glow
# from models.gans.gan_ops import load_pretrained_net
from models.networks.glow import Glow
from models.gans.gan_ops import load_pretrained_net

from utils.helper import Helper
from utils.provider import DataProvider

n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x

def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    return x

def postprocess0(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()

def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    z = torch.normal(mean, torch.exp(logs) * temperature)

    return z

def get_CIFAR10(augment=True, datapath='../data', download=True):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    test_dataset = datasets.CIFAR10(
        datapath,
        train=False,
        transform=test_transform,
        # target_transform=one_hot_encode,
        download=download,
    )

    return test_dataset


class Decoder:
    def __init__(self, opt):
        self.device = opt.device
        self.pretrained_path = os.path.join(opt.checkpoint_dir, 'net_G.pt')
        self.netG = Glow().to(opt.device)

    def load_networks(self, epoch=0):
        self.netG.load_state_dict(torch.load(self.pretrained_path))
        self.netG.set_actnorm_init() # correct already
        self.netG.eval()

    def sample(self, batch_size=None, target_class=0):
        # z: [, 48, 4, 4] mean ~ 0, exp(log) ~ 1
        with torch.no_grad():
            # z = torch.normal(64, 48, 4, 4).to(self.device)
            # images = self.netG(y_onehot=None, temperature=1, reverse=True)
            # if z is None:
            #     mean, logs = self.prior(z, None)
            #     z = gaussian_sample(mean, logs, temperature)
            x = self.netG(None, temperature=1, reverse=True, size=batch_size)
        # Helper.save_images(x, '../results', 'glow', 0)
        return x

    def compute_nll(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=6)
    
        nlls = []
        for x,y in dataloader:
            x = x.to(self.device)
            y = None
            with torch.no_grad():
                _, nll, _ = self.netG(x, y_onehot=y)
                nlls.append(nll)
            
        return torch.cat(nlls).cpu()

        

if __name__ == '__main__':
    class opt: pass
    setattr(opt, 'device', torch.device('cuda:0'))
    setattr(opt, 'checkpoint_dir', '../results/checkpoints/udecoder/flow')

    model = Decoder(opt)
    # cifar10 = DataProvider.load_dataset('cifar10', 32, '../data', train=False) # transform differently
    cifar10 = get_CIFAR10()
    cifar_nll = model.compute_nll(cifar10)
    print("CIFAR NLL", torch.mean(cifar_nll))