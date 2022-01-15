import os, math, platform, PIL
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from torchvision import datasets, transforms
from copy import deepcopy
import constant
import sys
import functools
from utils.celeba_loader import get_loader, get_celeba_dataset

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle




class CIFAR10NoisyLabels(datasets.CIFAR10):
    """CIFAR10 Dataset with noisy labels.

    Args:
        noise_type (string): Noise type (default: 'symmetric').
            The value is either 'symmetric' or 'asymmetric'.
        noise_rate (float): Probability of label corruption (default: 0.0).
        seed (int): Random seed (default: 12345).
        
    This is a subclass of the `CIFAR10` Dataset.
    """

    def __init__(self,
                 noise_type='symmetric',
                 noise_rate=0.0,
                 seed=12345,
                 **kwargs):
        super(CIFAR10NoisyLabels, self).__init__(**kwargs)
        self.seed = seed
        self.num_classes = 10
        self.flip_pairs = np.asarray([[9, 1], [2, 0], [4, 7], [3, 5], [5, 3]])

        if noise_rate > 0:
            if noise_type == 'symmetric':
                self.symmetric_noise(noise_rate)
            elif noise_type == 'asymmetric':
                self.asymmetric_noise(noise_rate)
            else:
                raise ValueError(
                    'expected noise_type is either symmetric or asymmetric '
                    '(got {})'.format(noise_type))

    def symmetric_noise(self, noise_rate):
        """Insert symmetric noise.

        For all classes, ground truth labels are replaced with uniform random
        classes.
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        mask = np.random.rand(len(targets)) <= noise_rate
        rnd_targets = np.random.choice(self.num_classes, mask.sum())
        targets[mask] = rnd_targets
        targets = [int(target) for target in targets]
        self.targets = targets

    def asymmetric_noise(self, noise_rate):
        """Insert asymmetric noise.

        Ground truth labels are flipped by mimicking real mistakes between
        similar classes. Following `Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach`_, 
        ground truth labels are replaced with
        
        * truck -> automobile,
        * bird -> airplane,
        * deer -> horse
        * cat -> dog
        * dog -> cat

        .. _Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
            https://arxiv.org/abs/1609.03683
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        for i, target in enumerate(targets):
            if target in self.flip_pairs[:, 0]:
                if np.random.uniform(0, 1) <= noise_rate:
                    idx = int(np.where(self.flip_pairs[:, 0] == target)[0])
                    targets[i] = self.flip_pairs[idx, 1]
        targets = [int(x) for x in targets]
        self.targets = targets

    def T(self, noise_type, noise_rate):
        if noise_type == 'symmetric':
            T = (torch.eye(self.num_classes) * (1 - noise_rate) +
                 (torch.ones([self.num_classes, self.num_classes]) /
                  self.num_classes * noise_rate))
        elif noise_type == 'asymmetric':
            T = torch.eye(self.num_classes)
            for i, j in self.flip_pairs:
                T[i, i] = 1 - noise_rate
                T[i, j] = noise_rate
        return T


class DataProvider: 
    @staticmethod
    def load_class_dataset(dataset, data_class, data_size=None, new_label=None):
        indices = np.argwhere(np.asarray(dataset.targets) == data_class)
        indices = indices.reshape(len(indices))
        if not(data_size is None):
            np.random.shuffle(indices)
            indices = indices[:data_size]
        if not(new_label is None):
            t = np.asarray(dataset.targets)
            t[indices] = new_label
            dataset.targets = list(t)
        return Subset(dataset, indices)

    @staticmethod
    def load_dataset(data_type, img_size, data_dir, train=True, noise=None, deterministic=False, num_attrs=None):
        transform_mnist = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        
        transform_3D = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5))])

        transform_imagenet = transforms.Compose([
            t for t in [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                (not deterministic) and transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
            ] if t is not False
        ]) 

        def normalize(x):
            x = 2 * ((x * 255. / 256.) - .5)
            x += torch.zeros_like(x).uniform_(0, 1. / 128)
            return x
        
        transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Lambda(normalize)])
        if data_type == constant.MNIST:
            dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform_mnist)
        elif data_type == constant.CIFAR10:
            if not(noise is None):
                data_noise_type, noise_ratio, data_seed = noise
                Dataset = functools.partial(CIFAR10NoisyLabels,
                        noise_type=data_noise_type,
                        noise_rate=noise_ratio,
                        seed=data_seed)
                dataset = Dataset(root=data_dir, train=train, download=True, transform=transform_cifar)
            else:
                dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform_3D)
        elif data_type == constant.CIFAR100:
            dataset = datasets.CIFAR100(data_dir, train=train, download=True, transform=transform_3D)
        elif constant.CELEBA == data_type:
            mode = 'train' if train else 'test'
            dataset = get_celeba_dataset(mode, data_dir + '/celebA', img_size, num_attrs)
        
        return dataset


class LoaderProvider(object):
    def __init__(self, opt):
        self.opt = opt
    
    def load_dataset_complexity(self, dataset, label_ratio=1):
        def _load_class_data(dataset, data_class, nlabels):
            inds = np.argwhere(np.asarray(dataset.targets) == data_class)
            inds = inds.reshape(len(inds))
            inds = inds[:nlabels]
            return Subset(dataset, inds)
        def _load_semi_dataset(dataset, nlabels):
            t = np.asarray(dataset.targets)
            for c in range(self.opt.num_classes):
                inds = np.argwhere(np.asarray(dataset.targets) == c)
                inds = inds.reshape(len(inds))
                unlabel_inds = inds[nlabels:]
                t[unlabel_inds] = -1
            dataset.targets = list(t)
            return dataset
        def _load_subset(dataset, nlabels):
            s = []
            for c in range(self.opt.num_classes):
                inds = np.argwhere(np.asarray(dataset.targets) == c)
                inds = inds.reshape(len(inds))
                inds = inds[:nlabels]
                s = s + inds.tolist()
            return Subset(dataset, np.asarray(s))

        # main entry
        if label_ratio == 0 and self.opt.is_train: # random-label
            print('Random labels')
            dataset.targets = np.random.randint(0, 10, size=len(dataset)).tolist()
            return dataset

        if self.opt.gan_type == constant.DECODER:
            # unlabeled
            dataset.targets = [-1] * len(dataset)
        else:
            if label_ratio == 1:
                if self.opt.gan_type == constant.GANREP:
                    if self.opt.gan_class > -1:
                        dataset = _load_class_data(dataset, self.opt.gan_class, nlabels=-1)
            # elif label_ratio == 0: # random-label
            #     print('Random labels')
            #     dataset.targets = np.random.randint(0, 9, size=len(dataset)).tolist()
            else:
                if label_ratio > 1:
                    nlabels = int(label_ratio) # 10
                else:
                    nlabels = int(label_ratio * len(dataset) // self.opt.num_classes)
                if self.opt.gan_type == constant.GANREP:
                    dataset = _load_class_data(dataset, self.opt.gan_class, nlabels)
                else:
                    # dataset = _load_semi_dataset(dataset, nlabels)
                    dataset = _load_subset(dataset, nlabels)
        return dataset

    def make_noise_dataset(self, dataset, noise_ratio):
        # main entry: noise ration <= 1
        nnoises = int(noise_ratio * len(dataset) / self.opt.num_classes)
        n_half = self.opt.num_classes // 2
        t = np.asarray(dataset.targets)
        for c in range(n_half):
            inds = np.argwhere(np.asarray(dataset.targets) == c)
            inds = inds.reshape(len(inds))
            noise_inds = inds[:nnoises]
            t[noise_inds] = np.random.randint(low=n_half, high=self.opt.num_classes, size=nnoises)
        dataset.targets = list(t)
        return dataset


    def get_data_loader(self, train=True):
        # if self.opt.data_type == constant.CELEBA:
        #     celeba_dir = os.path.join(self.opt.data_dir, 'celebA')
        #     mode = 'train' if train else 'test'
        #     return get_loader(celeba_dir, self.opt.num_attrs, image_size=self.opt.img_size, batch_size=self.opt.batch_size, mode=mode)
        if constant.EXP_COMPLEXITY == self.opt.exp_mode:
            if self.opt.label_ratio == 1:
                dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train, num_attrs=self.opt.num_attrs)
                if self.opt.gan_type in [constant.GANREP, constant.SRGAN] or self.opt.num_classes == 1:
                    dataset = DataProvider.load_class_dataset(dataset, self.opt.gan_class)
                elif self.opt.num_classes == 2:
                    print('2-class dataset')
                    data = []
                    for k in range(2):
                        data.append(DataProvider.load_class_dataset(dataset, k))
                    dataset = ConcatDataset(data)
                elif self.opt.imbalance:
                    print('Imbalance dataset')
                    data = []
                    for k in range(10):
                        if k < 2:
                            s = 500 if k < 2 else 2500
                            data.append(DataProvider.load_class_dataset(dataset, k, data_size=s))
                        else:
                            data.append(DataProvider.load_class_dataset(dataset, k))
                    dataset = ConcatDataset(data)
            else:
                full_dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train)
                dataset = self.load_dataset_complexity(full_dataset, self.opt.label_ratio)
        else:
            # a same dataset for all models
            if constant.EXP_ASYM_NOISE == self.opt.exp_mode:
                self.opt.data_noise_type = 'asymmetric'
            else:
                self.opt.data_noise_type = 'symmetric'
            dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train, noise=(self.opt.data_noise_type, self.opt.noise_ratio, self.opt.data_seed))
            # inrep
            if self.opt.gan_type == constant.GANREP:
                inds = np.argwhere(np.asarray(dataset.targets) == self.opt.gan_class)
                inds = inds.reshape(len(inds))
                dataset = Subset(dataset, inds)
        dataloader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=4)
        return dataloader