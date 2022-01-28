import os
import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset
from torchvision import datasets, transforms
import functools
import constant
from datasets.cifar10_noisylabels import CIFAR10NoisyLabels

class DataProvider: 
    @staticmethod
    def load_imbalanced_dataset(dataset, oversampling=10):
        labels = dataset.targets
        ind_0 = np.argwhere(np.asarray(labels) == 0)[:, 0].tolist()
        ind_1 = np.argwhere(np.asarray(labels) == 1)[:, 0].tolist()
        major_indices = np.argwhere(np.asarray(labels) > 1)[:, 0]
        major_classes = Subset(dataset, major_indices)

        num_minor_samples = int(0.1 * len(ind_0))
        minor_indices = ind_0[:num_minor_samples] + ind_1[:num_minor_samples]
        minor_classes = Subset(dataset, minor_indices)
        return ConcatDataset([major_classes] + [minor_classes] * oversampling)

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

        def normalize(x):
            x = 2 * ((x * 255. / 256.) - .5)
            x += torch.zeros_like(x).uniform_(0, 1. / 128)
            return x
        
        transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Lambda(normalize)])
        
        if data_type == constant.MNIST:
            dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform_mnist)
        elif data_type == constant.CIFAR10:
            if noise is not None:
                data_noise_type, noise_ratio, data_seed = noise
                Dataset = functools.partial(CIFAR10NoisyLabels,
                        noise_rate=noise_ratio,
                        seed=data_seed)
                dataset = Dataset(root=data_dir, train=train, download=True, transform=transform_cifar)
            else:
                dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform_3D)
        elif data_type == constant.CIFAR100:
            dataset = datasets.CIFAR100(data_dir, train=train, download=True, transform=transform_3D)
        return dataset