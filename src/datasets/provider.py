import numpy as np
from torch.utils.data import DataLoader, Subset
import constant
from datasets.dataset_provider import DataProvider

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
        
class LoaderProvider(object):
    def __init__(self, opt):
        self.opt = opt
    
    def load_dataset_complexity(self, dataset, label_ratio=1):
        def _load_class_data(dataset, data_class, nlabels):
            inds = np.argwhere(np.asarray(dataset.targets) == data_class)
            inds = inds.reshape(len(inds))
            inds = inds[:nlabels]
            return Subset(dataset, inds)

        def _load_subset(dataset, nlabels):
            s = []
            for c in range(self.opt.num_classes):
                inds = np.argwhere(np.asarray(dataset.targets) == c)
                inds = inds.reshape(len(inds))
                inds = inds[:nlabels]
                s = s + inds.tolist()
            return Subset(dataset, np.asarray(s))

        # main entry
        if self.opt.gan_type == constant.UGAN:
            # unlabeled
            dataset.targets = [-1] * len(dataset)
            return dataset

        if label_ratio == 1:
            if self.opt.gan_type == constant.GANREP:
                if self.opt.gan_class > -1:
                    dataset = _load_class_data(dataset, self.opt.gan_class, nlabels=-1)
        else:
            if label_ratio > 1:
                nlabels = int(label_ratio) # 10
            else:
                nlabels = int(label_ratio * len(dataset) // self.opt.num_classes)
            if self.opt.gan_type == constant.GANREP:
                dataset = _load_class_data(dataset, self.opt.gan_class, nlabels)
            else:
                dataset = _load_subset(dataset, nlabels)
        return dataset


    def get_data_loader(self, train=True):
        if constant.UGAN == self.opt.gan_type:
            dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train, num_attrs=self.opt.num_attrs)
            dataset.targets = [-1] * len(dataset)
        elif constant.EXP_COMPLEXITY == self.opt.exp_mode:
            if self.opt.label_ratio == 1:
                dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train, num_attrs=self.opt.num_attrs)
                if self.opt.gan_type in [constant.GANREP] or self.opt.num_classes == 1:
                    dataset = DataProvider.load_class_dataset(dataset, self.opt.gan_class)
            else:
                full_dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train)
                dataset = self.load_dataset_complexity(full_dataset, self.opt.label_ratio)
        elif constant.EXP_IMBALANCE == self.opt.exp_mode:
            dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train)
            dataset = DataProvider.load_imbalanced_dataset(dataset)
        elif constant.EXP_ASYM_NOISE == self.opt.exp_mode:
            # a same dataset for all models
            self.opt.data_noise_type = 'asymmetric'
            dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train, noise=(self.opt.data_noise_type, self.opt.noise_ratio, self.opt.data_seed))
            # inrep
            if self.opt.gan_type == constant.GANREP:
                inds = np.argwhere(np.asarray(dataset.targets) == self.opt.gan_class)
                inds = inds.reshape(len(inds))
                dataset = Subset(dataset, inds)
        print("Length: ", len(dataset))
        dataloader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, drop_last=True, num_workers=4)
        return dataloader