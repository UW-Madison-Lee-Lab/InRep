import os, torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader

from utils.provider import DataProvider
from utils.helper import Helper
from models.model_ops import GANDataset
from metrics.fid_score import FID
from metrics.inception import InceptionV3
from metrics.cas import Classifiers

from evals.eval_ops import mix_sample_class, mix_sample, load_gan
import constant


class BaseScorer(object):
    def __init__(self, opt):
        self.sign = 1 # for desending scores
        self.num_classes = opt.num_classes
        if opt.data_type in [constant.TINY, constant.IMAGENET]:
            self.num_samples = 50 
        else:
            self.num_samples = 10000 // self.num_classes

        if opt.eval_mode == constant.INTRA_FID:
            if opt.data_type == constant.IMAGENET:
                self.num_samples = 1000
            elif opt.data_type == constant.TINY:
                self.num_samples = 100
            

    def validate(self, gan, logf):
        pass


class PrecisionScorer(BaseScorer):
    def __init__(self, opt):
        BaseScorer.__init__(self, opt)
        self.sign = -1
        netC = Classifiers(opt, opt.real_classifier_dir)
        valid = netC.load_network(pretrained=True)
        if valid:
            print('Load classifier')
        else:
            print('No classifier')
            train_set = DataProvider.load_dataset(opt.data_type, opt.img_size, opt.data_dir, train=True, num_attrs=opt.num_attrs)
            val_set = DataProvider.load_dataset(opt.data_type, opt.img_size, opt.data_dir, train=False, num_attrs=opt.num_attrs)
            train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True)
            val_accuracy = netC.train(train_loader, val_loader)
            netC.load_network()
            print('Eval on real data -- Accuracy: {:.4f}'.format(val_accuracy))
        netC.net.eval()
        self.netC = netC
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = opt.device
    
    def test(self, inputs, targets):
        with torch.no_grad():
            logits = self.netC.net(inputs.contiguous())
            loss = self.criterion(logits, targets)
        precs = Helper.accuracy(logits, targets, topk=(1, ))
        return loss, precs[0]

    def evaluate(self, dataloader, is_transform=False):
        out = self.netC.evaluate(dataloader, is_transform)
        return out

    def validate(self, gan, logf):
        with torch.no_grad():
            data, labels = mix_sample(gan, self.num_samples, self.num_classes)
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            data = torch.tensor(data).to(self.device)
            # apply transforms
            data = self.netC.normalize(data)
            loss, prec = self.test(data, labels)
            Helper.log(logf, "=== GAN-Test -- loss {:.4f} prec {:.4f}".format(loss, prec))
        return prec


class FIDScorer(BaseScorer):
    def __init__(self, opt):
        BaseScorer.__init__(self, opt)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx]).to(opt.device)
        self.real_path = os.path.join(opt.real_classifier_dir, 'stats_real.npy')
        self.device = opt.device
        if not(os.path.isfile(self.real_path)):
            self.real_dataset = DataProvider.load_dataset(opt.data_type, opt.img_size, opt.data_dir, train=False, num_attrs=opt.num_attrs)
        else:
            self.real_dataset = None
    
    def validate(self, gan, logf):
        data, labels = mix_sample(gan, self.num_samples, self.num_classes)
        fake_dataset = GANDataset(torch.tensor(data), torch.tensor(labels, dtype=torch.long))
        score = FID(self.inception_model, [self.real_path, ""], [self.real_dataset, fake_dataset], 64, self.device, 2048, True)
        Helper.log(logf, "=== FID score {:.4f}".format(score))
        return score


class ClassFIDScorer(BaseScorer):
    def __init__(self, opt, target_class):
        BaseScorer.__init__(self, opt)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx]).to(opt.device)
        self.real_path = os.path.join(opt.real_classifier_dir, 'stats_real_{}.npy'.format(target_class))
        self.target_class = target_class
        self.device, self.data_type, self.img_size, self.data_dir, self.num_attrs = opt.device, opt.data_type, opt.img_size, opt.data_dir, opt.num_attrs

    def test(self):
        full_dataset = DataProvider.load_dataset(self.data_type, self.img_size, self.data_dir, train=False)
        score = FID(self.inception_model, [self.real_path, ""], [None, full_dataset], 64, self.device, 2048, True)
        print("\n=== FID score -- All {:.4f}".format(score))
        for c in range(self.opt.num_classes):
            fake_dataset = DataProvider.load_class_dataset(deepcopy(full_dataset), c)
            score = FID(self.inception_model, [self.real_path, ""], [None, fake_dataset], 64, self.device, 2048, True)
            print("\n=== class FID score -- class {} {:.4f}".format(c, score))
    
    def validate(self, gan, logf):
        if not(os.path.isfile(self.real_path)):
            full_dataset = DataProvider.load_dataset(self.data_type, self.img_size, self.data_dir, train=False, num_attrs=self.num_attrs)
            real_dataset = DataProvider.load_class_dataset(full_dataset, self.target_class)
        else:
            real_dataset = None
        
        data, labels = mix_sample_class(gan, self.num_samples, self.target_class)
        fake_dataset = GANDataset(torch.tensor(data), torch.tensor(labels, dtype=torch.long))
        score = FID(self.inception_model, [self.real_path, ""], [real_dataset, fake_dataset], 64, self.device, 2048, True)
        Helper.log(logf, "\n=== Class FID score {:.4f}".format(score))
        return score

