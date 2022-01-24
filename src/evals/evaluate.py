import os
import numpy as np
# from numpy.lib.type_check import real

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split

from datasets.dataset_provider import DataProvider
from utils.helper import Helper
from models.model_ops import GANDataset
from metrics.fid_score import FID, get_activations
from metrics.inception import InceptionV3
from metrics.cas import Classifiers
from metrics.precision_recall import knn_precision_recall_features
from metrics.cas import Classifiers

from evals.eval_ops import load_gan, mix_sample, mix_sample_class, mix_sample_class_ugan
import constant

class Tester():
    def __init__(self, opt, load_data=False):
        self.opt, self.load_data = opt, load_data
        if opt.data_type in [constant.CIFAR100]:
            self.num_samples = 100
        else:
            self.num_samples = 1000
        self.offset = opt.label_ratio if opt.exp_mode == constant.EXP_COMPLEXITY else opt.noise_ratio
        self.log_file = opt.eval_path + '.txt'
    
    def get_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, drop_last=False, num_workers=2)

    def evaluate(self):
        # evaluatation mode
        if self.opt.eval_mode == constant.VISUAL:
            print('Eval Visual')
            self.save_samples(num_images=10)
            return None

        if self.opt.eval_mode == constant.CAS: # Fitting capacity
            print('Eval CAS')
            score = self._evaluate_cas()
            message = '{} {:.4f}'.format(self.offset, score)
            logf = open(self.log_file, 'a+')
            Helper.log(logf, message)
            return None

        if self.opt.eval_mode == constant.PR: 
            # load inception
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            inception_model = InceptionV3([block_idx]).to(self.opt.device)
            # for each model
            print('Eval Precision-Recall')
            # dataset
            real_dataset = self._load_real_data() 
            fake_dataset = self._load_fake_data()
            # features
            real_features = get_activations(real_dataset, inception_model, batch_size=50, dims=2048, device=self.opt.device)
            generated_features = get_activations(fake_dataset, inception_model, batch_size=50, dims=2048, device=self.opt.device)
            # calculate
            k = 10
            state = knn_precision_recall_features(torch.Tensor(real_features), torch.Tensor(generated_features), nhood_sizes=[k])
            p, r = state['precision'][0], state['recall'][0]
            message = "Precision: {:.4f} -- Recall {:.4f}".format(p, r)
            logf = open(self.log_file, 'a+')
            Helper.log(logf, message)
            score = 1/(1/p + 1/r)
            # for both two cases
            message = '{} {:.4f}'.format(self.offset, score)
            logf = open(self.log_file, 'a+')
            Helper.log(logf, message)
            return None
        
        def get_path(c=-1):
            suffix = '' if c == -1 else '_' + str(c)
            real_path = os.path.join(self.opt.real_classifier_dir, 'stats_real{}.npy'.format(suffix))
            fake_path = os.path.join(self.opt.sample_dir, 'stats_fake{}.npy'.format(suffix))
            return real_path, fake_path

        if self.opt.eval_mode == constant.FID: # FID
            # load inception
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            inception_model = InceptionV3([block_idx]).to(self.opt.device)
            print('Eval FID @ ' + str(self.opt.gan_class))
            if self.opt.gan_type in [constant.UGAN, constant.INREP] and self.opt.gan_class >= 0:
                real_path, fake_path = get_path(self.opt.gan_class)
                full_dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=False, num_attrs=self.opt.num_attrs)
                real_dataset = DataProvider.load_class_dataset(full_dataset, self.opt.gan_class, data_size=500)
            else:
                real_path, fake_path = get_path()
                real_dataset = self._load_real_data() if not(os.path.isfile(real_path)) else None

            fake_dataset = self._load_fake_data() # modify here
            score = FID(inception_model, [real_path, fake_path], [real_dataset, fake_dataset], 64, self.opt.device, 2048, True)
            message = '{} @ {} {:.4f}'.format(self.offset, self.opt.gan_class, score)
            logf = open(self.log_file, 'a+')
            Helper.log(logf, message)
            return None
        
        if self.opt.eval_mode == constant.INTRA_FID: # classFID
            print('Eval Class FID')
            # load inception
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            inception_model = InceptionV3([block_idx]).to(self.opt.device)
            # real data
            full_dataset = None
            real_dataset = None
            # fake data
            if self.opt.gan_class > -1:
                test_classes = [self.opt.gan_class]
            else:
                test_classes = np.arange(self.opt.num_classes)
            fake_data_lst, fake_label_lst = self._load_fake_data_class(test_classes)
            scores = []
            #####TESTING########
            for c, _ in enumerate(test_classes):
                real_path, fake_path = get_path(c)
                # fake data
                fake_dataset = GANDataset(torch.tensor(fake_data_lst[c]), torch.tensor(fake_label_lst[c], dtype=torch.long))
                # real data
                if not(os.path.isfile(real_path)):
                    if full_dataset is None:
                        full_dataset = DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=False, num_attrs=self.opt.num_attrs)
                    real_dataset = DataProvider.load_class_dataset(full_dataset, c)
                # score
                s = FID(inception_model, [real_path, ""], [real_dataset, fake_dataset], 64, self.opt.device, 2048, True)
                # append to the scores
                scores.append(s)
            # print out
            message = '{} {}'.format(self.offset, scores)
            logf = open(self.log_file, 'a+')
            Helper.log(logf, message)
            # mean score
            score = np.asarray(scores).mean()
            # for both two cases
            message = '{} {:.4f}'.format(self.offset, score)
            logf = open(self.log_file, 'a+')
            Helper.log(logf, message)
            return None


    def _evaluate_cas(self, lime=False):
        # real data
        if self.opt.data_type in [constant.CIFAR100]:
            self.num_samples = 100
        else:
            self.num_samples = 1000

        real_dataset = self._load_real_data()
        fake_dataset = self._load_fake_data()
        n = len(real_dataset)
        test_size = int(n*0.8)
        lengths = [test_size, n - test_size]
        test_set, val_set = random_split(real_dataset, lengths)
        train_set = fake_dataset
        
        train_loader = self.get_loader(train_set)
        val_loader = self.get_loader(val_set)
        test_loader = self.get_loader(test_set)
        # train classifier
        netC = Classifiers(self.opt, True)
        val_accuracy = netC.train(train_loader, val_loader)
        # load the best model
        netC.load_network()
        if lime:
            netC.explain(test_loader, 'test')
        else:
            loss, test_accuracy = netC.evaluate(test_loader)
            print('Testset -- Loss {:.4f} Accuracy: {:.4f}'.format(loss, test_accuracy))
            return test_accuracy

    def _load_fake_data_class(self, test_classes=0):
        fake_data_path = os.path.join(self.opt.sample_dir, 'gan_samples_classwise.npy')
        if self.load_data and os.path.isfile(fake_data_path):
            dataset = np.load(fake_data_path, allow_pickle=True)
            dataset = dataset.item()
            data_lst, label_lst = dataset['data'], dataset['labels']
        else:
            gan = load_gan(self.opt)
            data_lst, label_lst = [], []
            if test_classes == []: test_classes = np.arange(self.opt.num_classes)

            for target_class in test_classes:
                data, labels = mix_sample_class(gan, self.num_samples, target_class)
                data_lst.append(data)
                label_lst.append(labels)
            np.save(fake_data_path, {'data': data_lst, 'labels': label_lst})
            
        return data_lst, label_lst

    def _load_fake_data(self, epoch=-1, dataloader=None):
        fake_data_path = os.path.join(self.opt.sample_dir, 'gan_samples_{}.npy'.format(epoch))
        if self.load_data and os.path.isfile(fake_data_path):
            dataset = np.load(fake_data_path, allow_pickle=True)
            dataset = dataset.item()
            data, labels = dataset['data'], dataset['labels']
        else:
            gan = load_gan(self.opt, epoch)
            if self.opt.gan_type in [constant.SRGAN, constant.INREP] and self.opt.gan_class >= 0:
                data, labels = mix_sample_class(gan, self.num_samples, self.opt.gan_class)
            elif self.opt.gan_type == constant.UGAN and self.opt.gan_class >= 0:
                if self.opt.decoder_type == constant.BIGGAN:
                    data, labels = mix_sample(gan, self.num_samples, 1)
                else:
                    netC = Classifiers(self.opt, self.opt.real_classifier_dir)
                    valid = netC.load_network(pretrained=True)
                    netC.net.eval()
                    if not valid:
                        raise NotImplementedError
                    elif dataloader is not None:
                        l, p = netC.evaluate(dataloader)
                        print('Accuracy of pretrained model: {:.4f}'.format(p))
                    data, labels = mix_sample_class_ugan(gan, netC, self.num_samples, self.opt.gan_class)
            else:
                data, labels = mix_sample(gan, self.num_samples, self.opt.num_classes)
            np.save(fake_data_path, {'data': data, 'labels': labels})
        datasets = GANDataset(torch.tensor(data), torch.tensor(labels, dtype=torch.long))
        return datasets

    def _load_real_data(self, train=False):
        return DataProvider.load_dataset(self.opt.data_type, self.opt.img_size, self.opt.data_dir, train=train, num_attrs=self.opt.num_attrs)

    def save_samples(self, num_images=5):
        gan = load_gan(self.opt)
        if constant.UGAN == self.opt.gan_type:
            samples = gan.sample(num_images)
        else:
            samples = []
            nclasses = 3
            num_images = 5
            for k in range(nclasses):
                samples.append(gan.sample(num_images, target_class=k))
            samples = torch.cat(samples, dim=0)

        save_path = self.opt.eval_path + '{}.png'.format(self.opt.label_ratio)
        kwgs = {'padding': 1}
        save_image(samples.cpu(), save_path, nrow=num_images, normalize=True)