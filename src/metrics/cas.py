import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
import time, sys, os
import numpy as np
from utils.helper import Helper, AverageMeter
from models.model_ops import init_net
import constant

from metrics.resnets import resnet18 as ResNet18

from PIL import Image
from torchvision import models, transforms
import torchvision



def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
    ])
    return transf


def get_preprocess_transform():
    transf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(32),
    ])
    return transf

# resize and take the center part of image to what our model expects
# def get_input_transform():
# 	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
# 									std=[0.229, 0.224, 0.225])
# 	transf = transforms.Compose([
# 		transforms.Resize((256, 256)),
# 		transforms.CenterCrop(224),
# 		transforms.ToTensor(),
# 		normalize
# 	])
# 	return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()


class Classifiers(object):
    def __init__(self, opt, cas=True):
        self.opt = opt
        if cas:
            if self.opt.data_type in [constant.MNIST, constant.FASHION]:
                self.net = Mnistnet(self.opt.num_classes)
            elif self.opt.data_type == constant.IMAGENET:
                self.net = models.resnet152(pretrained=True).to(opt.device)
            elif self.opt.data_type == constant.CIFAR10:
                # self.net = ResNet18(self.opt.img_size, self.opt.img_channel, self.opt.num_classes)
                self.net = ResNet18(False, False, opt.device)
            else: # CIFAR100
                self.net = Lenet_32(n_channels=3, n_classes=100).to(opt.device)
                # self.net = ResNet18(False, False, opt.device, num_classes=100)
            self.checkpoint_dir = os.path.join(self.opt.checkpoint_dir, 'cas') 
            Helper.try_make_dir(self.checkpoint_dir)
        else: # gan-test
            netname = {
                'cifar100': "cifar100_resnet56",
                'cifar10' : "cifar10_resnet32"
            }[opt.data_type]
            self.net = torch.hub.load("chenyaofo/pytorch-cifar-models", netname, pretrained=True)
        self.net = self.net.to(self.opt.device)

    def load_network(self, pretrained=False):
        if self.opt.data_type in [constant.IMAGENET, constant.CIFAR100]:
            print('Load from Torch Zoo')
            return True
        name = 'resnet18.pt'if pretrained else 'best_net.pth'
        net_path = os.path.join(self.checkpoint_dir, name)
        if os.path.isfile(net_path):
            dct = torch.load(net_path)
            if not pretrained:
                dct = dct['model'] 
            self.net.load_state_dict(dct)
            return True
        print('No check point at ', net_path)
        return False

    def train(self, trainloader, valloader):
        # optimizer = optim.Adam(self.net.parameters(), lr=self.opt.c_lr, betas=(0, 0.999))
        optimizer = optim.SGD(self.net.parameters(), lr=0.1, weight_decay=1e-4)
        criterionCE = nn.CrossEntropyLoss()
        elapsed_time, best_accuracy, total_steps = 0, -np.inf, len(trainloader)
        nepochs = self.opt.nepochs
            
        for e in range(nepochs):
            epoch = e + 1
            start_time = time.time()
            self.net.train()
            losses, top1 = AverageMeter(), AverageMeter()
            # Learning rate
            lr = Helper.learning_rate(self.opt.c_lr, epoch, factor=60)
            Helper.update_lr(optimizer, lr)
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                cur_iter = (epoch - 1) * total_steps + batch_idx
                # if first epoch use warmup
                if epoch - 1 <= self.opt.warmup_epochs:
                    this_lr = self.opt.c_lr * float(cur_iter) / (self.opt.warmup_epochs * len(trainloader))
                    Helper.update_lr(optimizer, this_lr)

                inputs = Variable(inputs, requires_grad=True).to(self.opt.device)
                targets = Variable(targets).long().to(self.opt.device)
                logits = self.net(inputs)

                alpha = 0.01
                # l1_norm = sum(p.abs().sum() for p in model.parameters())
                # l2_norm = 0
                # for p in self.net.parameters():
                #     l2_norm += torch.norm(p)
                loss = criterionCE(logits, targets) #+ alpha * l2_norm

                precs = Helper.accuracy(logits, targets, topk=(1,))
                prec1 = precs[0]
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % self.opt.nsteps_log == 0:
                    sys.stdout.write('\r')
                    sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.6f Acc@1: %.3f' % (epoch, self.opt.nepochs, batch_idx+1,
                                    total_steps, losses.avg, top1.avg))
                    sys.stdout.flush()

            elapsed_time += time.time() - start_time
            print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))
            print('|Learning Rate: ' + str(lr))
            l, acc1 = self.evaluate(valloader)
            print('Evaluating on Val-set: Loss: {:.6f} Acc@1: {:.3f}\n'.format(l, acc1))
            if acc1 > best_accuracy:
                best_accuracy = acc1
                Helper.save_networks(self.net, self.checkpoint_dir, epoch, l, prec1, loss, acc1, True)
                if best_accuracy > 99:
                    return best_accuracy
            if loss.item() < 0.0001:
                return best_accuracy

        return best_accuracy

    def normalize(self, x):
        #  = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        x = x * 0.5 + 0.5 # [0, 1]
        if self.opt.data_type == constant.IMAGENET:
            x = F.interpolate(x, size=224)
        mean = constant.means[self.opt.data_type]
        std = constant.stds[self.opt.data_type]
        for c in range(3):
            x[:, c, ...] = (x[:, c, ...] - mean[c])/std[c]
        return x

    def evaluate(self, dataloader, is_transform=False):
        self.net.eval()
        criterion = nn.CrossEntropyLoss()
        losses, top1 = AverageMeter(), AverageMeter()
        for _, (inputs, targets) in enumerate(dataloader):
            # print(batch_idx)
            targets = Variable(targets).long().to(self.opt.device)
            inputs = Variable(inputs).to(self.opt.device)
            inputs = self.normalize(inputs)
            with torch.no_grad():
                logits = self.net(inputs)
                loss = criterion(logits, targets)
            precs = Helper.accuracy(logits, targets, topk=(1, ))
            prec1 = precs[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

        return losses.avg, top1.avg

    def batch_predict(self, images):
        self.net.eval()
        batch =  torch.stack(tuple(torch.Tensor(np.asarray(preprocess_transform(transforms.ToPILImage(mode='RGB')(i)))) for i in images), dim=0)
        batch = batch.to(self.opt.device)
        batch = torch.transpose(batch, 1, 3)
        batch = (batch - 0.5) / 0.5
        logits = self.net(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def explain(self, dataloader, name):
        explainer = lime_image.LimeImageExplainer()
        self.net.eval()
        criterion = nn.CrossEntropyLoss()
        for _, (inputs, targets) in enumerate(dataloader):
            # print(batch_idx)
            targets = Variable(targets).long().to(self.opt.device)
            inputs = Variable(inputs).to(self.opt.device)
            with torch.no_grad():
                logits = self.net(inputs)
                loss = criterion(logits, targets)
            precs = Helper.accuracy(logits, targets, topk=(1, ))
            prec = precs[0]
            if prec == 0:
                img = inputs[0] #.type(torch.DoubleTensor)
                img = 0.5 * img + 0.5
                img = transforms.ToPILImage()(img.detach().cpu()).convert("RGB")
                img = np.array(pill_transf(img))
                # img = np.transpose(img, (2, 1, 0))
                explanation = explainer.explain_instance(img, self.batch_predict, top_labels=1, hide_color=0, num_samples=1000)
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
                img_boundry = mark_boundaries(temp/255.0, mask)
                plt.imsave('../results/explain-{}.png'.format(name), img_boundry)
                break


# MNIST
class Mnistnet(nn.Module):
    def __init__(self, num_class):
        super(Mnistnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# class Lenet_32(nn.Module):
#     def __init__(self, n_channels=3, n_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(n_channels, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, n_classes)

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv2(out))
#         out = F.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
#         return out


class Lenet_32(nn.Module):
    def __init__(self, n_channels=3, n_classes=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 6, 5),
            nn.ReLU(), 
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 6, 5),
            nn.ReLU(), 
            nn.BatchNorm2d(6),
            # nn.Dropout2d(0.5)
        )
        self.fc = nn.Linear(3456, n_classes)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out