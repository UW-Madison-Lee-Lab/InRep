from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_ops import onehot
from .common import (CustomConv2d, CustomEmbedding, CustomLinear,
                     ResidualBlock, OptimizedResidualBlock, CondResidualBlock, UncondResidualBlock,
                     global_pooling, init)
from .condbatchnorm import CondBatchNorm2d, CondBatchNorm1d


class Generator(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 channels=256,
                 image_size=32,
                 num_classes=10,
                 spectral_norm=False):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        self.linear1 = CustomLinear(latent_dim,
                                    channels * (image_size // 8) *
                                    (image_size // 8),
                                    spectral_norm=spectral_norm)
        self.block2 = CondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.block3 = CondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.block4 = CondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.norm5 = nn.BatchNorm2d(channels)
        self.relu5 = nn.ReLU()
        self.conv5 = CustomConv2d(channels,
                                  3,
                                  3,
                                  spectral_norm=spectral_norm,
                                  residual_init=False)
        self.tanh = nn.Tanh()

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim)

    def sample_y(self, batch_size):
        return torch.LongTensor(batch_size).random_(self.num_classes)

    def forward(self, input, label):
        output = input
        output = self.linear1(output)
        output = output.view(-1, self.channels, self.image_size // 8,
                             self.image_size // 8)
        output = self.block2(output, label)
        output = self.block3(output, label)
        output = self.block4(output, label)
        output = self.norm5(output)
        output = self.relu5(output)
        output = self.conv5(output)
        output = self.tanh(output)
        return output



class ACGANDiscriminator(nn.Module):
    def __init__(self,
                 num_classes=10,
                 channels=128,
                 dropout=False,
                 spectral_norm=False,
                 pooling='mean'):
        super(ACGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.dropout = dropout
        self.spectral_norm = spectral_norm
        self.pooling = pooling

        self.block1 = OptimizedResidualBlock(3,
                                             channels,
                                             3,
                                             spectral_norm=spectral_norm)
        self.block2 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='down',
                                    spectral_norm=spectral_norm)
        self.block3 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        self.block4 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        self.relu5 = nn.ReLU()
        self.linear5dis = CustomLinear(channels,
                                       1,
                                       spectral_norm=spectral_norm)
        self.linear5cls = CustomLinear(channels, num_classes)

    def forward(self, input, dropout=None):
        if dropout is None:
            dropout = self.dropout
        output = input
        output = self.block1(output)
        output = self.block2(output)
        output = F.dropout(output, p=0.2, training=dropout)
        output = self.block3(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.block4(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.relu5(output)
        out_feat = global_pooling(output, 'mean')
        output = global_pooling(output, self.pooling)
        out_dis = self.linear5dis(output)
        out_cls = self.linear5cls(out_feat)
        return out_dis.squeeze(), out_cls.squeeze(), out_feat


class ProjectionDiscriminator(nn.Module):
    def __init__(self,
                 num_classes=10,
                 channels=128,
                 dropout=False,
                 spectral_norm=False,
                 pooling='mean'):
        super(ProjectionDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.dropout = dropout
        self.spectral_norm = spectral_norm
        self.pooling = pooling

        self.block1 = OptimizedResidualBlock(3,
                                             channels,
                                             3,
                                             spectral_norm=spectral_norm)
        self.block2 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='down',
                                    spectral_norm=spectral_norm)
        self.block3 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        self.block4 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        self.relu5 = nn.ReLU()
        self.linear5 = CustomLinear(channels,
                                    1,
                                    bias=False,
                                    spectral_norm=spectral_norm)
        self.embed5 = CustomEmbedding(num_classes,
                                      channels,
                                      spectral_norm=spectral_norm)

    def forward(self, input, label, dropout=None):
        if dropout is None:
            dropout = self.dropout
        output = input
        output = self.block1(output)
        output = self.block2(output)
        output = F.dropout(output, p=0.2, training=dropout)
        output = self.block3(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.block4(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.relu5(output)
        out_feat = global_pooling(output, 'mean')
        output = global_pooling(output, self.pooling)
        out_dis = self.linear5(output)
        out_dis += torch.sum(self.embed5(label) * output, dim=1, keepdim=True)
        return out_dis.squeeze(), out_feat


class UncondGenerator(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 channels=256,
                 image_size=32,
                 num_classes=10,
                 spectral_norm=False):
        super(UncondGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        self.linear1 = CustomLinear(latent_dim,
                                    channels * (image_size // 8) *
                                    (image_size // 8),
                                    spectral_norm=spectral_norm)
        self.block2 = UncondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.block3 = UncondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.block4 = UncondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.norm5 = nn.BatchNorm2d(channels)
        self.relu5 = nn.ReLU()
        self.conv5 = CustomConv2d(channels,
                                  3,
                                  3,
                                  spectral_norm=spectral_norm,
                                  residual_init=False)
        self.tanh = nn.Tanh()

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim)

    def sample_y(self, batch_size):
        return torch.LongTensor(batch_size).random_(self.num_classes)

    def forward(self, input):
        output = input
        output = self.linear1(output)
        output = output.view(-1, self.channels, self.image_size // 8,
                             self.image_size // 8)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.norm5(output)
        output = self.relu5(output)
        output = self.conv5(output)
        output = self.tanh(output)
        return output


class UncondDiscriminator(nn.Module):
    def __init__(self,
                 num_classes=10,
                 channels=128,
                 dropout=False,
                 spectral_norm=False,
                 pooling='mean'):
        super(UncondDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.dropout = dropout
        self.spectral_norm = spectral_norm
        self.pooling = pooling

        self.block1 = OptimizedResidualBlock(3,
                                             channels,
                                             3,
                                             spectral_norm=spectral_norm)
        self.block2 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='down',
                                    spectral_norm=spectral_norm)
        self.block3 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        self.block4 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        self.relu5 = nn.ReLU()
        self.linear5dis = CustomLinear(channels,
                                       1,
                                       spectral_norm=spectral_norm)
        ##### Additional for inrep
        self.authen = CustomLinear(channels,
                                       num_classes,
                                       spectral_norm=spectral_norm)
        self.num_class = num_classes

    def embed(self, x):
        output = x
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.relu5(output)
        output = global_pooling(output, self.pooling)
        return output

    def predict(self, x, labels):
        batch_size = x.shape[0]
        x = self.embed(x)
        out = self.authen(x).squeeze()
        encoded_labels = onehot(labels, self.num_class).view(batch_size, self.num_class, 1)
        out = torch.bmm(out.view(batch_size, 1, self.num_class), encoded_labels).view(batch_size)
        return out

    def forward(self, x, dropout=None):
        if dropout is None:
            dropout = self.dropout
        output = x
        output = self.block1(output)
        output = self.block2(output)
        output = F.dropout(output, p=0.2, training=dropout)
        output = self.block3(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.block4(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.relu5(output)
        out_feat = global_pooling(output, 'mean')
        output = global_pooling(output, self.pooling)
        out_dis = self.linear5dis(output)
        return out_dis.squeeze(), out_feat



class inrepDiscriminator(nn.Module):
    def __init__(self,
                 num_classes=10,
                 channels=128,
                 dropout=False,
                 spectral_norm=False,
                 pooling='mean'):
        super(inrepDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.dropout = dropout
        self.spectral_norm = spectral_norm
        self.pooling = pooling

        self.block1 = OptimizedResidualBlock(3,
                                             channels,
                                             3,
                                             spectral_norm=spectral_norm)
        self.block2 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='down',
                                    spectral_norm=spectral_norm)
        # self.bn1 = CondBatchNorm2d(channels, num_classes)
        self.block3 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        # self.bn2 = CondBatchNorm2d(channels, num_classes)
        self.block4 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        # self.bn3 = CondBatchNorm2d(channels, num_classes)
        self.relu5 = nn.LeakyReLU(0.2)
        self.linear5 = CustomLinear(channels,
                                    1,
                                    bias=False,
                                    spectral_norm=spectral_norm)
        self.embed5 = CustomEmbedding(num_classes,
                                      channels,
                                      spectral_norm=spectral_norm)
        self.authen = CustomLinear(channels,
                                       num_classes,
                                       spectral_norm=spectral_norm)
        self.num_class = num_classes


    def forward(self, input, label, dropout=None):
        if dropout is None:
            dropout = self.dropout
        output = input
        output = self.block1(output)
        output = self.block2(output)
        # output = self.bn1(output, label)
        output = F.dropout(output, p=0.2, training=dropout)
        output = self.block3(output)
        output = F.dropout(output, p=0.5, training=dropout)
        # output = self.bn2(output, label)
        output = self.block4(output)
        output = F.dropout(output, p=0.5, training=dropout)
        # output = self.bn3(output, label)
        output = self.relu5(output)
        output = global_pooling(output, self.pooling)
        batch_size = input.shape[0]
        encoded_labels = onehot(label, self.num_class).view(batch_size, self.num_class, 1)
        out = self.authen(output).squeeze()
        out = torch.bmm(out.view(batch_size, 1, self.num_class), encoded_labels).view(batch_size)
        return out, None









class inrepDiscriminator_w_projection(nn.Module):
    def __init__(self,
                 num_classes=10,
                 channels=128,
                 dropout=False,
                 spectral_norm=False,
                 pooling='mean'):
        super(inrepDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.dropout = dropout
        self.spectral_norm = spectral_norm
        self.pooling = pooling

        self.block1 = OptimizedResidualBlock(3,
                                             channels,
                                             3,
                                             spectral_norm=spectral_norm)
        self.block2 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='down',
                                    spectral_norm=spectral_norm)
        self.block3 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        self.block4 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    spectral_norm=spectral_norm)
        self.relu5 = nn.ReLU()
        self.linear5 = CustomLinear(channels,
                                    1,
                                    bias=False,
                                    spectral_norm=spectral_norm)
        self.embed5 = CustomEmbedding(num_classes,
                                      channels,
                                      spectral_norm=spectral_norm)

    def forward(self, input, label, dropout=None):
        if dropout is None:
            dropout = self.dropout
        output = input
        output = self.block1(output)
        output = self.block2(output)
        output = F.dropout(output, p=0.2, training=dropout)
        output = self.block3(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.block4(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.relu5(output)
        out_feat = global_pooling(output, 'mean')
        output = global_pooling(output, self.pooling)
        out_dis = self.linear5(output)
        out_proj = torch.sum(self.embed5(label) * output, dim=1, keepdim=True)
        # out_dis = torch.sum(self.embed5(label) * output, dim=1, keepdim=True)
        out = out_dis + out_proj
        return out.squeeze(), out_feat





class UncondGenerator_v1(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 channels=256,
                 image_size=32,
                 num_classes=10,
                 spectral_norm=False):
        super(UncondGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        self.linear1 = CustomLinear(latent_dim,
                                    channels * (image_size // 8) *
                                    (image_size // 8),
                                    spectral_norm=spectral_norm)
        self.block2 = UncondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.block3 = UncondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.block4 = UncondResidualBlock(channels,
                                        channels,
                                        3,
                                        num_classes,
                                        resample='up',
                                        spectral_norm=spectral_norm)
        self.norm5 = nn.BatchNorm2d(channels)
        self.relu5 = nn.ReLU()
        self.conv5 = CustomConv2d(channels,
                                  3,
                                  3,
                                  spectral_norm=spectral_norm,
                                  residual_init=False)
        self.tanh = nn.Tanh()

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim)

    def sample_y(self, batch_size):
        return torch.LongTensor(batch_size).random_(self.num_classes)

    # def get_adapter(self):
    #     self.adapter = nn.Sequential(
    #         UnetSkipConnectionBlock(self.channels, self.channels//2),
    #         UnetSkipConnectionBlock(self.channels, self.channels//2),
    #         UnetSkipConnectionBlock(self.channels, self.channels//2)
    #     )

    # def forward(self, input):
    #     output = input
    #     output = self.linear1(output)
    #     output = output.view(-1, self.channels, self.image_size // 8,
    #                          self.image_size // 8)
    #     output = self.block2(output)
    #     output = self.block3(output)
    #     output = self.block4(output)
    #     output = self.norm5(output)
    #     output = self.relu5(output)
    #     output = self.conv5(output)
    #     output = self.tanh(output)
    #     return output
    
    def forward(self, input, adapter=None, labels=None):
        output = input
        output = self.linear1(output)
        if adapter is not None:
            output = output + adapter(output, labels)
        output = output.view(-1, self.channels, self.image_size // 8,
                             self.image_size // 8)
        output = self.block2(output) 
        output = self.block3(output) 
        output = self.block4(output)
        output = self.norm5(output)
        output = self.relu5(output)
        output = self.conv5(output)
        output = self.tanh(output)
        return output

    def forward_v0(self, input, adapter=None, labels=None):
        output = input
        output = self.linear1(output)
        output = output.view(-1, self.channels, self.image_size // 8,
                             self.image_size // 8)
        output = self.block2(output) 
        if adapter is not None:
            output = output + adapter[0](output, labels)
        output = self.block3(output) 
        if adapter is not None:
            output = output + adapter[1](output, labels)
        output = self.block4(output)
        if adapter is not None:
            output = output + adapter[2](output, labels)
        output = self.norm5(output)
        output = self.relu5(output)
        output = self.conv5(output)
        output = self.tanh(output)
        return output
