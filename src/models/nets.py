import constant
import torch
import torch.nn as nn
from models.model_ops import init_net, onehot, get_norm
# mod = import_module('models.networks')
# Gen = getattr(mod, data_type.upper() + "_Generator")# %%
# net = Gen(gan_type==2, noise_dim, img_size, out_channel, num_class)

# MineGAN
class Miner(nn.Module):
    def __init__(self, u_dim, z_dim, nclasses):
        super().__init__()
        embed_dim = 10
        h_dim = 128
        self.proj = nn.utils.spectral_norm(nn.Embedding(nclasses, embed_dim))
        self.fc = nn.Sequential(
            nn.Linear(u_dim + embed_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h_dim, z_dim)
        )
        init_net(self, init_type='orthogonal')

    def forward(self, u, labels):
        # embed y 
        y = self.proj(labels)
        x = torch.cat([u, y], dim=1)
        z = self.fc(x)
        return z

# Gan-reprogramming 
class Modifier(nn.Module):
    # h_\theta in the paper
    def __init__(self, u_dim, z_dim):
        super().__init__()
        h_dim = 128
        self.fc = nn.Sequential(
            nn.Linear(u_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h_dim, z_dim)
        )
        init_net(self, init_type='orthogonal')

    def forward(self, u):
        return self.fc(u)


# RepGAN
def define_M(cfgs):
    from .networks.modifer import ResNetModifier
    if cfgs.data_type == constant.TINY:
        nlayers = 5
    else:
        nlayers = 3
    return ResNetModifier(nlayers, cfgs.u_dim, cfgs.z_dim, cfgs.num_classes)


def define_G(cfgs):
    if cfgs.data_type == constant.IMAGENETTE:
        from .networks.big_resnet_repgan import Generator
        net = Generator(cfgs.z_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention, cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.g_init, cfgs.G_depth, False)
        return net

    if cfgs.data_type == constant.MNIST:
        from .networks.dcgan_mnist import Generator
        net = Generator(cfgs.z_dim, cfgs.img_size, cfgs.num_classes)
        return net

    if cfgs.decoder_type == constant.STYLEGAN:
        from .networks.stylegan import Generator
        net = Generator(256, 512, 8, channel_multiplier=2)
        return net

    if cfgs.gan_type == constant.CONTRAGAN:
        from .networks.big_resnet import Generator
        net = Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size,  cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
                            cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
                            cfgs.g_init, cfgs.G_depth, False)
        # from .networks.resnet import Generator
        # net = Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.img_channel, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
        #                     cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
        #                     cfgs.g_init, cfgs.G_depth, False)
        return net

    if cfgs.decoder_type == constant.GAN:
        if cfgs.gan_type in [constant.PROJGAN, constant.ACGAN]: # conditional gan
            from .networks.resnet_rcgan import Generator
        elif cfgs.gan_type in [constant.DECODER, constant.REPGAN, constant.GANREP, constant.TRANSFERGAN, constant.MINEGAN, constant.SRGAN]:
            from .networks.resnet_rcgan import UncondGenerator as Generator
        net = Generator(channels=256, image_size=cfgs.img_size, num_classes=cfgs.num_classes)
    else:
        from .networks.glow import Glow
        net = Glow()
    
    return net

def define_D(cfgs):
    if cfgs.data_type == constant.IMAGENETTE:
        from .networks.big_resnet_repgan import Discriminator
        net = Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
                            cfgs.activation_fn, cfgs.nonlinear_embed,
                            cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, False)
        return net

    if cfgs.data_type == constant.MNIST:
        from .networks.dcgan_mnist import Discriminator
        net = Discriminator(cfgs.gan_type, cfgs.num_classes)
        return net

    if cfgs.decoder_type == constant.STYLEGAN:
        from .networks.stylegan import Discriminator
        net = Discriminator(256, 2, num_classes=cfgs.num_classes)
        return net

    if cfgs.gan_type == constant.CONTRAGAN:
        from .networks.resnet import Discriminator
        net = Discriminator(cfgs.img_size, cfgs.img_channel, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
                            cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
                            cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, False)
        return net

    if cfgs.gan_type in [constant.PROJGAN]:
        from .networks.resnet_rcgan import ProjectionDiscriminator as Discriminator
    elif cfgs.gan_type in [constant.ACGAN, constant.TRANSFERGAN, constant.MINEGAN]:
        from .networks.resnet_rcgan import ACGANDiscriminator as Discriminator
    elif cfgs.gan_type in [constant.DECODER, constant.GANREP]:
        from .networks.resnet_rcgan import UncondDiscriminator as Discriminator
    elif cfgs.gan_type in [constant.REPGAN, constant.SRGAN]:
        from .networks.resnet_rcgan import RepGANDiscriminator as Discriminator
    net = Discriminator(num_classes=cfgs.num_classes, channels=128, spectral_norm=True, pooling='sum')
   
    return net