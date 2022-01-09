from collections import namedtuple

# pretrained
GAN = 'gan'
SCGAN = 'scgan'
BIGGAN = 'biggan'
STYLEGAN = 'stylegan'
FLOW = 'flow'
VAE = 'vae'

#cgan
DECODER = 'udecoder'
REPGAN = 'repgan'
REPGAN_AB = 'repgan_ab'
GANREP = 'ganrep'
TRANSFERGAN = 'transfergan'
MINEGAN = 'minegan'
ACGAN = 'acgan'
PROJGAN = 'projgan'
CONTRAGAN = 'contragan'
CGAN = 'cgan'
SRGAN = 'srgan'
INREP = 'inrep'


#exp
EXP_FULL = 'full'
EXP_COMPLEXITY = 'complexity'
EXP_ASYM_NOISE = 'asymnoise'
EXP_SYM_NOISE = 'symnoise'

#data
MNIST='mnist'
FASHION='fashion'
CIFAR10='cifar10'
CIFAR100='cifar100'
TINY='tiny'
CELEBA = 'celeba'
IMAGENET = 'imagenet'
IMAGENETTE = 'imagenette'


#eval
FID='fid'
IS='is'
INTRA_FID='intrafid'
GAN_TEST='gantest'
CAS='cas'
PR='pr'
VISUAL='visual'
PCA='pca'
TNSE='tsne'
LIME='lime'

PHASE_1 = 1
PHASE_2 = 2


means = {
	'cifar10': (0.4914, 0.4822, 0.4465),
    # mean = (0.485, 0.456, 0.406)
	'cifar100': (0.5071, 0.4867, 0.4408),
	'imagenet': (0.48145466, 0.4578275, 0.40821073),
	'tiny': (0.48145466, 0.4578275, 0.40821073)

}

stds = {
	'cifar10': (0.2023, 0.1994, 0.2010),
    # std = (0.229, 0.224, 0.225)
	'cifar100': (0.2675, 0.2565, 0.2761),
	'imagenet': (0.26862954, 0.26130258, 0.27577711),
	'tiny': (0.26862954, 0.26130258, 0.27577711)
}


num_classes = {
	'cifar10': 10,
	'cifar100': 100,
	'tiny': 200,
	'imagenet': 1000
}