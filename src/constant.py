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
INREP = 'inrep'
INREP_AB = 'inrep_ab'
GANREP = 'ganrep'
MINEGAN = 'minegan'
ACGAN = 'acgan'
PROJGAN = 'projgan'
CONTRAGAN = 'contragan'
CGAN = 'cgan'
SRGAN = 'srgan'


#exp
EXP_FULL = 'full'
EXP_COMPLEXITY = 'complexity'
EXP_ASYM_NOISE = 'asymnoise'

#data
MNIST='mnist'
FASHION='fashion'
CIFAR10='cifar10'
CIFAR100='cifar100'
CELEBA = 'celeba'


#eval
FID='fid'
INTRA_FID='intrafid'
CAS='cas'
PR='pr'
VISUAL='visual'

PHASE_1 = 1
PHASE_2 = 2


means = {
	'cifar10': (0.4914, 0.4822, 0.4465),
	'cifar100': (0.5071, 0.4867, 0.4408),
	'imagenet': (0.48145466, 0.4578275, 0.40821073),
	'tiny': (0.48145466, 0.4578275, 0.40821073)

}

stds = {
	'cifar10': (0.2023, 0.1994, 0.2010),
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