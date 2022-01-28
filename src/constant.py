from collections import namedtuple

# pretrained
GAN = 'gan'
STYLEGAN = 'stylegan'

#cgan
UGAN = 'ugan'
INREP = 'inrep'
INREP_AB = 'inrep_ab'
GANREP = 'ganrep'
MINEGAN = 'minegan'
ACGAN = 'acgan'
PROJGAN = 'projgan'
CONTRAGAN = 'contragan'
CGAN = 'cgan'


#exp
EXP_COMPLEXITY = 'complexity'
EXP_IMBALANCE = 'imbalance'
EXP_ASYM_NOISE = 'asymnoise'

#data
MNIST='mnist'
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
}

stds = {
	'cifar10': (0.2023, 0.1994, 0.2010),
	'cifar100': (0.2675, 0.2565, 0.2761),
}


num_classes = {
	'cifar10': 10,
	'cifar100': 100,
}