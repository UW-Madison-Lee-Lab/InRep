
import numpy as np
from torchvision import datasets
import torch

class CIFAR10NoisyLabels(datasets.CIFAR10):
    """CIFAR10 Dataset with noisy labels.

    Args:
        noise_type (string): Noise type (default: 'symmetric').
            The value is either 'symmetric' or 'asymmetric'.
        noise_rate (float): Probability of label corruption (default: 0.0).
        seed (int): Random seed (default: 12345).
        
    This is a subclass of the `CIFAR10` Dataset.
    """

    def __init__(self,
                 noise_rate=0.0,
                 seed=12345,
                 **kwargs):
        super(CIFAR10NoisyLabels, self).__init__(**kwargs)
        self.seed = seed
        self.num_classes = 10
        self.flip_pairs = np.asarray([[9, 1], [2, 0], [4, 7], [3, 5], [5, 3]])

        if noise_rate > 0:
            self.asymmetric_noise(noise_rate)

    def asymmetric_noise(self, noise_rate):
        """Insert asymmetric noise.

        Ground truth labels are flipped by mimicking real mistakes between
        similar classes. Following `Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach`_, 
        ground truth labels are replaced with
        
        * truck -> automobile,
        * bird -> airplane,
        * deer -> horse
        * cat -> dog
        * dog -> cat

        .. _Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
            https://arxiv.org/abs/1609.03683
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        for i, target in enumerate(targets):
            if target in self.flip_pairs[:, 0]:
                if np.random.uniform(0, 1) <= noise_rate:
                    idx = int(np.where(self.flip_pairs[:, 0] == target)[0])
                    targets[i] = self.flip_pairs[idx, 1]
        targets = [int(x) for x in targets]
        self.targets = targets

    def T(self, noise_rate):
        T = torch.eye(self.num_classes)
        for i, j in self.flip_pairs:
            T[i, i] = 1 - noise_rate
            T[i, j] = noise_rate
        return T


class CIFAR10ImbalancedLabels(datasets.CIFAR10):
    def __init__(self, path, transforms, train=True):
        super().__init__(path, train, download=True)
        self.transforms = transforms
        self.n_images_per_class = 5
        self.n_classes = 10
        self.new2old_indices = self.create_idx_mapping()

    def create_idx_mapping(self):
        label2idx = defaultdict(lambda: deque(maxlen=self.n_images_per_class))
        for original_idx in range(super().__len__()):
            _, label = super().__getitem__(original_idx)
            label2idx[label].append(original_idx)

        old_idxs = set(itertools.chain(*label2idx.values()))
        new2old_indices = {}
        for new_idx, old_idx in enumerate(old_idxs):
            new2old_indices[new_idx] = old_idx

        return new2old_indices

    def __len__(self):
        return len(self.new2old_indices)

    def __getitem__(self, index):
        index = self.new2old_indices[index]
        im, label = super().__getitem__(index)
        return self.transforms(im), label