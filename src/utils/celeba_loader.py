from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os, csv
import random
import torchvision


def load_data(fpath):
    data_lst = []
    with open(fpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                data_lst.append([row[0], int(row[1])])
                line_count += 1
    print(f'Processed {line_count} lines.')
    return data_lst

def save_dataset(data_lst, fpath):
    with open(fpath, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(['filename', 'label'])
        writer.writerows(data_lst)

def get_indexed_label(l):
    return int("".join(str(x) for x in l), 2)

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, data_dir, num_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.data_dir = data_dir
        self.num_attrs = num_attrs
        self.image_dir = data_dir + '/img_align_celeba'
        self.attr_path = data_dir + '/list_attr_celeba.txt'
        self.dataset_path = data_dir + '/{}_dataset_{}.csv'.format(mode, self.num_attrs)
        self.transform = transform
        self.attr2idx = {}
        self.idx2attr = {}
        if os.path.isfile(self.dataset_path):
            dataset = load_data(self.dataset_path)
        else:
            dataset = self.preprocess(mode)
        
        self.data = []
        self.targets = []
        self.num_images = len(dataset)
        for i in range(self.num_images):
            self.targets.append(dataset[i][1])
            self.data.append(dataset[i][0])
        
    def preprocess(self, mode):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        train_dataset = []
        test_dataset = []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = [int(values[k] == '1') for k in range(self.num_attrs)]
            idx = get_indexed_label(label)
            if (i+1) < 2000:
                test_dataset.append([filename, idx])
            else:
                train_dataset.append([filename, idx])

        print('Finished preprocessing the CelebA dataset... And Store')
        # store training set and testing set
        save_dataset(train_dataset, self.data_dir + '/train_dataset_{}.csv'.format(self.num_attrs))
        save_dataset(test_dataset, self.data_dir + '/test_dataset_{}.csv'.format(self.num_attrs))
        if mode == 'train':
            return train_dataset
        else:
            return test_dataset

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename = self.data[index]
        label = self.targets[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_celeba_dataset(mode, data_dir, image_size, num_attrs):
    transform = []
    crop_size = 178
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return CelebA(data_dir, num_attrs, transform, mode)


def get_loader(data_dir, num_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA(data_dir, num_attrs, transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  drop_last=True,
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    selected_attrs = ['Male']
    attr_path = '../../data/celebA/list_attr_celeba.txt'
    celeba_image_dir = '../../data/celebA/img_align_celeba'
    celeba_crop_size = 178
    image_size = 256 #128
    batch_size = 16
    mode = 'train'
    num_workers = 1
    celeba_loader = get_loader(celeba_image_dir, attr_path, selected_attrs, celeba_crop_size, image_size, batch_size, 'CelebA', mode, num_workers)
    iters = iter(celeba_loader)
    data = iters.next()
    torchvision.utils.save_image(data[0].cpu(), '../../results/samples/stylegan/male.png', nrow=4, normalize=True) 