
import numpy as np
import torch
from models.gans.gan_ops import get_gan
from metrics.cas import Classifiers


def load_gan(opt, epoch=-1):
    # model
    gan, _ = get_gan(opt)
    gan.load_networks(epoch)
    return gan

def mix_sample(gan, sample_size, num_class):
    batch_size = 128
    if sample_size < batch_size:
        batch_size = sample_size
    niters = 1 + (sample_size - 1) // batch_size
    nsamples = niters * batch_size
    data = []
    labels = []
    for k in range(num_class):
        for _ in range(niters):
            samples = gan.sample(batch_size, target_class=k)
            if not samples.min().isnan():
                data.append(samples.data.cpu().numpy())
                labels.append(np.ones(batch_size) * k)
            del samples
    # labels = np.outer(np.arange(num_class), np.ones(nsamples)).flatten()
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels

def mix_sample_class(gan, sample_size, target_class):
    batch_size = 128
    if sample_size < batch_size:
        batch_size = sample_size
    niters = 1 + (sample_size - 1) // batch_size
    nsamples = niters * batch_size
    labels = (target_class * np.ones(nsamples)).flatten()
    data = []
    for _ in range(niters):
        samples = gan.sample(batch_size, target_class=target_class)
        samples = samples.data.cpu().numpy()
        data.append(samples)
        del samples
    data = np.concatenate(data, axis=0)
    return data, labels

def mix_sample_class_ugan(gan, netC, sample_size, target_class):
    batch_size = 128
    if sample_size < batch_size:
        batch_size = sample_size
    niters = 1 + (sample_size - 1) // batch_size
    nsamples = niters * batch_size
    data = []
    counter = 0
    while counter < nsamples:
        samples = gan.sample(batch_size)
        with torch.no_grad():
            logits = netC.net(samples)
        preds = np.argmax(logits.data.cpu().numpy(), axis=1) 
        samples = samples.data.cpu().numpy()
        samples = samples[preds==target_class]
        data.append(samples)
        counter += samples.shape[0]
        del samples
    data = np.concatenate(data, axis=0)
    labels = (target_class * np.ones(counter)).flatten()
    return data, labels
