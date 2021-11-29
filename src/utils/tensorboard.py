import numpy as np
import torch

from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
import torch.nn.functional as F
from PIL import Image

from tqdm import tqdm

def log_losses_tensorboard(writer, loss_dict, i_iter, type_):
    for loss_name, loss_value in loss_dict.items():
        writer.add_scalar(f'{type_}/{loss_name}', to_numpy(loss_value), i_iter)

def draw_in_tensorboard(writer, images, i_iter, pred_main, label, num_classes, type_):
    grid_image = make_grid(images[:4].clone().cpu().data, 4, normalize=True)
    writer.add_image(f'Image/Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(
                   torch.from_numpy(np.array(colorize_mask(np.asarray(np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)),
                   3,
                   normalize=False, range=(0, 255)

                           )
    writer.add_image(f'Prediction/Prediction - {type_}', grid_image, i_iter)

    grid_image = make_grid(
                   torch.from_numpy(np.array(colorize_mask(np.asarray(label.data[0], dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)),
                   3,
                   normalize=False, range=(0, 255)

                           )
    writer.add_image(f'GroundTruth/GroundTruth - {type_}', grid_image, i_iter)

def print_loss(loss_dict, i_iter):
    list_strings = []
    for loss_name, loss_value in loss_dict.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()
