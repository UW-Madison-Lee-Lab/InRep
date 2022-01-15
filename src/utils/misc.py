import numpy as np
import random
import math
import os
import sys
import shutil
import warnings
from os.path import dirname, abspath, exists, join
from scipy import linalg
from datetime import datetime
from tqdm import tqdm
from itertools import chain
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image



class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v
    return res_dict


def setattr_cls_from_kwargs(cls, kwargs):
    kwargs = flatten_dict(kwargs)
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def dict2clsattr(train_configs, model_configs):
    cfgs = {}
    for k, v in chain(train_configs.items(), model_configs.items()):
        cfgs[k] = v

    class cfg_container: pass
    cfg_container.train_configs = train_configs
    cfg_container.model_configs = model_configs
    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container