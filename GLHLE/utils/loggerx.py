# -*- coding: UTF-8 -*-

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Union
import os.path as osp
import os
import time
from torchvision.utils import save_image
import math
import inspect
from torch._six import string_classes
import collections.abc as container_abcs
import warnings
from utils.ops import load_network


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):

    def __init__(self, save_root, enable_wandb=False, **kwargs):
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.enable_wandb = enable_wandb
        if enable_wandb:
            import wandb
            wandb.login(key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            try:
                wandb.init(dir=save_root, settings=wandb.Settings(_disable_stats=True, _disable_meta=True), **kwargs)
            except wandb.errors.CommError as e:
                print("wandb initialization failed: ", e)

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def append(self, module, name=None):
        self._modules.append(module)
        if name is None:
            name = get_varname(module)
        self._module_names.append(name)

    def checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        output_dict = {}
        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)
            output_dict[var_name] = var

        if self.enable_wandb:
            import wandb
            wandb.log(output_dict, step)

        print(output_str)

    def msg_str(self, output_str):
        print(str(output_str))

    def save_image(self, img, n_iter, sample_type):
        if isinstance(img, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img.cpu())
        fname = f'{str(n_iter).zfill(7)}-{sample_type}.png'
        img.save(osp.join(self.images_save_dir, fname))
        self.msg_str(f'save {fname} to {self.images_save_dir}')
