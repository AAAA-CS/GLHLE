# -*- coding: UTF-8 -*-

import math

from torch import nn
import torch
from typing import Union
from torch._six import string_classes
import torch.nn.functional as F
import collections.abc as container_abcs


def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [convert_to_cuda(d) for d in data]
    else:
        return data



def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


def convert_to_ddp(modules: Union[list, nn.Module], **kwargs):
    if isinstance(modules, list):
        modules = [x.cuda() for x in modules]
    else:
        modules = modules.cuda()
    modules = torch.nn.DataParallel(modules)

    return modules
