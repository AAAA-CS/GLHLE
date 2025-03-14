import torch
import numpy as np
import random

import sys
import yaml
import os.path as osp
from datetime import datetime

from torch.backends import cudnn

from models.basic_template import TrainTask
from models import model_dict

if __name__ == '__main__':
    config_path = 'models/glhle/configs/GLHLE.yml'
    with open(config_path) as f:
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())
    MODEL = model_dict[configs['model_name']]
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args('')
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)

    if opt.run_name is None:
        opt.run_name = osp.basename(config_path)[:-4]
    opt.run_name = '{}-{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), opt.run_name)
    for k in configs:
        setattr(opt, k, configs[k])

    seed = opt.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    model = MODEL(opt)
    model.fit()
