from __future__ import print_function

import os
import os.path as osp
import argparse
import warnings

import torch
import numpy as np
import tqdm


from utils.ops import convert_to_cuda
from utils.loggerx import LoggerX
from HSI_dataset import data_processing
from cg import tosegment

class TrainTask(object):

    def __init__(self, opt):
        self.opt = opt
        self.cur_epoch = 1
        self.logger = LoggerX(save_root=osp.join('./ckpt', opt.run_name),
                              enable_wandb=opt.wandb,
                              config=opt,
                              project=opt.project_name,
                              entity=opt.entity,
                              name=opt.run_name)
        self.feature_extractor = None
        self.set_loader()
        self.set_model()

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')
        parser.add_argument('--wandb', help='wandb', action='store_true')
        parser.add_argument('--project_name', help='wandb project_name', type=str, default='noisylabel')
        parser.add_argument('--entity', help='wandb project_name', type=str, default='maple_1202')
        parser.add_argument('--run_name', type=str, help='each run name')
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

        # optimization
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
        parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

        # learning rate
        parser.add_argument('--learning_rate', type=float, default=0.05, help='base learning rate')
        parser.add_argument('--learning_eta_min', type=float, default=0.01, help='base learning rate')
        parser.add_argument('--lr_decay_gamma', type=float, default=0.1)

        parser.add_argument('--step_lr', help='step_lr', action='store_true')

        parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup epochs')
        parser.add_argument('--num_devices', type=int, default=-1, help='warmup epochs')

        # dataset
        parser.add_argument('--dataset', type=str, default='UP', help='dataset')
        parser.add_argument('--num_cluster', type=int, help='num_cluster')
        parser.add_argument('--percent', type=int, default=24)
        parser.add_argument('--train_size', type=int, default=76)
        parser.add_argument('--patch_length', type=int, default=11)
        parser.add_argument('--noise_type', type=str, default='sym')
        parser.add_argument('--input_channel', type=int, default=103)
        parser.add_argument('--num_classes', type=int, default=9)

        parser.add_argument('--n_segments', type=int, default=100)

        return parser

    @staticmethod
    def build_options():
        pass

    def set_loader(self):
        opt = self.opt
        all_iter, BAND, CLASSES_NUM, data_shape = data_processing(Dataset=opt.dataset,
                                                                   batch_size=opt.batch_size,
                                                                   PATCH_LENGTH=opt.patch_length,
                                                                  flag='testAll')

        train_loader, valida_loader, test_loader, train_indices = data_processing(Dataset=opt.dataset,
                                                                   batch_size=opt.batch_size,
                                                                   PATCH_LENGTH=opt.patch_length,
                                                                   Train_size=opt.train_size,
                                                                   flag='train', noise_type=opt.noise_type,
                                                                   noise_ratio=opt.percent)
        segment = tosegment(Dataset= opt.dataset,n_segments=opt.n_segments)

        print(f'set train dataloader with {len(train_loader)} iterations...')
        print(f'set valida dataloader with {len(valida_loader)} iterations...')
        print(f'set test dataloader with {len(test_loader)} iterations...')
        labels = np.array(train_loader.dataset.labels)
        self.segment = segment
        self.all_iter = all_iter
        self.data_shape = data_shape
        self.train_indices = train_indices
        self.test_loader = test_loader
        self.valida_loader = valida_loader
        self.train_loader = train_loader
        self.iter_per_epoch = len(train_loader)
        self.num_classes = len(np.unique(labels[labels >= 0]))
        self.num_samples = len(labels)
        self.gt_labels = torch.from_numpy(labels).cuda()
        self.num_cluster = self.num_classes if opt.num_cluster is None else opt.num_cluster
        opt.num_cluster = self.num_cluster
        self.psedo_labels = torch.zeros((self.num_samples,)).long().cuda()
        print('load {} images...'.format(self.num_samples))

    def fit(self):
        opt = self.opt

        n_iter = 1
        self.cur_epoch = 1
        # training routine
        self.progress_bar = tqdm.tqdm(total=self.iter_per_epoch * opt.epochs, initial=n_iter)

        self.psedo_labeling(n_iter)
        i = 0
        while True:
            for inputs in self.train_loader:
                image1, image2, image3, labels, index = convert_to_cuda(inputs)
                indices = index
                inputs = [image1, image2, image3, labels]
                self.adjust_learning_rate(n_iter)
                self.train(inputs, indices, n_iter)
                self.progress_bar.refresh()
                self.progress_bar.update()
                n_iter += 1
                i = i + 1
            i = 0
            cur_epoch = self.cur_epoch
            self.logger.msg([cur_epoch, ], n_iter)

            self.psedo_labeling(n_iter)
            self.test(n_iter)

            self.cur_epoch += 1
            if self.cur_epoch > opt.epochs:
                break

    def set_model(opt):
        pass


    def test(self):
        pass

    def train(self, inputs, indices, n_iter):
        pass

    def cosine_annealing_LR(self, n_iter):
        opt = self.opt

        epoch = n_iter / self.iter_per_epoch
        max_lr = opt.learning_rate
        min_lr = max_lr * opt.learning_eta_min
        # warmup
        if epoch < opt.warmup_epochs:
            lr = opt.learning_rate * epoch / opt.warmup_epochs
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((epoch - opt.warmup_epochs) * np.pi / opt.epochs))
        return lr


    def step_LR(self, n_iter):
        opt = self.opt
        lr = opt.learning_rate * (opt.lr_decay_gamma ** (
            max(0, (n_iter - opt.warmup_epochs * self.iter_per_epoch) // self.iter_per_epoch)))
        return lr


    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        if opt.step_lr:
            lr = self.step_LR(n_iter)
        else:
            lr = self.cosine_annealing_LR(n_iter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.logger.msg([lr, ], n_iter)


    def psedo_labeling(self, n_iter):
        pass

    def collect_params(self, *models, exclude_bias_and_bn=True):
        param_list = []
        for model in models:
            for name, param in model.named_parameters():
                param_dict = {
                    'name': name,
                    'params': param,
                }
                if exclude_bias_and_bn and any(s in name for s in ['bn', 'bias']):
                    param_dict.update({'weight_decay': 0., 'lars_exclude': True})
                param_list.append(param_dict)
        return param_list
