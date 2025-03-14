import matplotlib.pyplot as plt
import torch
import argparse
import copy
import torch.nn.functional as F
import tqdm
import numpy as np


from models.basic_template import TrainTask

from .glhle_wrapper import SimCLRWrapper
from utils.ops import convert_to_cuda
from models import model_dict
from sklearn import metrics

from cg import one_iter_true


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    list_raw_sum_safe = np.where(list_raw_sum == 0, 1e-10, list_raw_sum)
    each_acc = np.nan_to_num(np.true_divide(list_diag, list_raw_sum_safe))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


@model_dict.register('glhle')
class GLHLE(TrainTask):

    def set_model(self):
        opt = self.opt

        kwargs = {'input_channel': opt.input_channel, 'num_classes': opt.num_classes, 'fea_dim': opt.feat_dim,
                  'T': opt.temp,
                  'num_cluster': self.num_cluster, 'mixup_alpha': opt.mixup_alpha, 'num_samples': self.num_samples,
                  'scale1': opt.scale1, 'scale2': opt.scale2}
        if opt.arch == 'simclr':
            glhle = SimCLRWrapper(**kwargs)
        else:
            raise NotImplemented
        glhle.register_buffer('pseudo_labels', self.gt_labels.cpu())

        params = list(glhle.parameters())

        # optimizer = torch.optim.SGD(params=params, lr=opt.learning_rate, momentum=opt.momentum,
        #                             weight_decay=opt.weight_decay)
        optimizer = torch.optim.Adam(params=params, lr=opt.learning_rate)

        glhle = glhle.cuda()
        self.logger.modules = [glhle, optimizer]
        self.glhle = glhle
        self.optimizer = optimizer

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')

        parser.add_argument('--temp', type=float, help='temp for contrastive loss')
        parser.add_argument('--scale1', type=float)
        parser.add_argument('--scale2', type=float)
        parser.add_argument('--contrastive_loss_weight', type=float, default=1, help='contrastive_loss_weight')
        parser.add_argument('--align_loss_weight', type=float, default=1, help='align_loss_weight')
        parser.add_argument('--mixup_alpha', type=float, default=1.0)
        parser.add_argument('--k_value', type=float, default=20)
        parser.add_argument('--arch', type=str, default='simclr', help='simclr')

        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        is_warmup = not (self.cur_epoch >= opt.warmup_epochs)
        self.glhle.warmup = is_warmup

        images = inputs
        self.glhle.train()

        im_w = images[0]
        im_q = images[1]
        im_k = images[2]

        # compute loss
        contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss = self.glhle(im_w, im_q, im_k, indices)

        self.optimizer.zero_grad()
        loss = contrastive_loss.mean() * opt.contrastive_loss_weight + \
               cls_loss1.mean() + \
               ent_loss.mean() + \
               ne_loss.mean() + \
               opt.align_loss_weight * (align_loss.mean() + cls_loss2.mean())

        loss.backward()
        self.optimizer.step()
        self.logger.msg([contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss], n_iter)


    @torch.no_grad()
    def extract_features(self, model, loader):
        opt = self.opt
        features = torch.zeros(len(loader.dataset), opt.feat_dim).cuda()
        all_labels = torch.zeros(len(loader.dataset)).cuda()
        cluster_labels = torch.zeros(len(loader.dataset), self.num_cluster).cuda()

        model.eval()
        encoder = model.model_scratch

        local_features = []
        local_labels = []
        local_cluster_labels = []
        for inputs in loader:
            images, labels, index = convert_to_cuda(inputs)
            local_labels.append(labels)
            x_projector, x_classifier = encoder(images)
            local_cluster_labels.append(F.softmax(x_classifier, dim=1))
            local_features.append(F.normalize(x_projector, dim=1))
        local_features = torch.cat(local_features, dim=0)
        local_labels = torch.cat(local_labels, dim=0)
        local_cluster_labels = torch.cat(local_cluster_labels, dim=0)

        indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()

        features.index_add_(0, indices, local_features)
        all_labels.index_add_(0, indices, local_labels.float())
        cluster_labels.index_add_(0, indices, local_cluster_labels.float())

        features = F.normalize(features, dim=1)
        labels = all_labels.long().cuda()
        return features, cluster_labels, labels

    def hist(self, assignments, is_clean, labels, n_iter, sample_type='context_assignments_hist'):
        fig, ax = plt.subplots()
        ax.hist(assignments[is_clean, labels[is_clean]].cpu().numpy(), label='clean', bins=100, alpha=0.5)
        ax.hist(assignments[~is_clean, labels[~is_clean]].cpu().numpy(), label='noisy', bins=100, alpha=0.5)
        ax.legend()
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        self.logger.save_image(img, n_iter, sample_type=sample_type)
        plt.close()

    @torch.no_grad()
    def psedo_labeling(self, n_iter):
        opt = self.opt

        self.logger.msg_str('Generating the psedo-labels')
        print("glhle-----------------psedo_labeling")
        labels = self.gt_labels.long()

        confidence, context_assignments, features, cluster_labels, = self.correct_labels(self.glhle, labels)
        self.glhle.confidences.copy_(confidence.float())
        epoch = int(n_iter / self.iter_per_epoch)
        if epoch == opt.epochs:
            self.evaluate(self.glhle, features, confidence, cluster_labels, labels, context_assignments, n_iter)

    def evaluate(self, model, features, confidence, cluster_labels, labels, context_assignments, n_iter):
        opt = self.opt
        model.eval()
        clean_labels = torch.Tensor(np.array(self.train_loader.dataset.labels_gt))

        is_clean = clean_labels.cpu().numpy() == labels.cpu().numpy()
        self.hist(context_assignments, is_clean, labels, n_iter)
        clean_labels = clean_labels.cuda()
        train_acc = (torch.argmax(cluster_labels, dim=1) == clean_labels).float().mean()
        test_features, test_cluster_labels, test_labels = self.extract_features(model, self.test_loader)
        test_acc = (test_labels == torch.argmax(test_cluster_labels, dim=1)).float().mean()

        test_cluster_labels_tensor = test_cluster_labels.clone().detach()
        OA = metrics.accuracy_score(torch.argmax(test_cluster_labels_tensor.cpu(), dim=1),
                                    test_labels.cpu().detach().numpy())

        confusion_matrix = metrics.confusion_matrix(torch.argmax(test_cluster_labels_tensor.cpu(), dim=1),
                                                    test_labels.cpu().detach().numpy())
        Each_acc, AA = aa_and_each_accuracy(confusion_matrix)
        Kappa = metrics.cohen_kappa_score(torch.argmax(test_cluster_labels_tensor.cpu(), dim=1),
                                          test_labels.cpu().detach().numpy())

        noise_accuracy = ((confidence > 0.5) == (clean_labels == labels)).float().mean()

        from sklearn.metrics import roc_auc_score
        context_noise_auc = roc_auc_score(is_clean, confidence.cpu().numpy())
        self.logger.msg([noise_accuracy, context_noise_auc, train_acc, test_acc, OA, AA, Kappa], n_iter)
        print(Each_acc)

        # Draw full classification map

        # def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
        #     fig, ax = plt.subplots()
        #     numlabel = np.array(label)
        #     v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
        #     ax.set_axis_off()
        #     ax.xaxis.set_visible(False)
        #     ax.yaxis.set_visible(False)
        #     fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        #     foo_fig = plt.gcf()  # 'get current figure'
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #     foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
        #     plt.close()
        #     sio.savemat(name + '.mat', mdict={'Result': label})
        #
        # testAll_features, testAll_cluster_labels, testAll_labels = self.extract_features(model,self.all_iter)
        # _, testAll_pred = torch.max(testAll_cluster_labels, dim=1)
        # testAll_pred_cpu = testAll_pred.cpu().numpy()
        #
        # All_labels = testAll_pred_cpu.reshape((self.data_shape[0], self.data_shape[1]))
        # Draw_Classification_Map(All_labels + 1, 'HC140+60sym')



    def correct_labels(self, model, labels):
        opt = self.opt
        features, cluster_labels, _ = self.extract_features(model, self.valida_loader)

        labels_one_hot = F.one_hot(labels, num_classes=self.num_cluster)
        Y_graph = one_iter_true(self.segment, self.train_indices, features,
                                           labels_one_hot,k=opt.k_value, classes=self.num_cluster)

        Y_graph = torch.Tensor(Y_graph).cuda()

        confidence, context_assignments, centers = self.noise_detect(Y_graph, labels, features)

        model.prototypes.copy_(centers)
        model.context_assignments.copy_(context_assignments.float())

        return confidence, context_assignments, features, cluster_labels

    def noise_detect(self, cluster_labels, labels, features):
        opt = self.opt

        centers = F.normalize(cluster_labels.T.mm(features), dim=1)
        context_assignments_logits = features.mm(centers.T)
        context_assignments = F.softmax(context_assignments_logits, dim=1)
        confidence = context_assignments[torch.arange(labels.size(0)), labels]
        return confidence, context_assignments, centers


    def test(self, n_iter):
        pass
