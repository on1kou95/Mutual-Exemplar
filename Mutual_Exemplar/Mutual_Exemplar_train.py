# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:02:24 2022

@author: loua2
"""

import argparse
import os
from datetime import datetime
from distutils.dir_util import copy_tree
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from data import image_loader
from loss import loss_sup, loss_diff
from metrics import dice_coef
from utils import get_logger, create_dir
from contrastive_loss import ConLoss, contrastive_loss_sup, ConLoss2, ContrastiveLoss_in
from model.projector import projectors, classifier
from model.pretrained_unet import preUnet, preUnet_a, preUnet_b, preUnet_l1, \
    preUnet_l2, preUnet_l3
from model.UNet import Unet,Unet2,Unet3
import numpy as np
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=(256, 256), help='training dataset size')
parser.add_argument('--dataset', type=str, default='kvasir', help='dataset name')
parser.add_argument('--split', type=float, default=1, help='training data ratio')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--ratio', type=float, default=0.1, help='labeled data ratio')
opt = parser.parse_args()
# pixel_wise_contrastive_loss_criter = ConLoss()
contrastive_loss_sup_criter = contrastive_loss_sup()


def adjust_lr(optimizer_1, optimizer_2, optimizer_3, init_lr, epoch, max_epoch):
    lr_ = init_lr * (1.0 - epoch / max_epoch) ** 0.9
    for param_group in optimizer_1.param_groups:
        param_group['lr'] = lr_
    for param_group in optimizer_2.param_groups:
        param_group['lr'] = lr_  # 可以选择不同的调整策略或参数
    for param_group in optimizer_3.param_groups:
        param_group['lr'] = lr_  # 可以选择不同的调整策略或参数


class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.best_dice_coeff_3 = False
        self.model_1 = preUnet_l1()
        self.model_2 = preUnet_l1()
        self.model_3 = preUnet_l1()
        # # 指定 model_1 中不希望更新的 resnet 层
        # non_trainable_layers_model_1 = [
        #     "resnet.conv1", "resnet.bn1", "resnet.relu", "resnet.maxpool",
        #     "resnet.layer1", "resnet.layer2", "resnet.layer3", "resnet.layer4",
        #     "resnet.avgpool", "resnet.fc"
        # ]
        #
        # # 指定 model_2 中不希望更新的 resnet 层
        # non_trainable_layers_model_2 = [ "resnet.layer2","resnet.layer3"
        # ]
        #
        # # 设置 model_1 中指定层为不可训练
        # for name, param in self.model_1.named_parameters():
        #     if any(layer in name for layer in non_trainable_layers_model_1):
        #         param.requires_grad = False
        #
        # # 设置 model_2 中指定层为不可训练
        # for name, param in self.model_2.named_parameters():
        #     if any(layer in name for layer in non_trainable_layers_model_2):
        #         param.requires_grad = False

        # self.optimizer_1 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_1.parameters()), lr=1e-2)
        # self.optimizer_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_2.parameters()), lr=1e-2)
        self.optimizer_1 = torch.optim.Adam(self.model_1.parameters(), lr=1e-2)  # 例如，1e-4
        self.optimizer_2 = torch.optim.Adam(self.model_2.parameters(), lr=1e-2)  # 例如，1e-4
        self.optimizer_3 = torch.optim.Adam(self.model_3.parameters(), lr=1e-2)  # 例如，1e-4
        self.projector_1 = projectors()
        self.projector_2 = projectors()
        self.projector_3 = projectors()
        self.classifier_1 = classifier()
        self.classifier_2 = classifier()
        self.classifier_3 = classifier()
        self.classifier_1a = classifier()
        self.classifier_2a = classifier()
        self.classifier_3a = classifier()
        self.classifier_1b = classifier()
        self.classifier_2b = classifier()
        self.classifier_3b = classifier()
        self.best_mIoU, self.best_dice_coeff = 0, 0
        self._init_configure()
        self._init_logger()

    def _init_configure(self):
        with open('configs/config.yml') as fp:
            self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        log_dir = 'logs/' + opt.dataset + '/train/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.image_save_path_1 = log_dir + "/saved_images_1"
        self.image_save_path_2 = log_dir + "/saved_images_2"
        self.image_save_path_3 = log_dir + "/saved_images_3"

        create_dir(self.image_save_path_1)
        create_dir(self.image_save_path_2)
        create_dir(self.image_save_path_3)

        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def set_trainable(self, model, layer_names, is_trainable):
        """设置特定层为可训练或不可训练状态"""
        for name, param in model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = is_trainable
    def print_trainable_status(self, model):
        print("Trainable layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)
        print("Non-trainable layers:")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print("\t", name)

    def run(self):
        print('Generator Learning Rate: {} Critic Learning Rate'.format(opt.lr))
        self.model_1.cuda()
        self.model_2.cuda()
        self.model_3.cuda()

        params = list(self.model_1.parameters()) + list(self.model_2.parameters()) + list(self.model_3.parameters())

        optimizer = torch.optim.Adam(params, lr=opt.lr)

        image_root = './data/' + opt.dataset + '/train/images/'
        gt_root = './data/' + opt.dataset + '/train/masks/'
        val_img_root = './data/' + opt.dataset + '/test/images/'
        val_gt_root = './data/' + opt.dataset + '/test/masks/'

        self.logger.info("Split Percentage : {} Labeled Data Ratio : {}".format(opt.split, opt.ratio))
        train_loader_1, train_loader_2, unlabeled_train_loader, unlabeled_train_loader_2, val_loader = image_loader(image_root, gt_root,
                                                                                          val_img_root, val_gt_root,
                                                                                          opt.batchsize, opt.trainsize,
                                                                                          opt.split, opt.ratio)
        self.logger.info(
            "train_loader_1 {} train_loader_2 {} unlabeled_train_loader {} unlabeled_train_loader_2 {}val_loader {}".format(len(train_loader_1),
                                                                                                 len(train_loader_2),
                                                                                                 len(unlabeled_train_loader),
                                                                                                 len(unlabeled_train_loader_2),
                                                                                                 len(val_loader)))
        print("Let's go!")

        best_dice_index = None
        for epoch in range(1, opt.epoch):
            # if epoch == 1:
            #     print(f"Model 1's layer trainable status at epoch {epoch}:")
            #     self.print_trainable_status(self.model_1)
            #     print(f"Model 2's layer trainable status at epoch {epoch}:")
            #     self.print_trainable_status(self.model_2)
            # # 在第10个epoch时，将 model_1 中的特定层设置为可训练
            # if epoch == 10:
            #     self.set_trainable(self.model_1, ["resnet.conv1", "resnet.bn1", "resnet.relu", "resnet.maxpool",
            # "resnet.layer1", "resnet.layer2", "resnet.layer3", "resnet.layer4",
            # "resnet.avgpool", "resnet.fc"], True)
            #     # 重置优化器以包含新的可训练参数
            #     self.optimizer_1 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_1.parameters()), lr=1e-1)
            #     print(f"Model 1's layer trainable status at epoch {epoch}:")
            #     self.print_trainable_status(self.model_1)
            # if epoch == 15:
            #     self.set_trainable(self.model_2, ["resnet.layer3"], True)
            #     # 重置优化器以包含新的可训练参数
            #     self.optimizer_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_2.parameters()), lr=1e-1)
            #     print(f"Model 2's layer trainable status at epoch {epoch}:")
            #     self.print_trainable_status(self.model_2)
            # if epoch == 20:
            #     self.set_trainable(self.model_2, ["resnet.layer2"], True)
            #     # 重置优化器以包含新的可训练参数
            #     self.optimizer_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_2.parameters()), lr=1e-1)
            #     print(f"Model 2's layer trainable status at epoch {epoch}:")
            #     self.print_trainable_status(self.model_2)

            running_loss = 0.0
            running_dice_val_1 = 0.0
            running_dice_val_2 = 0.0
            running_dice_val_3 = 0.0
            if epoch > 1:
                dice_scores = [mdice_coeff_1, mdice_coeff_2, mdice_coeff_3]
                best_dice_index = dice_scores.index(max(dice_scores))

            for i, data in enumerate(zip(train_loader_1, train_loader_2, unlabeled_train_loader, unlabeled_train_loader_2)):

                inputs_S1, labels_S1 = data[0][0], data[0][1]
                inputs_S2, labels_S2 = data[1][0], data[1][1]
                inputs_U, labels_U = data[2][0], data[2][1]
                inputs_U2, labels_U2 = data[3][0], data[3][1]
                inputs_S1 = inputs_S1.cuda()
                labels_S1 = labels_S1.cuda()
                inputs_S2 = inputs_S2.cuda()
                labels_S2 = labels_S2.cuda()
                inputs_U = inputs_U.cuda()
                inputs_U2 = inputs_U2.cuda()
                # optimizer.zero_grad()
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                self.optimizer_3.zero_grad()
                # Train Model 1
                prediction_1,dirc_8_1,dirc_1_1 = self.model_1(inputs_S1)
                prediction_1_1 = torch.sigmoid(prediction_1)

                # prediction_1a = self.model_1(inputs_S2)
                # prediction_1_1a = torch.sigmoid(prediction_1a)

                feat_1,_,_ = self.model_1(inputs_U)
                u_prediction_1 = torch.sigmoid(feat_1)

                # Train Model 2
                prediction_2,dirc_8_2,dirc_1_2 = self.model_2(inputs_S1)
                prediction_2_2 = torch.sigmoid(prediction_2)

                # prediction_2a = self.model_2(inputs_S1)
                # prediction_2_2a = torch.sigmoid(prediction_2a)

                feat_2,_,_ = self.model_2(inputs_U)
                u_prediction_2 = torch.sigmoid(feat_2)
                ####################
                # Train Model 3
                prediction_3,dirc_8_3,dirc_1_3 = self.model_3(inputs_S1)
                prediction_3_3 = torch.sigmoid(prediction_3)

                # prediction_3a = self.model_3(inputs_S1)
                # prediction_3_2a = torch.sigmoid(prediction_3a)

                feat_3,_,_ = self.model_3(inputs_U)
                u_prediction_3 = torch.sigmoid(feat_3)
                ####################
                self.projector_1.cuda()
                self.projector_2.cuda()
                self.projector_3.cuda()
                self.classifier_1.cuda()
                self.classifier_2.cuda()
                self.classifier_3.cuda()
                self.classifier_1a.cuda()
                self.classifier_2a.cuda()
                self.classifier_3a.cuda()
                self.classifier_1b.cuda()
                self.classifier_2b.cuda()
                self.classifier_3b.cuda()
                # feat_q = self.projector_1(feat_1)
                # feat_k = self.projector_2(feat_2)
                # feat_e = self.projector_3(feat_3)
                # feat_l_q = self.classifier_1(prediction_1)
                # feat_l_k = self.classifier_2(prediction_2)
                # feat_l_e = self.classifier_3(prediction_3)
                feat_l_qa = self.classifier_1a(dirc_8_1)
                feat_l_ka = self.classifier_2a(dirc_8_2)
                feat_l_ea = self.classifier_3a(dirc_8_3)
                # feat_l_qb = self.classifier_1b(dirc_1_1)
                # feat_l_kb = self.classifier_2b(dirc_1_2)
                # feat_l_eb = self.classifier_3b(dirc_1_3)
                Loss_sup_1 = loss_sup(prediction_1_1, labels_S1)
                Loss_sup_2 = loss_sup(prediction_2_2, labels_S1)
                Loss_sup_3 = loss_sup(prediction_3_3, labels_S1)
                # Loss_diff_1 = loss_diff(u_prediction_1, u_prediction_2, opt.batchsize)
                # Loss_diff_2 = loss_diff(u_prediction_1, u_prediction_3, opt.batchsize)
                # Loss_diff_3 = loss_diff(u_prediction_2, u_prediction_3, opt.batchsize)
                if best_dice_index is not None:
                    # 根据DICE最高的模型，计算Loss_diff
                    if best_dice_index == 0:  # 第一个模型DICE最高
                        Loss_diff_1 = loss_diff(u_prediction_2, u_prediction_1, opt.batchsize)
                        Loss_diff_2 = loss_diff(u_prediction_3, u_prediction_1, opt.batchsize)
                    elif best_dice_index == 1:  # 第二个模型DICE最高
                        Loss_diff_1 = loss_diff(u_prediction_1, u_prediction_2, opt.batchsize)
                        Loss_diff_2 = loss_diff(u_prediction_3, u_prediction_2, opt.batchsize)
                    else:  # 第三个模型DICE最高
                        Loss_diff_1 = loss_diff(u_prediction_1, u_prediction_3, opt.batchsize)
                        Loss_diff_2 = loss_diff(u_prediction_2, u_prediction_3, opt.batchsize)
                else:
                    # 第一个epoch，比较所有模型
                    Loss_diff_1 = loss_diff(u_prediction_1, u_prediction_2, opt.batchsize)
                    Loss_diff_2 = loss_diff(u_prediction_1, u_prediction_3, opt.batchsize)
                    Loss_diff_3 = loss_diff(u_prediction_2, u_prediction_3, opt.batchsize)
                # Loss_contrast = pixel_wise_contrastive_loss_criter(feat_q, feat_k)
                # Loss_contrast_a = pixel_wise_contrastive_loss_criter(feat_q, feat_e)
                # Loss_contrast_b = pixel_wise_contrastive_loss_criter(feat_e, feat_k)
                # Loss_contrast_2c = contrastive_loss_sup_criter(feat_l_q, feat_l_k)
                # Loss_contrast_2b = contrastive_loss_sup_criter(feat_l_q, feat_l_e)
                # Loss_contrast_2a = contrastive_loss_sup_criter(feat_l_k, feat_l_e)
                # encoder_loss_12 = contrastive_loss_sup_criter(feat_l_q, feat_l_ka)
                # encoder_loss_13 = contrastive_loss_sup_criter(feat_l_q, feat_l_ea)
                # encoder_loss_21 = contrastive_loss_sup_criter(feat_l_k, feat_l_qa)
                # encoder_loss_23 = contrastive_loss_sup_criter(feat_l_k, feat_l_ea)
                # encoder_loss_31 = contrastive_loss_sup_criter(feat_l_e, feat_l_qa)
                # encoder_loss_32 = contrastive_loss_sup_criter(feat_l_e, feat_l_ka)
                encoder_loss_12a = contrastive_loss_sup_criter(feat_l_qa, feat_l_ka)
                encoder_loss_13a = contrastive_loss_sup_criter(feat_l_ka, feat_l_ea)
                encoder_loss_23a = contrastive_loss_sup_criter(feat_l_ea, feat_l_qa)
                # encoder_loss_12b = contrastive_loss_sup_criter(feat_l_qb, feat_l_kb)
                # encoder_loss_13b = contrastive_loss_sup_criter(feat_l_qb, feat_l_eb)
                # encoder_loss_23b = contrastive_loss_sup_criter(feat_l_kb, feat_l_eb)

                seg_loss = (
                            0.1 * Loss_sup_1 + 0.1 * Loss_sup_2 + 0.1 * Loss_sup_3
                            + 0.05 * Loss_diff_1 + 0.05 * Loss_diff_2 + 0.05 * Loss_diff_3
                            # + 0.1 * Loss_contrast + 0.1 * Loss_contrast_a + 0.1 * Loss_contrast_b
                            # + 0.1 * Loss_contrast_2c + 0.1 * Loss_contrast_2b + 0.1 * Loss_contrast_2a
                            +0.05*(encoder_loss_12a+encoder_loss_13a+encoder_loss_23a)
                            # +0.05*(encoder_loss_12b+encoder_loss_13b+encoder_loss_23b)
                            # +0.05*(encoder_loss_12+encoder_loss_13+encoder_loss_21+encoder_loss_23+encoder_loss_31+encoder_loss_32)
                            )

                seg_loss.backward(retain_graph=True)
                running_loss = running_loss + seg_loss.item()
                # optimizer.step()
                self.optimizer_1.step()  # 更新model_1的参数
                self.optimizer_2.step()  # 更新model_2的参数
                self.optimizer_3.step()  # 更新model_3的参数

                # adjust_lr(optimizer, opt.lr, epoch, opt.epoch)
                adjust_lr(self.optimizer_1, self.optimizer_2, self.optimizer_3, opt.lr, epoch, opt.epoch)

            epoch_loss = running_loss / (len(train_loader_1) + len(train_loader_2))
            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, opt.epoch, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    images = Variable(images)
                    gts = Variable(gts)
                    images = images.cuda()
                    gts = gts.cuda()

                    prediction_1,_,_ = self.model_1(images)
                    prediction_1 = torch.sigmoid(prediction_1)

                    prediction_2,_,_ = self.model_2(images)
                    prediction_2 = torch.sigmoid(prediction_2)

                    prediction_3,_,_ = self.model_3(images)
                    prediction_3 = torch.sigmoid(prediction_3)

                dice_coe_1 = dice_coef(prediction_1, gts)
                running_dice_val_1 = running_dice_val_1 + dice_coe_1
                dice_coe_2 = dice_coef(prediction_2, gts)
                running_dice_val_2 = running_dice_val_2 + dice_coe_2
                dice_coe_3 = dice_coef(prediction_3, gts)
                running_dice_val_3 = running_dice_val_3 + dice_coe_3

            epoch_dice_val_1 = running_dice_val_1 / len(val_loader)

            self.logger.info('Validation dice coeff model 1: {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Validation_1/DSC', epoch_dice_val_1, epoch)

            epoch_dice_val_2 = running_dice_val_2 / len(val_loader)

            self.logger.info('Validation dice coeff model 2: {}'.format(epoch_dice_val_2))
            self.writer.add_scalar('Validation_1/DSC', epoch_dice_val_2, epoch)

            epoch_dice_val_3 = running_dice_val_3 / len(val_loader)

            self.logger.info('Validation dice coeff model 3: {}'.format(epoch_dice_val_3))
            self.writer.add_scalar('Validation_1/DSC', epoch_dice_val_3, epoch)

            mdice_coeff_1 = epoch_dice_val_1
            mdice_coeff_2 = epoch_dice_val_2
            mdice_coeff_3 = epoch_dice_val_3

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True

                if not os.path.exists(self.image_save_path_1):
                    os.makedirs(self.image_save_path_1)

                copy_tree(self.image_save_path_1, self.save_path + '/best_model_predictions_1')
                self.patience = 0
            else:
                self.save_best_model_1 = False
                self.patience = self.patience + 1

            if self.best_dice_coeff_2 < mdice_coeff_2:
                self.best_dice_coeff_2 = mdice_coeff_2
                self.save_best_model_2 = True

                if not os.path.exists(self.image_save_path_2):
                    os.makedirs(self.image_save_path_2)

                copy_tree(self.image_save_path_2, self.save_path + '/best_model_predictions_2')
                self.patience = 0
            else:
                self.save_best_model_2 = False
                self.patience = self.patience + 1

            if self.best_dice_coeff_3 < mdice_coeff_3:
                self.best_dice_coeff_3 = mdice_coeff_3
                self.save_best_model_3 = True

                if not os.path.exists(self.image_save_path_3):
                    os.makedirs(self.image_save_path_3)

                copy_tree(self.image_save_path_3, self.save_path + '/best_model_predictions_3')
                self.patience = 0
            else:
                self.save_best_model_3 = False
                self.patience = self.patience + 1

            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                torch.save(self.model_1.state_dict(), Checkpoints_Path + '/Model_1.pth')
            if self.save_best_model_2:
                torch.save(self.model_2.state_dict(), Checkpoints_Path + '/Model_2.pth')
            if self.save_best_model_3:
                torch.save(self.model_3.state_dict(), Checkpoints_Path + '/Model_3.pth')

            self.logger.info(
                'current best dice coef model 1 {}, model 2 {}, model 3 {}'.format(self.best_dice_coeff_1,
                                                                                   self.best_dice_coeff_2,
                                                                                   self.best_dice_coeff_3))
            self.logger.info('current patience :{}'.format(self.patience))


if __name__ == '__main__':
    train_network = Network()
    train_network.run()