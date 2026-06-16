"""训练模型"""
import os
import time
import shutil
import logging
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer

from criterions.plcc_loss import PLCCLoss, L2RankLoss
from utils.my_meters import AverageMeter
from .test import test_rank, test_regress


def train(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module,
          criterion: nn.Module, optimizer: Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler, args, flag, k_fold):
    """
    :param train_loader: 训练数据集
    :param test_loader: 验证数据集
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 学习率调整
    :param args: 超参
    :return:
    """

    writer = SummaryWriter(args.logdir)

    best_val_acc1 = 0
    for epoch in range(args.epochs):

        lr_current = scheduler.get_last_lr()

        if flag == 'rank':
            train_loss, train_acc1, train_time = train_rank_epoch(train_loader, model, criterion, optimizer, epoch, lr_current[0], args)
            val_acc1, val_loss, val_time, _ = test_rank(test_loader, model, criterion, args)
            scheduler.step()

            # 保存当前及最好的acc@1的checkpoint
            if val_acc1 > best_val_acc1:
                best_val_acc1 = max(val_acc1, best_val_acc1)
                ckpts_dir = os.path.join(args.ckpts, args.database, 'rank_' + args.arch + '_fold_' + str(k_fold) + '_margin' + str(args.margin) + '.pth')
                torch.save(model.state_dict(), ckpts_dir)

            writer.add_scalar('learning rate', lr_current[0], epoch)
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Accuracy', train_acc1, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', val_acc1, epoch)
            writer.flush()

        if flag == 'regress':
            train_loss, train_acc, train_time = train_regress_epoch(train_loader, model, criterion, optimizer, epoch, lr_current[0], args)
            val_acc, val_loss, val_time, _ = test_regress(test_loader, model, criterion, args, k_fold, save_flag=False)
            scheduler.step()

            # 保存当前及最好的plcc的checkpoint
            if val_acc[0] > best_val_acc1:
                best_val_acc1 = max(val_acc[0], best_val_acc1)
                ckpts_dir = os.path.join(args.ckpts, args.database, 'fineT_' + args.arch + '_fold_' + str(k_fold) + '_margin' + str(args.margin) + '.pth')
                torch.save(model.state_dict(), ckpts_dir)

                logging.info("Update the best Test results on epoch{:d}:  PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(epoch, val_acc[0], val_acc[1], val_acc[2], val_acc[3]))
                logging.info("Saving model's parameters!!!")
                # print("Saving model's parameters!!!")

            writer.add_scalar('learning rate', lr_current[0], epoch)
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/PLCC', train_acc[0], epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/PLCC', val_acc[0], epoch)
            writer.flush()

    writer.close()

    return


def train_rank_epoch(train_loader, model, criterion, optimizer, epoch, lr, args):

    model.train()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('RankAcc@1', ':6.2f')

    start_time = time.time()
    for i, (image_pair, mos_pair, wb_mos_pair1, wb_mos_pair2, path_pair) in tqdm(enumerate(train_loader), total= len(train_loader), smoothing=0.9):
        if args.cuda:
            image_pair = image_pair.cuda(args.gpus)  # non_blocking=True
            wb_mos_pair1 = wb_mos_pair1.cuda(args.gpus)
            wb_mos_pair2 = wb_mos_pair2.cuda(args.gpus)

        img_size = image_pair.shape
        imgs = image_pair.view(-1, img_size[2], img_size[3], img_size[4])

        mos_pair1 = wb_mos_pair1[:, np.newaxis]
        mos_pair2 = wb_mos_pair2[:, np.newaxis]

        mos_output = model(imgs)

        # mos_output = torch.flatten(mos_output, 1)
        # average the projection features
        mos_output = mos_output.view(img_size[0], img_size[1], 1)
        mos_output = torch.mean(mos_output, dim=1)

        # compute loss
        loss = criterion(mos_output, mos_pair1)
        # loss_rank = criterion(mos_output, mos_pair1)
        # loss_wb = torch.mean(nn.functional.relu(mos_pair1-mos_output) + nn.functional.relu(mos_output - mos_pair2))
        # loss = loss_rank + 0.1 * loss_wb

        optimizer.zero_grad()  # clear gradients for next train
        torch.autograd.backward(loss)
        optimizer.step()

        acc1, _, _ = criterion.accuracy(mos_output, mos_pair1)
        batch_size = img_size[0] / 2
        losses.update(loss.detach().cpu().item(), batch_size)
        top1.update(acc1.item(), batch_size)
        #  记录时间
        batch_time.update(time.time() - start_time)
        start_time = time.time()

    logging.info('Epoch %d , learning rate: %f, averaged training loss: %.4f, using time %.3f' % (epoch + 1, lr, losses.avg, batch_time.sum))

    return losses.avg, top1.avg, batch_time.sum


def train_regress_epoch(train_loader, model, criterion, optimizer, epoch, lr, args):

    model.train()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('RankAcc@1', ':6.2f')

    # y_output = np.zeros(len(train_loader))
    # y_label = np.zeros(len(train_loader))

    y_output = list([])
    y_label = list([])

    l2rank = L2RankLoss().cuda(args.gpus)
    plccLoss = PLCCLoss().cuda(args.gpus)

    start_time = time.time()
    for i, (images, mos, names) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        if args.cuda:
            images = images.cuda(args.gpus)  # non_blocking=True
            mos = mos.cuda(args.gpus)

        # y_label[i] = mos.item()
        y_label.extend(torch.flatten(mos.detach().cpu()))

        img_size = images.shape
        images = images.view(-1, img_size[2], img_size[3], img_size[4])
        mos = mos[:, np.newaxis]

        mos_output = model(images)

        # mos_output = torch.flatten(mos_output, 1)
        # average the projection features
        mos_output = mos_output.view(img_size[0], img_size[1], 1)
        mos_output = torch.mean(mos_output, dim=1)
        # y_output[i] = mos_output.item()
        y_output.extend(torch.flatten(mos_output.detach().cpu()))

        # compute loss
        # loss = criterion(mos_output, mos)
        loss1 = plccLoss(mos_output, mos)
        loss2 = l2rank(mos_output, mos)
        loss = -1.0 * loss1 + loss2
        
        optimizer.zero_grad()  # clear gradients for next train
        torch.autograd.backward(loss)
        optimizer.step()

        batch_size = img_size[0]
        losses.update(loss.detach().cpu().item(), batch_size)
        # print(losses,names)
        
        # top1.update(acc1.item(), batch_size)
        #  记录时间
        batch_time.update(time.time() - start_time)
        start_time = time.time()

    plcc, srocc, krocc, mse = criterion.accuracy(np.array(y_output), np.array(y_label))
    logging.info('Epoch %d , learning rate: %f, averaged training loss: %.4f, using time %.3f' % (epoch + 1, lr, losses.avg, batch_time.sum))

    return losses.avg, [plcc, srocc, krocc, mse], batch_time.sum
