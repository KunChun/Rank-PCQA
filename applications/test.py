# -*- coding: utf-8 -*-
""" 模型测试脚本 """
import time
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import my_meters


def test_rank(test_loader: DataLoader, model: nn.Module, criterion: nn.Module, args: argparse.Namespace):
    """ 验证集、测试集 评估
    :param test_loader: 测试集DataLoader对象
    :param model: 待测试模型
    :param criterion: 损失函数
    :param args: 测试参数
    """
    batch_time = my_meters.AverageMeter('Time', ':6.3f')
    losses = my_meters.AverageMeter('Loss', ':.4e')
    top1 = my_meters.AverageMeter('Acc@1', ':6.2f')

    # 模型评估
    model.eval()
    total_paths, total_preds, total_probs = list(), list(), list()  # 样本的路径
    with torch.no_grad():
        start_time = time.time()
        for i, (image_pair, mos_pair, wb_mos_pair1, wb_mos_pair2, path_pair) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            if args.cuda:
                image_pair = image_pair.cuda(args.gpus)  # non_blocking=True
                wb_mos_pair1 = wb_mos_pair1.cuda(args.gpus)
                wb_mos_pair2 = wb_mos_pair2.cuda(args.gpus)

            img_size = image_pair.shape
            imgs = image_pair.view(-1, img_size[2], img_size[3], img_size[4])
            mos_output = model(imgs)
            mos_output = mos_output.view(img_size[0], img_size[1], 1)
            outputs = torch.mean(mos_output, dim=1)

            loss = criterion(outputs, wb_mos_pair1)

            # 统计准确率和损失函数
            acc1, pred, probs = criterion.accuracy(outputs, wb_mos_pair1)
            # 收集结果
            if args.evaluate:
                total_paths.extend(zip(path_pair[0::2], path_pair[1::2]))
                total_preds.extend(pred.detach().cpu().numpy())
                total_probs.extend(probs.detach().cpu().numpy())

            losses.update(loss.detach().cpu().item(), img_size[0] / 2)
            top1.update(acc1.item(), img_size[0] / 2)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

    logging.info(f' *** Averaged Testing Rank Acc ======= {top1.avg:.3f}======= and loss {losses.avg:.3f} with time {batch_time.sum:.3f}')

    return top1.avg, losses.avg, batch_time.sum, zip(total_paths, total_preds, total_probs)


def test_regress(test_loader: DataLoader, model: nn.Module, criterion: nn.Module, args: argparse.Namespace, fold_num, save_flag):
    batch_time = my_meters.AverageMeter('Time', ':6.3f')
    losses = my_meters.AverageMeter('Loss', ':.4e')
    # top1 = my_meters.AverageMeter('Acc@1', ':6.2f')

    # 模型评估
    model.eval()
    y_output = np.zeros(len(test_loader))
    y_label = np.zeros(len(test_loader))

    pred_all = np.array([])
    target_all = np.array([])
    name_all = np.array([])

    total_paths, total_preds, total_probs = list(), list(), list()  # 样本的路径
    with torch.no_grad():
        start_time = time.time()
        for i, (images, mos, names) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            if args.cuda:
                images = images.cuda(args.gpus)  # non_blocking=True
                mos = mos.cuda(args.gpus)

            y_label[i] = mos.item()
            name_all = np.append(name_all, names)

            img_size = images.shape
            images = images.view(-1, img_size[2], img_size[3], img_size[4])
            mos = mos[:, np.newaxis]

            mos_output = model(images)

            # mos_output = torch.flatten(mos_output, 1)
            # average the projection features
            mos_output = mos_output.view(img_size[0], img_size[1], 1)
            mos_output = torch.mean(mos_output, dim=1)
            y_output[i] = mos_output.item()

            # compute loss
            loss = criterion(mos_output, mos)
            batch_size = img_size[0]
            losses.update(loss.detach().cpu().item(), batch_size)

            #  记录时间
            batch_time.update(time.time() - start_time)
            start_time = time.time()

    plcc, srocc, krocc, mse = criterion.accuracy(y_output, y_label)

    logging.info(f' *** Testing Regress plcc, srocc, krocc, mse ======= {plcc},  {srocc},  {krocc},  {mse} ======= and loss {losses.avg:.3f} with time {batch_time.sum:.3f}')

    if save_flag:
        name_all = name_all.reshape(-1, 1)
        target_all = y_label.reshape(-1, 1)
        pred_all = y_output.reshape(-1, 1)
        all_results = np.concatenate((name_all, target_all, pred_all), axis=1)
        results2 = pd.DataFrame(columns=['Plyname', 'MOS', 'Pred'], data=all_results)

        exp_dir = Path(args.results + args.database)
        exp_dir.mkdir(exist_ok=True)

        results2.to_csv(str(exp_dir) + '/' + args.arch + '_final_test_score_fold' + str(fold_num) + '_margin' + str(args.margin) + '.csv', index=False)
        # results2.to_csv(str(exp_dir) + '/A_final_test_score_fold'+str(fold_num)+'.csv', index=False)

    return [plcc, srocc, krocc, mse], losses.avg, batch_time.sum, zip(total_paths, total_preds, total_probs)

# if __name__=='__main__':
#
#     exp_dir = Path('../results/'+ 'sjtu')
#     exp_dir.mkdir(exist_ok=True)