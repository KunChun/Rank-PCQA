import os
import argparse
import random
import logging
import datetime
from pathlib import Path

from torchvision import transforms
import numpy as np
import pandas as pd
import torch
from utils.set_seed import set_rand_seed
from utils.my_logger import generate_logger

import criterions
import applications
from my_models.get_network import get_model
from utils.my_dataset import RankDataset, RegressDataset, MyDataloader


def rank_train(args, fold_num, flag='rank'):

    # get data info
    img_dir = os.path.join(args.root, args.database)
    train_idx_info = os.path.join(args.root, 'index_info', args.database, 'rank_train_'+str(fold_num)+'.txt')
    test_idx_info = os.path.join(args.root, 'index_info', args.database, 'rank_test_'+str(fold_num)+'.txt')

    # get model
    logging.info(f"=> creating model '{args.arch}'")
    model = get_model(args.arch, args.pretrained, args.num_classes)

    # get training tools
    criterion = criterions.RankingLoss(args=args)  # 对比排序损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if args.cuda:
        logging.info(f"The current fold : {fold_num}, use GPU: {args.gpus} for Rank training~~~")
        torch.cuda.set_device(args.gpus)
        model = model.cuda(args.gpus)
        criterion.cuda(args.gpus)

    if args.train:
        transformations_train = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_dataset = RankDataset(img_dir=img_dir, datainfo_path=train_idx_info, transform=transformations_train)
        train_loader = MyDataloader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=MyDataloader.my_collate_fn, num_workers=args.workers)
        test_dataset = RankDataset(img_dir=img_dir, datainfo_path=test_idx_info, transform=transformations_test)
        test_loader = MyDataloader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=MyDataloader.my_collate_fn, num_workers=args.workers)
        applications.train(train_loader, test_loader, model, criterion, optimizer, scheduler, args, flag, fold_num)

    if args.evaluate:
        transformations_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        test_dataset = RankDataset(img_dir=img_dir, datainfo_path=test_idx_info, transform=transformations_test)
        test_loader = MyDataloader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=MyDataloader.my_collate_fn, num_workers=args.workers)
        applications.test_rank(test_loader, model, criterion, args)

    return


def finetune_train(args, fold_num, flag='regress'):

    # get data info
    img_dir = os.path.join(args.root, args.database)
    trian_idx_info = os.path.join(args.root, 'index_info', args.database, 'train_' + str(fold_num) + '.csv')
    test_idx_info = os.path.join(args.root, 'index_info', args.database, 'test_' + str(fold_num) + '.csv')

    # get model
    logging.info(f"=> Loading model from ckpts'{args.arch}'")
    model = get_model(args.arch, pretrained=True, num_class=args.num_classes)
    state_dict = torch.load(os.path.join(args.ckpts, args.database, 'rank_' + args.arch + '_fold_' + str(fold_num) + '_margin' + str(args.margin) + '.pth'))
    # state_dict = torch.load(os.path.join(args.ckpts, 'pclpcqa', 'rank_' + args.arch + '_fold_' + str(fold_num) + '_margin' + str(args.margin) + '.pth'))
    model.load_state_dict(state_dict)

    # get training tools
    criterion = criterions.RegressionLoss()  # 回归损失，用于微调
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

    if args.cuda:
        logging.info(f"The current fold : {fold_num}, use GPU: {args.gpus} for Regress training~~~")
        torch.cuda.set_device(args.gpus)
        model = model.cuda(args.gpus)
        criterion.cuda(args.gpus)

    if args.train:
        transformations_train = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_dataset = RegressDataset(img_dir=img_dir, datainfo_path=trian_idx_info, transform=transformations_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        test_dataset = RegressDataset(img_dir=img_dir, datainfo_path=test_idx_info, transform=transformations_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        applications.train(train_loader, test_loader, model, criterion, optimizer, scheduler, args, flag, fold_num)

    # if args.evaluate:
    #     test_dataset = RegressDataset(img_dir=img_dir, datainfo_path=test_idx_info)
    #     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    #     applications.test_regress(test_loader, model, criterion, args, save_flag=False)

    return


def evaluate_results(args, fold_num):
    # get data info
    img_dir = os.path.join(args.root, args.database)
    trian_idx_info = os.path.join(args.root, 'index_info', args.database, 'train_' + str(fold_num) + '.csv')
    test_idx_info = os.path.join(args.root, 'index_info', args.database, 'test_' + str(fold_num) + '.csv')

    # get model
    logging.info(f"=> Loading best model from ckpts-'{args.database}'")
    model = get_model(args.arch, pretrained=False, num_class=args.num_classes)
    state_dict = torch.load(os.path.join(args.ckpts, args.database, 'fineT_' + args.arch + '_fold_' + str(fold_num) + '_margin' + str(args.margin) + '.pth'))
    model.load_state_dict(state_dict)

    # get training tools
    criterion = criterions.RegressionLoss()  # 回归损失，用于微调
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if args.cuda:
        logging.info(f"The current fold : {fold_num}, use GPU: {args.gpus} for Regress testing~~~")
        torch.cuda.set_device(args.gpus)
        model = model.cuda(args.gpus)
        criterion.cuda(args.gpus)

    transformations_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_dataset = RegressDataset(img_dir=img_dir, datainfo_path=test_idx_info, transform=transformations_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    results, _, __, ___ = applications.test_regress(test_loader, model, criterion, args, fold_num, save_flag=True)

    return results


def my_parse_args():
    parser = argparse.ArgumentParser(description="training")

    parser.add_argument('--database', default='lspcqa', type=str, help='数据集: sjtu, wpc, pclpcqa, ...')
    parser.add_argument('--k_fold', default=5, type=int, help='9 for the SJTU-PCQA, 5 for the WPC and pclpcqa, 4 for the WPC2.0')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='训练epoch数，默认：85')
    parser.add_argument('--arch', metavar='ARCH', default='resnet34', help='模型结构，默认：efficientnet-b0')  # efficientnet-b0
    parser.add_argument('--margin', default=0.1, type=float, help='margin ranking loss的margin值，默认为：0.0')
    parser.add_argument('--learning_rate', default=0.0001, type=float, metavar='LR', help='初始学习率，默认：1e-4')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='学习率动量')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='网络权重衰减正则项，默认: 1e-4', dest='weight_decay')
    parser.add_argument('--batch_size', default=10, type=int, metavar='N', help='训练batch size大小，默认：256')

    parser.add_argument('--root', default='data/', metavar='DIR', help='数据集根路径')
    parser.add_argument('--ckpts', default='ckpts_all_loss/', metavar='DIR', help='checkpoint模型保存路径')
    parser.add_argument('--results', default='results/', metavar='DIR', help='results结果保存路径')
    parser.add_argument('--logdir', default='logs/', type=str, metavar='PATH', help='Tensorboard日志目录，默认 logs')

    parser.add_argument('--train', default=True, dest='train', action='store_true', help='是否训练，默认True, 训练时自带测试部分')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true', help='在测试集上评估模型')
    parser.add_argument('--image_size', default=[224, 224], type=int, nargs='+', dest='image_size', help='模型输入尺寸[H, W]，默认：[224, 224]')
    parser.add_argument('--num_classes', default=1, type=int, help='分支数，或者说最大分值数，默认：1')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='数据加载进程数，默认：8')
    parser.add_argument('--seed', default=2024, type=int, help='训练或测试时，使用随机种子保证结果的可复现，默认不使用')
    parser.add_argument('-g', '--gpus', default=0, type=int, help='每个节点使用的GPU数量，可通过设置环境变量（CUDA_VISIBLE_DEVICES=1）限制使用哪些/单个GPU')
    # parser.add_argument('--rank', default=-1, type=int, help='分布式训练的当前节点的序号')
    parser.add_argument('--cuda', default=True, dest='cuda', action='store_true', help='是否使用cuda进行模型推理，默认 True，会根据实际机器情况调整')

    parser.add_argument('--warmup', default=5, type=int, metavar='W', help='warm-up迭代数')
    parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='训练过程中的信息打印，每隔多少个batch打印一次，默认: 50')
    parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true', help='是否使用预训练模型，默认不使用')

    args = parser.parse_args()

    return args


if __name__=='__main__':

    args = my_parse_args()
    set_rand_seed(args.seed)

    generate_logger(f"{args.logdir}{args.database}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{args.gpus}.log")
    logging.info(f'args: {args}')

    all_fold_res = np.array([])
    for k_id in range(1, args.k_fold + 1):
        args.epochs = 10
        # args.learning_rate = 0.00001
        rank_train(args, k_id)  # 排序训练

        # 调整学习率，epoch，
        args.learning_rate = 0.00001
        args.epochs = 100
        # args.bathch_size = 10
        # args.database = 'wpc'
        finetune_train(args, k_id)  # 微调训练
        
        res = evaluate_results(args, k_id)  # 测试结果
        res = np.array(res)
        if k_id == 1:
            all_fold_res = res
        else:
            all_fold_res = np.vstack((all_fold_res, res))
    
    mean_res = all_fold_res.mean(axis=0)
    mean_res = mean_res.reshape(1, -1)
    all_results = np.vstack((all_fold_res, mean_res))
    # all_results = np.concatenate((all_fold_res, mean_res), axis=1)
    results2 = pd.DataFrame(columns=['PLCC', 'SROCC', 'KROCC', 'MSE'], data=all_results)
    exp_dir = Path(f'{args.results}{args.database}')
    exp_dir.mkdir(exist_ok=True)
    results2.to_csv(str(exp_dir) + '/'+ args.arch + '_final_test_score_All_margin' + str(args.margin) +'.csv', index=False)
