import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random

from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator

from PIL import Image
from tqdm import tqdm



class RegressDataset(data.Dataset):
    """Read data from the original dataset for feature extraction
    :param img_dir: 图像
    :param datainfo_path:
    """

    def __init__(self, img_dir, datainfo_path, transform, crop_size=224, img_length_read=4):
        super(RegressDataset, self).__init__()
        dataInfo = pd.read_csv(datainfo_path, header=0, sep=',', index_col=False, encoding="utf-8-sig")
        self.ply_name = dataInfo[['name']]
        self.ply_mos = dataInfo['mos']
        self.crop_size = crop_size
        self.img_dir = img_dir
        self.transform = transform
        self.img_length_read = img_length_read
        self.length = len(self.ply_name)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img_name = self.ply_name.iloc[idx, 0]
        frames_dir = os.path.join(self.img_dir, img_name)

        img_channel = 3
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size

        img_length_read = self.img_length_read
        transformed_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])

        # read images
        img_read_index = 0
        for i in range(img_length_read):
            # load images
            imge_name2 = os.path.join(frames_dir, str(i) + '.png')
            if os.path.exists(imge_name2):
                # print(imge_name)
                read_frame2 = Image.open(imge_name2)
                read_frame2 = read_frame2.convert('RGB')
                read_frame2 = self.transform(read_frame2)
                transformed_img[i] = read_frame2
                img_read_index += 1
            else:
                print(imge_name2 + ' - Image do not exist!')

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                transformed_img[j] = transformed_img[img_read_index - 1]

        y_mos = self.ply_mos.iloc[idx]
        y_label = torch.FloatTensor(np.array(y_mos))

        # return transformed_img, selected_patches, y_label
        return transformed_img, y_label, img_name


class RankDataset(data.Dataset):
    """ 排序图片文件夹Dataset类 """

    def __init__(self, img_dir, datainfo_path, transform, crop_size=224, img_length_read=4):
        """ 排序图片文件夹Dataset类
        :param img_dir: 排序图像所在文件夹，
        :param datainfo_path: 该文件夹下有 rank.txt 标签文件，每行格式如下：
                     good_image_path.jpg,bad_image_path.jpg, mos1, mos2, w_mos1, b_mos1, w_mos2, b_mos2
                     图片格式不一定为jpg格式，以上表示左边图片质量 > 右边图片质量
        """
        super(RankDataset, self).__init__()
        self.crop_size = crop_size
        self.img_dir = img_dir
        self.transform = transform
        self.img_length_read = img_length_read

        # file_name = 'rank.txt'
        self.label_lines = list()
        with open(datainfo_path) as label_file:
            label_file = label_file.readlines()[1:]
            for line in label_file:
                self.label_lines.append([p.strip() for p in line.strip().split(',')])

    def __len__(self):
        return len(self.label_lines)

    def __getitem__(self, index: int):
        """ 读取数据，得到 图像，分数，文件路径
        :param index: Index
        :return tuple: (image_pair, score_pair, path_pair) 一对图像，对应的DMOS分数，对应的路径
        """
        label_line = self.label_lines[index]

        frames_dir_1 = os.path.join(self.img_dir, label_line[0])
        frames_dir_2 = os.path.join(self.img_dir, label_line[1])

        img_channel = 3
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size
        img_length_read = self.img_length_read

        transformed_img_1 = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        transformed_img_2 = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])

        img_read_index1 = 0
        img_read_index2 = 0

        for i in range(img_length_read):
            # load images
            imge_name1 = os.path.join(frames_dir_1, str(i) + '.png')
            imge_name2 = os.path.join(frames_dir_2, str(i) + '.png')

            if os.path.exists(imge_name1):
                # print(imge_name)
                read_frame1 = Image.open(imge_name1)
                read_frame1 = read_frame1.convert('RGB')
                read_frame1 = self.transform(read_frame1)
                transformed_img_1[i] = read_frame1
                img_read_index1 += 1
            else:
                print(imge_name1 + ' - Image do not exist!')

            if os.path.exists(imge_name2):
                # print(imge_name)
                read_frame2 = Image.open(imge_name2)
                read_frame2 = read_frame2.convert('RGB')
                read_frame2 = self.transform(read_frame2)
                transformed_img_2[i] = read_frame2
                img_read_index2 += 1
            else:
                print(imge_name2 + ' - Image do not exist!')

        if img_read_index1 < img_length_read:
            for j in range(img_read_index1, img_length_read):
                transformed_img_1[j] = transformed_img_1[img_read_index1 - 1]

        if img_read_index2 < img_length_read:
            for j in range(img_read_index2, img_length_read):
                transformed_img_2[j] = transformed_img_2[img_read_index2 - 1]

        mos_pair = [torch.FloatTensor(np.array(float(label_line[2]))), torch.FloatTensor(np.array(float(label_line[3])))]
        # wb_mos_pair1 = [torch.FloatTensor(np.array(float(label_line[4]))), torch.FloatTensor(np.array(float(label_line[5])))]
        # wb_mos_pair2 = [torch.FloatTensor(np.array(float(label_line[6]))), torch.FloatTensor(np.array(float(label_line[7])))]
        # left control
        wb_mos_pair1 = [torch.FloatTensor(np.array(float(label_line[4]))), torch.FloatTensor(np.array(float(label_line[6])))]
        # right control
        wb_mos_pair2 = [torch.FloatTensor(np.array(float(label_line[5]))), torch.FloatTensor(np.array(float(label_line[7])))]

        image_pair = [transformed_img_1, transformed_img_2]
        path_pair = [frames_dir_1, frames_dir_2]

        return image_pair, mos_pair, wb_mos_pair1, wb_mos_pair2, path_pair


class MyDataloader(DataLoader):
    """ 使用prefetch_generator包提供的数据预加载功能 """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    @staticmethod
    def my_collate_fn(batch):
        """ 将图像对、DMOS分值对、路径对的列表，进行数据整理为图像列表和路径列表
        :param batch: 图像对、路径对的列表，[[[image_a0, image_a1], [score_a0, score_a1], [path_a0, path_a1]], ...]
        :return 图像列表和路径列表，
                [image_a0, image_a1, image_b0, image_b1, ...]
                [score_a0, score_a1, score_b0, score_b1, ...]
                [path_a0, path_a1, path_b0, path_b1, ...]
        """
        image_sequence = []
        score_sequence = []
        wbscore_sequence1 = []
        wbscore_sequence2 = []
        path_sequence = []
        for image_pair, score_pair, wb_pair1, wb_pair2, path_pair in batch:
            image_sequence.extend(image_pair)
            score_sequence.extend(score_pair)
            wbscore_sequence1.extend(wb_pair1)
            wbscore_sequence2.extend(wb_pair2)
            path_sequence.extend(path_pair)

        # 列表的Tensor堆成一个Tensor，列表成为新维
        image_sequence = torch.utils.data.dataloader.default_collate(image_sequence)
        score_sequence = torch.utils.data.dataloader.default_collate(score_sequence)
        wbscore_sequence1 = torch.utils.data.dataloader.default_collate(wbscore_sequence1)
        wbscore_sequence2 = torch.utils.data.dataloader.default_collate(wbscore_sequence2)
        path_sequence = torch.utils.data.dataloader.default_collate(path_sequence)
        return image_sequence, score_sequence, wbscore_sequence1, wbscore_sequence2, path_sequence


if __name__ == '__main__':

    img_dir = '../data/sjtu/'
    # train_filename_list = '../data/index_info/sjtu/train_' + str(1) + '.csv'
    train_filename_list = '../data/index_info/sjtu/rank_train_' + str(1) + '.txt'

    # train_dataset = RegressDataset(img_dir=img_dir, datainfo_path=train_filename_list)
    train_dataset = RankDataset(img_dir=img_dir, datainfo_path=train_filename_list)

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=7, shuffle=True, num_workers=1)
    train_loader = MyDataloader(dataset=train_dataset, batch_size=7, shuffle=True, collate_fn=MyDataloader.my_collate_fn, num_workers=1)

    # for i, (imgs, mos, names) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
    for i, (imgs, mos, wb_mos1, wb_mos2, _) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        print(imgs, mos, wb_mos1, wb_mos2, _)
        break
        # print(imgs, mos, wb_mos1, wb_mos2)