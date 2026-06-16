
""" PyTorch官方提供的预定义模型及自定义模型 """
import logging

import torch
import torch.nn as nn
from torchvision import models

from .res2net import res2net50
from .resnet import resnet34, resnet50, resnet101

models_map = {'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101, 'res2net50': res2net50}


def get_model(name, pretrained=False, num_class=1, **kwargs):
    """ 获取指定名称的模型
    :param name: 指定模型名称
    :param pretrained: 是否加载预训练模型
    :param kwargs: num_classes等
    :return 指定名称的模型
    """

    # if name in models_map:
    #     model = models_map[name](pretrained=pretrained)
    # else:
    #     model = models.__dict__[name](**kwargs)

    model = models.__dict__[name](pretrained=pretrained)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)

        # if pretrained:
        #     model_path = f'pretrained/{name}.pth'
        #     state_dict = torch.load(model_path)
        #     state_dict.pop('fc.weight')
        #     state_dict.pop('fc.bias')
        #     acc = model.load_state_dict(state_dict, strict=False)
        #     del state_dict
        #     assert set(acc.missing_keys) == {'fc.weight', 'fc.bias'}, 'issue loading pretrained weights'
        #     logging.info(f"=> using pre-trained model '{model_path}'")

    return model

