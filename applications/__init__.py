# -*- coding: utf-8 -*-
""" 主程序main.py调用的应用
train: 基于训练集、验证集的模型训练
test: 基于测试集的模型测试
"""

from .test import test_rank, test_regress
from .train import train