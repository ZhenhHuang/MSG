# -*- coding: utf-8 -*-
# @Time    : 2024/4/6  15:08
# @Author  : WanQiQi
# @FileName: SpikingGCNN.py
# @Software: PyCharm
"""
    Description:
        
"""
import os
import time
import argparse
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from torch_geometric.datasets import Planetoid

from GraphEncoding import GCNencoder


# 定义SNN模型
class SGCNN(nn.Module):
    def __init__(self, input_size, num_classes, tau):
        super().__init__()

        # 输入尺寸和分类数现在是可配置的
        self.input_size = input_size
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, self.num_classes, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    # 加载数据集
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    tau = 100.0
    # 脉冲序列编码
    spike_seq = GCNencoder(dataset,tau)
    print(spike_seq)
    print(spike_seq.size())
    spike_seq = spike_seq.float()

    # input_size是脉冲序列中每个元素的特征数量
    # num_classes是目标分类数量
    input_size = spike_seq.size(1)
    num_classes = dataset.num_classes
    net = SGCNN(input_size=input_size, num_classes=num_classes, tau=tau)

    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    output = net(spike_seq)
    print(sum(output))

    # # 训练模型
    # for epoch in range(100):  # 假设训练100个epoch
    #     optimizer.zero_grad()
    #     output = net(spike_seq)
    #     print(output)
    #     # loss
