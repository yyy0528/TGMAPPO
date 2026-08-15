import copy
import numpy as np

import torch
import torch.nn as nn

#这个函数用于初始化神经网络模块的权重和偏置
def init(module: nn.Module, weight_init, bias_init, gain: float = 1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

#这个函数创建指定数量的模块副本，使用ModuleList来管理。
def get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#这个函数处理输入数据类型，
# 如果输入是NumPy数组则转换为PyTorch张量，否则保持原样。
def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
