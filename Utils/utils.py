import logging
import os
import time
import torch
import pickle
import numpy as np


def save_log(obj, path):
    """
    保存日志对象到指定路径的文件中。

    参数:
    - obj: 要保存的对象。
    - path: 文件路径，用于存储日志。
    """
    with open(path, "wb") as f:  # 以二进制写入模式打开文件
        pickle.dump(obj, f)  # 使用 pickle 序列化并保存对象


def read_pkl(path):
    """
    从指定路径读取 pickle 文件并返回数据。

    参数:
    - path: 文件路径，用于读取数据。

    返回:
    - data: 反序列化后的数据。
    """
    with open(path, 'rb') as f:  # 以二进制读取模式打开文件
        data = pickle.load(f)  # 使用 pickle 反序列化文件内容
    return data


def save_model(model, args):
    """
    保存 PyTorch 模型的状态字典到本地。

    参数:
    - model: 要保存的 PyTorch 模型。
    - args: 参数对象，包含模型的配置（例如 `backbone` 和是否启用 FFT）。
    """
    if not os.path.exists("./checkpoints"):  # 检查存储路径是否存在
        os.makedirs("./checkpoints")  # 如果不存在，则创建目录

    # 根据是否启用 FFT 模式选择文件名
    if not args.fft:
        torch.save(model.state_dict(), "./checkpoints/{}.tar".format(args.backbone))
    else:
        torch.save(model.state_dict(), "./checkpoints/{}FFT.tar".format(args.backbone))


def accuracy(outputs, targets):
    """
    计算有标签数据的准确率。

    参数:
    - outputs (tensor): 模型预测输出，通常是 logits。
    - targets (tensor): 实际标签，可能包含 -1（表示未标记的数据）。

    返回:
    - acc (float): 有标签数据的分类准确率（百分比）。
    """
    # 获取有标签的样本数，确保最小值为一个很小的数以避免除零错误
    labeled_minibatch_size = max(targets.ne(-1).sum(), 1e-8)

    # 获取模型预测的类别索引
    pre = torch.max(outputs.cpu(), 1)[1].numpy()  # 将 logits 转化为类别索引
    y = targets.data.cpu().numpy()  # 获取真实标签的 numpy 表示

    # 计算预测值与真实值匹配的数量并计算准确率
    acc = ((pre == y).sum() / labeled_minibatch_size) * 100
    return acc


def randomseed(seed):
    """
    设置随机种子以确保实验的可重复性。

    参数:
    - seed (int): 随机种子值。
    """
    torch.manual_seed(seed)  # 设置 CPU 上的随机种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 上的随机种子


class Meter(object):
    """
    通用计量器类，用于记录和存储不同类别的值。
    """

    def __init__(self):
        # 初始化一个字典，用于存储多个类别的数据
        self.meter = {}

    def update(self, key, value):
        """
        更新指定类别的数据。
        如果该类别已存在，则追加新值；
        如果该类别不存在，则创建一个新的列表并添加值。

        参数:
        - key: 类别名称（字符串）。
        - value: 要记录的值。
        """
        if key in self.meter:
            self.meter[key].append(value)  # 类别已存在，追加值
        else:
            self.meter[key] = [value]  # 类别不存在，初始化并添加值

    def reset(self):
        """
        重置计量器，清空所有数据。
        """
        self.meter = {}


class AverageMeter(object):
    """
    平均值计量器类，用于计算和记录一个指标的平均值、总和和历史值。
    """

    def __init__(self, name) -> None:
        """
        初始化计量器。

        参数:
        - name: 计量器的名称，用于区分不同的计量对象。
        """
        self.name = name
        self.reset()  # 重置所有计量数据

    def reset(self):
        """
        重置计量器的数据，包括平均值、总和、计数器和历史记录。
        """
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数器
        self.history = []  # 历史值记录

    def update(self, val, n=1):
        """
        更新计量器的数据。
        根据输入的值和权重，计算新的总和和平均值，并存储历史记录。

        参数:
        - val: 当前更新的值。
        - n: 当前值的权重，默认是 1。
        """
        self.sum += val * n  # 更新总和
        self.count += n  # 更新计数器
        self.avg = self.sum / self.count  # 计算新的平均值
        self.history.append(val)  # 添加到历史记录
