"""
Author: Xiaohan Chen
Mail: cxh_bb@outlook.com
"""

import torch
from torch.utils.data import Dataset
from Datasets.Augmentation import *
import numpy as np

class BaseDataset(Dataset):
    """
    BaseDataset 类：用于将字典格式的数据集转换为 PyTorch 数据集。

    功能：
    - 读取字典格式的数据集。
    - 提供统一的样本访问接口。
    - 将数据和标签转换为 PyTorch 张量。
    """

    def __init__(self, dataset):
        """
        初始化 BaseDataset 对象。

        参数：
        - dataset (dict): 字典格式的数据集，其中键为类别或标签，值为对应类别的样本数据（数组或列表）。

        功能：
        - 将字典格式的数据集存储为类属性。
        - 计算数据集包含的类别数量。
        - 读取数据并将其存储为统一的样本和标签格式。
        """
        super(BaseDataset, self).__init__()  # 调用父类构造函数
        self.dataset = dataset  # 存储数据集
        self.classes = len(dataset)  # 数据集中的类别数量
        self.x, self.y = self._read(dataset)  # 读取并整理数据和标签

    def _read(self, dataset):
        """
        从字典格式的数据集中提取样本和标签。

        参数：
        - dataset (dict): 字典格式的数据集。

        返回：
        - x (ndarray): 包含所有样本数据的 NumPy 数组。
        - y (ndarray): 每个样本对应的标签数组（整数值）。

        逻辑：
        - 将字典中所有类别的数据连接成一个数组 `x`。
        - 生成标签数组 `y`，每个类别的标签值对应类别索引。
        """
        # 将字典中所有类别的数据拼接成一个数组
        x = np.concatenate([dataset[key] for key in dataset.keys()])
        y = []  # 存储每个样本的标签
        for i, key in enumerate(dataset.keys()):
            number = len(dataset[key])  # 当前类别样本数量
            y.append(np.tile(i, number))  # 生成与样本数量相同的类别标签
        y = np.concatenate(y)  # 将标签列表拼接成一个数组
        '''
        假设 dataset 中的数据结构为：
        dataset = {
        "class1": [array1, array2],        # 2 个样本
        "class2": [array3, array4, array5]  # 3 个样本
        }
        
        x = np.array([
        [array1],       # class1 的第一个样本
        [array2],       # class1 的第二个样本
        [array3],       # class2 的第一个样本
        [array4],       # class2 的第二个样本
        [array5]        # class2 的第三个样本
        ])

        遍历后的标签数组生成过程：
        第一次循环：i=0, key="class1"
        number = 2（class1 有 2 个样本）
        np.tile(0, 2) 生成 [0, 0]
        y.append([0, 0])，此时 y = [[0, 0]]
        第二次循环：i=1, key="class2"
        number = 3（class2 有 3 个样本）
        np.tile(1, 3) 生成 [1, 1, 1]
        y.append([1, 1, 1])，此时 y = [[0, 0], [1, 1, 1]]
        拼接最终标签
        
        在后续代码中，这些列表会通过 np.concatenate 拼接成一个完整的标签数组：
        y = np.concatenate(y)
        # y = [0, 0, 1, 1, 1]
        '''
        return x, y#将字典拆分为x和y，x-->(labal数*number,1,data_length)的三维数组，y-->（labal数*number）的一维数组，x[i][][]对应着y[i]

    def __len__(self):
        """
        获取数据集中样本的总数。

        返回：
        - count (int): 数据集中样本的总数。
        """
        count = 0
        for key in self.dataset.keys():
            count += len(self.dataset[key])  # 累加每个类别的样本数量
        return count

    def __getitem__(self, index):
        """
        根据索引获取数据集中的一个样本。

        参数：
        - index (int): 样本索引。

        返回：
        - data (Tensor): 样本数据，类型为 PyTorch 张量。
        - label (Tensor): 样本标签，类型为 PyTorch 张量。
        """
        # 根据索引提取数据和标签
        data = self.x[index]
        label = self.y[index]

        # 将数据和标签转换为 PyTorch 张量
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)
        return data, label

class AugmentDasetsetTFPair(BaseDataset):
    """
    AugmentDasetsetTFPair 类，用于生成经过时域和频域增强的数据对。

    继承自：
    - BaseDataset：一个自定义的基础数据集类，提供 `self.x` 和 `self.y` 等属性。

    方法：
    - __getitem__(self, index): 根据索引返回一个样本，包含增强后的时域数据、频域数据以及对应的标签。
    """

    def __getitem__(self, index):#训练过程中，每当pytorch利用此方法取得数据集中一条数据时，自动进行数据增强
        """
        获取数据集中的一个样本并进行数据增强。

        参数：
        - index: int，样本索引，用于从数据集中定位特定样本。

        返回：
        - data_t: tensor，经过时域增强的样本数据。
        - data_f: tensor，经过频域增强的样本数据。
        - label: tensor，样本对应的标签。
        """
        # 从数据集中提取原始数据和对应标签
        data = self.x[index]  # 原始数据（NumPy 数组形式）
        label = self.y[index]  # 对应标签（通常是类别或标注）

        # 将 NumPy 数组转换为 PyTorch 张量并设置数据类型为 float32
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)

        # 对数据进行时域增强（如平移、缩放等操作）
        data_t = random_waveform_transforms(data)#[1, 1024]
        # 对数据进行频域增强（如频率滤波、频率变换等操作）
        data_f = frequency_transforms(data)#[1, 512]
        # 返回增强后的时域数据、频域数据以及标签
        return data_t, data_f, label


def relabel_dataset(args, dataset):#dataset:10*（num_train，1，data_length）
    """
    对数据集进行部分重标记，将部分标签标记为 -1，模拟半监督学习中的未标记数据。

    参数:
    - args: 包含训练相关参数的对象。
    - dataset: 数据集对象，包含输入数据和标签。

    返回:
    - labeled_indices: 包含所有已标记样本的索引。
    - unlabeled_indices: 包含所有未标记样本的索引。
    """
    # 存储未标记样本的索引
    unlabeled_idx = []

    # 每个类别的样本数量
    num_data_per_class = args.num_train

    # 每个类别未标记样本的数量
    num_unlabeled_per_class = args.num_train - args.num_labels

    # 类别标签列表
    classes = list(int(i) for i in args.labels.split(","))
    num_class = len(classes)


    idx = np.arange(num_data_per_class * num_class).reshape(num_class, num_data_per_class)
    #idx为num-class,num_data_per_class形状的如下数组：
    '''
    
    [[ 0  1  2  3  4]
    [ 5  6  7  8  9]
    [10 11 12 13 14]]
    ...
    
    '''

    # 对每个类别的样本索引进行随机打乱，并选择部分样本作为未标记数据
    for i in range(len(idx)):
        idx[i] = np.random.permutation(idx[i])
        unlabeled_idx.append(idx[i][:num_unlabeled_per_class])


    unlabeled_idx = np.array(unlabeled_idx).reshape(-1)#把unlabeled_idx转成一维数组
    dataset.y[unlabeled_idx] = -1#把索引数组中，下标在unlabeled_idx中的所有索引改为-1

    # 将未标记样本的索引与已标记样本的索引分开
    unlabeled_indices = set(unlabeled_idx)
    labeled_indices = set(np.arange(num_data_per_class * num_class)) - unlabeled_indices

    return list(labeled_indices), list(unlabeled_indices)#返回哪些索引属于被标记的，哪些属于未标记的
