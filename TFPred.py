"""
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
"""
import warnings

# 屏蔽特定的 UserWarning
warnings.filterwarnings("ignore", message="No audio backend is available.")

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import *

import Configs.TFPrediction as parms
from Preparedata import PU
from Preparedata import PU2
from Datasets import Dataset
from Models import ResNet1D
from Losses.CrossCorrelation import CrossCorrelationLoss
from Utils import utils
from Utils.logger import setlogger

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support

def load_data(args):
    # 使用PU.PUloader加载数据集，根据args中的配置返回一个包含数据的字典
    datadict = PU.PUloader(args)#处理后每个label对应的datadict[label]中都存有N片1*data_length的数据，即共10*（N，1，data_length），其中N为args.num_train + args.num_validation + args.num_test的总和

    # 将数据集随机打乱，确保训练、验证和测试集的数据分布随机
    np.random.seed(28)  # 固定随机种子以保证实验可复现
    datadict = {key: np.random.permutation(datadict[key]) for key in datadict.keys()}#（N，1，data_length）-->（N，1，data_length）但每条1*data_length内部不变，但上下顺序改变

    # 将数据集划分为训练集、验证集和测试集
    # 根据args中的参数分别选取前num_train个、接下来的num_validation个和最后的num_test个样本
    train_datadict = {key: datadict[key][:args.num_train] for key in datadict.keys()}#10*（num_train，1，data_length）
    val_datadict = {key: datadict[key][args.num_train: args.num_train + args.num_validation] for key in datadict.keys()}#10*（num_num_validation，1，data_length）
    test_datadict = {key: datadict[key][-args.num_test:] for key in datadict.keys()}#10*（num_test，1，data_length）

    # 创建训练数据集，使用AugmentDatasetTFPair类为训练数据生成数据增强
    train_dataset = Dataset.AugmentDasetsetTFPair(train_datadict)
    # 创建评估数据集，使用BaseDataset类，不对数据进行增强
    evaluate_dataset = Dataset.BaseDataset(train_datadict)
    # 创建验证数据集，使用BaseDataset类，不对数据进行增强
    val_dataset = Dataset.BaseDataset(val_datadict)
    # 创建测试数据集，使用BaseDataset类，不对数据进行增强
    test_dataset = Dataset.BaseDataset(test_datadict)

    # 对评估数据集重新标记，返回标记数据的索引；这里的标记可能用于半监督或自监督学习
    labeled_indices, _ = Dataset.relabel_dataset(args, evaluate_dataset)#labeled_indices，一个一维数组
    # 使用SubsetRandomSampler根据标记索引采样，SubsetRandomSampler 是 PyTorch 提供的一个采样器类，允许从给定的索引列表中随机采样数据。
    sampler = torch.utils.data.SubsetRandomSampler(labeled_indices)

    # 创建训练数据加载器，支持多线程加载、批量处理、数据增强和随机打乱
    train_loader = DataLoader(
        train_dataset,  # 1. 训练数据集
        batch_size=args.batch_size,  # 2. 每个批次加载的样本数量
        shuffle=True,  # 3. 是否在每个 epoch 开始时打乱数据
        num_workers=args.num_workers,  # 4. 数据加载时使用的子进程数
        pin_memory=True,  # 5. 是否将数据保存在固定内存中
        drop_last=True  # 6. 是否丢弃最后一个不完整批次
    )

    # 创建评估数据加载器，使用采样器代替shuffle，确保加载的数据是标记过的
    evaluate_loader = DataLoader(evaluate_dataset, batch_size=args.batch_size, sampler=sampler,
                                 num_workers=args.num_workers, pin_memory=True)
    # 创建验证数据加载器，不打乱数据顺序
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    # 创建测试数据加载器，不打乱数据顺序
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # 返回所有数据加载器，供主函数使用
    return train_loader, evaluate_loader, val_loader, test_loader


# ===== Define encoder =====
class ModelBase(nn.Module):
    '''
    定义一个基本的编码器模型 (Encoder)，用于将输入数据编码为指定维度的特征向量。
    '''

    def __init__(self, dim=128) -> None:
        '''
        初始化 ModelBase 模型。

        参数:
        - dim (int): 输出特征向量的维度，默认值为 128。
        '''
        super().__init__()

        # 定义主网络结构为一维版本的 ResNet18，不使用归一化层
        self.net = ResNet1D.resnet18(norm_layer=None)

        # 定义一个展平层，将多维数据展平成一维向量
        self.flatten = nn.Flatten()

        # 定义一个全连接层，将 ResNet 输出的 512 维特征映射到目标维度 dim
        self.fc = nn.Linear(512, dim)

    def forward(self, x):#encoderT:[256, 1, 1024];encoderF:[256, 1, 512]
        '''
        定义前向传播过程。

        参数:
        - x (Tensor): 输入数据，形状为 [batch_size, channels, sequence_length]。

        返回:
        - x (Tensor): 编码后的特征向量，形状为 [batch_size, dim]。
        '''

        # 将输入数据通过 ResNet 网络提取特征
        x = self.net(x)#encoderT:[256, 1, 1024]-->[256, 512];encoderF:[256, 1, 512]-->[256, 512]

        # 将 ResNet 的输出展平为一维向量
        x = self.flatten(x)#encoderT:[256, 512]-->[256, 512];encoderF:[256, 512]-->[256, 512]

        # 通过全连接层将展平后的特征映射到指定维度
        x = self.fc(x)#encoderT:[256, 512]-->[256, 128];encoderF:[256, 512]-->[256, 128]

        # 返回编码后的特征向量
        return x#[256, 128]


class TFPrediction(nn.Module):
    """
    定义一个用于时域（T）和频域（F）特征提取及预测的神经网络模型。
    """

    def __init__(self, dim=128) -> None:
        """
        初始化网络结构。

        参数:
        - dim (int): 输出特征维度的大小，默认为 128。
        """
        super().__init__()
        # 定义两个特征提取器，一个用于处理时域信号，一个用于处理频域信号
        self.encoderT = ModelBase()  # 时域特征提取器
        self.encoderF = ModelBase()  # 频域特征提取器

        # 定义频域特征的预测模块，由全连接层、批归一化层和激活函数组成
        self.PredictionF = nn.Sequential(
            nn.Linear(dim, 256),           # 全连接层，将输入特征维度从 dim 映射到 256
            nn.BatchNorm1d(256),           # 批归一化，稳定训练过程
            nn.ReLU(inplace=True),         # ReLU 激活函数，增加非线性
            nn.Linear(256, dim),           # 全连接层，将特征维度从 256 映射回 dim
        )

    def forward(self, x_t, x_f):#x_t:[256, 1, 1024],x_f:[256, 1, 512]
        """
        前向传播函数，定义数据流的计算逻辑。

        参数:
        - x_t: 时域输入特征张量
        - x_f: 频域输入特征张量

        返回:
        - x_t: 经过时域特征提取后的特征
        - x_f: 经过频域特征提取和预测后的特征
        """
        # 使用时域特征提取器处理时域输入
        x_t = self.encoderT(x_t)#[256, 1, 1024]-->[256, 128]
        # 使用频域特征提取器处理频域输入
        x_f = self.encoderF(x_f)#[256, 1, 512]-->[256, 128]
        # 使用预测模块对频域特征进行进一步预测
        x_f = self.PredictionF(x_f)#(256,128)-->(256,128)

        # 返回提取后的时域特征和频域特征
        return x_t, x_f  # x_t=[256,128],x_f=[256,128]


# 辅助函数（不对外暴露）
def _save_correlation_analysis(outputs, labels, class_names, save_path='.'):
    """
    计算并保存两种相关性结果：
    1. 单变量相关性（每个类别的预测输出 vs 真实标签的one-hot编码）
    2. 互相关矩阵（所有真实类别 vs 所有预测类别的相关性）

    参数：
        outputs: 模型输出矩阵 (n_samples, n_classes)
        labels: 真实标签数组 (n_samples,)
        class_names: 类别名称列表
        save_path: 结果保存路径

    生成文件：
        1. output_label_correlation.csv - 单变量相关性
        2. label_prediction_correlation_matrix.csv - 互相关矩阵
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    num_classes = outputs.shape[1]

    # ============================================
    # 1. 计算单变量相关性（原始功能）
    # ============================================
    single_corr_results = []
    for class_idx in range(num_classes):
        # 计算当前类别的预测输出与真实标签的相关性
        corr, _ = pearsonr(
            outputs[:, class_idx],
            (labels == class_idx).astype(float)
        )
        single_corr_results.append({
            'Class_Index': class_idx,
            'Class_Name': class_names[class_idx],
            'Correlation': round(corr, 4)  # 关键修改：添加round
        })

    # 保存单变量相关性结果
    single_corr_df = pd.DataFrame(single_corr_results)
    single_corr_df.to_csv(
        os.path.join(save_path, 'output_label_correlation.csv'),
        index=False,
        float_format='%.4f'  # 关键修改：统一输出格式
    )

    # ============================================
    # 2. 计算互相关矩阵（新增功能）
    # ============================================
    corr_matrix = np.zeros((num_classes, num_classes))

    # 预先计算所有真实类别的one-hot编码
    true_labels_onehot = np.zeros((len(labels), num_classes))
    for i in range(num_classes):
        true_labels_onehot[:, i] = (labels == i).astype(float)

    # 计算互相关矩阵
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            corr, _ = pearsonr(
                outputs[:, pred_class],
                true_labels_onehot[:, true_class]
            )
            # 保留4位小数
            corr_matrix[true_class, pred_class] = round(corr, 4)

    # 创建带标签的DataFrame
    matrix_df = pd.DataFrame(
        corr_matrix,
        index=[f"True_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names]
    )

    # 保存互相关矩阵
    matrix_df.to_csv(
        os.path.join(save_path, 'label_prediction_correlation_matrix.csv'),
        float_format = '%.4f'  # 确保CSV文件中的数字格式
    )

    return {
        'single_correlation': single_corr_df,
        'correlation_matrix': matrix_df
    }


def test_evaluate_hotmap(args, model, dataloader, criterion, device):
    """
    原始测试评估函数（保持参数和返回值不变）
    新增功能：
    1. 计算每个类别的precision/recall/f1
    2. 将详细指标保存到TXT文件
    3. 保持原有相关性分析功能
    """
    model.eval()
    lossmeter = utils.AverageMeter("test_loss")
    accmeter = utils.AverageMeter("test_acc")

    # 原有收集器
    output_collector = [] if hasattr(args, 'save_correlation') else None
    label_collector = [] if hasattr(args, 'save_correlation') else None

    # 新增收集器（用于类别指标计算）
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)

            # 原有逻辑
            loss = criterion(output, y.long())
            acc = utils.accuracy(output, y)
            lossmeter.update(loss.item())
            accmeter.update(acc)

            # 原有数据收集
            if hasattr(args, 'save_correlation'):
                output_collector.append(output.cpu().numpy())
                label_collector.append(y.cpu().numpy())

            # 新增：收集预测结果和真实标签
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 原有相关性分析保存
    if hasattr(args, 'save_correlation') and hasattr(args, 'class_names'):
        _save_correlation_analysis(
            outputs=np.vstack(output_collector),
            labels=np.concatenate(label_collector),
            class_names=args.class_names,
            save_path=getattr(args, 'output_dir', '.')
        )

    # 新增：保存每个类别的详细指标
    if len(all_preds) > 0:
        # 计算每个类别的指标
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        # 准备写入内容
        content = []
        content.append("=== Per-Class Metrics ===")

        # 添加每个类别的指标
        for i in range(len(precision)):
            label_name = args.class_names[i] if hasattr(args, 'class_names') else f"Class {i}"
            content.append(f"\n[{label_name}]")
            content.append(f"Samples:    {support[i]}")
            content.append(f"Precision: {precision[i]:.4f}")
            content.append(f"Recall:    {recall[i]:.4f}")
            content.append(f"F1 Score:  {f1[i]:.4f}")

        # 写入TXT文件
        output_dir = getattr(args, 'output_dir', './')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'per_class_metrics.txt'), 'w') as f:
            f.write('\n'.join(content))

    return accmeter.avg, lossmeter.avg  # 严格保持原始返回值


# 测试模型的性能评估函数
def test_evaluate(args, model, dataloader, criterion, device):
    """
    在测试集或验证集上评估模型性能。

    Args:
        args: 包含超参数等设置信息的命令行参数。
        model: 要评估的模型。
        dataloader: 数据加载器，提供测试或验证数据。
        criterion: 损失函数，用于计算预测与真实标签之间的误差。
        device: 指定计算设备（CPU 或 GPU）。

    Returns:
        accmeter.avg: 测试集上的平均准确率。
        lossmeter.avg: 测试集上的平均损失值。
    """
    model.eval()  # 将模型设置为评估模式，禁用 Dropout 和 BatchNorm 等操作。
    lossmeter = utils.AverageMeter("test_loss")  # 创建用于记录测试损失的工具类。
    accmeter = utils.AverageMeter("test_acc")    # 创建用于记录测试准确率的工具类。

    # 禁用梯度计算，加速推理过程并减少内存占用。
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):  # 遍历测试集中的所有样本。
            x, y = x.to(device), y.to(device)   # 将数据移动到指定设备（CPU 或 GPU）。
            output = model(x)                   # 前向传播，获取模型输出。
            loss = criterion(output, y.long())  # 计算损失值。
            acc = utils.accuracy(output, y)     # 计算预测的准确率。

            lossmeter.update(loss.item())  # 更新损失值的记录器。
            accmeter.update(acc)           # 更新准确率的记录器。

    return accmeter.avg, lossmeter.avg  # 返回测试集的平均准确率和平均损失值。

# 训练过程中模型性能评估函数
def train_evaluate(args, model, dataloader, optimizer, criterion, device):
    """
    在训练集上评估模型性能，同时更新模型参数。

    Args:
        args: 包含超参数等设置信息的命令行参数。
        model: 要训练的模型。
        dataloader: 数据加载器，提供训练数据。
        optimizer: 优化器，用于更新模型参数。
        criterion: 损失函数，用于计算预测与真实标签之间的误差。
        device: 指定计算设备（CPU 或 GPU）。

    Returns:
        accmeter.avg: 训练集上的平均准确率。
        lossmeter.avg: 训练集上的平均损失值。
    """
    model.train()  # 将模型设置为训练模式，启用 Dropout 和 BatchNorm 等操作。
    lossmeter = utils.AverageMeter("train_loss")  # 创建用于记录训练损失的工具类。
    accmeter = utils.AverageMeter("train_acc")    # 创建用于记录训练准确率的工具类。

    # 使用 tqdm 显示训练进度条。
    with tqdm(total=len(dataloader), ncols=70, leave=False) as pbar:
        for i, (x, y) in enumerate(dataloader):  # 遍历训练集中的所有样本。
            x, y = x.to(device), y.to(device)   # 将数据移动到指定设备（CPU 或 GPU）。
            output = model(x)                   # 前向传播，获取模型输出。
            loss = criterion(output, y.long())  # 计算损失值。
            acc = utils.accuracy(output, y)     # 计算预测的准确率。

            optimizer.zero_grad()               # 清空优化器中存储的梯度。
            loss.backward()                     # 反向传播，计算梯度。
            optimizer.step()                    # 使用优化器更新模型参数。

            lossmeter.update(loss.item())  # 更新损失值的记录器。
            accmeter.update(acc)           # 更新准确率的记录器。

            pbar.update()  # 更新进度条。

    return accmeter.avg, lossmeter.avg  # 返回训练集的平均准确率和平均损失值。

def main_evaluate(args):
    """
    对 TFPred 模型进行半监督评估。

    参数:
    - args: 命令行参数，包含评估所需的配置信息。

    工作流程:
    1. 设置计算设备（GPU 或 CPU）。
    2. 加载评估集、验证集和测试集。
    3. 初始化模型，加载预训练的检查点，并调整分类器层。
    4. 设置损失函数、优化器和学习率调度器。
    5. 运行指定轮数的评估，跟踪最佳验证集准确率并记录结果。
    """

    # 使用 GPU 如果可用；否则使用 CPU
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    with open('evaluate-data.txt', 'w') as file:  # 'w'表示写入模式，会覆盖原有内容
        file.write('\n')

    # 加载数据：评估集、验证集和测试集
    _, evaluate_loader, val_loader, test_loader = load_data(args)

    # 初始化模型，类别数等于标签数
    classes = len(args.labels.split(","))
    model = ModelBase(dim=classes).to(device)

    # 加载预训练模型的检查点
    checkpoint = torch.load("./History/TFPred_checkpoint.pth", map_location="cpu")

    # 处理检查点中的键，保留编码器的权重并移除其他无关项
    for k in list(checkpoint.keys()):
        if k.startswith('encoderT'):
            # 去掉键的前缀
            checkpoint[k[len("encoderT."):]] = checkpoint[k]
        # 删除重命名或未使用的键
        del checkpoint[k]

    # 再次清理检查点，移除分类器相关的键
    for k in list(checkpoint.keys()):
        if k.startswith('fc'):
            del checkpoint[k]

    # 加载模型的状态字典，并验证是否有缺失键
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    assert missing_keys == ["fc.weight", "fc.bias"]  # 确保仅分类器的参数丢失

    # 初始化分类器层的权重和偏置
    model.fc.weight.data.normal_(mean=0.0, std=0.1)
    model.fc.bias.data.zero_()

    # 将分类器和其他模型参数分组，分别设置不同的学习率
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)

    # 定义交叉熵损失函数并移动到 GPU
    criterion = nn.CrossEntropyLoss().cuda()

    # 设置优化器，分类器和主干网络使用不同学习率
    param_groups = [
        dict(params=classifier_parameters, lr=args.classifier_lr),
        dict(params=model_parameters, lr=args.backbone_lr)
    ]
    optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=5e-4)

    # 使用余弦退火学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tune_max_epochs)

    # 初始化最佳验证集准确率
    best_acc = 0.0
    test_acc = 0.0

    # 开始记录评估过程
    logging.info(">>>>> TFPred 半监督评估开始 ...")

    # 遍历指定的最大训练轮数
    for epoch in range(args.tune_max_epochs):
        # 在评估集上进行训练
        train_acc, train_loss = train_evaluate(args, model, evaluate_loader, optimizer, criterion, device)

        # 在验证集上评估模型
        val_acc, val_loss = test_evaluate(args, model, val_loader, criterion, device)

        # 更新学习率
        lr_scheduler.step()

        # 如果验证集准确率高于当前最佳值，更新最佳准确率并测试模型
        if val_acc > best_acc:
            best_acc = val_acc
            test_acc, _ = test_evaluate(args, model, test_loader, criterion, device)

        # 记录当前轮次的结果
        logging.info(f"Epoch: {epoch+1}/{args.tune_max_epochs}, train loss: {train_loss:.4f}, "
                     f"train_acc: {train_acc:6.2f}%, val loss: {val_loss:.4f}, val_acc: {val_acc:6.2f}%")

        with open('evaluate-data.txt', 'a') as file:  # # 'a'表示追加模式
            file.write(f"{val_acc:6.2f}\n")  # 换行后追加新内容z

    # 输出最佳验证集准确率和测试集准确率
    logging.info(f"最佳验证集准确率: {best_acc:6.2f}%, 测试集准确率: {test_acc:6.2f}%")
    logging.info("=" * 15 + "TFPred 评估完成!" + "=" * 15)
    _, _ = test_evaluate_hotmap(args, model, test_loader, criterion, device)


def train(args, model, train_loader, criterion, optimizer, device):
    """
    定义模型的训练函数。

    参数:
    - args: 包含训练配置的参数对象。
    - model: 需要训练的模型。
    - train_loader: 数据加载器，提供训练数据。
    - criterion: 损失函数，用于计算损失值。
    - optimizer: 优化器，用于更新模型的参数。
    - device: 设备信息（CPU 或 GPU），用于加速计算。

    返回:
    - lossmeter.avg: 训练过程中损失的平均值。
    """
    # 设置模型为训练模式（启用 dropout 和 batchnorm 等训练特性）
    model.train()

    # 初始化损失值记录器，用于计算和存储损失的平均值
    lossmeter = utils.AverageMeter("train_loss")

    # 使用 tqdm 显示训练进度条
    with tqdm(total=len(train_loader), ncols=70, leave=False) as pbar:
        # 遍历训练数据加载器
        for i, (x_t, x_f, _) in enumerate(train_loader):#x_t为[256, 1, 1024]的三维数组，储存数据，x_f形状[256, 1, 512]，第三个返回值为[256]的一维数组，储存标签索引
            # 将数据移动到指定设备（CPU 或 GPU）
            x_t, x_f = x_t.to(device), x_f.to(device)

            # 前向传播，获取模型的输出
            x_t, x_f = model(x_t, x_f)

            # 计算损失值
            loss = criterion(x_t, x_f)

            # 清除优化器的梯度缓存
            optimizer.zero_grad()

            # 反向传播，计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # 更新损失记录器
            lossmeter.update(loss.item())

            # 更新进度条
            pbar.update()

    # 返回损失的平均值
    return lossmeter.avg


def main(args):
    # 选择设备：如果有GPU可用，则使用GPU，否则使用CPU
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用gpu比cpu更慢？，已检测cuda安装正确，cudnn也安装了
    device = torch.device("cpu")
    #if device==torch.device("cuda:0"):
        #print("available")
    #else:
        #print("not available")
    #print(torch.__version__)  # 查看 PyTorch 版本
    #print(torch.backends.cudnn.version())  # 查看 cuDNN 版本
    #print(torch.cuda.is_available())  # 检查 CUDA 是否可用

    with open('pre-trainloss.txt', 'w') as file:  # 'w'表示写入模式，会覆盖原有内容
        file.write('\n')

    # 加载数据，train_loader用于训练数据，其他部分可能是验证或测试数据（未使用）
    train_loader, _, _, _ = load_data(args)

    # 初始化模型，并将模型加载到指定设备上
    model = TFPrediction().to(device)

    # 定义优化器，使用随机梯度下降（SGD）优化器，学习率、动量和权重衰减参数从args中读取
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # 定义学习率调度器，使用余弦退火学习率调整方法
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)

    # 定义损失函数，CrossCorrelationLoss通常用于自监督学习
    criterion = CrossCorrelationLoss()

    # 开始训练日志记录
    logging.info(">>>>> TFPred Pre-training ...")
    best_loss = 1e9  # 初始化最优损失为一个很大的值

    # 开始训练循环
    for epoch in range(args.max_epochs):
        # 执行一个训练周期，返回当前周期的训练损失
        train_loss = train(args, model, train_loader, criterion, optimizer, device)

        # 更新学习率（如果设置了学习率调度器）
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 保存当前最优模型的参数
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "./History/TFPred_checkpoint.pth")

        # 记录当前周期的损失和学习率
        logging.info(f"Epoch: {epoch + 1:>3}/{args.max_epochs}, train_loss: {train_loss:.4f}, "
                     f"current lr: {lr_scheduler.get_last_lr()[0]:.6f}")

        with open('pre-trainloss.txt', 'a') as file:  # 'a'表示追加模式
            file.write(f"{train_loss:.4f}\n")  # 换行后追加新内容z

    # 训练完成的日志记录
    logging.info("=" * 15 + "TFPred Pre-training Done!" + "=" * 15)


if __name__ == "__main__":

    # 解析命令行参数
    args = parms.parse_args()

    # 如果"History"文件夹不存在，则创建，用于保存模型检查点
    if not os.path.exists("./History"):
        os.makedirs("./History")

    # 设置日志记录器，确保日志文件夹存在
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger("./logs/TFPred.log")

    # 记录所有命令行参数
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    # 根据模式执行相应的操作
    if args.mode == "train":
        # 如果模式为训练，则调用main函数进行预训练
        main(args)
    elif args.mode == "tune":
        # 如果模式为调参，则调用main_evaluate函数进行评估
        main_evaluate(args)
    elif args.mode == "train_then_tune":
        # 如果模式为先训练再调参，先调用main函数预训练，再调用main_evaluate函数评估
        main(args)
        main_evaluate(args)
