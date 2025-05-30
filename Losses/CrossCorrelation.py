import torch
import torch.nn as nn

class CrossCorrelationLoss(nn.Module):
    """
    定义一个用于计算特征之间的交叉相关性损失的损失函数。

    主要思想:
    - 通过交叉相关矩阵来评估特征的去冗余性。
    - 希望交叉相关矩阵的对角线值接近 1，非对角线值接近 0。
    """
    def __init__(self, lambda_param=5e-3):
        """
        初始化损失函数。

        参数:
        - lambda_param (float): 用于控制非对角线惩罚项的权重。
        """
        super(CrossCorrelationLoss, self).__init__()
        self.lambda_param = lambda_param  # 非对角线元素的惩罚系数

    def forward(self, x1, x2):
        """
        前向传播，计算交叉相关性损失。

        参数:
        - x1 (torch.Tensor): 输入特征 1，形状为 [N, D]。
        - x2 (torch.Tensor): 输入特征 2，形状为 [N, D]。

        返回:
        - loss (torch.Tensor): 计算得到的交叉相关性损失标量。
        """
        device = x1.device  # 获取张量所在的设备

        # 检查输入特征的形状是否一致
        x1_shape, x2_shape = x1.size(), x2.size()
        assert x1_shape == x2_shape, "输入特征的形状必须一致"
        N, D = x1.shape  # N 为样本数，D 为特征维度

        # 特征归一化，确保均值为 0，标准差为 1
        x1 = (x1 - x1.mean(0)) / x1.std(0)
        x2 = (x2 - x2.mean(0)) / x2.std(0)

        # 计算交叉相关矩阵，c 的形状为 [D, D]
        c = torch.mm(x1.T, x2) / N

        # 损失计算
        # 对角线元素应接近 1，非对角线元素应接近 0
        c_diff = (c - torch.eye(D, device=device)).pow(2)  # 差的平方
        '''
        torch.eye(D, device=device)：创建一个形状为 [D, D] 的单位矩阵。单位矩阵的对角线元素为 1，其他元素为 0。
        c - torch.eye(D, device=device)：将交叉相关矩阵 c 与单位矩阵做差，得到每个元素的差值。这个差值的目标是：对角线元素接近 1，非对角线元素接近 0，因此差值应该尽量小。
        .pow(2)：对差值进行平方操作，目的是将差异放大并突出较大的差异。
        '''

        # 对非对角线元素施加额外惩罚
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        '''
        ~torch.eye(D, dtype=bool)：创建一个布尔矩阵，非对角线元素为 True，对角线元素为 False。
        c_diff[~torch.eye(D, dtype=bool)]：选择 c_diff 中的非对角线元素（即那些差值不在对角线上的元素）。
        *= self.lambda_param：对非对角线的差值施加额外的惩罚，乘以一个超参数 lambda_param，控制对非对角线元素差异的惩罚强度。
        '''

        # 求和得到最终的损失值
        loss = c_diff.sum()

        return loss
