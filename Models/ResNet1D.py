# one-dimentional ResNet source code reference: https://github.com/ZhaoZhibin/UDTL/blob/master/models/resnet18_1d.py

import torch
import torch.nn as nn  # 引入PyTorch的神经网络模块
import torch.utils.model_zoo as model_zoo  # 用于从网络加载预训练模型

# 预训练模型的URL字典
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# 定义3x1卷积层
def conv3x1(in_planes, out_planes, stride=1):
    """3x3卷积（实际上是1D卷积），带有padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)  # 使用padding=1确保卷积输出的长度不变


# 定义1x1卷积层
def conv1x1(in_planes, out_planes, stride=1):
    """1x1卷积"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 定义ResNet的基本块 (BasicBlock)，用于 ResNet-18 或 ResNet-34
class BasicBlock(nn.Module):
    expansion = 1  # 基本块的扩展系数，表示输出通道数与输入通道数的倍数关系

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        """
        BasicBlock的初始化方法，包含卷积层、批归一化层和ReLU激活层。

        参数：
        - inplanes: 输入通道数
        - planes: 输出通道数
        - stride: 卷积步幅
        - downsample: 用于处理步幅变化的下采样模块（如果需要）
        - norm_layer: 归一化层，默认为BatchNorm1d
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d  # 默认使用批归一化

        # 第一个卷积层 + 批归一化 + 激活函数
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层 + 批归一化
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = norm_layer(planes)

        # 下采样和步幅
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入，后续与输出相加进行残差连接

        # 通过第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 通过第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果有下采样，改变输入的形状
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)  # 最后的激活

        return out  # 返回输出


# 定义ResNet的瓶颈结构 (Bottleneck)，用于 ResNet-50, ResNet-101, ResNet-152
class Bottleneck(nn.Module):
    expansion = 4  # 扩展系数，瓶颈结构的输出通道数是输入通道数的4倍

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        """
        Bottleneck的初始化方法，包含多个卷积层、批归一化和激活函数。

        参数：
        - inplanes: 输入通道数
        - planes: 基础通道数
        - stride: 卷积步幅
        - downsample: 用于处理步幅变化的下采样模块（如果需要）
        - norm_layer: 归一化层，默认为BatchNorm1d
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d  # 默认使用批归一化

        # 1x1卷积（将输入通道数降维）
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)

        # 3x1卷积（特征提取）
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = norm_layer(planes)

        # 1x1卷积（升维，增加通道数）
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

        # 下采样
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入，后续与输出相加进行残差连接

        # 通过第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 通过第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 通过第三个卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果有下采样，改变输入的形状
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)  # 最后的激活

        return out  # 返回输出

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        """
        初始化 CBAMLayer 模块。

        参数:
        - channel (int): 输入特征图的通道数。
        - reduction (int, optional): 用于通道注意力的缩减比例，默认值为 16。
        - spatial_kernel (int, optional): 空间注意力卷积的核大小，默认值为 7。
        """
        super(CBAMLayer, self).__init__()

        # Channel Attention: 通过全局池化压缩空间维度
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化，得到每个通道的最大值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化，得到每个通道的均值

        # 共享的 MLP 网络，通过卷积来模拟全连接层
        self.mlp = nn.Sequential(
            # 使用 1x1 卷积代替线性层进行通道压缩
            nn.Conv2d(channel, channel // reduction, 1, bias=False),  # 将通道数缩小
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数
            nn.Conv2d(channel // reduction, channel, 1, bias=False)  # 恢复到原来的通道数
        )

        # Spatial Attention: 空间注意力，通过最大池化和平均池化的特征图来获得空间权重
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)  # 使用卷积生成空间权重图
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，将输出值压缩到 [0, 1] 之间

    def forward(self, x):
        """
        前向传播函数，进行通道和空间注意力的加权计算。

        参数:
        - x (Tensor): 输入张量，形状为 [batch_size, channel, height, width]。

        返回:
        - x (Tensor): 经过通道和空间注意力加权后的输出张量。
        """
        # 计算通道注意力
        max_out = self.mlp(self.max_pool(x))  # 最大池化后经过 MLP
        avg_out = self.mlp(self.avg_pool(x))  # 平均池化后经过 MLP
        channel_out = self.sigmoid(max_out + avg_out)  # 通道注意力通过 sigmoid 激活

        # 将输入与通道注意力进行加权
        x = channel_out * x  # 通道注意力加权

        # 计算空间注意力
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 沿着通道维度求最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 沿着通道维度求平均值

        # 将最大池化和平均池化结果拼接，生成空间注意力
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))  # 空间注意力通过 sigmoid 激活

        # 将输入与空间注意力进行加权
        x = spatial_out * x  # 空间注意力加权

        return x  # 返回加权后的输出张量

class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

class ResNet(nn.Module):
    """
    1D版本的ResNet，用于处理一维数据，例如时间序列或语音信号。
    参考: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18
    """
    def __init__(self, block, layers, in_channel=1, zero_init_residual=False, norm_layer=None):
        """
        初始化ResNet模型。

        参数:
        - block: 使用的残差模块类型，可以是BasicBlock或Bottleneck。
        - layers: 每一层残差模块的数量列表，例如[2, 2, 2, 2]表示4层，每层2个模块。
        - in_channel: 输入数据的通道数，默认为1。
        - zero_init_residual: 是否对残差分支的最后一层BN进行零初始化。
        - norm_layer: 使用的归一化层类型，默认为nn.BatchNorm1d。
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        # 初始卷积层和参数设置
        self.inplanes = 64  # 初始通道数
        self.conv1 = nn.Conv1d(
            in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )  # 7x1卷积
        self.bn1 = self._norm_layer(self.inplanes)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化层

        # 定义残差模块的层
        self.layer1 = self._make_layer(block, 64, layers[0])  # 第一层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 第二层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 第三层
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 第四层

        # 自适应全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 输出固定为(512, 1)

        # #CBAM注意力模块
        # self.cbam1 = CBAMLayer(64)
        # self.cbam2 = CBAMLayer(128)
        # self.cbam3 = CBAMLayer(256)
        self.cbam4 = CBAMLayer(512)
        # self.alpha1 = nn.Parameter(torch.tensor(1.0))
        # self.alpha2 = nn.Parameter(torch.tensor(1.0))
        # self.alpha3 = nn.Parameter(torch.tensor(1.0))
        self.alpha4 = nn.Parameter(torch.tensor(1.0))

        # #CAnet模块
        # self.ca_block1 = CA_Block(channel=512, h=1, w=32)
        # self.ca_block2 = CA_Block(channel=512, h=1, w=16)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 可选：对最后的BN层进行零初始化
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # Bottleneck的第三个BN层
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # BasicBlock的第二个BN层

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建残差模块的层。

        参数:
        - block: 残差模块类型。
        - planes: 输出通道数。
        - blocks: 残差模块的数量。
        - stride: 第一个模块的步幅，默认为1。

        返回:
        - 一个由多个残差模块组成的nn.Sequential对象。
        """
        norm_layer = self._norm_layer
        downsample = None;

        # 如果输入与输出形状不同，需要使用downsample调整
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 第一块残差模块，可能包含下采样
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        # 其余残差模块
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        定义ResNet的前向传播。

        参数:
        - x: 输入张量，形状为(batch_size, in_channel, sequence_length)。

        返回:
        - 输出特征张量，形状为(batch_size, 512)。
        """
        #以下数据均为encoderT的数据，encoderF以此类推：[256, 1, 1024]
        # 初始卷积、批归一化和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self.layer1(x)#[256, 64, 256]-->[256, 64, 256]

        # x = x.unsqueeze(2)  # [256, 64, 256]-->[256, 64, 1, 256]
        # origin1 = x
        # # CBAM注意力机制
        # x = self.cbam1.forward(x) #[256, 64, 1, 256]-->[256, 64, 1, 256]
        # x = origin1 + self.alpha1 * x
        # x = x.squeeze(2)  # [256, 64, 1, 256]-->[256, 64, 256]

        x = self.layer2(x)#[256, 64, 256]-->[256, 128, 128]

        # x = x.unsqueeze(2)  # [256, 128, 128]-->[256, 128, 1, 128]
        # origin2 = x
        # # CBAM注意力机制
        # x = self.cbam2.forward(x)  # [256, 128, 1, 128]-->[256, 128, 1, 128]
        # x = origin2 + self.alpha2 * x
        # x = x.squeeze(2)  # [256, 128, 1, 128]-->[256, 128, 128]

        x = self.layer3(x)#[256, 128, 128]-->[256, 256, 64]

        # x = x.unsqueeze(2)  # [256, 256, 64]-->[256, 256, 1, 64]
        # origin3 = x
        # # CBAM注意力机制
        # x = self.cbam3.forward(x)  # [256, 256, 1, 64]-->[256, 256, 1, 64]
        # x = origin3 + self.alpha3 * x
        # x = x.squeeze(2)  # [256, 256, 1, 64]-->[256, 256, 64]

        x = self.layer4(x)#[256, 256, 64]-->[256, 512, 32]

        x = x.unsqueeze(2)  # [256, 512, 32]-->[256, 512, 1, 32]
        origin4 = x
         #CBAM注意力机制
        x = self.cbam4.forward(x)  # [256, 512, 1, 32]-->[256, 512, 1, 32]
        x = origin4 + self.alpha4 * x
        x = x.squeeze(2)  # [256, 512, 1, 32]-->[256, 512, 32]


        #x = x.unsqueeze(2) # 对于encoderT[256, 512, 32]-->[256, 512, 1, 32] 对于encoderF[256, 512, 16]-->[256, 512, 1, 16]
        # CAnet注意力机制
        #if x.shape[3]==32:
            #x = self.ca_block1.forward(x) #[256, 512, 1, 32]-->[256, 512, 1, 32]
        #elif x.shape[3]==16:
            #x = self.ca_block2.forward(x)  # [256, 512, 1, 16]-->[256, 512, 1, 16]
        #x = x.squeeze(2) # [256, 512, 1, 32]-->[256, 512, 32]

        # 平均池化和展平输出
        x = self.avgpool(x)#[256, 512, 32]-->[256, 512, 1]
        x = x.view(x.size(0), -1)#[256, 512, 1]-->[256, 512]

        return x


def resnet18(pretrained=False, **kwargs):
    """
    构造一个 ResNet-18 模型。

    参数:
    - pretrained (bool): 如果为 True，则加载一个在 ImageNet 数据集上预训练的模型权重。
    - kwargs: 其他可选参数，传递给 ResNet 类。

    返回:
    - model (ResNet): 构造的 ResNet-18 模型实例。
    """
    # 定义一个 ResNet-18 模型，使用 BasicBlock 作为基础块，并指定每个层的 block 数目。
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    # 如果指定加载预训练模型，则从指定的 URL 加载预训练权重。
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model


# 定义一个只提取特征的 ResNet18 网络，不包括最后的分类层
class resnet18_features(nn.Module):
    def __init__(self, pretrained=False):
        """
        初始化 resnet18_features 模块。

        参数:
        - pretrained (bool): 如果为 True，则加载预训练的 ResNet-18 模型。
        """
        super(resnet18_features, self).__init__()

        # 构造一个 ResNet-18 模型，并根据参数决定是否加载预训练权重。
        self.model_resnet18 = resnet18(pretrained)

        # 定义特征的输出维度为 512，这是 ResNet 的最后一个卷积层输出的通道数。
        self.__in_features = 512

    def forward(self, x):
        """
        前向传播过程。

        参数:
        - x (Tensor): 输入张量，形状为 [batch_size, channels, sequence_length]。

        返回:
        - x (Tensor): 提取的特征，形状为 [batch_size, 512]。
        """
        # 使用 ResNet-18 提取特征。
        x = self.model_resnet18(x)
        return x

    def output_num(self):
        """
        返回模型的输出特征的维度。

        返回:
        - int: 输出特征维度，固定为 512。
        """
        return self.__in_features
