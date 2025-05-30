import numpy as np
from scipy.io import loadmat

#每个工况有10种数据。如下：
#正常0hp 为一类
#0.007   0hp   内圈   滚珠  外圈
#0.014   0hp   内圈   滚珠  外圈
#0.021   0hp   内圈   滚珠  外圈
RDBdata = ['97.mat','105.mat','118.mat','130.mat','169.mat','185.mat','197.mat','209.mat','222.mat','234.mat']
midpathname = ['Normal Baseline Data','12k Drive End Bearing Fault Data','12k Drive End Bearing Fault Data','12k Drive End Bearing Fault Data',
               '12k Drive End Bearing Fault Data','12k Drive End Bearing Fault Data','12k Drive End Bearing Fault Data','12k Drive End Bearing Fault Data',
               '12k Drive End Bearing Fault Data','12k Drive End Bearing Fault Data']
# 采用驱动端数据
data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                'X185_DE_time','X197_DE_time','X209_DE_time','X222_DE_time','X234_DE_time']
columns_name = ['de_normal','de_7_inner','de_7_ball','de_7_outer','de_14_inner','de_14_ball','de_14_outer','de_21_inner','de_21_ball','de_21_outer']

def _normalization(data, normalization):
    # 检查是否指定归一化方法为 "0-1"
    if normalization == "0-1":
        # 将数据归一化到 [0, 1] 范围：
        # 减去数据的最小值，将数据范围平移到以 0 为起点
        # 然后除以 (最大值 - 最小值)，将数据压缩到 [0, 1] 区间
        data = (data - data.min()) / (data.max() - data.min())
    return data

def _transformation(sub_data, backbone):
    # 检查指定的特征提取骨干网络类型
    if backbone in ("ResNet1D"):
        # 如果使用 ResNet1D 模型：
        # 在数据的第一个维度上增加一个新轴，将数据的形状从 (N,) 转换为 (1, N)
        # 这是为了符合 ResNet1D 的输入要求（通常需要增加通道维度）
        sub_data = sub_data[np.newaxis, :]
    else:
        # 如果指定的 backbone 类型未实现，抛出 NotImplementedError 异常
        # 提示用户指定的模型未实现
        raise NotImplementedError(f"Model {backbone} is not implemented.")

    # 返回经过变换的数据
    return sub_data

def read_file(path, label):

    data = loadmat(path)
    dataList = data[data_columns[label]].reshape(-1)

    return dataList[:119808]

def PU2(datadir, load, data_length, labels, window, normalization, backbone, number):
    # 数据存储路径
    path = datadir + "/" + 'cwru' + '/'

    # 初始化数据集字典，键为标签，值为空列表；{0：[],1:[],2:[],.......}
    dataset = {label: [] for label in labels}

    for label in labels:
        filename = RDBdata[label]
        subset_path = path + midpathname[label] + '/' + filename

        # 从 MAT 文件中读取数据
        mat_data = read_file(subset_path, label)

        # 对读取的数据进行归一化处理
        mat_data = _normalization(mat_data, normalization)

        # 初始化滑动窗口的起点和终点
        start, end = 0, data_length  # 输入参数，正常是1024

        # 获取数据长度并计算滑动窗口的终止位置
        length = mat_data.shape[0]
        endpoint = data_length + number * window  # 1024+200*512

        # 如果终止位置超出数据长度，抛出异常提示
        if endpoint > length:
            raise Exception(f"Sample number {number} exceeds signal length.")

        # 使用滑动窗口分割数据并进行变换
        while end < endpoint:
            # 提取窗口内的数据段
            sub_data = mat_data[start:end].reshape(-1, )  # 切出一片1024个数据，并将其转换为一个1*1024的行向量

            # 对分割的数据进行变换（根据指定的骨干网络类型， ResNet1D）
            sub_data = _transformation(sub_data, backbone)  # 将（1024）-->(1,1024)变为二维

            # 将处理后的数据段添加到对应标签的数据集中
            dataset[label].append(sub_data)

            # 滑动窗口向前移动
            start += window  # 步长512，第一次取0-1024，第二次取512-1536
            end += window

        # 将每个标签的数据集转换为 numpy 数组，并设置数据类型为 float32
        dataset[label] = np.array(dataset[label], dtype="float32")  # 处理后dataset[label]中存有number片1*1024的数据，即（200，1，1024）
    # 返回处理后的数据集
    return dataset  # 每个dataset[label]都是一个（200，1，1024）的数据


def PU2loader(args):
    # 将args.labels（一个逗号分隔的字符串）解析为整数列表
    # 用于指定需要加载的数据的标签集合
    label_set_list = list(int(i) for i in args.labels.split(","))#0.1.2.3.4.5.6.7.8.9

    # 计算所需的数据总量：训练集、验证集和测试集的总和
    num_data = args.num_train + args.num_validation + args.num_test#140+20+40

    # 创建PU类的实例，加载数据集
    # 参数包括数据目录、是否加载已有数据、数据长度、标签集合、窗口大小、是否归一化、特征提取骨干网络和数据总量
    dataset = PU2(
        args.datadir,        # 数据存储路径
        args.load,           # 一个数字，判断从哪个工况中选择数据集
        args.data_length,    # 每个样本的信号长度
        label_set_list,      # 训练的类别标签
        args.window,         # 窗口大小，用于切片或特征提取
        args.normalization,  # 是否对数据进行归一化处理
        args.backbone,       # 特征提取骨干网络的名称或类型
        num_data             # 总的数据量
    )
    # 处理后每个label对应的dataset[label]都中存有num_data 片1*1024的原始数据，即共10*（200 ，1，1024）（label默认值0-9的整数）
    # 返回加载的数据集对象
    return dataset