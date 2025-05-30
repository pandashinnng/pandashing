import numpy as np
from scipy.io import loadmat

RDBdata = ['K004','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI14','KI16','KI17','KI18','KI21']

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
#state = WC[0] #WC[0] can be changed to different working conditions
#（1）转速1500rpm，负载扭矩0.7Nm，径向力1000 N（代号WC1）;
#（2）转速900rpm，负载扭矩0.7Nm，径向力1000N（代号WC2）;
#（3） 转速1500rpm，负载扭矩0.1Nm，径向力1000N（代号WC3）;
#（4） 转速1500rpm，负载扭矩0.7Nm，径向力400N（代号WC4）

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

def read_file(path, filename):
    # 使用loadmat函数加载MAT文件，path是文件路径
    # 从加载的MAT文件中提取指定键（filename）对应的数据
    # 数据的结构层次为 [0][0][2][0][6][2]，这取决于MAT文件的具体存储格式
    data = loadmat(path)[filename][0][0][2][0][6][2]
    # 将提取的数据展开为一维数组
    return data.reshape(-1,)


'''def read_file(path, filename):
    """
    从 MAT 文件中读取数据，并逐层打印索引链 [0][0][2][0][6][2] 的内容。
    参数：
        path: MAT 文件路径。
        filename: MAT 文件中数据的键名。
    返回：
        提取的数据（展开为一维数组）。
    """
    # 加载 MAT 文件
    mat_data = loadmat(path)
    print(f"Loaded MAT file. Top-level keys: {list(mat_data.keys())}")

    # 提取指定键的数据
    data = mat_data[filename]
    print(f"\n=== Initial data for key '{filename}' ===")
    print(f"Type: {type(data)}, Shape: {data.shape}, dtype: {data.dtype}")

    # 逐层解析索引链
    indices = [0, 0, 2, 0, 6, 2]
    current_data = data
    for i, idx in enumerate(indices):
        try:
            # 进入下一层
            next_data = current_data[idx]

            # 打印当前层信息
            print(f"\n=== Layer {i + 1}: Index [{idx}] ===")
            print(f"Type: {type(next_data)}")
            if isinstance(next_data, np.ndarray):
                print(f"Shape: {next_data.shape}")
                print(f"dtype: {next_data.dtype}")
                if next_data.dtype.names:
                    print(f"Structured fields: {next_data.dtype.names}")
            elif isinstance(next_data, (np.generic, int, float, str)):
                print(f"Value: {next_data}")
            else:
                print(f"Value: {next_data}")

            # 更新当前数据
            current_data = next_data
        except (IndexError, KeyError, TypeError) as e:
            print(f"\n[ERROR] Failed at layer {i + 1} (index [{idx}]): {str(e)}")
            print("Possible reasons:")
            print("- Index out of bounds")
            print("- Trying to index a non-iterable object")
            print("- MATLAB struct field accessed by number instead of name")
            break

    # 返回最终数据（展开为一维数组）
    return current_data.reshape(-1, )'''

def PU(datadir, load, data_length, labels, window, normalization, backbone, number):
    """
    加载并处理 PU 数据集。

    参数：
    - datadir: 数据目录路径。
    - load: 数据加载状态的标识符，用于决定加载的文件属于哪个工况：WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
    - data_length: 每个样本的信号长度
    - labels: 标签集合，用于指定加载的类别。
    - window: 时间窗口大小。如果不进行数据增强，window=1024
    - normalization: 数据归一化方式（如 "0-1" 或其他方法）。
    - backbone: 主干特征提取方法，用于对数据进行特定的变换（如 ResNet1D）。特征提取主干网络
    - number: 所需的数据总量：训练集、验证集和测试集的总和

    返回：
    - dataset: 处理后的数据集，字典形式，键为标签，值为对应标签的数据数组。
    """
    # 数据存储路径
    path = datadir + "/"

    # 根据加载标识符，从 WC 字典中获取对应工况（如 "N15_M07_F04"）
    state = WC[load]

    # 初始化数据集字典，键为标签，值为空列表；{0：[],1:[],2:[],.......}
    dataset = {label: [] for label in labels}

    # 遍历标签集合，逐个加载和处理每个类别的数据
    for label in labels:
        # 根据标签和实验状态构建文件路径和文件名
        filename = state + '_' + RDBdata[label] + '_' + '1'
        subset_path = path + RDBdata[label] + '/' + filename + '.mat'

        # 从 MAT 文件中读取数据
        mat_data = read_file(subset_path, filename)

        # 对读取的数据进行归一化处理
        mat_data = _normalization(mat_data, normalization)

        # 初始化滑动窗口的起点和终点
        start, end = 0, data_length#输入参数，正常是1024

        # 获取数据长度并计算滑动窗口的终止位置
        length = mat_data.shape[0]
        endpoint = data_length + number * window#1024+300*512

        # 如果终止位置超出数据长度，抛出异常提示
        if endpoint > length:
            raise Exception(f"Sample number {number} exceeds signal length.")

        # 使用滑动窗口分割数据并进行变换
        while end < endpoint:
            # 提取窗口内的数据段
            sub_data = mat_data[start:end].reshape(-1,)#切出一片1024个数据，并将其转换为一个1*1024的行向量

            # 对分割的数据进行变换（根据指定的骨干网络类型， ResNet1D）
            sub_data = _transformation(sub_data, backbone)#将（1024）-->(1,1024)变为二维

            # 将处理后的数据段添加到对应标签的数据集中
            dataset[label].append(sub_data)

            # 滑动窗口向前移动
            start += window#步长512，第一次取0-1024，第二次取512-1536
            end += window


        # 将每个标签的数据集转换为 numpy 数组，并设置数据类型为 float32
        dataset[label] = np.array(dataset[label], dtype="float32")#处理后dataset[label]中存有number片1*1024的数据，即（300，1，1024）
    # 返回处理后的数据集
    return dataset#每个dataset[label]都是一个（300，1，1024）的数据


def PUloader(args):
    # 将args.labels（一个逗号分隔的字符串）解析为整数列表
    # 用于指定需要加载的数据的标签集合
    label_set_list = list(int(i) for i in args.labels.split(","))#0.1.2.3.4.5.6.7.8.9

    # 计算所需的数据总量：训练集、验证集和测试集的总和
    num_data = args.num_train + args.num_validation + args.num_test

    # 创建PU类的实例，加载数据集
    # 参数包括数据目录、是否加载已有数据、数据长度、标签集合、窗口大小、是否归一化、特征提取骨干网络和数据总量
    dataset = PU(
        args.datadir,        # 数据存储路径
        args.load,           # 一个数字，判断从哪个工况中选择数据集
        args.data_length,    # 每个样本的信号长度
        label_set_list,      # 训练的类别标签
        args.window,         # 窗口大小，用于切片或特征提取
        args.normalization,  # 是否对数据进行归一化处理
        args.backbone,       # 特征提取骨干网络的名称或类型
        num_data             # 总的数据量
    )
    # 处理后每个label对应的dataset[label]都中存有num_data 片1*1024的原始数据，即共10*（300 ，1，1024）（label默认值0-9的整数）
    # 返回加载的数据集对象
    return dataset
