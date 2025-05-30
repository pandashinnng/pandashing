'''
Define hyperparameters.
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters')

    # 数据集相关参数
    parser.add_argument("--datadir", type=str, default="./paper_datasets", help="数据目录路径")
    parser.add_argument("--dataset", type=str, default="PU", choices=["PU"], help="数据集名称")
    parser.add_argument("--load", type=int, default=3, help="工况编号")
    parser.add_argument("--num_train", type=int, default=210, help="每个类别的训练样本数量")
    parser.add_argument("--num_validation", type=int, default=30, help="每个类别的验证样本数量")
    parser.add_argument("--num_test", type=int, default=60, help="每个类别的测试样本数量")
    parser.add_argument("--num_labels", type=int, default=3, help="每个类别有标签的样本数量")
    parser.add_argument("--ratio_labels", type=float, default=0.01, help="有标签样本的比例")
    parser.add_argument("--data_length", type=int, default=1024, help="每个样本的信号长度")
    parser.add_argument("--labels", type=str, default="0,1,2,3,4,5,6,7,8,9", help="训练的类别标签")

    # 数据预处理相关参数
    parser.add_argument("--window", type=int, default=512, help="时间窗口大小。如果不进行数据增强，window=1024")
    parser.add_argument("--normalization", type=str, default="0-1", choices=["0-1"], help="数据归一化方式")
    parser.add_argument("--backbone", type=str, default="ResNet1D", choices=["ResNet1D"], help="特征提取主干网络")

    # 训练相关参数
    parser.add_argument("--mode", type=str, default="train_then_tune", choices=["train", "tune", "train_then_tune", "evaluate"], help="训练模式")
    parser.add_argument("--max_epochs", type=int, default=500, help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批处理大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument('--lr_scheduler', type=str, default='stepLR', choices=['stepLR'], help='学习率调度器类型')
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载器的工作线程数量")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="优化器类型")

    # 调优相关参数
    parser.add_argument("--tune_max_epochs", type=int, default=100, help="线性评估的最大训练轮数")
    parser.add_argument("--backbone_lr", type=float, default=5e-3, help="主干网络的学习率")
    parser.add_argument("--classifier_lr", type=float, default=5e-3, help="分类器的学习率")

    # 超参数
    parser.add_argument('--gamma', type=float, default=0.8, help='学习率调度器的衰减参数（适用于step和exp模式）')
    parser.add_argument('--steps', type=int, default=60, help='学习率的衰减步长（适用于step和stepLR模式）')

    #热图相关参数
    parser.add_argument("--save_correlation",type=bool,default=True,help="启用相关性分析保存")
    parser.add_argument('--class_names',
                        type=lambda s: s.split(),  # 自动分割字符串
                        default=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                        help='类别名称列表（用空格分隔），例如："cat dog bird"')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./',
                        help='分析结果保存目录（默认：./results）')


    # # 数据集相关参数
    # parser.add_argument("--datadir", type=str, default="/home/datasets", help="数据目录路径")
    # parser.add_argument("--dataset", type=str, default="PU", choices=["PU"], help="数据集名称")
    # parser.add_argument("--load", type=int, default=0, help="工况编号")
    # parser.add_argument("--num_train", type=int, default=500, help="每个类别的训练样本数量")
    # parser.add_argument("--num_validation", type=int, default=100, help="每个类别的验证样本数量")
    # parser.add_argument("--num_test", type=int, default=100, help="每个类别的测试样本数量")
    # parser.add_argument("--num_labels", type=int, default=30, help="每个类别有标签的样本数量")
    # parser.add_argument("--ratio_labels", type=float, default=0.01, help="有标签样本的比例")
    # parser.add_argument("--data_length", type=int, default=512, help="每个样本的信号长度")
    # parser.add_argument("--labels", type=str, default="0,1,2,3,4,5,6,7,8,9", help="训练的类别标签")
    #
    # # 数据预处理相关参数
    # parser.add_argument("--window", type=int, default=384, help="时间窗口大小。如果不进行数据增强，window=1024")
    # parser.add_argument("--normalization", type=str, default="0-1", choices=["0-1"], help="数据归一化方式")
    # parser.add_argument("--backbone", type=str, default="ResNet1D", choices=["ResNet1D"], help="特征提取主干网络")
    #
    # # 训练相关参数
    # parser.add_argument("--mode", type=str, default="train", choices=["train", "tune", "train_then_tune", "evaluate"], help="训练模式")
    # parser.add_argument("--max_epochs", type=int, default=500, help="最大训练轮数")
    # parser.add_argument("--batch_size", type=int, default=128, help="批处理大小")
    # parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    # parser.add_argument('--lr_scheduler', type=str, default='stepLR', choices=['stepLR'], help='学习率调度器类型')
    # parser.add_argument("--num_workers", type=int, default=2, help="数据加载器的工作线程数量")
    # parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="优化器类型")
    #
    # # 调优相关参数
    # parser.add_argument("--tune_max_epochs", type=int, default=100, help="线性评估的最大训练轮数")
    # parser.add_argument("--backbone_lr", type=float, default=5e-3, help="主干网络的学习率")
    # parser.add_argument("--classifier_lr", type=float, default=0.1, help="分类器的学习率")
    #
    # # 超参数
    # parser.add_argument('--gamma', type=float, default=0.8, help='学习率调度器的衰减参数（适用于step和exp模式）')
    # parser.add_argument('--steps', type=int, default=60, help='学习率的衰减步长（适用于step和stepLR模式）')
    # python TFPred.py \
    # --datadir "dataset path"
    # --mode "train_then_tune" \
    # --load 3 \
    # --num_train 210 \
    # --num_validation 30 \
    # --num_test 60 \
    # --num_labels 3 \
    # --data_length 1024 \
    # --window 512 \
    # --max_epochs 500 \
    # --batch_size 256 \
    # --lr 1e-2 \
    # --normalization '0-1' \
    # --tune_max_epochs 100 \
    # --backbone_lr 5e-3 \
    # --classifier_lr 5e-3 \
    args = parser.parse_args()

    return args
