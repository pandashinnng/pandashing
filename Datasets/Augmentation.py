'''
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
'''

import numpy as np
import torch
import torchaudio
from typing import Optional

def probability():
    '''
    return a probability
    '''
    return np.random.random()


def FFT(signal: torch.tensor):
    '''
    对输入的波形信号进行快速傅里叶变换（FFT），并返回处理后的频谱信号。

    参数:
    - signal (torch.tensor): 输入的二维张量，表示波形信号。
                              每一行对应一个样本的时间序列。

    返回:
    - torch.tensor: 处理后的频谱信号，每一行对应一个样本的频域表示。
    '''

    # 确保输入信号为二维张量（batch_size, sequence_length）
    assert len(signal.size()) == 2

    # 对信号进行快速傅里叶变换 (FFT)
    signal = torch.fft.fft(signal)

    # 计算频谱幅值，归一化幅值范围，并乘以 2
    # (公式: 2 * |FFT信号| / 信号长度)
    signal = 2 * torch.abs(signal) / len(signal)

    # 仅保留频谱的前半部分（真实频率部分）
    signal = signal[:, :int(signal.size(-1) / 2)]

    # 将直流分量（频率为 0 的分量）置为 0，避免直流偏移的影响
    signal[:, 0] = 0.

    return signal


def ToTensor(signal):
    '''
    Numpy to Tensor
    '''
    return torch.from_numpy(signal).float()



def AddGaussianSNR(signal: torch.tensor):
    """
        为输入信号添加高斯噪声，模拟信噪比 (Signal-to-Noise Ratio, SNR) 范围内的噪声污染。

        参数:
        - signal (torch.tensor): 输入的信号张量，一维或二维。

        返回:
        - torch.tensor: 添加了高斯噪声的信号，形状与输入信号一致。
        """
    # 最小和最大信噪比（以分贝为单位）
    min_snr = 3  # (int): 最小信噪比，单位 dB
    max_snr = 30  # (int): 最大信噪比，单位 dB

    # 获取信号长度
    signal_length = signal.size(-1)

    # 获取信号的设备类型（CPU/GPU），确保返回的噪声在同一设备上
    device = signal.device

    # 在[min_snr, max_snr]范围内随机选择一个信噪比值
    snr = np.random.randint(min_snr, max_snr, dtype=int)

    # 计算信号的均方根值（RMS）
    clear_rms = torch.sqrt(torch.mean(torch.square(signal)))

    # 根据信噪比计算噪声的 RMS
    # 公式: noise_rms = clear_rms / (10^(snr/20))
    a = float(snr) / 20  # 将信噪比从 dB 转换为线性比例
    noise_rms = clear_rms / (10 ** a)

    # 根据噪声 RMS 生成高斯噪声
    noise = torch.normal(0.0, noise_rms, size=(signal_length,))

    # 返回加噪后的信号，确保噪声张量与信号张量位于相同设备
    return signal + noise.to(device)


def Shift(signal: torch.tensor):
    """
    随机平移信号，用于数据增强。

    参数:
    - signal (torch.tensor): 输入的信号张量，一维或二维。

    返回:
    - torch.tensor: 平移后的信号，形状与输入信号一致。
    """
    # 随机生成平移因子，范围在 [0, 0.3] 之间
    shift_factor = np.random.uniform(0, 0.3)

    # 根据平移因子计算最大平移步数
    # 平移步数 = 信号长度 × 平移因子
    shift_length = round(signal.size(-1) * shift_factor)

    # 在 [-shift_length, shift_length] 范围内随机选择平移步数
    num_places_to_shift = round(np.random.uniform(-shift_length, shift_length))

    # 使用 torch.roll 进行信号的循环平移
    # dims=-1 表示在最后一个维度上进行平移
    shifted_signal = torch.roll(signal, num_places_to_shift, dims=-1)

    return shifted_signal


def TimeMask(signal: torch.tensor):
    signal_length = signal.size(-1)
    device = signal.device
    signal_copy = signal.clone()
    fill_value = 0.

    mask_factor = np.random.uniform(0.1,0.45)
    max_mask_length = int(signal_length * mask_factor)  # maximum mask band length
    mask_length = np.random.randint(max_mask_length) # randomly choose a mask band length
    mask_start = np.random.randint(0, signal_length) # randomly choose a mask band start point

    while mask_start + mask_length > max_mask_length:
        mask_start = np.random.randint(0, signal_length)
    mask = torch.arange(signal_length, device=device)
    mask = (mask >= mask_start) & (mask < mask_start + mask_length)
    
    signal_copy = signal_copy.masked_fill(mask, fill_value)
    
    return signal_copy

def Fade(signal: torch.tensor):
    signal_length = signal.size(-1)
    fade_in_length = int(0.3 * signal_length)
    fade_out_length = int(0.3 * signal_length)

    device = signal.device
    signal_copy = signal.clone()

    # fade in
    if probability() > 0.5:
        fade_in = torch.linspace(0,1,fade_in_length,device=device)
        fade_in = torch.log10(0.1 + fade_in) + 1 # logarithmic
        ones = torch.ones(signal_length - fade_in_length, device=device)
        fade_in_mask = torch.cat((fade_in, ones))
        signal_copy *= fade_in_mask
    
    # fade out
    if probability() > 0.5:
        fade_out = torch.linspace(1,0,fade_out_length,device=device)
        fade_out = torch.log10(0.1 + fade_out) + 1 # logarithmic
        ones = torch.ones(signal_length - fade_out_length, device=device)
        fade_out_mask = torch.cat((ones, fade_out))
        signal_copy *= fade_out_mask
    
    return signal_copy

def Gain(signal: torch.tensor):
    gain_min = 0.5
    gain_max = 1.5
    gain_factor = np.random.uniform(gain_min, gain_max)

    signal_copy = signal.clone()

    return signal_copy * gain_factor

def Flip(signal: torch.tensor):
    if len(signal.size()) == 1:
        signal_copy = torch.flip(signal, dims=[0])
    elif len(signal.size()) == 2:
        signal_copy = torch.flip(signal, dims=[1])
    else:
        raise Exception(f"{signal.size()} dimentinal signal time masking is not implemented, "
                        "please try 1 or 2 dimentional signal.")
    
    return signal_copy

def PolarityInversion(signal: torch.tensor):
    '''
    Multiply the signal by -1.
    '''
    signal_copy = signal.clone()
    return - signal_copy


def random_waveform_transforms(signal: torch.tensor):
    '''
    随机波形信号变换
    对输入的波形信号应用各种随机增强操作，每种操作具有预定义的概率。
    '''

    # 以 80% 的概率添加高斯噪声
    # 在信号中添加高斯噪声，模拟真实环境中的噪声干扰。
    if probability() > 0.2:
        signal = AddGaussianSNR(signal)

    # 以 30% 的概率对信号进行时序偏移
    # 将信号在时间轴上移动，模拟时间偏差或数据对齐误差。
    if probability() > 0.7:
        signal = Shift(signal)

    # 以 50% 的概率对信号进行时间掩蔽
    # 随机遮盖信号的一部分，增强模型的抗时序噪声能力。
    if probability() > 0.5:
        signal = TimeMask(signal)

    # 以 50% 的概率对信号进行渐变处理
    # 使信号的振幅逐渐增强或减弱，模拟信号强度的自然变化。
    if probability() > 0.5:
        signal = Fade(signal)

    # 以 30% 的概率对信号进行增益调整
    # 增加或减少信号的整体振幅，模拟不同的录音强度。
    if probability() > 0.7:
        signal = Gain(signal)

    # 以 20% 的概率对信号进行水平翻转
    # 水平翻转信号，通常用于音频波形增强。
    if probability() > 0.8:
        signal = Flip(signal)

    # 以 20% 的概率对信号进行极性翻转
    # 将信号的振幅上下颠倒，模拟信号反相的情况。
    if probability() > 0.8:
        signal = PolarityInversion(signal)

    return signal


def frequency_transforms(signal: torch.tensor):
    '''
    NOTE: input waveform signal.
    '''
    signal = FFT(signal)
    
    return signal