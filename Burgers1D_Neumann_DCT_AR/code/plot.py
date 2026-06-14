import pandas as pd
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker
import torch
import math
import os
from matplotlib.ticker import AutoMinorLocator


def calculate_fig_size(Aspect_Ratio, word_width=210, word_margins=25.4, ):
    '''
    计算出图时图的大小
    :param Aspect_Ratio: 目标图的横纵比
    :param word_width: 文档宽度
    :param word_margins: 文档横向页边距
    :return:
    '''
    fig_width = (word_width - 2 * word_margins) / 25.4
    fig_lenth = fig_width / Aspect_Ratio
    return [fig_width, fig_lenth]


def round_down_auto(number):
    if number == 0:
        return 0
    decimals = -int(math.floor(math.log10(abs(number))))
    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def ewma(data, alpha=0.5):
    return pd.Series(data).ewm(alpha=alpha).mean().to_numpy()


def plot_history(loss_list, test_loss_list, alpha):
    """
    绘制训练和测试的损失曲线。
    """
    # 设置绘图的全局字体
    plt.rcParams['font.family'] = 'DejaVu Serif'
    x_axis = loss_list[:, 0:1].squeeze()
    Aspect_Ratio = float(1 / 1)
    fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0] / 2, calculate_fig_size(Aspect_Ratio)[1] / 2))
    fontdict = {'fontsize': 6, 'fontweight': 'normal', 'fontname': 'DejaVu Serif'}

    loss_data = loss_list[:, 4:5].squeeze()
    loss_f = loss_list[:, 3:4].squeeze()
    loss_init = loss_list[:, 2:3].squeeze()
    loss_all = loss_data + loss_f + loss_init

    test_loss_data = test_loss_list[:, 4:5].squeeze()

    # 绘制训练数据
    ax.semilogy(x_axis, ewma(loss_init, alpha), linestyle='-.',
                color=(254 / 255, 183 / 255, 5 / 255, 0.8),
                linewidth=0.8, label='I.C.Loss')
    ax.semilogy(x_axis, ewma(loss_data, alpha), linestyle='-.',
                color=(19 / 255, 103 / 255, 158 / 255, 0.8),
                linewidth=0.8, label='Data.Loss')
    ax.semilogy(x_axis, ewma(loss_f, alpha), linestyle='-.',
                color=(42 / 255, 157 / 255, 142 / 255, 0.8),
                linewidth=0.8, label='E.Loss')
    ax.semilogy(x_axis, ewma(loss_all, alpha), linestyle='-',
                color=(239 / 255, 65 / 255, 67 / 255, 0.9),
                linewidth=0.8, label='Total Loss')

    ax.set_xlabel('Epochs', fontdict=fontdict, labelpad=4)
    ax.set_ylabel('MSE', fontdict=fontdict, labelpad=4)
    ax.set_xlim(0, x_axis[-1])
    ax.legend(fontsize=6, frameon=False, markerscale=3, loc='upper left')
    ax.tick_params(axis='x', labelsize=6, direction='in', length=2)
    ax.tick_params(axis='y', labelsize=6, direction='in', length=2)
    ax.locator_params(axis='x', nbins=20)

    # 创建副 y 轴
    ax2 = ax.twinx()
    ax2.semilogy(x_axis, ewma(test_loss_data, alpha), linestyle='-',
                 color=(32 / 255, 152 / 255, 58 / 255, 0.9),
                 linewidth=0.8, label='Test data Loss')
    ax2.set_ylabel(r'Mean relative $L_2$ error', fontdict=fontdict, labelpad=4)
    ax2.legend(fontsize=6, frameon=False, markerscale=3, loc='upper right')

    # 设置边框的宽度
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='x', labelsize=6, direction='in', length=1, which='minor')
    ax.tick_params(axis='y', labelsize=6, direction='in', length=1, which='minor')
    ax.minorticks_on()

    save_path = 'history.png'
    fig.savefig(save_path, bbox_inches='tight', transparent=True, dpi=400)
    plt.close()


def plot_solutions(data, i):
    '''
    绘制数据的热图表示

    :param data: ndarray, 输入数据，形状为 [nt, nx]
    '''

    plt.rcParams['font.family'] = 'DejaVu Serif'

    # 计算图的宽高比
    Aspect_Ratio = 1 / 1.7

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0] / 2, calculate_fig_size(Aspect_Ratio)[1] / 2))

    # 禁用轴刻度和标签
    ax.axis('off')

    # 获取数据的最大值以设定 vmax
    max_error = data.max()

    # 绘制数据的热图
    h = ax.imshow(data.T, interpolation='nearest',
                  extent=[0, 1, 0, 1],
                  origin='lower', aspect='auto', vmin=0, vmax=3)

    # 添加色标以帮助理解数据值
    cbar = plt.colorbar(h, ax=ax, orientation='vertical')
    cbar.set_label('Value')

    # 保存图形
    plt.savefig(f'test_residual_{i}.png', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
