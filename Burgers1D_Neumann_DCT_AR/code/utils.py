import math
from collections import deque
from pathlib import Path

import scipy.io
import numpy as np
import h5py
import torch
import random
import os
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from matplotlib import pyplot as plt, ticker, gridspec
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
import re


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self.h5 = False
        self._load_file()

    def _load_file(self):

        if self.file_path[-3:] == '.h5':
            self.data = h5py.File(self.file_path, 'r')
            self.h5 = True

        else:
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except:
                self.data = h5py.File(self.file_path, 'r')
                self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if self.h5:
            x = x[()]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


def FC1d(u, order=5):
    if not 1 <= order <= 5:
        raise ValueError(f"`order` must be between 1 and 5; got {order}")

    AQ1 = MatReader(f"FC_data/AlQl_d{order}_C_25.mat", to_cuda=True).read_field('AlQl').double()
    AQ2 = MatReader(f"FC_data/ArQr_d{order}_C_25.mat", to_cuda=True).read_field('ArQr').double()

    u1 = torch.einsum("xy,bcy->bcx", AQ1, u[..., :order])
    u2 = torch.einsum("xy,bcy->bcx", AQ2, u[..., -order:])
    return torch.cat([u, u1 + u2], dim=-1)


def save_and_record(num_epochs, optimizer, scheduler, model, filename='EndModel', ):
    state = {
        'epoch': num_epochs,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, filename + '.pth.tar')
    torch.save(model, filename + '.pkl')


def save_checkpoint(model, LogIter, optimizer, scheduler, loss_list, test_loss_list, lr_list, model_save_record,
                    grad, filename='checkpoint'):
    state = {
        'epoch': LogIter,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_list': loss_list,
        'test_loss_list': test_loss_list,
        'lr_list': lr_list,
        'model_save_record': model_save_record,
        'grad': grad
    }
    torch.save(state, filename + '.pth.tar')
    if filename[-4:] == 'best':
        torch.save(model, filename + '.pkl')


def analyze_epoch_loss_simple(epoch_markers, loss_values):
    """
    简化版epoch loss分析，支持一维/二维loss_values输入
    输入:
        epoch_markers: [n_batches,] 标记每个batch所属的epoch（整数）
        loss_values:   [n_batches,] 或 [n_batches, n_loss_types] 的loss值

    输出:
        epoch_stats: {
            'mean': [n_epochs, (n_loss_types)],
            'median': [...],
            'max': [...],
            'min': [...],
            'std': [...]
        }
        intra_anomalies: [异常字典列表]
        inter_anomalies: [突变字典列表]
    """
    unique_epochs = np.unique(epoch_markers)

    # 确保loss_values是二维（n_batches, n_loss_types）
    if loss_values.ndim == 1:
        loss_values = loss_values[:, np.newaxis]
    n_loss_types = loss_values.shape[1]

    # 按epoch重组数据
    epoch_loss = []  # 最终形状: [n_epochs, n_batches_in_epoch, n_loss_types]
    batches_per_epoch = []
    for epoch in unique_epochs:
        mask = (epoch_markers == epoch)
        epoch_loss.append(loss_values[mask])
        batches_per_epoch.append(np.sum(mask))
    max_batches = max(batches_per_epoch)

    # 计算统计量
    epoch_stats = {}
    for stat_name, stat_func in [('mean', np.mean), ('median', np.median),
                                 ('max', np.max), ('min', np.min), ('std', np.std)]:
        # 对每个epoch的batch维度计算统计量
        epoch_stats[stat_name] = np.array([stat_func(epoch, axis=0) for epoch in epoch_loss])

    # 1. 检测epoch内异常
    intra_anomalies = []
    for epoch_idx, epoch in enumerate(unique_epochs):
        for loss_type in range(n_loss_types):
            losses = epoch_loss[epoch_idx][:, loss_type]
            median = np.median(losses)
            mad = stats.median_abs_deviation(losses)
            if mad == 0:
                continue

            z_scores = np.abs(losses - median) / mad
            anomalies = np.where(z_scores > 3.0)[0]
            if len(anomalies) > 0:
                intra_anomalies.append({
                    'epoch': epoch,
                    'loss_type': loss_type,
                    'batch_indices': anomalies,
                    'max_z_score': np.max(z_scores)
                })

    # 2. 检测epoch间异常
    inter_anomalies = []
    for loss_type in range(n_loss_types):
        means = epoch_stats['mean'][:, loss_type]
        changes = np.abs(np.diff(means)) / (means[:-1] + 1e-8)
        median_change = np.median(changes)
        mad_change = stats.median_abs_deviation(changes)

        z_scores = changes / (median_change + mad_change + 1e-8)
        anomalies = np.where(z_scores > 3.0)[0]

        for trans in anomalies:
            inter_anomalies.append({
                'transition': (unique_epochs[trans], unique_epochs[trans + 1]),
                'loss_type': loss_type,
                'change_ratio': changes[trans],
                'epoch_means': (means[trans], means[trans + 1])
            })

    return epoch_stats, intra_anomalies, inter_anomalies


def plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array, title="training"):
    """改进的绘图函数，包含：
    1. 学习率曲线（副坐标轴）
    2. 坐标轴颜色与loss类型一致
    3. 只标记每个epoch中最高loss异常点

    [loss.item(), loss_f.item(), total_loss.item(), e]
    绘制data_loss:[:,0]
    绘制training loss：[:,2]
    绘制[test_l2, e]
    绘制lr变化 lr_list.append([current_lr, e])
    grad：[e, loss.item(), grad_norm.cpu()]
    """
    loss_array = np.array(loss_list)
    lr_array = np.array(lr_list)
    test_loss_list = np.array(test_loss_list)
    grad_array = np.array(grad_array)

    unique_epochs = np.unique(loss_array[:, -1].astype(int))

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    loss_types = ['Training Loss', 'Data Loss', 'Testing Loss']
    colors = ['#1f77b4', '#8A0011', '#107A38']  # 标准matplotlib颜色
    lr_color = '#9467bd'  # 学习率曲线颜色
    plot_data = [[loss_array[:, -1].astype(int), loss_array[:, 2], lr_array[:, -1].astype(int), lr_array[:, 0]],
                 # 绘制训练总损失
                 [grad_array[:, 0].astype(int), grad_array[:, 1], grad_array[:, 0].astype(int), grad_array[:, 2]],
                 # 绘制自回归过程中的data损失和梯度变化
                 [test_loss_list[:, -1].astype(int), test_loss_list[:, 0], lr_array[:, -1].astype(int), lr_array[:, 0]],
                 # 绘制测试l2损失
                 ]
    plot_label = [['Training MSE', 'Learning Rate'], ['Training l2 data error', 'grad_norm'],
                  ['Testing l2 data error', 'Learning Rate']]

    for i in trange(len(plot_data)):
        # 主坐标轴设置（保持与loss类型相同颜色）
        # region 绘制基础loss散点图
        data = plot_data[i]
        ax = axes[i]
        if data[0][0] == data[0][1]:
            ax.scatter(data[0],
                       data[1],
                       alpha=0.2, color=colors[i], s=10)
            epoch_stats, intra_anomalies, inter_anomalies = analyze_epoch_loss_simple(
                data[0], data[1])
            ax.plot(unique_epochs, epoch_stats['mean'],
                    color='black', label=f'Epoch Mean of {plot_label[i][0]}', linewidth=1)
        else:
            ax.plot(data[0], data[1],
                    color=colors[i], label=plot_label[i][0], linewidth=1)
        # endregion
        # region 主坐标轴标签设置
        # 主坐标轴标签设置（与loss类型同色）

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Error', fontsize=12, color=colors[i])
        ax.set_yscale('log')
        # ax.tick_params(axis='x', colors=colors[i])
        ax.tick_params(axis='y', colors=colors[i])
        ax.set_title(f"{loss_types[i]}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        # endregion
        # region 绘制副坐标轴数据
        # 添加学习率曲线（副坐标轴）
        ax2 = ax.twinx()
        ax2.plot(data[2], data[3],
                 color=lr_color,
                 linestyle=':',
                 linewidth=1.5,
                 alpha=1,
                 label=plot_label[i][1])
        ax2.set_ylabel(plot_label[i][1], fontsize=12, color=lr_color)
        # ax2.set_yscale('log')
        ax2.tick_params(axis='y', colors=lr_color)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=10,
                  loc='upper right',
                  framealpha=0.9)
        # endregion

    plt.tight_layout()
    plt.savefig(f'{title}_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('plot succeed')


class CurriculumScheduler:
    def __init__(self,
                 max_t_train,
                 min_steps=20,
                 warmup_epochs=0,
                 rollback_prob=0.1,
                 rollback_decay=0.95):
        """
        课程学习调度器
        :param max_t_train: 最大时间步数（原始t_train）
        :param min_steps: 初始最小预测步数
        :param warmup_epochs: 达到最大步数需要的epoch数
        :param rollback_prob: 回滚概率
        :param rollback_decay: 回滚概率衰减系数
        """
        self.max_t_train = max_t_train
        self.min_steps = min_steps
        self.warmup_epochs = warmup_epochs
        self.rollback_prob = rollback_prob
        self.rollback_decay = rollback_decay
        self.current_t_train = min_steps
        self.base_schedule = [
            min(
                max_t_train,
                min_steps + int(epoch * (max_t_train - min_steps) / warmup_epochs)
            )
            for epoch in range(warmup_epochs + 1)  # +1防止索引越界
        ]
        if (self.max_t_train - self.min_steps) < self.warmup_epochs:
            epoch_per_t = self.warmup_epochs / (self.max_t_train - self.min_steps)
            print(
                f"Curriculum training starts at {self.min_steps} steps, increasing by {epoch_per_t} epoch every t for {self.warmup_epochs} epochs to reach {self.max_t_train} steps")
        else:
            avg_step = (self.max_t_train - self.min_steps) / self.warmup_epochs
            print(
                f"Curriculum training starts at {self.min_steps} steps, increasing by ~{avg_step:.1f} steps on average over {self.warmup_epochs} epochs to reach {self.max_t_train} steps")

    def update(self, epoch):
        """每个epoch开始时更新当前t_train"""
        # 基础线性增长
        new_t_train = self.base_schedule[min(epoch, self.warmup_epochs)]

        # 随机回滚机制
        p = random.random()
        if p < self.rollback_prob and new_t_train > self.min_steps:
            rollback_risol = 0.5 + 0.25 * random.random()
            self.current_t_train = max(
                self.min_steps,
                int(rollback_risol * new_t_train)  # 回滚到50%-100%之间
            )

            self.rollback_prob *= self.rollback_decay  # 衰减回滚概率
        else:
            self.current_t_train = min(new_t_train, self.max_t_train)

        return int(self.current_t_train)


class RobustAdaptiveGradientClipper:
    def __init__(self, initial_max_norm=500.0, window_size=30, trim_k=5, multiplier=10):
        """
        Args:
            initial_max_norm: 初始梯度裁剪阈值
            window_size: 滑动窗口大小（记录最近多少次训练的梯度范数）
            trim_k: 去掉窗口中最大和最小的各K个值
            multiplier: 去尾均值乘以该倍数得到新阈值
        """
        self.max_norm = initial_max_norm
        self.window_size = window_size
        self.trim_k = trim_k
        self.multiplier = multiplier
        self.grad_norm_history = deque(maxlen=window_size)  # 滑动窗口

    def step(self, model):
        # 1. 计算当前梯度范数（不裁剪，仅测量）
        current_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        self.grad_norm_history.append(current_norm.item())

        # 2. 如果窗口未填满，使用初始阈值
        if len(self.grad_norm_history) < self.window_size:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            return self.max_norm

        # 3. 计算去尾均值
        sorted_norms = np.sort(self.grad_norm_history)
        trimmed_norms = sorted_norms[self.trim_k: -self.trim_k]  # 去掉前K大和前K小
        trimmed_mean = np.mean(trimmed_norms)

        # 4. 更新裁剪阈值
        self.max_norm = max(trimmed_mean * self.multiplier, 1e-3)  # 避免阈值过小

        # 5. 执行裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        return self.max_norm


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_min=1e-9
                                    ):
    def lr_lambda(current_step):
        # print('num_warmup_steps:', num_warmup_steps)
        if current_step < num_warmup_steps:  # 线性预热
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦衰减
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            lr_min,
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    return LambdaLR(optimizer, lr_lambda)


class CustomFormatter(ticker.ScalarFormatter):
    '''
    设置cbar的科学计数法表示
    '''

    def __init__(self, useMathText=True, powerlimits=(-1, 1)):
        super().__init__(useMathText=useMathText)
        self.set_powerlimits(powerlimits)
        self.set_scientific(True)

    def __call__(self, x, pos=None):
        # 缩放数值以适应科学计数法的基数部分
        scale = np.power(10, -self.orderOfMagnitude)
        return f'{x * scale:.1f}'


class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        # 清理控制字符
        cleaned_instr = re.sub('\n|\x1b.*|\r', '', instr)
        # 将清理后的信息存储到 _desc
        self._desc += cleaned_instr

    def read(self, b):
        ret = self._desc
        self._desc = f'batch {b}:'
        return ret

    def flush(self):
        pass


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


def plot_solution_with_error(ref, pred, plt_name, save_svg=False, Aspect_Ratio=1 / 1.4):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param ref:
    :param x:
    :param t:
    :param path:
    :param title:
    :param plt_name:
    :param save_svg:
    :return:
    '''
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #  ################# 图标题 ##################
    fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0], calculate_fig_size(Aspect_Ratio)[1]))
    error = np.abs(ref - pred)
    ref = (ref - ref.min()) / (ref.max() - ref.min())
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    gs0 = gridspec.GridSpec(2, 2)
    gs0.update(top=1, bottom=0, left=0, right=1, wspace=0.2)

    #  region 数值解
    ax1 = plt.subplot(gs0[0, 0])
    h = ax1.imshow(ref, interpolation='nearest', cmap='viridis',
                   extent=[0, 1, -1, 1],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Numerical Solution', fontsize=10)
    #  ################# 坐标轴 ##################
    ax1.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax1.set_ylabel(r'$x$', fontweight=400, size=10, labelpad=-5)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.tick_params(labelsize=10)
    # endregion
    #  region 预测解
    ax2 = plt.subplot(gs0[0, 1])
    h = ax2.imshow(pred, interpolation='nearest', cmap='viridis',
                   extent=[0, 1, -1, 1],
                   origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=9))
    cbar.ax.tick_params(labelsize=10)
    #  ################# 坐标轴 ##################
    ax2.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.set_yticklabels([])
    ax2.set_title('Predicted Solution', fontsize=10)
    ax2.tick_params(labelsize=10)
    # endregion
    # region error

    ax3 = plt.subplot(gs0[1, 0])
    h2 = ax3.imshow(error, interpolation='nearest',  # cmap='rainbow',
                    extent=[0, 1, -1, 1],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h2, cax=cax)
    cbar.set_ticks(np.linspace(0, 0.1, num=9))
    ax3.set_title('Absolute Error', fontsize=10)
    ax3.set_xlabel('$t$', labelpad=2)
    ax3.set_ylabel('$x$', labelpad=2)
    cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')  # 调整偏移文本的位置
    # endregion
    # region solution_in_time
    #     gs_nested = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1, 1])
    gs_nested = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs0[1, 1], wspace=0.6, hspace=0.3)

    len_x = ref.shape[0]
    len_t = ref.shape[1]
    x = np.linspace(-1, 1, len_x)
    t = np.linspace(0, 1, len_t)

    t_1 = t[int(0.5 * (len_t - 1))]
    t_2 = t[int(0.8 * (len_t - 1))]
    t_3 = t[-1]

    solution_1_1 = ref[:, int(0.5 * (len_t - 1))]
    solution_2_1 = ref[:, int(0.8 * (len_t - 1))]
    solution_3_1 = ref[:, -1]

    solution_1_2 = pred[:, int(0.5 * (len_t - 1))]
    solution_2_2 = pred[:, int(0.8 * (len_t - 1))]
    solution_3_2 = pred[:, -1]

    ax4 = plt.subplot(gs_nested[0, 0])
    ax4.plot(x, solution_1_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
             label='Numerical Solution')
    ax4.plot(x, solution_1_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
             label='Predicted Solution')
    ax4.set_xticklabels([])
    ax4.set_ylabel(r'$u(x,t)$')
    ax4.set_title(f'$t = {format(t_1, ".1f")}$', fontsize=10)
    plt.axis('equal')
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([0, 1])
    ax4.set_aspect('auto')

    ax5 = plt.subplot(gs_nested[1, 0])
    ax5.plot(x, solution_2_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
             label='Numerical Solution')
    ax5.plot(x, solution_2_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
             label='Predicted Solution')
    ax5.set_xticklabels([])
    ax5.set_ylabel(r'$u(x,t)$')
    ax5.set_title(f'$t = {format(t_2, ".1f")}$', fontsize=10)
    plt.axis('equal')
    ax5.set_xlim([-1, 1])
    ax5.set_ylim([0, 1])
    ax5.set_aspect('auto')

    ax6 = plt.subplot(gs_nested[2, 0])
    line1, = ax6.plot(x, solution_3_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
                      label='Numerical Solution')
    line2, = ax6.plot(x, solution_3_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
                      label='Predicted Solution')
    ax6.set_xlabel(r'$x$')
    ax6.set_ylabel(r'$u(x,t)$')
    title = ax6.set_title(f'$t = {format(t_3, ".1f")}$', fontsize=10)
    title.set_position([0.5, 1.0])
    plt.axis('equal')
    ax6.set_xlim([-1, 1])
    ax6.set_ylim([0, 1])
    ax6.set_aspect('auto')
    for ax in [ax4, ax5, ax6]:
        pos = ax.get_position()
        pos.x0 += 0.1  # 向右移动
        ax.set_position(pos)
    plt.tight_layout()
    #     fig.legend(labels, loc='center', bbox_to_anchor=(0.5, 0.5))
    fig.legend(handles=[line1, line2], loc='center right', ncol=2, frameon=False)
    # endregion

    if save_svg:
        plt.savefig(f'{plt_name}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{plt_name}.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def plot_solution_with_Du(ref, pred, plt_name, du_pred, du_pred_mean, du_yy, du_yy_mean, mean_l2, save_svg=False,
                          Aspect_Ratio=1 / 1.4, ):
    '''
    绘制全局的残差分布，与solution绘制不同的是，由于数值普遍较小，在cbar刻度上采用科学计数法的表示，保留了两位小数；
    :param ref:
    :param x:
    :param t:
    :param path:
    :param title:
    :param plt_name:
    :param save_svg:
    :return:
    '''
    plt.rcParams['font.family'] = 'DejaVu Serif'
    #  ################# 图标题 ##################
    fig, ax = plt.subplots(figsize=(calculate_fig_size(Aspect_Ratio)[0], calculate_fig_size(Aspect_Ratio)[1]))
    ref = (ref - ref.min()) / (ref.max() - ref.min())
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    gs0 = gridspec.GridSpec(2, 2)
    gs0.update(top=1, bottom=0, left=0, right=1, wspace=0.2)

    #  region 数值解
    ax1 = plt.subplot(gs0[0, 0])
    h = ax1.imshow(du_yy, interpolation='nearest',  # cmap='rainbow',
                   extent=[0, 1, -1, 1],
                   origin='lower', aspect='auto')
    # print('mean_l2 in plot:', mean_l2)
    ax1.set_title(f'Ref Residual:{du_yy_mean:.2e}', fontsize=10)
    #  ################# 坐标轴 ##################
    ax1.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax1.set_ylabel(r'$x$', fontweight=400, size=10, labelpad=-5)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.tick_params(labelsize=10)
    # endregion
    #  region DU_pred
    ax2 = plt.subplot(gs0[0, 1])
    h = ax2.imshow(du_pred, interpolation='nearest', cmap='viridis',
                   extent=[0, 1, -1, 1],
                   origin='lower', aspect='auto')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.set_ticks(np.linspace(0, np.max(du_pred), num=9))
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_formatter(CustomFormatter())
    cbar.ax.yaxis.get_offset_text().set_position((2.7, 1))  # 调整偏移文本的位置
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')
    #  ################# 坐标轴 ##################
    ax2.set_xlabel(r'$t$', fontweight=400, size=10, labelpad=-5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.set_yticklabels([])
    ax2.set_title(f'Pred Residual:{du_pred_mean:.2e}', fontsize=10)
    ax2.tick_params(labelsize=10)
    # endregion
    # region Solution:{mean_l2:.2e}
    ax3 = plt.subplot(gs0[1, 0])
    h2 = ax3.imshow(ref, interpolation='nearest', cmap='viridis',
                    extent=[0, 1, -1, 1],
                    origin='lower', aspect='auto', vmin=0, vmax=1)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h2, cax=cax)
    cbar.set_ticks(np.linspace(0, 1, num=9))
    ax3.set_title(f'Solution:{mean_l2:.2e}', fontsize=10)
    ax3.set_xlabel('$t$', labelpad=2)
    ax3.set_ylabel('$x$', labelpad=2)
    # endregion
    # region solution_in_time
    #     gs_nested = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1, 1])
    gs_nested = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs0[1, 1], wspace=0.6, hspace=0.3)

    len_x = ref.shape[0]
    len_t = ref.shape[1]
    x = np.linspace(-1, 1, len_x)
    t = np.linspace(0, 1, len_t)

    t_1 = t[int(0.5 * (len_t - 1))]
    t_2 = t[int(0.8 * (len_t - 1))]
    t_3 = t[-1]

    solution_1_1 = ref[:, int(0.5 * (len_t - 1))]
    solution_2_1 = ref[:, int(0.8 * (len_t - 1))]
    solution_3_1 = ref[:, -1]

    solution_1_2 = pred[:, int(0.5 * (len_t - 1))]
    solution_2_2 = pred[:, int(0.8 * (len_t - 1))]
    solution_3_2 = pred[:, -1]

    ax4 = plt.subplot(gs_nested[0, 0])
    ax4.plot(x, solution_1_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
             label='Numerical Solution')
    ax4.plot(x, solution_1_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
             label='Predicted Solution')
    ax4.set_xticklabels([])
    ax4.set_ylabel(r'$u(x,t)$')
    ax4.set_title(f'$t = {format(t_1, ".1f")}$', fontsize=10)
    plt.axis('equal')
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([0, 1])
    ax4.set_aspect('auto')

    ax5 = plt.subplot(gs_nested[1, 0])
    ax5.plot(x, solution_2_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
             label='Numerical Solution')
    ax5.plot(x, solution_2_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
             label='Predicted Solution')
    ax5.set_xticklabels([])
    ax5.set_ylabel(r'$u(x,t)$')
    ax5.set_title(f'$t = {format(t_2, ".1f")}$', fontsize=10)
    plt.axis('equal')
    ax5.set_xlim([-1, 1])
    ax5.set_ylim([0, 1])
    ax5.set_aspect('auto')

    ax6 = plt.subplot(gs_nested[2, 0])
    line1, = ax6.plot(x, solution_3_1, linestyle='-', color=(45 / 255, 12 / 255, 126 / 255), linewidth=2,
                      label='Numerical Solution')
    line2, = ax6.plot(x, solution_3_2, linestyle='--', color=(255 / 255, 106 / 255, 125 / 255), linewidth=2,
                      label='Predicted Solution')
    ax6.set_xlabel(r'$x$')
    ax6.set_ylabel(r'$u(x,t)$')
    title = ax6.set_title(f'$t = {format(t_3, ".1f")}$', fontsize=10)
    title.set_position([0.5, 1.0])
    plt.axis('equal')
    ax6.set_xlim([-1, 1])
    ax6.set_ylim([0, 1])
    ax6.set_aspect('auto')
    for ax in [ax4, ax5, ax6]:
        pos = ax.get_position()
        pos.x0 += 0.1  # 向右移动
        ax.set_position(pos)
    plt.tight_layout()
    #     fig.legend(labels, loc='center', bbox_to_anchor=(0.5, 0.5))
    fig.legend(handles=[line1, line2], loc='center right', ncol=2, frameon=False)
    # endregion

    if save_svg:
        plt.savefig(f'{plt_name}.svg', format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{plt_name}.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def plot_visualization_results_data(visualize_results, save_dir=None):
    """
    绘制可视化结果

    Args:
        visualize_results: 包含 9 个样本的列表
        save_dir: 保存路径（可选）
    """
    # 按类别分组
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    best_samples = [r for r in visualize_results if r['category'] == 'best']
    mid_samples = [r for r in visualize_results if r['category'] == 'mid']
    worst_samples = [r for r in visualize_results if r['category'] == 'worst']

    # ========== 图1-3：pred vs yy 对比（每类别一张图）==========
    for category, samples in [('best', best_samples), ('mid', mid_samples), ('worst', worst_samples)]:
        plot_pred_vs_yy_steps_v2(samples, category, save_dir)


def plot_pred_vs_yy_steps_v2(samples, category, save_dir=None, n_plots=4):
    """
    绘制 pred vs yy 对比图 - 自适应时间步版本

    每个 sample 一行：
    - n_plots 个子图（选取关键时间步）+ 1个误差直方图
    - GT: 蓝色虚线, Pred: 红色实线

    时间步选取策略：
    - 第 1 个时间步
    - 第 1/2 位置的时间步
    - 第 3/4 位置的时间步
    - 最后一个时间步

    参数:
        samples: list of dict, 每个包含 'pred', 'yy', 'index', 'l2_error'
        category: str, 类别名称
        save_dir: str, 保存路径
        n_plots: int, 绘制的时间步数量（默认4）
    """
    n_samples = len(samples)

    fig, axes = plt.subplots(n_samples, n_plots + 1, figsize=(4 * (n_plots + 1), 3.5 * n_samples))

    # 确保 axes 是二维的
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, sample in enumerate(samples):
        pred = sample['pred']  # [nx, n_steps]
        yy = sample['yy']  # [nx, n_steps]
        idx = sample['index']
        l2_error = sample['l2_error']

        nx, nt = pred.shape

        # 选取关键时间步索引
        if nt <= n_plots:
            # 如果时间步数 <= n_plots，直接用所有时间步
            step_indices = list(range(nt))
        else:
            # 选取: 第1个, 1/2位置, 3/4位置, 最后一个
            step_indices = [
                0,  # 第 1 个
                nt // 2,  # 1/2 位置
                (3 * nt) // 4,  # 3/4 位置
                nt - 1  # 最后一个
            ]
            # 确保索引唯一且排序
            step_indices = sorted(list(set(step_indices)))

        # CGL点
        i = np.arange(nx)
        x_cgl = np.cos(np.pi * i / (nx - 1))[::-1]  # [-1, 1]

        # 计算误差
        error = np.abs(pred - yy)

        # 统一 y 轴范围
        y_min = min(pred.min(), yy.min())
        y_max = max(pred.max(), yy.max())
        y_margin = (y_max - y_min) * 0.1

        # ========== 绘制选定的时间步 ==========
        for plot_idx, step in enumerate(step_indices):
            ax = axes[row, plot_idx]

            # GT: 蓝色虚线, Pred: 红色实线
            ax.plot(x_cgl, pred[:, step], color='red', linestyle='-',
                    label='Pred', linewidth=2)
            ax.plot(x_cgl, yy[:, step], color='blue', linestyle='--',
                    label='GT', linewidth=2)

            step_max_err = error[:, step].max()

            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_xlabel('x', fontsize=9)
            ax.set_title(f't={step + 1}/{nt}\nmax_err={step_max_err:.2e}', fontsize=9)
            ax.grid(True, alpha=0.3)

            if plot_idx == 0:
                ax.set_ylabel(f'Sample {idx}\nu', fontsize=9)

            ax.legend(loc='upper right', fontsize=7)

        # 如果实际绘制的时间步少于 n_plots，隐藏多余的子图
        for plot_idx in range(len(step_indices), n_plots):
            axes[row, plot_idx].axis('off')

        # ========== 误差直方图 ==========
        ax_hist = axes[row, n_plots]
        all_errors = error.flatten()

        ax_hist.hist(all_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax_hist.axvline(all_errors.mean(), color='red', linestyle='--', linewidth=1.5,
                        label=f'Mean: {all_errors.mean():.2e}')
        ax_hist.axvline(np.median(all_errors), color='orange', linestyle='--', linewidth=1.5,
                        label=f'Median: {np.median(all_errors):.2e}')

        ax_hist.set_xlabel('Absolute Error', fontsize=9)
        ax_hist.set_ylabel('Count', fontsize=9)
        ax_hist.set_title(f'L2={l2_error:.2e}\nMax={all_errors.max():.2e}', fontsize=9)
        ax_hist.legend(loc='upper right', fontsize=7)
        ax_hist.grid(True, alpha=0.3)

    fig.suptitle(f'{category.upper()} - Prediction vs Ground Truth',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy_steps.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy_steps.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


class ResolutionPhaseManager:
    def __init__(self, phases):
        self.phases = sorted(phases, key=lambda x: x["start_epoch"])
        self.current_phase_idx = 0
        self.last_resolutions = None
        self.max_epoch = max(phase["start_epoch"] for phase in phases)

    def get_resolutions(self, epoch):
        """根据当前epoch返回应激活的分辨率"""
        # 找到最后一个满足 start_epoch <= epoch 的phase
        for phase in reversed(self.phases):
            if epoch >= phase["start_epoch"]:
                return phase["resolutions"]
        return self.phases[0]["resolutions"]  # 默认返回第一个phase


def plot_visualization_results(visualize_results, save_dir=None, plot_residual=True):
    """
    绘制可视化结果

    Args:
        visualize_results: 包含 9 个样本的列表
        save_dir: 保存路径（可选）
    """
    # 按类别分组
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    best_samples = [r for r in visualize_results if r['category'] == 'best']
    mid_samples = [r for r in visualize_results if r['category'] == 'mid']
    worst_samples = [r for r in visualize_results if r['category'] == 'worst']

    # ========== 图1-3：pred vs yy 对比（每类别一张图）==========
    for category, samples in [('best', best_samples), ('mid', mid_samples), ('worst', worst_samples)]:
        plot_pred_vs_yy(samples, category, save_dir)

    # ========== 图4：残差对比（9 个样本）==========
    if plot_residual:
        plot_residual_comparison(visualize_results, save_dir)


def plot_pred_vs_yy(samples, category, save_dir=None):
    """
    绘制 pred vs yy 对比图
    每个样本一行，从左到右依次是：参考解、预测解、绝对误差、时间切片
    """
    # 设置字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'SimSun']
    plt.rcParams['mathtext.fontset'] = 'stix'

    n_samples = len(samples)
    fig = plt.figure(figsize=(16, 4 * n_samples))

    # 两个GridSpec：前三列紧凑，第四列单独（间距5%）
    gs_left = GridSpec(n_samples, 3, figure=fig,
                       left=0.05, right=0.65, hspace=0.35, wspace=0.12)
    gs_right = GridSpec(n_samples, 1, figure=fig,
                        left=0.70, right=0.90, hspace=0.35)

    for row, sample in enumerate(samples):
        pred = sample['pred']  # [nx, nt]
        yy = sample['yy']  # [nx, nt]
        l2_error = sample['l2_error']

        # Case 编号
        case_name = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][row] if row < 10 else str(row + 1)

        nx, nt = pred.shape

        # CGL点 (Chebyshev-Gauss-Lobatto)
        x_cgl = np.linspace(-1, 1, nx)

        # 均匀时间网格
        t_grid = np.linspace(0, 1, nt)

        # 统一色标范围
        vmin = min(pred.min(), yy.min())
        vmax = max(pred.max(), yy.max())
        error = np.abs(pred - yy)

        T, X = np.meshgrid(t_grid, x_cgl)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # ==================== 列1：参考解 ====================
        ax1 = fig.add_subplot(gs_left[row, 0])
        im1 = ax1.pcolormesh(T, X, yy, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Ground Truth\nCase {case_name}', fontsize=12)
        ax1.set_xlabel(r'$\mathit{t}$', fontsize=12)
        ax1.set_ylabel(r'$\mathit{x}$', fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-1, 1)
        ax1.tick_params(labelsize=10)

        # ==================== 列2：预测解 ====================
        ax2 = fig.add_subplot(gs_left[row, 1])
        im2 = ax2.pcolormesh(T, X, pred, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Prediction\n$L_2$ Error: {l2_error:.4e}', fontsize=12)
        ax2.set_xlabel(r'$\mathit{t}$', fontsize=12)
        ax2.set_yticklabels([])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-1, 1)
        ax2.tick_params(labelsize=10)

        # 图2右侧添加colorbar（与图2等高）
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.formatter = ScalarFormatter(useMathText=True)
        cbar2.formatter.set_powerlimits((-2, 2))
        cbar2.update_ticks()
        cbar2.ax.tick_params(labelsize=10)

        # ==================== 列3：绝对误差 ====================
        ax3 = fig.add_subplot(gs_left[row, 2])
        im3 = ax3.pcolormesh(T, X, error, shading='auto', cmap='Blues')
        ax3.set_title(f'Absolute Error\nMax: {error.max():.2e}', fontsize=12)
        ax3.set_xlabel(r'$\mathit{t}$', fontsize=12)
        ax3.set_yticklabels([])
        ax3.set_xlim(0, 1)
        ax3.set_ylim(-1, 1)
        ax3.tick_params(labelsize=10)

        # 使用 make_axes_locatable 确保 colorbar 与子图等高
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cbar3 = plt.colorbar(im3, cax=cax3)
        cbar3.formatter = ScalarFormatter(useMathText=True)
        cbar3.formatter.set_powerlimits((-2, 2))
        cbar3.update_ticks()
        cbar3.ax.tick_params(labelsize=10)

        # ==================== 列4：时间切片对比 ====================
        ax4 = fig.add_subplot(gs_right[row, 0])
        # 时间步索引: t=0, t=0.1, t=0.3, t=1
        t_indices = [int(0.1 * (nt - 1)), int(0.3 * (nt - 1)), nt - 1]
        t_values = [0.1, 0.3, 1.0]
        gt_colors = ['#453681', '#26828E', '#4FC36A', '#BEDF26']
        pred_colors = ['#493FC5', '#71CDD9', '#98DCA8', '#E2F09A']

        # 画 IC (t=0)
        ax4.plot(x_cgl, yy[:, 0], linestyle='-', color=gt_colors[0],
                 label=r'GT $\mathit{t}$=0 (IC)', linewidth=2.0)

        for i, (t_idx, t_val) in enumerate(zip(t_indices, t_values)):
            label_gt = rf'GT $\mathit{{t}}$={t_val:.2f}'
            label_pred = rf'Pred $\mathit{{t}}$={t_val:.2f}'

            ax4.plot(x_cgl, yy[:, t_idx], linestyle='-', color=gt_colors[i + 1],
                     label=label_gt, linewidth=2.0)
            ax4.plot(x_cgl, pred[:, t_idx], linestyle='--', color=pred_colors[i + 1],
                     label=label_pred, linewidth=2.0, dashes=(1, 2))

        ax4.set_xlabel(r'$\mathit{x}$', fontsize=12)
        ax4.set_ylabel(r'$\mathit{u}$', fontsize=12)
        ax4.set_title('Time Slices Comparison', fontsize=12)
        ax4.set_xlim(-1, 1)
        ax4.tick_params(labelsize=10)
        ax4.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

        n_grid_lines = min(17, nx)
        grid_indices = np.linspace(0, nx - 1, n_grid_lines, dtype=int)
        for gi in grid_indices:
            ax4.axvline(x=x_cgl[gi], color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    fig.suptitle(f'{category.upper()} Samples - Prediction vs Ground Truth',
                 fontsize=14, fontweight='bold', y=0.995)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy.pdf', bbox_inches='tight')
        print(f"Saved: {save_dir}/{category}_pred_vs_yy.png/pdf")

    plt.show()
    plt.close()


def plot_residual_comparison(visualize_results, save_dir=None):
    """绘制残差对比图"""
    sorted_results = (
            [r for r in visualize_results if r.get('category') == 'best'] +
            [r for r in visualize_results if r.get('category') == 'mid'] +
            [r for r in visualize_results if r.get('category') == 'worst']
    )

    n_samples = len(sorted_results)
    if n_samples == 0:
        print("警告：没有可绘制的样本")
        return

    fig = plt.figure(figsize=(15, 3 * n_samples))
    gs = GridSpec(n_samples, 3, figure=fig, hspace=0.35, wspace=0.25)

    for row, sample in enumerate(sorted_results):
        pred = sample['pred']  # [nx, nt]
        yy = sample['yy']  # [nx, nt]
        pred_du = sample['pred_du']  # [nx, nt]
        yy_du = sample['yy_du']  # [nx, nt]
        idx = sample['index']
        category = sample['category']
        l2_error = sample['l2_error']

        error = np.abs(pred - yy)

        # 各自独立的色标范围
        pred_du_abs_max = np.abs(pred_du).max()
        yy_du_abs_max = np.abs(yy_du).max()

        # 列1：pred_du（使用自己的色标）
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(pred_du, aspect='auto', cmap='RdBu_r',
                         vmin=-pred_du_abs_max, vmax=pred_du_abs_max,
                         extent=[0, 1, -1, 1], origin='lower')
        ax1.set_ylabel(f'{category.upper()}\nSample {idx}\nx', fontsize=9)
        if row == 0:
            ax1.set_title('Pred Residual (Du)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('t')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax1.text(0.02, 0.98, f'Mean: {np.mean(np.abs(pred_du)):.2e}\nMax: {pred_du_abs_max:.2e}',
                 transform=ax1.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 列2：yy_du（使用自己的色标）
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(yy_du, aspect='auto', cmap='RdBu_r',
                         vmin=-yy_du_abs_max, vmax=yy_du_abs_max,
                         extent=[0, 1, -1, 1], origin='lower')
        if row == 0:
            ax2.set_title('GT Residual (Du)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('t')
        ax2.set_ylabel('x')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax2.text(0.02, 0.98, f'Mean: {np.mean(np.abs(yy_du)):.2e}\nMax: {yy_du_abs_max:.2e}',
                 transform=ax2.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 列3：绝对误差（使用自己的色标）
        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(error, aspect='auto', cmap='hot',
                         extent=[0, 1, -1, 1], origin='lower')
        if row == 0:
            ax3.set_title('|Pred - GT|', fontsize=11, fontweight='bold')
        ax3.set_xlabel('t')
        ax3.set_ylabel('x')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax3.text(0.02, 0.98, f'L2: {l2_error:.2e}\nMax: {error.max():.2e}',
                 transform=ax3.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Residual Comparison: Best (top 3) → Mid (middle 3) → Worst (bottom 3)',
                 fontsize=14, fontweight='bold', y=1.001)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/residual_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/residual_comparison.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


class CausalCurriculumScheduler:
    """结合课程学习、因果权重和自适应学习率的调度器"""

    def __init__(self,
                 max_t_train,
                 min_steps=20,
                 warmup_epochs=100,
                 # 课程学习参数
                 rollback_prob=0.1,
                 rollback_decay=0.95,
                 # 自适应门控参数（基于 plateau 检测）
                 adaptive_gate=False,
                 loss_plateau_patience=5,
                 loss_plateau_threshold=0.01,
                 force_expand_patience=15,
                 force_patience_early_ratio=1.5,  # 新增
                 force_patience_late_ratio=0.5,
                 # 因果权重参数
                 use_causal_weights=False,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 # 学习率调度参数
                 use_lr_schedule=False,
                 lr_boost=2.0,
                 lr_warmup_epochs=5,
                 lr_scheduler_patience=3,
                 lr_scheduler_factor=0.5,
                 lr_scheduler_threshold=1e-2,
                 lr_min_ratio=0.001,
                 log_file=None):
        """
        参数说明
        ----------
        基础课程学习：
            max_t_train: 最大时间步数
            min_steps: 初始最小预测步数
            warmup_epochs: 达到最大步数的参考epoch数（用于lr调度的进度计算）
            rollback_prob: 回滚概率
            rollback_decay: 回滚概率衰减系数

        自适应门控（adaptive_gate=True 启用）：
            loss_plateau_patience: 连续几个epoch loss不降才算plateau
            loss_plateau_threshold: 下降多少才算"有改善"（如0.01表示1%）
            force_expand_patience: 单个窗口最多待多少epoch

        因果权重（use_causal_weights=True 启用）：
            epsilon_start: 因果强度初始值（强约束）
            epsilon_end: 因果强度最终值（弱约束）

        学习率调度（use_lr_schedule=True 启用）：
            lr_boost: 窗口扩展时lr可提升到的最大倍数
            lr_warmup_epochs: 最大warmup持续epoch数
            lr_scheduler_patience: ReduceLROnPlateau的patience
            lr_scheduler_factor: ReduceLROnPlateau的衰减系数
            lr_scheduler_threshold: ReduceLROnPlateau的threshold
            lr_min_ratio: 最低lr比例

        退化模式
        ----------
        adaptive_gate=False, use_causal_weights=False, use_lr_schedule=False
        → 等价于原始 CurriculumScheduler
        """
        self.log_file = log_file
        print(f"DEBUG: log_file = {self.log_file}")
        # ============ 课程学习参数 ============
        self.max_t_train = max_t_train
        self.min_steps = min_steps
        self.warmup_epochs = warmup_epochs
        self.rollback_prob = rollback_prob
        self.rollback_prob_init = rollback_prob
        self.rollback_decay = rollback_decay

        # ============ 自适应门控参数 ============
        self.adaptive_gate = adaptive_gate
        self.loss_plateau_patience = loss_plateau_patience
        self.loss_plateau_threshold = loss_plateau_threshold
        self.force_expand_patience = force_expand_patience
        self.best_loss_in_window = float('inf')
        self.epochs_no_improve = 0
        self.epochs_in_window = 0

        # ============ 因果权重参数 ============
        self.use_causal_weights = use_causal_weights
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.current_epsilon = epsilon_start

        # ============ 学习率调度参数 ============
        self.use_lr_schedule = use_lr_schedule
        self.lr_boost = lr_boost
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_threshold = lr_scheduler_threshold
        self.lr_min_ratio = lr_min_ratio

        # lr调度状态
        self.optimizer = None
        self.lr_scheduler = None
        self.base_lr = None
        self.epochs_since_expansion = 0
        self.lr_at_window_start = None
        self.lr_target = None
        self.current_warmup_epochs = lr_warmup_epochs
        self.in_warmup = False

        # ============ 当前状态 ============
        self.current_t_train = min_steps
        self.prev_t_train = None
        self.current_epoch = 0

        # ============ 因果矩阵缓存 ============
        self._causal_matrices = {}
        self._print_config()

        # ============ 动态窗口最大epoch ============
        self.force_expand_patience = force_expand_patience
        self.force_patience_early_ratio = force_patience_early_ratio  # 新增
        self.force_patience_late_ratio = force_patience_late_ratio

    def _print_config(self):
        """打印并记录配置信息"""
        config_msg = [
            "=" * 70,
            "Causal Curriculum Scheduler Configuration",
            "=" * 70,
            f"  Time steps: {self.min_steps} → {self.max_t_train}",
            f"  Warmup epochs: {self.warmup_epochs}",
            f"  Rollback prob: {self.rollback_prob}",
        ]

        if self.adaptive_gate:
            config_msg.append(f"  Adaptive gate: ON")
            config_msg.append(f"    - plateau_patience: {self.loss_plateau_patience}")
            config_msg.append(f"    - plateau_threshold: {self.loss_plateau_threshold}")
            config_msg.append(f"    - force_expand_patience: {self.force_expand_patience}")
        else:
            config_msg.append(f"  Adaptive gate: OFF (linear expansion)")

        if self.use_causal_weights:
            config_msg.append(f"  Causal weights: ON")
            config_msg.append(f"    - epsilon: {self.epsilon_start} → {self.epsilon_end}")
        else:
            config_msg.append(f"  Causal weights: OFF")

        if self.use_lr_schedule:
            config_msg.append(f"  LR schedule: ON")
            config_msg.append(f"    - lr_boost: {self.lr_boost}")
            config_msg.append(f"    - lr_warmup_epochs: {self.lr_warmup_epochs}")
            config_msg.append(f"    - lr_scheduler_patience: {self.lr_scheduler_patience}")
            config_msg.append(f"    - lr_scheduler_factor: {self.lr_scheduler_factor}")
            config_msg.append(f"    - lr_min_ratio: {self.lr_min_ratio}")
        else:
            config_msg.append(f"  LR schedule: OFF")

        config_msg.append("=" * 70)

        # 打印到控制台
        for line in config_msg:
            print(line)
        # 写入日志文件
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n")
                for line in config_msg:
                    f.write(line + "\n")
                f.write("\n")

    def _log_event(self, message):
        """打印并写入日志"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    # ================================================================
    # 学习率调度相关
    # ================================================================

    def init_lr_scheduler(self, optimizer):
        """
        初始化学习率调度器（必须在训练开始前调用）
        :param optimizer: PyTorch optimizer
        """
        if not self.use_lr_schedule:
            return

        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]['lr']
        self.lr_at_window_start = self.base_lr
        self.lr_target = self.base_lr
        self.current_warmup_epochs = 0
        self.in_warmup = False
        self._create_new_scheduler()

        print(f"  [LR Scheduler] Initialized with base_lr={self.base_lr:.2e}")

    def _create_new_scheduler(self):
        """创建新的ReduceLROnPlateau调度器"""
        if self.optimizer is None:
            return

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            min_lr=self.base_lr * self.lr_min_ratio,
            verbose=True
        )

    def _get_target_lr_for_epoch(self):
        """根据全局epoch进度计算目标lr"""
        global_progress = min(self.current_epoch / max(self.warmup_epochs, 1), 1.0)
        lr_max = self.base_lr * self.lr_boost
        lr_min = self.base_lr * self.lr_min_ratio
        return lr_max - global_progress * (lr_max - lr_min)

    def _reset_lr_for_new_window(self):
        """新窗口开始时判断是否需要warmup"""
        if self.optimizer is None:
            return

        # 1. 记录当前lr
        self.lr_at_window_start = self.optimizer.param_groups[0]['lr']

        # 2. 根据全局进度计算目标lr
        epoch_target_lr = self._get_target_lr_for_epoch()

        # 3. 只有当前lr低于目标时才提升（方案2）
        if self.lr_at_window_start < epoch_target_lr:
            # 需要提升，启动warmup
            self.lr_target = epoch_target_lr
            self.in_warmup = True

            # 根据提升幅度计算warmup长度
            boost_ratio = self.lr_target / max(self.lr_at_window_start, 1e-10)
            warmup_scale = math.log(boost_ratio + 1) / math.log(self.lr_boost + 1)
            self.current_warmup_epochs = max(1, int(self.lr_warmup_epochs * warmup_scale))

            # 重置计数器，创建新scheduler
            self.epochs_since_expansion = 0
            self._create_new_scheduler()

            print(f"  [LR Scheduler] New window (progress={self.current_epoch / self.warmup_epochs:.1%}): "
                  f"{self.lr_at_window_start:.2e} ↑ {self.lr_target:.2e} "
                  f"(×{boost_ratio:.2f}), warmup={self.current_warmup_epochs} epochs")
        else:
            # 当前lr已经足够高，不需要提升
            self.in_warmup = False

    def _warmup_lr(self):
        """Warmup阶段：从上一个窗口的lr逐步增加到目标lr"""
        if self.optimizer is None:
            return

        if self.current_warmup_epochs <= 0:
            current_lr = self.lr_target
        else:
            progress = min(self.epochs_since_expansion / self.current_warmup_epochs, 1.0)
            current_lr = self.lr_at_window_start + progress * (self.lr_target - self.lr_at_window_start)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

    def step_lr_scheduler(self, loss):
        """
        更新学习率调度器（每个epoch结束时调用）
        :param loss: 当前epoch的loss
        """
        if not self.use_lr_schedule or self.optimizer is None:
            # print(f"DEBUG step_lr_scheduler: SKIP - use_lr_schedule={self.use_lr_schedule}, optimizer={self.optimizer}")
            return

        lr_before = self.optimizer.param_groups[0]['lr']
        # print(f"DEBUG step_lr_scheduler: loss={loss:.4e}, lr_before={lr_before:.2e}, in_warmup={self.in_warmup}")

        if self.in_warmup:
            self.epochs_since_expansion += 1
            # print(
            # f"DEBUG: In warmup, epochs_since_expansion={self.epochs_since_expansion}/{self.current_warmup_epochs}")

            if self.epochs_since_expansion <= self.current_warmup_epochs:
                # Warmup阶段：lr逐步增加
                self._warmup_lr()
            else:
                # Warmup结束，切换到ReduceLROnPlateau
                self.in_warmup = False
                self.lr_scheduler.step(loss)
                # print(f"DEBUG: Warmup ended, lr_scheduler.step() called")
        else:
            # 非warmup阶段：直接用ReduceLROnPlateau
            # print(f"DEBUG: Calling lr_scheduler.step(loss={loss:.4e})")
            self.lr_scheduler.step(loss)

        lr_after = self.optimizer.param_groups[0]['lr']
        # print(f"DEBUG step_lr_scheduler: lr_after={lr_after:.2e}")

        if lr_after < lr_before:
            # lr 下降了，重置 plateau 计数，给新 lr 机会
            self.epochs_no_improve = 0
            print(f"  [Scheduler] LR dropped {lr_before:.2e} → {lr_after:.2e}, reset plateau counter")

    # ================================================================
    # 因果权重相关
    # ================================================================

    def get_causal_matrix(self, n_t, device='cpu'):
        """获取因果累积矩阵（带缓存）"""
        if n_t not in self._causal_matrices:
            M = torch.tril(torch.ones(n_t, n_t), diagonal=-1)
            self._causal_matrices[n_t] = M
        return self._causal_matrices[n_t].to(device)

    def compute_causal_weights(self, residual_per_t):
        """
        计算因果权重
        :param residual_per_t: [n_t] 每个时间点的平均残差
        :return: [n_t] 因果权重（detached）
        """
        n_t = residual_per_t.shape[0]
        device = residual_per_t.device

        M = self.get_causal_matrix(n_t, device)
        cumsum_residual = M @ residual_per_t
        weights = torch.exp(-self.current_epsilon * cumsum_residual)

        return weights.detach()

    def compute_weighted_loss(self, residuals, reduce='mean'):
        """
        计算带因果权重的损失
        :param residuals: [n_t, ...] 每个时间点的残差（第一维是时间）
        :param reduce: 'mean' or 'sum'
        :return: (加权损失, 每个时间点的MSE)
        """
        # 计算每个时间点的MSE
        if residuals.dim() > 1:
            residual_per_t = (residuals ** 2).mean(dim=tuple(range(1, residuals.dim())))
        else:
            residual_per_t = residuals ** 2

        if self.use_causal_weights:
            weights = self.compute_causal_weights(residual_per_t)
            weighted_residual = weights * residual_per_t
        else:
            weighted_residual = residual_per_t

        if reduce == 'mean':
            return weighted_residual.mean(), residual_per_t.detach()
        else:
            return weighted_residual.sum(), residual_per_t.detach()

    # ================================================================
    # 核心调度逻辑
    # ================================================================

    def _check_plateau(self, current_loss):
        """
        检查是否达到plateau（loss不再下降）
        :return: True表示可以扩展，False表示继续当前窗口
        """
        # print(f"  DEBUG _check_plateau: epoch={self.current_epoch}, "
        #       f"epochs_in_window={self.epochs_in_window}, "
        #       f"epochs_no_improve={self.epochs_no_improve}, "
        #       f"loss={current_loss}")
        if current_loss is None:
            return False

        self.epochs_in_window += 1

        # 第一次有 loss 时，初始化 best_loss，不做判定
        if self.best_loss_in_window == float('inf'):
            self.best_loss_in_window = current_loss
            self.epochs_no_improve = 0  # 确保重置
            return False

        # 计算相对改善
        improvement = (self.best_loss_in_window - current_loss) / self.best_loss_in_window

        if improvement > self.loss_plateau_threshold:
            # 有改善
            self.best_loss_in_window = current_loss
            self.epochs_no_improve = 0
        else:
            # 没改善
            self.epochs_no_improve += 1

        # 判断是否扩展
        plateau_reached = self.epochs_no_improve >= self.loss_plateau_patience
        progress = self.current_t_train / self.max_t_train
        ratio = self.force_patience_early_ratio - progress * (
                self.force_patience_early_ratio - self.force_patience_late_ratio)
        dynamic_force_patience = max(5, int(self.force_expand_patience * ratio))

        force_expand = self.epochs_in_window >= dynamic_force_patience

        if plateau_reached:
            print(f"  [Scheduler] Plateau detected (no improve for {self.epochs_no_improve} epochs), expanding...")
        elif force_expand:
            print(
                f"  [Scheduler] Force expand (stayed {self.epochs_in_window}/{dynamic_force_patience} epochs in window)")

        return plateau_reached or force_expand

    def _reset_window_tracking(self, current_loss):
        """新窗口开始时重置追踪状态"""
        self.best_loss_in_window = current_loss if current_loss else float('inf')
        self.epochs_no_improve = 0
        self.epochs_in_window = 0

    def update(self, epoch, current_loss=None):
        """
        每个epoch开始时更新调度状态
        :param epoch: 当前epoch
        :param current_loss: 上一个epoch的loss（用于plateau检测）
        :return: 当前应该训练的时间步数
        """
        self.current_epoch = epoch

        # ============ 更新epsilon（如果启用因果权重）============
        if self.use_causal_weights:
            progress = min(epoch / max(self.warmup_epochs, 1), 1.0)
            self.current_epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

        # ============ 自适应门控：基于plateau检测 ============
        if self.adaptive_gate and current_loss is not None:
            can_expand = self._check_plateau(current_loss)

            if not can_expand and self.current_t_train < self.max_t_train:
                return int(self.current_t_train)

            # 可以扩展
            if can_expand and self.current_t_train < self.max_t_train:
                remaining = self.max_t_train - self.current_t_train
                progress = self.current_t_train / self.max_t_train

                # 前期小步（1-2步），后期大步（5-6步）
                base_step = 1 + int(progress * 5)
                expand_step = min(base_step, remaining)

                new_t_train = min(self.current_t_train + expand_step, self.max_t_train)

                if new_t_train > self.current_t_train:
                    msg = (f"\n{'=' * 60}\n"
                           f"[Epoch {self.current_epoch}] Window Expanded: "
                           f"{self.current_t_train} → {new_t_train} (+{expand_step})\n"
                           f"  Progress: {new_t_train}/{self.max_t_train} ({new_t_train / self.max_t_train:.1%})\n"
                           f"  Loss: {current_loss:.2e}")

                    if self.use_causal_weights:
                        msg += f", Epsilon: {self.current_epsilon:.3f}"

                    if self.use_lr_schedule and self.optimizer:
                        msg += f"\n  LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                        msg += f" → target: {self._get_target_lr_for_epoch():.2e}"

                    msg += f"\n{'=' * 60}"

                    self._log_event(msg)
                    self.current_t_train = new_t_train
                    self._reset_window_tracking(current_loss)

                    if self.use_lr_schedule:
                        self._reset_lr_for_new_window()

        else:
            # 非自适应模式：线性扩展（原始逻辑）
            step_size = (self.max_t_train - self.min_steps) / max(self.warmup_epochs, 1)
            target_t_train = min(
                self.max_t_train,
                self.min_steps + int(epoch * step_size)
            )

            # 随机回滚机制
            if random.random() < self.rollback_prob and target_t_train > self.min_steps:
                rollback_ratio = 0.5 + 0.25 * random.random()
                new_t_train = max(self.min_steps, int(rollback_ratio * target_t_train))
                self.rollback_prob *= self.rollback_decay
                print(f"  [Scheduler] Rollback at epoch {epoch}: {self.current_t_train} → {new_t_train}")
                self.current_t_train = new_t_train
            else:
                if target_t_train > self.current_t_train:
                    self.current_t_train = target_t_train
                    if self.use_lr_schedule:
                        self._reset_lr_for_new_window()

        self.prev_t_train = self.current_t_train
        return int(self.current_t_train)

    # ================================================================
    # 获取状态
    # ================================================================

    def get_current_lr(self):
        """获取当前学习率"""
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        return None

    def get_state(self):
        """获取当前完整状态（用于logging）"""
        state = {
            'epoch': self.current_epoch,
            't_train': self.current_t_train,
            'progress': self.current_t_train / self.max_t_train
        }

        if self.adaptive_gate:
            state['epochs_in_window'] = self.epochs_in_window
            state['epochs_no_improve'] = self.epochs_no_improve
            state['best_loss_in_window'] = self.best_loss_in_window

        if self.use_causal_weights:
            state['epsilon'] = self.current_epsilon

        if self.use_lr_schedule and self.optimizer is not None:
            state['lr'] = self.optimizer.param_groups[0]['lr']
            state['in_warmup'] = self.in_warmup
            state['target_lr'] = self._get_target_lr_for_epoch()

        return state

    def reset(self):
        """重置调度器状态"""
        self.current_t_train = self.min_steps
        self.prev_t_train = None
        self.current_epoch = 0
        self.rollback_prob = self.rollback_prob_init

        # 重置自适应门控状态
        self.best_loss_in_window = float('inf')
        self.epochs_no_improve = 0
        self.epochs_in_window = 0

        # 重置因果权重
        self.current_epsilon = self.epsilon_start

        # 重置lr调度
        self.current_warmup_epochs = 0
        self.in_warmup = False
        self.epochs_since_expansion = 0

        if self.base_lr is not None:
            self.lr_at_window_start = self.base_lr
            self.lr_target = self.base_lr

        if self.use_lr_schedule and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr
            self._create_new_scheduler()


if __name__ == '__main__':
    import torch
    import math
    import matplotlib.pyplot as plt
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 临时忽略冲突（可能影响性能）

    # 虚拟模型参数和优化器（仅用于演示）
    model_params = [torch.nn.Parameter(torch.randn(1))]
    optimizer = Adam(model_params, lr=1e-3)  # 初始学习率设为 1e-3（最大值）

    # 调度器参数
    num_training_steps = 1500  # 总训练步数
    num_warmup_steps = 1500 * 0.05  # 预热步数（如 50 或总步数的 5%）
    lr_min = 1e-8  # 最小学习率

    epoch_set = 1501
    bfe = math.ceil(1000 / 20)
    steps = bfe * epoch_set
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=steps * 0.02, num_training_steps=steps
    )

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps,
    #     lr_min=lr_min
    # )

    learning_rates = []
    for step in range(steps):
        # 获取当前学习率（需先调用 optimizer.step() 更新步数）
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])

    plt.figure(figsize=(10, 5))
    plt.plot(range(steps), learning_rates, label='Learning Rate')
    plt.axvline(x=num_warmup_steps, color='r', linestyle='--', label='Warmup End')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Schedule with Warmup')
    plt.legend()
    plt.grid(True)
    plt.show()
