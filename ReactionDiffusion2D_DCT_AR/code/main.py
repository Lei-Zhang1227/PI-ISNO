"""
0506：
	TEST4:
EXP3-用低分辨率的数据做pretrain
Exp3的原始分辨率是128*128*100
A.	低分辨率data训练
B.	低分辨率data训练+高分辨率pde的实例微调
C.	低分辨率data+高分辨率pde的协同训练
是直接输入低分辨率数据，还是对输出的高分辨率数据做下采样呢？？
D.	低分辨率data+高分辨率pde的协同训练+高分辨率pde的实例微调
E.	高分辨率pde训练
0507
整理一下代码逻辑，方便后续进行其他实验；
0508
	加载模型再训练的逻辑；
	Checkpoint的保存和加载；
0509
1.	使用 Xavier/Glorot 初始化（nn.init.xavier_normal_）和零初始化（nn.init.zeros_）对模型的线性层（nn.Linear）和卷积层进行参数初始化；
2.	在训练初期修改loss计算方式为mean；
3.	添加数据归一化操作，并且是样本单独归一化
0512
添加PDEloss，进行协同训练实验；

"""
import os
import sys
import pandas as pd
import time

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath('..'))
print(sys.path)
from argparse import ArgumentParser

print("模块导入完成: from argparse import ArgumentParser")
import yaml
import shutil
import re
from tqdm import trange

print("模块导入完成: tqdm")
from loss import *

print("模块导入完成: from loss import *")
from utils import *

print("模块导入完成: from utils import *")
from datetime import datetime

print("模块导入完成: from datetime import datetime")
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

print("模块导入完成: from torch.utils.data import Dataset")
from NOs_dict.models import CosNO_II as Model
import warnings

print("模块导入完成: from NOs_dict.models import CosNO_II as Model")
from utilities import *

print("模块导入完成: from utilities import *")
import h5py
import math
import torch
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer
import random
import math


class TeacherForcingScheduler:
    def __init__(self,
                 initial_ratio=1.0,
                 final_ratio=0.0,
                 decay_steps=10000,
                 time_decay_factor=0.9,
                 mode='linear',
                 disable_tf=False):  # 新增：完全禁用TF的开关
        """
        Args:
            initial_ratio: 初始TF比例 (1.0=100%使用真实值)
            final_ratio: 最终TF比例
            decay_steps: 衰减步数
            time_decay_factor: 时间步衰减因子 (0.9表示较远时间步更快放弃TF)
            mode: 'linear'/'exponential' 衰减模式
            disable_tf: 是否完全禁用Teacher Forcing (True=完全禁用)
        """
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.decay_steps = decay_steps
        self.time_decay_factor = time_decay_factor
        self.mode = mode
        self.disable_tf = disable_tf  # 新增：完全禁用开关
        self.global_step = 0

    def get_ratio(self, t_step=None):
        """获取当前全局TF比例（如果禁用TF则始终返回0）"""
        if self.disable_tf:
            return 0.0  # 完全禁用时，比例固定为0

        progress = min(self.global_step / self.decay_steps, 1.0)

        if self.mode == 'linear':
            current_ratio = self.initial_ratio - progress * (self.initial_ratio - self.final_ratio)
        else:  # exponential
            current_ratio = self.initial_ratio * (self.final_ratio / self.initial_ratio) ** progress

        # 时间步衰减
        if t_step is not None:
            inverted_t = (91 - t_step) / 10
            current_ratio *= (self.time_decay_factor ** inverted_t)

        return max(current_ratio, 0.0)

    def step(self):
        """更新训练步数"""
        self.global_step += 1

    def should_use_teacher_forcing(self, t_step=None):
        """决定当前是否使用TF（禁用时始终返回False）"""
        if self.disable_tf:
            return False  # 完全禁用时，永远不使用TF
        return random.random() < self.get_ratio(t_step)


# print("模块导入完成: import h5py")
# print(f"当前运行的文件: {os.path.abspath(__file__)}")
# print(f"Python 路径: {sys.path}")


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'])
        return loss


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


lap_2d_op = [[[[0, 0, -1 / 12, 0, 0],
               [0, 0, 4 / 3, 0, 0],
               [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
               [0, 0, 4 / 3, 0, 0],
               [0, 0, -1 / 12, 0, 0]]]]


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=5, name=''):
        super(Conv2dDerivative, self).__init__()
        self.deno = deno
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)
        # 固定的导数算子
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()
        self.deno = deno
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)
        # 固定的导数算子
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class loss_generator(nn.Module):
    ''' 用于物理损失计算 '''

    def __init__(self, dt=(1.0 / 2), dx=(1.0 / 100)):
        super(loss_generator, self).__init__()
        self.dx = dx

        # 空间导数算子，转到 device
        self.laplace = Conv2dDerivative(
            DerFilter=lap_2d_op,
            deno=(dx ** 2),
            kernel_size=5,
            name='laplace_operator').to(device)

        # 时间导数算子，转到 device
        self.dt = Conv1dDerivative(
            DerFilter=[[[-1, 1, 0]]],
            deno=(dt * 1),
            kernel_size=3,
            name='partial_t').to(device)

    def get_phy_Loss(self, output):
        '''
        计算物理残差，输入 shape 为 [time, channel, height, width]
        '''
        # 空间导数
        laplace_u = self.laplace(output[0:-2, 0:1, :, :])
        laplace_v = self.laplace(output[0:-2, 1:2, :, :])

        # 时间导数 - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height, width, channel, time]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = self.dt(u_conv1d)  # 时间步减少2
        u_t = u_t.reshape(leny, lenx, 1, lent - 2)
        u_t = u_t.permute(3, 2, 0, 1)  # [time-2, channel, height, width]

        # 时间导数 - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)
        v_conv1d = v_conv1d.reshape(lenx * leny, 1, lent)
        v_t = self.dt(v_conv1d)
        v_t = v_t.reshape(leny, lenx, 1, lent - 2)
        v_t = v_t.permute(3, 2, 0, 1)

        # 对应区域
        u = output[0:-2, 0:1, 2:-2, 2:-2]
        v = output[0:-2, 1:2, 2:-2, 2:-2]

        # 保证形状一致
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # Gray-Scott 模型参数
        Du = 0.001
        Dv = 0.005
        k = 0.005

        f_u = (Du * laplace_u + u - (u ** 3) - k - v - u_t)
        f_v = (Dv * laplace_v + u - v - v_t)
        return f_u, f_v

    def get_phy_LossII(self, output):
        '''
        计算物理残差，输入 shape 为 [sample, time, channel, height, width]
        '''
        # 获取输入形状
        num_samples = output.shape[0]  # 样本数
        time_steps = output.shape[1]  # 时间步数
        channels = output.shape[2]  # 通道数
        height = output.shape[3]  # 高度
        width = output.shape[4]  # 宽度

        # 调整输入形状以适应卷积操作
        # 将 sample 和 time 维度合并为 batch 维度
        output_reshaped = output.reshape(num_samples * time_steps, channels, height,
                                         width)  # [sample * time, channel, height, width]

        # 空间导数
        laplace_u = self.laplace(output_reshaped[:, 0:1, :, :])  # [sample * time, 1, height, width]
        laplace_v = self.laplace(output_reshaped[:, 1:2, :, :])  # [sample * time, 1, height, width]

        # 恢复 sample 和 time 维度
        laplace_u = laplace_u.reshape(num_samples, time_steps, 1, height - 4,
                                      width - 4)  # [sample, time, 1, height, width]
        laplace_v = laplace_v.reshape(num_samples, time_steps, 1, height - 4,
                                      width - 4)  # [sample, time, 1, height, width]
        laplace_u = laplace_u[:, 0:-2, :, :, :]
        laplace_v = laplace_v[:, 0:-2, :, :, :]
        # 时间导数 - u
        u = output[:, :, 0:1, 2:-2, 2:-2]  # [sample, time, 1, height-4, width-4]
        u_conv1d = u.permute(0, 3, 4, 2, 1)  # [sample, height-4, width-4, 1, time]
        u_conv1d = u_conv1d.reshape(num_samples * (height - 4) * (width - 4), 1,
                                    time_steps)  # [sample * (height-4) * (width-4), 1, time]
        u_t = self.dt(u_conv1d)  # [sample * (height-4) * (width-4), 1, time-2]
        u_t = u_t.reshape(num_samples, height - 4, width - 4, 1,
                          time_steps - 2)  # [sample, height-4, width-4, 1, time-2]
        u_t = u_t.permute(0, 4, 3, 1, 2)  # [sample, time-2, 1, height-4, width-4]

        # 时间导数 - v
        v = output[:, :, 1:2, 2:-2, 2:-2]  # [sample, time, 1, height-4, width-4]
        v_conv1d = v.permute(0, 3, 4, 2, 1)  # [sample, height-4, width-4, 1, time]
        v_conv1d = v_conv1d.reshape(num_samples * (height - 4) * (width - 4), 1,
                                    time_steps)  # [sample * (height-4) * (width-4), 1, time]
        v_t = self.dt(v_conv1d)  # [sample * (height-4) * (width-4), 1, time-2]
        v_t = v_t.reshape(num_samples, height - 4, width - 4, 1,
                          time_steps - 2)  # [sample, height-4, width-4, 1, time-2]
        v_t = v_t.permute(0, 4, 3, 1, 2)  # [sample, time-2, 1, height-4, width-4]

        # 提取对应区域
        u = output[:, 0:-2, 0:1, 2:-2, 2:-2]  # [sample, time-2, 1, height-4, width-4]
        v = output[:, 0:-2, 1:2, 2:-2, 2:-2]  # [sample, time-2, 1, height-4, width-4]

        # 保证形状一致
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # Gray-Scott 模型参数
        Du = 0.001
        Dv = 0.005
        k = 0.005

        # 计算物理残差
        f_u = (Du * laplace_u + u - (u ** 3) - k - v - u_t)  # [sample, time-2, 1, height-4, width-4]
        f_v = (Dv * laplace_v + u - v - v_t)  # [sample, time-2, 1, height-4, width-4]

        return f_u, f_v


def loss_gen(output, loss_func):
    '''计算物理损失'''
    # 周期性边界条件的 padding
    # output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
    # output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_LossII(output)
    # residual_stitasticII(f_u)
    # residual_stitasticII(f_v)
    loss = mse_loss(f_u[:, 10:, :, :, :], torch.zeros_like(f_u[:, 10:, :, :, :]).to(device)) + \
           mse_loss(f_v[:, 10:, :, :, :], torch.zeros_like(f_v[:, 10:, :, :, :]).to(device))
    return f_u, f_v, loss


class FNODatasetMult(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False, test_ratio=0.1
                 ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        self.file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5'
        self.t_step = reduced_resolution_t

        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])

        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape) - 1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

            # Extract spatial dimension of data
            dim = len(data.shape) - 2

            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(seed_group["grid"]["x"], dtype='f')
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                grid = torch.stack((X, Y), axis=-1)
            elif dim == 3:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                z = np.array(seed_group["grid"]["z"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z)
                grid = torch.stack((X, Y, Z), axis=-1)

        return data[..., ::self.t_step, :][..., :self.initial_step, :], data[..., ::self.t_step, :], grid


def load_model_and_params(model_path, args):
    # 自动检测设备，若无GPU则映射到CPU
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=map_location)

    # 打印模型文件内容
    print("\n===== Loaded Checkpoint Contents =====")
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor of shape {value.shape}")
        elif isinstance(value, list):
            print(f"{key}: List of length {len(value)}")
        else:
            print(f"{key}: {value}")
    print("====================================\n")

    # 提取模型参数
    model_params = {
        'batch_size': checkpoint.get('batch_size', 5),
        'learning_rate': checkpoint.get('learning_rate', 1e-3),
        'width': checkpoint.get('width', 24),
        'modes': checkpoint.get('modes', 24),
        'sub': checkpoint.get('sub', 1),
        'weight_decay': checkpoint.get('weight_decay', 1e-4),
        'epochs': checkpoint.get('epochs', None)
    }

    # 初始化模型并加载权重
    model = Model(args.initial_step * 2 + 2, model_params['modes'], model_params['width'], 1, out_channels=2, dim=2,
                  triL=0).to(device)
    model.load_state_dict(checkpoint['model'])

    # 打印模型结构
    print("\n===== Model Architecture =====")
    print(model)
    print("====================================\n")

    # 打印每层的参数数量
    print("===== Parameters Per Layer =====")
    layer_param_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            layer_param_counts[name] = param_count
            print(f"{name}: {param_count} parameters")
    print("====================================\n")

    # 计算总参数量
    total_params = sum(layer_param_counts.values())
    print(f"Total Parameters: {total_params}")
    print("====================================\n")

    print(f"Model loaded from {model_path} with parameters: {model_params}")
    return model, model_params


class FNODatasetMultII(Dataset):
    def __init__(self, filename, initial_step=10, if_test=False, test_ratio=0.1):
        """
        优化后的数据集类，预先计算并缓存网格数据

        :param filename: 数据文件路径
        :param initial_step: 初始时间步数
        :param if_test: 是否为测试集
        :param test_ratio: 测试集比例
        """
        self.file_path = filename
        self.initial_step = initial_step

        # 加载网格数据并预计算网格矩阵
        with h5py.File(self.file_path, 'r') as h5_file:
            # 加载下采样后的网格坐标
            x_coords = torch.tensor(np.array(h5_file["grid/x"]), dtype=torch.float32)
            y_coords = torch.tensor(np.array(h5_file["grid/y"]), dtype=torch.float32)

            # 预计算网格矩阵 (只执行一次)
            X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
            self.grid = torch.stack((X, Y), axis=-1)  # [x, y, 2]

            # 获取样本列表
            data_list = sorted([k for k in h5_file.keys() if k not in ["grid", "x", "y"]])

        # 划分训练/测试集
        test_idx = int(len(data_list) * (1 - test_ratio))
        self.data_list = np.array(data_list[test_idx:] if if_test else data_list[:test_idx])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        高效的数据加载方法，直接使用预计算的网格
        """
        with h5py.File(self.file_path, 'r') as h5_file:
            sample_group = h5_file[self.data_list[idx]]

            # 加载下采样数据
            data = torch.tensor(np.array(sample_group["data"]), dtype=torch.float32)

            # 转换为 [x, y, t, v] 格式
            data = data.permute(1, 2, 0, 3)

            # 分割初始条件和完整数据
            input_data = data[..., :self.initial_step, :]
            full_data = data

        # 直接返回预计算的网格
        return input_data, full_data, self.grid


# class FNODatasetMultII(Dataset):
#     def __init__(self, filename, initial_step=10, if_test=False, test_ratio=0.1,
#                  norm_type='channel', norm_stats=None):
#         """
#         改进的数据集类，支持数据归一化
#
#         :param filename: 数据文件路径
#         :param initial_step: 初始时间步数
#         :param if_test: 是否为测试集
#         :param test_ratio: 测试集比例
#         :param norm_type: 归一化类型 ('channel', 'global')
#         :param norm_stats: 预计算的归一化统计量 (mean, std)
#         """
#         self.file_path = filename
#         self.initial_step = initial_step
#         self.norm_type = norm_type
#         self.norm_stats = norm_stats  # (mean, std)
#
#         # 加载网格数据并预计算网格矩阵
#         with h5py.File(self.file_path, 'r') as h5_file:
#             # 加载下采样后的网格坐标
#             x_coords = torch.tensor(np.array(h5_file["grid/x"]), dtype=torch.float32)
#             y_coords = torch.tensor(np.array(h5_file["grid/y"]), dtype=torch.float32)
#
#             # 预计算网格矩阵 (只执行一次)
#             X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
#             self.grid = torch.stack((X, Y), axis=-1)  # [x, y, 2]
#
#             # 获取样本列表
#             data_list = sorted([k for k in h5_file.keys() if k not in ["grid", "x", "y"]])
#
#         # 划分训练/测试集
#         test_idx = int(len(data_list) * (1 - test_ratio))
#         self.data_list = np.array(data_list[test_idx:] if if_test else data_list[:test_idx])
#
#         # 如果不是测试集且没有提供归一化统计量，则计算归一化参数
#         if not if_test and norm_stats is None:
#             self._compute_norm_stats()
#
#     def _compute_norm_stats(self):
#         """预计算归一化统计量 (基于训练集)"""
#         all_data = []
#         with h5py.File(self.file_path, 'r') as h5_file:
#             for name in self.data_list[:100]:  # 使用前100个样本计算统计量
#                 data = np.array(h5_file[name]["data"])
#                 all_data.append(data)
#
#         all_data = np.stack(all_data)  # [sample, t, x, y, v]
#
#         if self.norm_type == 'channel':
#             # 通道独立归一化
#             mean = all_data.mean(axis=(0, 1, 2, 3), keepdims=True)  # [1, 1, 1, 1, v]
#             std = all_data.std(axis=(0, 1, 2, 3), keepdims=True)
#         else:  # 'global'
#             # 全局归一化
#             mean = all_data.mean()
#             std = all_data.std()
#
#         self.norm_stats = (torch.tensor(mean, dtype=torch.float32),
#                            torch.tensor(std, dtype=torch.float32))
#
#     def _normalize(self, data):
#         """应用归一化"""
#         if self.norm_stats is None:
#             return data
#
#         mean, std = self.norm_stats
#         if self.norm_type == 'channel':
#             # 保持原始维度 [x, y, t, v]
#             return (data - mean) / (std + 1e-8)
#         else:
#             # 全局归一化
#             return (data - mean) / (std + 1e-8)
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def __getitem__(self, idx):
#         """
#         改进的数据加载方法，包含归一化处理
#         """
#         with h5py.File(self.file_path, 'r') as h5_file:
#             sample_group = h5_file[self.data_list[idx]]
#             data = torch.tensor(np.array(sample_group["data"]), dtype=torch.float32)
#             data = data.permute(1, 2, 0, 3)  # [x, y, t, v]
#
#         # 应用归一化
#         norm_data = self._normalize(data)
#
#         # 分割初始条件和完整数据
#         input_data = norm_data[..., :self.initial_step, :]
#         full_data = norm_data
#
#         return input_data, full_data, self.grid
#
#     def get_denormalize_fn(self):
#         """获取反归一化函数"""
#         if self.norm_stats is None:
#             return lambda x: x
#
#         mean, std = self.norm_stats
#         if self.norm_type == 'channel':
#             return lambda x: x * std + mean
#         else:
#             return lambda x: x * std + mean


def load_samplesII(file_path, sample_keys):
    data_list = []
    with h5py.File(file_path, 'r') as f:
        for key in sample_keys:
            data = f[f'{key}/data'][:]
            data_list.append(data)
        x = f[f'{sample_keys[0]}/grid/x'][:]
        y = f[f'{sample_keys[0]}/grid/y'][:]
        t = f[f'{sample_keys[0]}/grid/t'][:]

    # 沿 sample 维度堆叠
    data_stacked = np.stack(data_list, axis=0)  # [N, time, height, width, channel]
    return data_stacked, x, y, t


def get_spatial_intervals(file_path):
    """
    从h5文件中获取空间坐标间隔(dx, dy)
    :param file_path: h5文件路径
    :return: (dx, dy) 空间间隔
    """
    with h5py.File(file_path, 'r') as h5_file:
        # 获取第一个样本的网格数据
        x_coords = np.array(h5_file["grid"]["x"], dtype='f')
        y_coords = np.array(h5_file["grid"]["y"], dtype='f')

        # 计算间隔
        dx = x_coords[1] - x_coords[0]
        dy = y_coords[1] - y_coords[0]
        print(x_coords[0], x_coords[-1])

    return dx, dy


def init_weights(m):
    """
    初始化模型参数
    Args:
        m: 模型或子模块
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # 对线性层和卷积层的权重进行 Kaiming 初始化
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            # 对偏置进行零初始化
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        # 对自定义可训练矩阵进行 Kaiming 初始化
        nn.init.xavier_normal_(m)


class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.gradient_data = defaultdict(list)
        self._register_hooks()

    def _register_hooks(self):
        """为所有参数注册梯度钩子"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(name))

    def _make_hook(self, name):
        """生成梯度记录钩子"""

        def hook(grad):
            self.gradient_data[name].append({
                'max': grad.abs().max().item(),
                'norm': grad.norm().item(),
                'mean': grad.abs().mean().item()
            })

        return hook

    def get_summary(self):
        """计算当前梯度统计摘要"""
        if not self.gradient_data:
            return {}

        # 全局统计
        all_norms = [v['norm'] for stats in self.gradient_data.values() for v in stats]
        summary = {
            'global_norm': torch.norm(torch.tensor(all_norms)).item(),
            'max_grad': max(v['max'] for stats in self.gradient_data.values() for v in stats),
            'mean_grad': torch.tensor([v['mean'] for stats in self.gradient_data.values() for v in stats]).mean().item()
        }

        # 按层统计
        for name, stats in self.gradient_data.items():
            summary[f'{name}_norm'] = stats[-1]['norm']  # 只保留最新值

        self.gradient_data.clear()  # 清空当前批次数据
        return summary


# 使用示例


def run(config):
    # region prepare
    ################################################################
    # prepare
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # endregion
    # region dataloader
    ################################################################
    # dataloader
    ################################################################
    data_config = config['data']
    batch_size = config['train']['batchsize']
    initial_step = config['train']['initial_step']
    file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_4_1.h5'
    # train_data = FNODatasetMultII(filename=file_path,
    #                               initial_step=initial_step, if_test=False)
    # val_data = FNODatasetMultII(filename=file_path,
    #                             initial_step=initial_step, if_test=True)
    #
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=batch_size,
    #                                            num_workers=8,
    #                                            shuffle=True,
    #                                            pin_memory=True,  # 启用固定内存加速传输
    #                                            persistent_workers=True)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=8, shuffle=False)

    train_data = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                                initial_step=initial_step,
                                reduced_resolution=1,
                                reduced_resolution_t=1,
                                reduced_batch=5)
    val_data = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                              initial_step=initial_step,
                              reduced_resolution=1,
                              reduced_resolution_t=1,
                              reduced_batch=5,
                              if_test=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, num_workers=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, num_workers=2, shuffle=False)
    # train_data = FNODatasetMultII(filename=file_path,
    #                               initial_step=initial_step, if_test=False)
    # val_data = FNODatasetMultII(filename=file_path,
    #                             initial_step=initial_step, if_test=True)
    #
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=5,
    #                                            num_workers=8,
    #                                            shuffle=True,
    #                                            pin_memory=True,  # 启用固定内存加速传输
    #                                            persistent_workers=True)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=8, shuffle=False)

    train_size, test_size = train_data.data_list.shape[0], val_data.data_list.shape[0]
    ntrain, ntest = train_size, test_size
    print('size-of-train/val:', train_size, test_size)
    # dx, dy = get_spatial_intervals(file_path)
    # print(f"x方向间隔: {dx}, y方向间隔: {dy}")
    # dt = 0.05  # 小时间步
    # train_size, test_size = train_data.data_list.shape[0], val_data.data_list.shape[0]
    # ntrain, ntest = train_size, test_size
    print(
        f'{datetime.now()} --- set dataset，batch size: {batch_size}, Train loader lens：{train_size}, Test loader lens：{test_size}')
    ################################################################
    # location
    ################################################################
    # endregion
    # region location
    first_dic = f"./{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    if args.pretrain is not None:
        shutil.copy('./information.yaml', f"./{config['prepare']['project']}/info_{args.pretrain}.yaml")
        # shutil.copy(f'./info_{args.pretrain}.yaml', f"./{config['prepare']['project']}")
        # 这里的information并不在目标文件夹内
    else:
        shutil.copy('./information.yaml', f"./{config['prepare']['project']}")
    os.chdir(first_dic)
    current_directory = os.getcwd()
    print(
        f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---, current_directory:{current_directory}")
    writer = SummaryWriter(log_dir="/output/logs")  #
    # endregion
    # region model
    ################################################################
    # model
    ################################################################
    # 这里有所不同的是，不再将这个一维含时的问题视为二维问题
    # 这个还是先暂且不论，跑通了再优化这些有的没的
    # _trans = PARTIAL(Wrapper, [dctI_SPFNO,dctI_SPFNO])
    # _itrans = PARTIAL(Wrapper, [idctI_SPFNO,idctI_SPFNO])
    # T = Transform(_trans, _itrans)
    # # 定义模型
    # Model = PARTIAL(SOL1dII, T)
    # input_channel = config['model']['input_channel'] * config['data']['initial_step'] + 1
    # model = Model(input_channel, config['model']['modes'], config['model']['width'],
    #               config['model']['bandwidth'], out_channels=config['model']['output_channel'],
    #               dim=config['model']['dim'], triL=config['model']['triL']).to(device)  # .to(torch.float32)
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    model = Model(10 * 2 + 2, modes, width, bandwidth, out_channels=2, dim=2, triL=0).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    # 模型初始化
    # model.apply(init_weights)
    # 验证一下初始化结果
    # print("FC1 weight mean/std:", model.fc1.weight.mean().item(), model.fc1.weight.std().item())
    # print("FC1 bias:", model.fc1.bias[:5])
    #
    # print("FC2 weight mean/std:", model.fc2.weight.mean().item(), model.fc2.weight.std().item())
    # print("FC2 bias:", model.fc2.bias[:5])
    #
    # print("Conv0 weights mean/std:", model.conv0.weights.mean().item(), model.conv0.weights.std().item())
    # print("W0 weight mean/std:", model.w0.weight.mean().item(), model.w0.weight.std().item())
    # print("W0 bias:", model.w0.bias[:5])
    # endregion
    # region 定义优化器
    ################################################################
    # 定义优化器
    ################################################################
    lr = config['train']['base_lr']
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if config['train']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config['train']['milestones'],
                                                         gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['train']['gamma'],
                                                               threshold=1e-2, patience=config['train']['patience'],
                                                               verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['patience'],
                                                    gamma=config['train']['gamma'])
    # 定义Teacher Force 调度器
    tf_scheduler = TeacherForcingScheduler(
        initial_ratio=1.0,
        final_ratio=0.0,
        decay_steps=500 * 90,
        time_decay_factor=0.9,
        disable_tf=False  # 关键参数：完全禁用
    )

    print(f'{datetime.now()} --- set optimizer, scheduler ---')
    # endregion
    # region load model
    ################################################################
    # load model
    ################################################################
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        # 获取 epoch
        epoch = checkpoint['epoch']
        print(f'模型【{args.pretrain}】已加载')
        if args.load_lr:
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
            scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率调度器状态
    else:
        loss_list = []
        test_loss_list = []
        lr_list = []
    # endregion
    # region loss
    ################################################################
    # loss
    ################################################################
    grad_monitor = GradientMonitor(model)
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    initial_step = int(config['train']['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    model_save_record = []
    best_error = 1000.0
    print(config['train']['size_average'])
    myloss = LpLoss(size_average=config['train']['size_average'])
    loss_fn = myloss

    if args.pretrain is not None:
        ebar = trange(epoch, epoch + config['train']['epochs'], desc="Epoch")
    else:
        ebar = trange(config['train']['epochs'], desc="Epoch")
    desc = DescStr()
    model.train()

    for e in ebar:
        # 内层进度条
        # Loss_f = 0.0
        # Loss_init = 0.0
        # Loss_data = 0.0
        # Loss_all = 0.0
        count = 0
        t00 = time.time()
        max_norm = 100.0
        train_iter = iter(train_loader)
        epoch_loss = 0.0
        for b in trange(len(train_loader), file=desc, desc="batch"):
            loss = 0
            t0 = time.time()
            """
            x:[b, nx, self.initial_step，1]
            y:[b, nx, nt, 1]
            grid:[b, nx, 1]
            """
            xx, yy, grid = next(train_iter)
            xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                       non_blocking=True)
            # print(grid)
            t1 = time.time()
            optimizer.zero_grad()
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
            outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
            # pred = yy[..., :initial_step, :]
            t2 = time.time()
            nx = xx.shape[1]
            pred = torch.empty((batch_size, nx, nx, 101, 2), device=xx.device)
            #  这里的代码逻辑需要优化一下
            pred[..., :initial_step, :] = yy[..., :initial_step, :]
            for t in range(initial_step, t_train):
                inp = xx.reshape(inp_shape)
                y = yy[..., t:t + 1, :]
                out = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
                _batch = out.size(0)
                # print('out[::4,::4,...].shape:', out[::4, ::4, ...].shape)
                loss += myloss(out.reshape(_batch, -1), y.reshape(_batch, -1))
                # 这个是叠加的data loss
                pred[..., t:t + 1, :] = out
                use_tf = tf_scheduler.should_use_teacher_forcing(t_step=t)
                if use_tf:
                    xx = yy[..., t + 1 - initial_step:t + 1, :]
                else:
                    xx = torch.cat((xx[..., 1:, :], out), dim=-2)

            assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
            # print(loss.item())
            # print('pred.shape:', pred.shape)
            t3 = time.time()
            if config['train']['loss_mode'] == 'both':
                pred_permuted = pred.permute(0, 3, 4, 1, 2)
                # print('output_permuted.shape:', pred_permuted.shape)
                loss_func = loss_generator(dt, dx).to(device)
                f_u, f_v, loss_phy = loss_gen(pred_permuted, loss_func)
                total_loss = (loss_phy * f_weight + loss / (t_train - initial_step) * data_weight).to(torch.float32)
            elif config['train']['loss_mode'] == 'data':
                total_loss = loss
                loss_phy = torch.Tensor([0.0])
            else:
                pred_permuted = pred.permute(0, 3, 4, 1, 2)
                # print('output_permuted.shape:', pred_permuted.shape)
                loss_func = loss_generator(dt, dx).to(device)
                f_u, f_v, loss_phy = loss_gen(pred_permuted, loss_func)
                total_loss = loss_phy
                assert not torch.isnan(loss_phy).any(), "NaN in loss"
            t4 = time.time()
            total_loss.backward()
            # region 梯度监控
            grad_stats = grad_monitor.get_summary()
            # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            # if total_norm > max_norm * 2:  # 如果梯度范数远大于 max_norm
            #     max_norm *= 1.1  # 适当增大 max_norm
            # elif total_norm < max_norm * 0.5:  # 如果梯度范数远小于 max_norm
            #     max_norm *= 0.9  # 适当减小 max_norm
            # endregion
            optimizer.step()
            t5 = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            loss_list.append([loss.item(), loss_phy.item(), total_loss.item()])
            new_desc = f"epoch {e + 1}: {desc.read(b)},Loss: {loss.item():.4e},Loss_phy: {loss_phy.item():.4e}, lr:{current_lr},Gradient Norm: {grad_stats['global_norm']:.1e}, Max Norm: {grad_stats['max_grad']:.1e},use_tf: {use_tf}"
            ebar.set_description(new_desc)
            batch_loss = loss.item()
            writer.add_scalar("Loss/Batch", batch_loss, b)
            epoch_loss += batch_loss
        writer.add_scalar("Loss/Epoch", epoch_loss / len(train_loader), e)
        #             print(
        #                 f'enter loader:{t0 - t00}, data loader: {t1 - t0}, prepare: {t2 - t1},autoregressive:{t3 - t2}, culcalate loss: {t4 - t3}, backward:{t5 - t4}')
        lr_list.append(current_lr)
        scheduler.step()
        if best_error > loss.item():
            best_error = loss.item()
            model_save_record.append([e, loss.item()])
            save_checkpoint(model, e, optimizer, scheduler, loss_list,
                            lr_list, test_loss_list, model_save_record, filename=f'checkpoint-best')
        if e % config['train']['verbose_interval'] == 0:
            model.eval()
            val_l2_step = 0
            val_l2_full = 0
            val_l2_full_4_1 = 0
            val_l2_full_2_1 = 0
            with torch.no_grad():
                for xx, yy, grid in val_loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)
                    outp_shape = inp_shape[:-1] + [1, -1]

                    pred = yy[..., :initial_step, :]
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    for t in range(initial_step, yy.shape[-2]):
                        inp = xx.reshape(inp_shape)
                        y = yy[..., t:t + 1, :]
                        # im = model(inp, grid)
                        im = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
                        _batch = im.size(0)
                        loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                        pred = torch.cat((pred, im), -2)

                        xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                    val_l2_step += loss.item()
                    _batch = yy.size(0)
                    _pred = pred[..., initial_step:t_train, :]
                    _yy = yy[..., initial_step:t_train, :]
                    val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                    val_l2_full_4_1 += loss_fn(_pred[:, :4, ::4, ...].reshape(_batch, -1),
                                               _yy[:, :4, ::4, ...].reshape(_batch, -1)).item()
                    val_l2_full_2_1 += loss_fn(_pred[:, :2, ::2, ...].reshape(_batch, -1),
                                               _yy[:, :2, ::2, ...].reshape(_batch, -1)).item()

            test_l2 = val_l2_full / ntest
            test_l2_4_1 = val_l2_full_4_1 / ntest
            test_l2_2_1 = val_l2_full_2_1 / ntest
            # train_list.append(train_l2)
            test_loss_list.append(test_l2)
            print(f'epoch:{e}, test_l2: {test_l2},test_l2_4_1: {test_l2_4_1},test_l2_2_1: {test_l2_2_1}')
            if e % config['train']['check_epochs'] == 0:
                save_checkpoint(model, e, optimizer, scheduler, loss_list,
                                lr_list, test_loss_list, model_save_record, filename=f'checkpoint-{e}')
            model.train()
    writer.close()
    print(f'{datetime.now()} --- training succeed ---')
    # endregion


def test_afer_train(model):
    '''
    加载不同分辨率的数据集进行测试（这个的一般做法是这样的吗？？还是直接用最高分辨率的数据集？anyway，试试再说）
    :param model:
    :return:
    '''
    pass


def error_talk(pre, ref):
    abs_err = torch.abs(pre - ref)
    temporal_stats = {
        'mean': abs_err.mean(dim=(1, 2, 4)),  # (1,91) 空间和通道平均
        'max': abs_err.max(dim=1)[0].max(dim=1)[0],  # (1,91,2)->(1,91)
        'std': abs_err.std(dim=(1, 2, 4))  # (1,91) 时空波动程度
    }
    pass


def test(config, args):
    """
    一个全面的评测函数，可以再三个不同分辨率上计算模型的l2误差
    :param config:
    :param args:
    :return:
    """
    # region prepare
    ################################################################
    # prepare
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # endregion
    # region dataloader
    ################################################################
    # dataloader
    ################################################################
    data_config = config['data']
    batch_size = config['train']['batchsize']
    initial_step = config['train']['initial_step']
    TEST_1 = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                            initial_step=initial_step,
                            reduced_resolution=1,
                            reduced_resolution_t=1,
                            reduced_batch=5,
                            if_test=True)
    loader_1 = torch.utils.data.DataLoader(TEST_1, batch_size=1, num_workers=126, shuffle=False)
    # val_data = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
    #                           initial_step=initial_step,
    #                           reduced_resolution=1,
    #                           reduced_resolution_t=1,
    #                           reduced_batch=5,
    #                           if_test=True)
    #
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, num_workers=126, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, num_workers=126, shuffle=False)

    TEST_2 = FNODatasetMultII(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_2_1.h5',
                              initial_step=initial_step, if_test=True)
    loader_2 = torch.utils.data.DataLoader(TEST_2, batch_size=1, num_workers=28, shuffle=False)

    TEST_3 = FNODatasetMultII(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_4_1.h5',
                              initial_step=initial_step, if_test=True)
    loader_3 = torch.utils.data.DataLoader(TEST_3, batch_size=1, num_workers=28, shuffle=False)
    test_size = TEST_1.data_list.shape[0]
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    first_dic = f"/code/ex3/{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    ################################################################
    # model
    ################################################################
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    model = Model(10 * 2 + 2, modes, width, bandwidth, out_channels=2, dim=2, triL=0).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    ################################################################
    # load model
    ################################################################
    if args.pretrain is not None:
        model_name = args.pretrain[:-8]
        model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
        checkpoint = torch.load(args.pretrain)
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']  # checkpoint['loss_for_train']
        test_loss_list = checkpoint.get('test_loss_list')  # checkpoint.get('test_loss_list')
        if test_loss_list is None:
            print("[Warning] Checkpoint does not contain 'test_loss_list'.")
            test_loss_list = []
        lr_list = checkpoint.get('lr_list')

        if lr_list is None:
            print("[Warning] Checkpoint does not contain 'lr_list'.")
            lr_list = []
        # 获取 epoch
        epoch = checkpoint.get('epoch', checkpoint.get('epochs', None))
        print('epoch:', epoch)
        if epoch is None:
            print("[Warning] Checkpoint does not contain 'epoch' or 'epochs'.")
            epoch = 0
        print(f'模型【{args.pretrain}】已加载')

    else:
        print(os.getcwd())
        checkpoint = torch.load('checkpoint-best.pth.tar')
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        # 获取 epoch
        epoch = checkpoint['epoch']
        model_name = 'checkpoint-best'
        print(f'模型【checkpoint-best.pth.tar】已加载')
    # endregion
    # region loss carve
    batchsize = config['train']['batchsize']
    bfe = 900 / batchsize
    plot_loss(loss_list, bfe, f'loss_carve_for_{model_name}')
    print('损失曲线绘制完成~')
    # endregion
    # region evaluate
    ################################################################
    # loss
    ################################################################
    initial_step = int(config['train']['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    desc = DescStr()
    val_l2_full = []
    first = True
    train_iter = iter(loader_1)
    global_stats = {
        'temporal_mean': torch.zeros(91),  # 各时间步平均误差 (100样本平均)
        'temporal_std': torch.zeros(91),  # 各时间步标准差
        'spatial_mean': torch.zeros((2, 91)),  # 各通道空间平均误差
        'max_error': torch.zeros(100),  # 每个样本的最大误差
        'error_growth_rate': []  # 各样本误差增长率
    }  # 用于误差分析
    for b in trange(len(loader_1)):
        """
        x:[b, nx, self.initial_step，1]
        y:[b, nx, nt, 1]
        grid:[b, nx, 1]
        """
        xx, yy, grid = next(train_iter)
        xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                   non_blocking=True)  # 确保数据在相同设备上
        inp_shape = list(xx.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
        outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
        pred = yy[..., :initial_step, :]
        for t in range(initial_step, t_train):
            inp = xx.reshape(inp_shape)
            out = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
            _batch = out.size(0)
            pred = torch.cat((pred, out), -2)
            xx = torch.cat((xx[..., 1:, :], out), dim=-2)
        assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
        if first:
            print(f'分辨率：{pred.shape}')
            first = False
        _yy = yy[..., initial_step + 1:t_train, :]  # if t_train is not -1
        _pred = pred[..., initial_step + 1:t_train, :]
        _batch = yy.size(0)
        val_l2_full.append(loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item())
        # 误差统计
        # abs_err = torch.abs(_pred - _yy)
        # global_stats['temporal_mean'] += abs_err.mean(dim=(0, 1, 2, 3))  # 各时间步平均
        # global_stats['temporal_std'] += abs_err.std(dim=(0, 1, 2, 3))
        # global_stats['spatial_mean'][0] += abs_err[..., 0].mean(dim=(0, 1, 2))  # 通道0
        # global_stats['spatial_mean'][1] += abs_err[..., 1].mean(dim=(0, 1, 2))  # 通道1
        # global_stats['max_error'][b] = abs_err.max()
        # # 4. 计算误差增长率 (指数拟合)
        # y = abs_err.mean(dim=(0, 1, 2, 3)).numpy()
        # growth_rate = np.polyfit(np.arange(91), np.log(y + 1e-8), 1)[0]
        # global_stats['error_growth_rate'].append(growth_rate)

    test_l2_full = [np.mean(val_l2_full), np.max(val_l2_full), np.min(val_l2_full), np.std(val_l2_full)]

    global_stats['temporal_mean'] /= len(loader_1)
    global_stats['temporal_std'] /= len(loader_1)
    global_stats['spatial_mean'] /= len(loader_1)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(10, 101), global_stats['temporal_mean'],
             'b-', label='Mean Error')
    plt.fill_between(np.arange(10, 101),
                     global_stats['temporal_mean'] - global_stats['temporal_std'],
                     global_stats['temporal_mean'] + global_stats['temporal_std'],
                     alpha=0.2, color='b')
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Global Temporal Error (Mean ± Std over 100 Samples)')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_test.png', dpi=200)

    # corr_coeff = np.corrcoef(
    #     np.tile(np.arange(91), 100),
    #     np.concatenate([sample_err.flatten() for sample_err in sample_errors])
    # )[0, 1]
    # print(f"Error-Time Correlation: {corr_coeff:.3f}")

    val_l2_2 = []
    first = True
    train_iter_2 = iter(loader_2)
    for b in trange(len(loader_2)):
        """
        x:[b, nx, self.initial_step，1]
        y:[b, nx, nt, 1]
        grid:[b, nx, 1]
        """
        xx, yy, grid = next(train_iter_2)
        xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                   non_blocking=True)  # 确保数据在相同设备上
        inp_shape = list(xx.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
        outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
        pred = yy[..., :initial_step, :]
        for t in range(initial_step, t_train):
            inp = xx.reshape(inp_shape)
            out = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
            _batch = out.size(0)
            pred = torch.cat((pred, out), -2)
            xx = torch.cat((xx[..., 1:, :], out), dim=-2)
        if first:
            print(f'分辨率：{pred.shape}')
            first = False
        assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
        _yy = yy[..., initial_step + 1:t_train, :]  # if t_train is not -1
        _pred = pred[..., initial_step + 1:t_train, :]
        _batch = yy.size(0)
        val_l2_2.append(loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item())
    test_l2_2 = [np.mean(val_l2_2), np.max(val_l2_2), np.min(val_l2_2), np.std(val_l2_2)]

    val_l2_4 = []
    first = True
    train_iter_3 = iter(loader_3)
    for b in trange(len(loader_3)):
        """
        x:[b, nx, self.initial_step，1]
        y:[b, nx, nt, 1]
        grid:[b, nx, 1]
        """
        xx, yy, grid = next(train_iter_3)
        xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                   non_blocking=True)  # 确保数据在相同设备上
        inp_shape = list(xx.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
        outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
        pred = yy[..., :initial_step, :]
        for t in range(initial_step, t_train):
            inp = xx.reshape(inp_shape)
            out = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
            _batch = out.size(0)
            pred = torch.cat((pred, out), -2)
            xx = torch.cat((xx[..., 1:, :], out), dim=-2)
        assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
        if first:
            print(f'分辨率：{pred.shape}')
            first = False
        _yy = yy[..., initial_step + 1:t_train, :]  # if t_train is not -1
        _pred = pred[..., initial_step + 1:t_train, :]
        _batch = yy.size(0)
        val_l2_4.append(loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item())
    test_l2_4 = [np.mean(val_l2_4), np.max(val_l2_4), np.min(val_l2_4), np.std(val_l2_4)]
    df = pd.DataFrame(
        [test_l2_4, test_l2_2, test_l2_full],  # 数据
        columns=["Mean", "Max", "Min", "Std"],  # 列名
        index=["Test_L2_4", "Test_L2_2", "Test_L2_full"]  # 行名（可选）
    )
    # 写入 CSV 文件
    if args.pretrain is not None:
        csv_name = re.sub(r'[^a-zA-Z0-9_]', '_', args.pretrain)
        df.to_csv(f"l2_loss_stats_for_{csv_name}.csv")

    else:
        df.to_csv(f"l2_loss_stats_for_best.csv")
    print('模型数值评估完成~')
    # endregion
    # region visualize
    ################################################################
    # visualize_results
    ################################################################
    # 示例：可视化第 idx 个样本的第 t 个时间步
    Nx = Ny = 128
    nx = np.linspace(-1, 1, Nx)
    ny = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(nx, ny)
    if args.visualize:
        visualize_indices = list(map(int, args.visualize.split(',')))
    else:
        visualize_indices = [0, 50, 90]
    for index in visualize_indices:
        test_xx, test_yy, test_grid = TEST_1[index]
        test_xx, test_yy, test_grid = test_xx.to(device, non_blocking=True), test_yy.to(device,
                                                                                        non_blocking=True), test_grid.to(
            device, non_blocking=True)
        test_xx = test_xx.unsqueeze(0)
        test_yy = test_yy.unsqueeze(0)
        test_grid = test_grid.unsqueeze(0)
        test_inp_shape = list(test_xx.shape)
        test_inp_shape = test_inp_shape[:-2]
        test_inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
        test_outp_shape = test_inp_shape[:-1] + [1, -1]
        print('test_yy.shape:', test_yy.shape)
        print('test_grid.shape:', test_grid.shape)
        print(test_inp_shape)
        test_pred = test_yy[..., :initial_step, :]
        for t in range(initial_step, t_train):
            inp = test_xx.reshape(test_inp_shape)
            test_out = model(torch.cat([inp, test_grid], dim=-1)).reshape(test_outp_shape)
            _batch = test_out.size(0)
            test_pred = torch.cat((test_pred, test_out), -2)
            test_xx = torch.cat((test_xx[..., 1:, :], test_out), dim=-2)
        assert test_pred.shape == test_yy.shape, f"Tensor shapes do not match: {test_pred.shape} != {test_yy.shape}"
        t_list = [11, 50, 100]  # 选择第5个时间步
        for t in t_list:
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f'{index}-{t}', fontsize=10, y=1.02)
            for i in range(2):
                true_vals = [test_yy[0, ..., t, i].cpu() for i in range(2)]
                pred_vals = [test_pred[0, ..., t, i].cpu().detach().numpy() for i in range(2)]
                vmin_true = min(t.min() for t in true_vals)
                vmax_true = max(t.max() for t in true_vals)
                vmin_pred = min(p.min() for p in pred_vals)
                vmax_pred = max(p.max() for p in pred_vals)

                ax1 = plt.subplot(2, 2, i + 1, aspect='equal')  # 设置为正方形
                im1 = plt.pcolor(X, Y, true_vals[i], cmap="jet", vmin=vmin_true, vmax=vmax_true)
                plt.title(f'True Channel {i + 1}')
                plt.colorbar(im1, ax=ax1)

                # 预测值子图
                ax2 = plt.subplot(2, 2, i + 3, aspect='equal')  # 设置为正方形
                im2 = plt.pcolor(X, Y, pred_vals[i], cmap="jet", vmin=vmin_pred, vmax=vmax_pred)
                plt.title(f'Predicted Channel {i + 1}')
                plt.colorbar(im2, ax=ax2)
            plt.tight_layout()
            # 保存图像
            save_path = f'visualization_index_{index}_t_{t}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭当前图像，避免内存泄漏

    plot_vector_field_comparison(
        model_field=test_pred.squeeze().detach().cpu().numpy(),
        ref_field=test_yy.squeeze().detach().cpu().numpy(),
        output_gif_path="model_vs_ref.gif",
        time_steps=range(0, 101),  # 仅绘制前20帧
        plot_type="streamplot",  # 或 "quiver"
        downsample_stride=4,  # 降采样提升性能
        frame_duration=200,  # 每帧200ms
    )
    print('可视化完成~')
    # endregion
    # region error talk

    # endregion


def plot_loss(loss_list, bfe, title):
    # 转换为 NumPy 数组
    if all(isinstance(x, (int, float)) for x in loss_list):
        loss_array = np.array(loss_list)
        has_nan = np.isnan(loss_array)
        nan_indices = np.where(has_nan)  # 返回所有NaN的位置索引

        if not has_nan.any():
            print("所有损失值均正常，未检测到NaN。")
            cut_len = (len(loss_array) // bfe) * bfe
        else:
            print("检测到NaN值，首次出现位置：")

            # 按训练步骤顺序检查
            for step in range(len(loss_array)):
                if np.isnan(loss_array[step]).any():
                    print(f"第 {step} 步的以下损失为NaN:")
                    cut_len = (step // bfe) * bfe
                    break  # 只报告首次出现的位置

        steps = np.arange(cut_len)  # 横轴：训练步数
        # ---- 子图1：Training Loss (MSE) ----
        max_epoch = steps[-1] // bfe
        num_ticks = 10  # 只显示10个刻度
        tick_positions = np.linspace(steps[0], steps[-1], num=num_ticks, dtype=int)
        tick_labels = [str(pos // bfe) for pos in tick_positions]

        plt.plot(steps, loss_array[:cut_len], label='Training Loss', marker='o', markersize=1, color='blue',
                 linewidth=0.5)
        plt.xlabel('Training Steps', fontsize=10)
        plt.ylabel('Loss Value', fontsize=10)
        plt.yscale('log')
        plt.xticks(tick_positions, tick_labels)

        # plt.ylim(loss_array[0])
        plt.title(f'{title.capitalize()} Loss (l2)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f'{title}_loss_list.png')
    else:
        loss_array = np.array(loss_list)
        has_nan = np.isnan(loss_array)
        nan_row = None
        if has_nan.any():
            nan_row = np.where(np.any(has_nan, axis=1))[0][0]
            print(f"检测到NaN值，首次出现于第 {nan_row} 步")
            # print("最近10步loss:", loss_array[nan_row - 10:nan_row] if nan_row >= 10 else loss_array[:nan_row])
        else:
            print("所有损失值均正常，未检测到NaN。")

        if nan_row is not None:
            cut_len = int((nan_row // bfe) * bfe)  # 截断到bfe的整数倍，且在异常前
            if cut_len == 0:  # 异常出现在最前面，保留1个bfe
                cut_len = bfe
        else:
            cut_len = int((len(loss_array) // bfe) * bfe)

        loss_array = loss_array[:cut_len, :]
        steps = np.arange(cut_len)
        epochs = steps // bfe

        # 创建画布和子图（1行3列）
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        labels = ['Training Loss', 'Physics Loss', 'Total Loss']
        colors = ['blue', 'green', 'red']
        markers = ['o', 's', '^']

        for i in range(3):
            axes[i].plot(epochs, loss_array[:, i], label=labels[i], marker=markers[i], markersize=1, alpha=0.5,
                         color=colors[i], linewidth=0.7)
            axes[i].set_xlabel('Epoch (step // %d)' % bfe, fontsize=10)
            axes[i].set_ylabel('Loss Value', fontsize=10)
            axes[i].set_yscale('log')
            axes[i].set_title(labels[i], fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.5)
            axes[i].legend(fontsize=9)

            # 标注异常位置
            if nan_row is not None and nan_row < len(epochs):
                anomaly_epoch = nan_row // bfe
                axes[i].axvline(anomaly_epoch, color='orange', linestyle='--', alpha=0.7, label='NaN出现')
                axes[i].legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{title}_loss_list.png', dpi=300)
        plt.close()


def debug(config, args):
    ################################################################
    # prepare
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    ################################################################
    # dataloader
    ################################################################
    data_config = config['data']
    batch_size = 5
    initial_step = config['train']['initial_step']
    file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5'
    train_data = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                                initial_step=initial_step,
                                reduced_resolution=1,
                                reduced_resolution_t=1,
                                reduced_batch=5)
    val_data = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                              initial_step=initial_step,
                              reduced_resolution=1,
                              reduced_resolution_t=1,
                              reduced_batch=5,
                              if_test=False)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               shuffle=True,
                                               pin_memory=True,  # 启用固定内存加速传输
                                               persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=8, shuffle=False)

    file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5'
    sample_key = ['0000', '0001', '0002']
    data, x, y, t = load_samplesII(file_path, sample_key)
    # print('u.shape, v.shape, x.shape, y.shape, t.shape:', u.shape, v.shape, x.shape, y.shape, t.shape)
    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = 0.05  # 小时间步
    train_size, test_size = train_data.data_list.shape[0], val_data.data_list.shape[0]
    ntrain, ntest = train_size, test_size
    print(
        f'{datetime.now()} --- set dataset，batch size: {batch_size}, Train loader lens：{train_size}, Test loader lens：{test_size}')
    ################################################################
    # location
    ################################################################
    first_dic = f"./{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    current_directory = os.getcwd()
    print(
        f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---, current_directory:{current_directory}")
    ################################################################
    # model
    ################################################################
    # 这里有所不同的是，不再将这个一维含时的问题视为二维问题
    # 这个还是先暂且不论，跑通了再优化这些有的没的
    # _trans = PARTIAL(Wrapper, [dctI_SPFNO,dctI_SPFNO])
    # _itrans = PARTIAL(Wrapper, [idctI_SPFNO,idctI_SPFNO])
    # T = Transform(_trans, _itrans)
    # # 定义模型
    # Model = PARTIAL(SOL1dII, T)
    # input_channel = config['model']['input_channel'] * config['data']['initial_step'] + 1
    # model = Model(input_channel, config['model']['modes'], config['model']['width'],
    #               config['model']['bandwidth'], out_channels=config['model']['output_channel'],
    #               dim=config['model']['dim'], triL=config['model']['triL']).to(device)  # .to(torch.float32)

    model = Model(10 * 2 + 2, 24, 24, 1, out_channels=2, dim=2, triL=0).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")

    # 模型初始化
    # model.apply(init_weights)
    # 验证一下初始化结果
    # print("FC1 weight mean/std:", model.fc1.weight.mean().item(), model.fc1.weight.std().item())
    # print("FC1 bias:", model.fc1.bias[:5])
    #
    # print("FC2 weight mean/std:", model.fc2.weight.mean().item(), model.fc2.weight.std().item())
    # print("FC2 bias:", model.fc2.bias[:5])
    #
    # print("Conv0 weights mean/std:", model.conv0.weights.mean().item(), model.conv0.weights.std().item())
    # print("W0 weight mean/std:", model.w0.weight.mean().item(), model.w0.weight.std().item())
    # print("W0 bias:", model.w0.bias[:5])
    ################################################################
    # 定义优化器
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['base_lr'], weight_decay=1e-4)
    if config['train']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config['train']['milestones'],
                                                         gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['train']['gamma'],
                                                               threshold=1e-2, patience=config['train']['patience'],
                                                               verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['patience'],
                                                    gamma=config['train']['gamma'])

    print(f'{datetime.now()} --- set optimizer, scheduler ---')
    ################################################################
    # load model
    ################################################################
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        for name, param in model.named_parameters():
            print(
                f"{name}: min={param.data.min():.6f}, max={param.data.max():.6f}, mean={param.data.mean():.6f}, NaN={torch.isnan(param.data).any()}")
        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        # 获取 epoch
        epoch = checkpoint['epoch']
        plot_loss(loss_list, args.pretrain)
        print(f'模型【{args.pretrain}】已加载')

        if args.load_lr:
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
            scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率调度器状态
    else:
        loss_list = []
        test_loss_list = []
        lr_list = []
    ################################################################
    # loss
    ################################################################
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    initial_step = int(config['train']['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=config['train']['size_average'])
    model.eval()
    max_norm = 100.0
    # for b in trange(1):
    #     loss = 0
    #     """
    #     x:[b, nx, self.initial_step，1]
    #     y:[b, nx, nt, 1]
    #     grid:[b, nx, 1]
    #     """
    #     xx, yy, grid = next(iter(train_loader))
    #     xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
    #                                                                                                non_blocking=True)  # 确保数据在相同设备上
    #     optimizer.zero_grad()
    #     inp_shape = list(xx.shape)
    #     inp_shape = inp_shape[:-2]
    #     inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
    #     outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
    #     # pred = yy[..., :initial_step, :]
    #     pred = torch.empty((batch_size, 128, 128, 101, 2), device=xx.device)
    #     #  这里的代码逻辑需要优化一下
    #     pred[..., :initial_step, :] = yy[..., :initial_step, :]
    #     for t in range(initial_step, t_train):
    #         inp = xx.reshape(inp_shape)
    #         # print(inp)
    #         # print(t, torch.isnan(inp).any())
    #         y = yy[..., t:t + 1, :]
    #         out = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
    #         _batch = out.size(0)
    #         # print('out[::4,::4,...].shape:', out[::4, ::4, ...].shape)
    #         loss += myloss(out[:, :4, ::4, ...].reshape(_batch, -1), y[:, :4, ::4, ...].reshape(_batch, -1))
    #         # 这个是叠加的data loss
    #         pred[..., t:t + 1, :] = out
    #         xx = torch.cat((xx[..., 1:, :], out), dim=-2)
    #         # print(out)
    #         # print(t,torch.isnan(out).any())
    #
    #     assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
    #     # print('pred.shape:', pred.shape)
    #     if config['train']['loss_mode'] == 'both':
    #         pred_permuted = pred.permute(0, 3, 4, 1, 2)
    #         # print('output_permuted.shape:', pred_permuted.shape)
    #         loss_func = loss_generator(dt, dx).to(device)
    #         f_u, f_v, loss_phy = loss_gen(pred_permuted, loss_func)
    #         total_loss = (loss_phy * f_weight + loss * data_weight).to(torch.float32)
    #     elif config['train']['loss_mode'] == 'data':
    #         total_loss = loss
    #         loss_phy = 0
    #     else:
    #         pred_permuted = pred.permute(0, 3, 4, 1, 2)
    #         # print('output_permuted.shape:', pred_permuted.shape)
    #         loss_func = loss_generator(dt, dx).to(device)
    #         f_u, f_v, loss_phy = loss_gen(pred_permuted, loss_func)
    #         total_loss = loss_phy
    #         assert not torch.isnan(loss_phy).any(), "NaN in loss"
    #     total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    #     if total_norm > max_norm * 2:  # 如果梯度范数远大于 max_norm
    #         max_norm *= 1.1  # 适当增大 max_norm
    #     elif total_norm < max_norm * 0.5:  # 如果梯度范数远小于 max_norm
    #         max_norm *= 0.9  # 适当减小 max_norm
    #     current_lr = optimizer.param_groups[0]['lr']
    #     print(
    #         f"Loss: {loss.item():.4e},Loss_phy: {loss_phy.item():.4e}, lr:{current_lr},Gradient Norm: {total_norm}, Max Norm: {max_norm}")


if __name__ == '__main__':
    print('enter')
    print(f"当前运行的文件: {os.path.abspath(__file__)}")
    print(f"Python 路径: {sys.path}")

    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='./information.yaml', help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, default='debug', help='train or test')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
    parser.add_argument('--visualize', type=str, default=None, help='index that need to visualize')
    parser.add_argument('--load_lr', action='store_true', help='pretrain model path')
    args = parser.parse_args()
    print("当前所有解析器参数:", [action.dest for action in parser._actions])
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(config)
        test(config, args)
    elif args.mode == 'test':
        test(config, args)
    else:
        debug(config, args)
