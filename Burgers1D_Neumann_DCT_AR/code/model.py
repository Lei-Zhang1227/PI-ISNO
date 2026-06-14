'''
去除所有print的版本
'''

'''
This project references the following open-source projects.
1. [SPFNO](https://github.com/liu-ziyuan-math/SPFNO) by Ziyuan Liu.
2. [spectral operator learning] (https://github.com/liu-ziyuan-math/spectral_operator_learning) by Ziyuan Liu.
3. [physics_informed FNO] (https://github.com/neuraloperator/physics_informed) by Zongyi Li.
4. [FNO] () by Zongyi Li.
5. [DCT] by
6. [FC-FNO] by Haydn Maust

Q: MODES 和 WIDTH分别代表什么，有什么联系？

对于齐次狄利克雷边界条件下的一维含时问题。采用二维的DST模型;
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import functools
import transforms
from utils import FC1d
import time

CONTINUATION_FUNC = lambda x: FC1d(x, 3)

class SpectralProjection1D(nn.Module):
    """DCT → learnable diagonal weights (truncated modes) → IDCT"""

    def __init__(self, T, channels, modes):
        super(SpectralProjection1D, self).__init__()
        self.T = T
        self.X_dims = np.arange(-1, 0)
        self.modes = modes
        self.weights = nn.Parameter(torch.ones(channels, modes))

    def forward(self, u):
        # u: [batch, channels, nx]
        b = self.T(u, self.X_dims)
        out = torch.zeros_like(b)
        out[..., :self.modes] = b[..., :self.modes] * self.weights
        return self.T.inv(out, self.X_dims)
        
class PseudoSpectra1d(nn.Module):
    '''
    (T, width, width, modes, bandwidth, triL)
    T=DCT
    in_channels=2
    modes=20
    width=50
    bandwidth=4
    '''

    def __init__(self, T, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super().__init__()

        self.T = T
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-1, 0)

        scale = 1 / (in_channels * out_channels)  # 1/50*50
        self.weights = nn.Parameter(scale * torch.rand(modes, in_channels, out_channels, bandwidth))
        '''
        将生成的随机数张量乘以缩放因子 scale，以调整随机数的范围。结果是一个形状为 (modes, in_channels, out_channels, bandwidth) 的张量，元素在 [0, scale) 区间内。
        这里为什么要×一个缩放因子？？？如果是为了后面爱因斯坦求和的平均的话，不是应该scale = 1 / (in_channels * bandwidth)更合理吗

        按下不表，这里p的形状是[20,50,50,4],对于其它谱算子层；
        对于self.convl，形状是[20,2,48,4]
        Q:按照FNO中的定义，这个参数矩阵应该是共轭对称的啊。
        A：在FNO中，fft的变换中如果要模拟核积分算子，那么需要矩阵是共轭对称的复值矩阵，但是这里实质上进行的是DCT/DST，所以不需要这样子操作？对于DCT/DST，矩阵是全实的
        '''
        # self.unfold = torch.nn.Unfold(kernel_size=bandwidth,
        #                               padding=triL)

    def quasi_diag_mul(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    '''
    xpad = x.unfold(-1, self.bandwidth, 1)
    这步就类似一个时间滑窗，这里输入始张量 x 的形状为 [20, 2, 23]（因为在b[..., :self.modes + self.bandwidth - 1]中进行了裁切）
    得到的新张量的形状就是：[20, 2, 20, 4]
    Q：但是为什么要进行unfold呢？？

    这里weight的形状为：[20,2,48,4]（对于层self.convl）
    然后进行爱因斯坦求和操作：
    这个求和操作就是用wight对输入张量的input channel和bandwidth进行逐元素相乘然后求和。
    对于输入张量x：[20, 2, 20, 4]
    进行加权的部分是i维度2，也就是input channel和第w维 4，窗口大小
    进行weight后相当于进行了升维，channel大小变成了weight中设定的o：out_channels。当然这个稍微复杂一点，还有mode的维度。
    总之，最后得到张量的维度是[batch size, out_channels, modes]
    '''

    def forward(self, u):
        #  # u: (1100, 2, 4097)
        batch_size, _, Nx = u.shape
        b = self.T(u, self.X_dims)
        '''
        函数为什么要老母猪带套？？？
        套来套去的结果就是，对b做了一次谱变换（正向的），self.X_dims的作用是选择变换的维度
        输入的张量形状是：(1100, 2, 4097)
        得到的张量形状是：(1100, 2, 4097)，并且是只留了实部的。
        '''
        out = torch.zeros((batch_size, self.out_channels, Nx), device=u.device, dtype=u.dtype)  # 第一次是50-2=48
        b = F.pad(b, (self.triL, 0, 0, 0, 0, 0))  # 本来是个填充流程，但是由于 self.triL=0， 所以实质上没有填充的；
        out[..., :self.modes] = self.quasi_diag_mul(b[..., :self.modes + self.bandwidth - 1], self.weights)
        '''
        b[..., :self.modes + self.bandwidth - 1] 只选择前23个
        out：形状为[batch size, out_channels, modes],[20,48,4097], 
        前[batch size, out_channels, modes][20,48,20]个是计算得到的非零值，其余都是0；
        '''
        u = self.T.inv(out, self.X_dims)
        '''
        u的形状保持不变：
        然后做逆变换变换回去，所以这个层的主要就是对得到傅里叶层进行了一个加权求和？
        我还是不明白为啥正变换要求实值。---当然是因为要实现DCT啦
        '''
        return u


class PseudoSpectra1d_complex(nn.Module):
    '''
    (T, width, width, modes, bandwidth, triL)
    T=DCT
    in_channels=2
    modes=20
    width=50
    bandwidth=4
    '''

    def __init__(self, T, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super().__init__()

        self.T = T
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-1, 0)

        scale = 1 / (in_channels * out_channels)  # 1/50*50
        self.weights = nn.Parameter(
            scale * torch.rand(modes, in_channels, out_channels, bandwidth, dtype=torch.cfloat))

    def quasi_diag_mul(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        #  # u: (1100, 2, 4097)
        batch_size, _, Nx = u.shape
        b = self.T(u, self.X_dims)
        out = torch.zeros((batch_size, self.out_channels, Nx), device=u.device, dtype=b.dtype)  # 第一次是50-2=48
        b = F.pad(b, (self.triL, 0, 0, 0, 0, 0))  # 本来是个填充流程，但是由于 self.triL=0， 所以实质上没有填充的；
        out[..., :self.modes] = self.quasi_diag_mul(b[..., :self.modes + self.bandwidth - 1], self.weights)
        u = self.T.inv(out, self.X_dims)
        return u


class PseudoSpectra(nn.Module):
    '''
    与源代码的不同之处：
    1. 将t维度另外处理，不做谱变化，同样也失去了时间超分辨性；
    '''

    def __init__(self, T, dim, in_channels, out_channels, t_dim, modes, bandwidth=1, triL=0):
        super(PseudoSpectra, self).__init__()

        self.T = T
        self.dim = dim  # 1
        self.in_channels = in_channels  # 3
        self.out_channels = out_channels  # 25-3
        self.modes = modes  # 25
        self.bandwidth = int(bandwidth)  # 4
        self.triL = triL  # 0
        self.X_dims = np.arange(-dim, 0)
        self.t_dim = t_dim

        # print([(l, 0) for l in triL])
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.rand(t_dim, in_channels * bandwidth.prod().item(), out_channels,
                                                       modes.prod().item(), dtype=torch.float32))  # size：[3, 22, 25]
        self.unfold = torch.nn.Unfold(kernel_size=bandwidth, padding=triL)
        # 虽然这个操作看起来有些许复杂，但是最后得到的结果和直接reshape没啥差别。只是把xy的二位数据拉平罢辽
        self.X_slices = [slice(None), slice(None), slice(None)] + [slice(freq) for freq in modes]
        self.pad_slices = [slice(None), slice(None), slice(None)] + [slice(freq) for freq in
                                                                     modes + bandwidth - 1 - triL * 2]
        '''
        那么同理，这里的操作也是一个切片函数
        self.pad_slices相当于[:,:,0:24,0:24]
        '''

    def quasi_diag_mul(self, x, weights):
        # xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bitx, tiox->botx", x, weights)

    def forward(self, u):
        # 输入张量的szie是[20 3 101 1024], 如果是含时问题的话，时间通道也再第二个维度
        batch_size = u.shape[0]
        b = self.T(u, self.X_dims[::-1])  # 分别进行进行一次离散正弦变换 size: [20, 3, 101, 1024]
        out = torch.zeros(batch_size, self.out_channels, *u.shape[2:], device=u.device,
                          dtype=u.dtype)  # [20, 22, 101, 1024]
        out[self.X_slices] = self.quasi_diag_mul(b[self.pad_slices], self.weights).reshape(
            batch_size, self.out_channels, self.t_dim, *self.modes)  # size: [20 29 130 130]， 其中除[20,29,24,24]外均为0
        u = self.T.inv(out, self.X_dims)  # size: [20 29 130 130]
        return u


class PseudoSpectra2D(nn.Module):
    def __init__(self, T, dim, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super(PseudoSpectra2D, self).__init__()

        self.T = T
        self.dim = dim
        self.in_channels = in_channels if isinstance(in_channels, (int, np.integer)) else in_channels[0]
        self.out_channels = out_channels if isinstance(out_channels, (int, np.integer)) else out_channels[0]
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-dim, 0)
        self.scale = 1 / (self.in_channels * self.out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels,
                                    self.modes[0], self.modes[1],
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels,
                                    self.modes[0], self.modes[1],
                                    dtype=torch.cfloat))

        # self.weights1 = nn.Parameter(
        #     self.scale * torch.rand(self.in_channels * bandwidth.prod().item(), self.out_channels,
        #                             self.modes.prod().item(),
        #                             dtype=torch.cfloat))
        # self.weights2 = nn.Parameter(
        #     self.scale * torch.rand(self.in_channels * bandwidth.prod().item(), self.out_channels,
        #                             self.modes.prod().item(),
        #                             dtype=torch.cfloat))
        # self.unfold = torch.nn.Unfold(kernel_size=bandwidth, padding=triL)
        # 虽然这个操作看起来有些许复杂，但是最后得到的结果和直接reshape没啥差别。只是把xy的二位数据拉平罢辽
        self.X_slices1 = [slice(None), slice(None)] + [slice(freq) for freq in modes]
        self.X_slices2 = [slice(None), slice(None), slice(-modes[0], None)] + [slice(freq) for freq in modes[1:]]
        self.pad_slices1 = [slice(None), slice(None)] + [slice(freq) for freq in modes + bandwidth - 1 - triL * 2]
        self.pad_slices2 = [slice(None), slice(None), slice(-(modes + bandwidth - 1 - triL * 2)[0], None)] + [
            slice(freq) for freq in (modes + bandwidth - 1 - triL * 2)[1:]]

    def quasi_diag_mul(self, input, weights):
        # xpad = self.unfold(input)  # 这里unfold的目的究竟何在？？
        # out = torch.einsum("bix, iox->box", xpad, weights)
        # print('input.shape,weights.shape:',input.shape,weights.shape)
        out = torch.einsum("bixy,ioxy->boxy", input, weights)
        return out

    def forward(self, u):
        # 输入张量的szie是[20 3 130 130]
        batch_size = u.shape[0]
        b = self.T(u, self.X_dims[::-1])  # 先进行一次离散正弦变换 size: [20 3 130 130]
        out = torch.zeros(batch_size, self.out_channels, *b.shape[2:], device=b.device,
                          dtype=b.dtype)  # size: [20 29 130 130]
        out[:, :, :self.modes[0], :self.modes[1]] = self.quasi_diag_mul(b[:, :, :self.modes[0], :self.modes[1]],
                                                                        self.weights1).reshape(
            batch_size, self.out_channels, *self.modes)  # size: [20 29 130 130]， 其中除[20,29,24,24]外均为0
        out[:, :, -self.modes[0]:, :self.modes[1]] = self.quasi_diag_mul(b[:, :, -self.modes[0]:, :self.modes[1]],
                                                                         self.weights2).reshape(
            batch_size, self.out_channels, *self.modes)
        u = self.T.inv(out, self.X_dims)  # size: [20 29 130 130]
        return u


class ZerosFilling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(1, device=x.device)


class SOL(nn.Module):
    '''
    对于exp3：
    model = Model(initial_step*2+2, modes, width, bandwidth, out_channels=2, dim = 2, triL=triL).to(device)
    in_channels=10*2+2=22 这个channel就是10个时间步*2个变量+x+y
    modes=24
    width=24
    bandwidth=1
    out_channels=2
    dim=2
    '''

    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=1, skip=True, triL=0):
        super(SOL, self).__init__()

        modes = np.array([modes] * dim) if isinstance(modes, int) else np.array(modes)
        bandwidth = np.array([bandwidth] * dim) if isinstance(bandwidth, int) else np.array(bandwidth)
        triL = np.array([triL] * dim) if isinstance(triL, int) else np.array(triL)

        self.modes = modes
        self.width = width
        self.triL = triL
        self.T = T
        self.dim = dim
        self.X_dims = np.arange(-dim, 0)
        self.Print = True
        if dim == 1:
            convND = nn.Conv1d
        elif dim == 2:
            convND = nn.Conv2d
        elif dim == 3:
            convND = nn.Conv3d

        self.conv0 = PseudoSpectraII(T, dim, width, width, modes, bandwidth, triL)
        self.conv1 = PseudoSpectraII(T, dim, width, width, modes, bandwidth, triL)
        self.conv2 = PseudoSpectraII(T, dim, width, width, modes, bandwidth, triL)
        self.conv3 = PseudoSpectraII(T, dim, width, width, modes, bandwidth, triL)

        self.convl = PseudoSpectraII(T, dim, in_channels, width - in_channels, modes, bandwidth, triL)
        self.w0 = convND(width, width, 1)  #
        self.w1 = convND(width, width, 1)
        self.w2 = convND(width, width, 1)
        self.w3 = convND(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()  # skip的话就是0，不skip的话就是原值

    def forward(self, x):

        # [batch, XYZ, c] -> [batch, c, XYZ]

        x = x.permute(0, -1, *self.X_dims - 1)  # 就是把channel提到了第二个维度，所以此时的输入tensor形状为：[20 3 130 130]
        x_1 = torch.cat([x, F.gelu(self.convl(x))], dim=1)
        x_2 = self.skip(x_1) + F.gelu(self.w0(x_1) + self.conv0(x_1))
        x_3 = self.skip(x_2) + F.gelu(self.w1(x_2) + self.conv1(x_2))
        x_4 = self.skip(x_3) + F.gelu(self.w2(x_3) + self.conv2(x_3))
        x_5 = self.skip(x_4) + F.gelu(self.w3(x_4) + self.conv3(x_4))
        x_6 = x_5.permute(0, 2, 3, 1)
        x_7 = self.fc1(x_6)
        x_7 = F.gelu(x_7)
        x_8 = self.fc2(x_7)
        x_8 = x_8.permute(0, -1, *self.X_dims - 1)
        out = self.T(self.T.inv(x_8, self.X_dims[1::]), self.X_dims[:0:-1])
        # 对于多维的问题，在最后的滤波环节去除时间维度的变换，只对空间维度进行滤波
        if self.Print:
            print('--------------- data shape ---------------\n')
            variables = [x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, out]
            for i, var in enumerate(variables):
                print(f'--x_{i}: {var.shape},type: {var.dtype}')
            self.Print = False
        return out


class SOL_1D(nn.Module):
    '''
    exp 1D with time:
    1D Burgers'Equation with Homogeneous Dirichlet Boundary Condition.
    tips：
    1. dim = 1, 只在nx维度进行变换；
    2. input size：[sample, nt, nx, 3], 其中最后一个维度包括u、x、t, u0在t维度上重复nt次
    3. output size: [sample, nt, nx]
    4. 使用的变换：[DSTI]; 二阶导用延拓后的傅里叶谱方法计算？
    5. 由于在最后一步又加了谱滤波器，所以不计算dQ，直接对输出进行谱微分；
    '''

    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=1, skip=True, triL=0, flat=True):
        super(SOL_1D, self).__init__()
        '''
        这三句就是把modes、bandwidth、triL转换为array：
        [modes] * dim 创建一个包含 dim 个 modes 值的列表。例如，如果 modes = 5 和 dim = 3，则结果为 [5, 5, 5]。
        np.array([modes] * dim) 将这个列表转换为一个 NumPy 数组。例如，结果为 array([5, 5, 5])。
        如果它已经是数组了，则不做改变；
        '''
        self.modes = modes
        self.width = width
        self.triL = triL
        self.T = T
        self.dim = dim
        self.X_dims = np.arange(-dim, 0)
        self.Print = True
        if dim == 1:
            convND = nn.Conv1d
        elif dim == 2:
            convND = nn.Conv2d
        elif dim == 3:
            convND = nn.Conv3d

        self.conv0 = PseudoSpectra(T, dim, width, width, t_dim, modes, bandwidth, triL)
        # dim=2,width=24,24,modes=16,bandwidth=4,triL=0
        self.conv1 = PseudoSpectra(T, dim, width, width, t_dim, modes, bandwidth, triL)
        self.conv2 = PseudoSpectra(T, dim, width, width, t_dim, modes, bandwidth, triL)
        self.conv3 = PseudoSpectra(T, dim, width, width, t_dim, modes, bandwidth, triL)

        self.convl = PseudoSpectra(T, dim, in_channels, width - in_channels, t_dim, modes, bandwidth, triL)
        self.w0 = convND(width, width, 1)  #
        self.w1 = convND(width, width, 1)
        self.w2 = convND(width, width, 1)
        self.w3 = convND(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()  # skip的话就是0，不skip的话就是原值
        print('--------------- model paras shape ---------------\n')
        print(f'--self.convl.weights.shape: {self.convl.weights.shape}')
        for i in range(4):
            conv_layer = getattr(self, f'conv{i}')
            print(f'--self.conv{i}.weights.shape: {conv_layer.weights.shape}')
        for i in range(4):
            w_layer = getattr(self, f'w{i}')
            for param in w_layer.parameters():
                print(f'--self.w{i}.param.shape: {param.shape}')

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # 就是把channel提到了第二个维度，所以此时的输入tensor形状为：[20 3 130 130]
        x_1 = torch.cat([x, F.gelu(self.convl(x))], dim=1)
        x_2 = self.skip(x_1) + F.gelu(self.w0(x_1) + self.conv0(x_1))
        x_3 = self.skip(x_2) + F.gelu(self.w1(x_2) + self.conv1(x_2))
        x_4 = self.skip(x_3) + F.gelu(self.w2(x_3) + self.conv2(x_3))
        x_5 = self.skip(x_4) + F.gelu(self.w3(x_4) + self.conv3(x_4))
        x_6 = x_5.permute(0, 2, 3, 1)
        x_7 = self.fc1(x_6)
        x_7 = F.gelu(x_7)
        x_8 = self.fc2(x_7)
        out = self.T(self.T.inv(x_8, self.X_dims - 1), self.X_dims - 1)
        if self.Print:
            print('--------------- data shape ---------------\n')
            variables = [x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, out]
            for i, var in enumerate(variables):
                print(f'--x_{i}: {var.shape}')
            self.Print = False
        return out


class SOLII(nn.Module):
    '''
    对于exp3：
    model = Model(initial_step*2+2, modes, width, bandwidth, out_channels=2, dim = 2, triL=triL).to(device)
    in_channels=10*2+2=22 这个channel就是10个时间步*2个变量+x+y
    modes=24
    width=24
    bandwidth=1
    out_channels=2
    dim=2
    '''

    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=2, skip=True, triL=0, flat=True):
        super(SOLII, self).__init__()

        modes = np.array([modes]) if isinstance(modes, int) else np.array(modes)
        bandwidth = np.array([bandwidth]) if isinstance(bandwidth, int) else np.array(bandwidth)
        triL = np.array([triL]) if isinstance(triL, int) else np.array(triL)
        # width = np.array([width])
        print('width:', width)

        self.modes = modes
        self.width = width
        self.triL = triL
        self.bandwidth = bandwidth
        self.T = T
        self.dim = dim
        self.X_dims = np.arange(-dim, 0)
        self.Print = True
        if dim == 1:
            convND = nn.Conv1d
        elif dim == 2:
            convND = nn.Conv2d
        elif dim == 3:
            convND = nn.Conv3d

        self.convl = PseudoSpectra2D(T, dim, width[0], width[0] - in_channels, modes[0], bandwidth[0], triL[0])

        self.sp_convs = nn.ModuleList([PseudoSpectra2D(T, dim, in_size, out_size, modes, bandwidth, tril)
                                       for in_size, out_size, modes, bandwidth, tril
                                       in zip(self.width[0:], self.width[1:], self.modes,
                                              self.bandwidth, self.triL)])
        self.ws = nn.ModuleList([convND(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.width[0:], self.width[1:])])
        self.fc0 = nn.Linear(in_channels, width[0])
        self.fc1 = nn.Linear(width[-1], 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()  # skip的话就是0，不skip的话就是原值
        self.flat = nn.ModuleList([nn.Linear(in_size, out_size)
                                   for in_size, out_size in
                                   zip(self.width[0:], self.width[1:])]) if flat else self.skip

    def forward(self, x0):
        x = self.fc0(x0)
        # print('x.shape:', x.shape)
        x = x.permute(0, -1, *self.X_dims - 1)
        # print('x.shape:', x.shape)
        x = torch.cat([x0.permute(0, -1, *self.X_dims - 1), F.gelu(self.convl(x))], dim=1)
        for i, (speconv, w, flat) in enumerate(zip(self.sp_convs, self.ws, self.flat)):
            x_1 = speconv(x)
            x_2 = w(x)
            x = flat(x.permute(0, *self.X_dims, 1)).permute(0, -1, *self.X_dims - 1) + F.gelu(x_1 + x_2)
            variables = [x, x_1, x_2]
            # for i, var in enumerate(variables):
            #     print(f'--speconv_{i}: {var.shape},type: {var.dtype}')
        x_3 = x.permute(0, 2, 3, 1)
        x_4 = self.fc1(x_3)
        x_5 = F.gelu(x_4)
        x_6 = self.fc2(x_5)
        x_6 = x_6.permute(0, -1, *self.X_dims - 1)
        out = self.T(self.T.inv(x_6, self.X_dims[1::]), self.X_dims[:0:-1])
        if self.Print:
            print('--------------- data shape ---------------\n')
            variables = [x, x_1, x_2, x_3, x_4, x_5, x_6, out]
            for i, var in enumerate(variables):
                print(f'--x_{i}: {var.shape},type: {var.dtype}')
            self.Print = False
        # 对于多维的问题，在最后的滤波环节去除时间维度的变换，只对空间维度进行滤波
        # 最后输出的形状是[b,nt,nx]
        out = out.permute(0, 2, 3, 1)
        return out


class SOLII_2(nn.Module):
    '''
    对于exp3：
    model = Model(initial_step*2+2, modes, width, bandwidth, out_channels=2, dim = 2, triL=triL).to(device)
    in_channels=10*2+2=22 这个channel就是10个时间步*2个变量+x+y
    modes=24
    width=24
    bandwidth=1
    out_channels=2
    dim=2
    '''

    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=2, skip=True, triL=0, flat=True):
        super(SOLII_2, self).__init__()

        modes = np.array([modes]) if isinstance(modes, int) else np.array(modes)
        bandwidth = np.array([bandwidth]) if isinstance(bandwidth, int) else np.array(bandwidth)
        triL = np.array([triL]) if isinstance(triL, int) else np.array(triL)
        # width = np.array([width])
        print('width:', width)

        self.modes = modes
        self.width = width
        self.triL = triL
        self.bandwidth = bandwidth
        self.T = T
        self.dim = dim
        self.X_dims = np.arange(-dim, 0)
        self.Print = True
        if dim == 1:
            convND = nn.Conv1d
        elif dim == 2:
            convND = nn.Conv2d
        elif dim == 3:
            convND = nn.Conv3d

        self.convl = PseudoSpectra2D(T, dim, in_channels, width[0] - in_channels, modes[0], bandwidth[0], triL[0])
        self.sp_convs = nn.ModuleList([PseudoSpectra2D(T, dim, in_size, out_size, modes, bandwidth, tril)
                                       for in_size, out_size, modes, bandwidth, tril
                                       in zip(self.width[0:], self.width[1:], self.modes,
                                              self.bandwidth, self.triL)])
        self.ws = nn.ModuleList([convND(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.width[0:], self.width[1:])])
        self.fc1 = nn.Linear(width[-1], 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()  # skip的话就是0，不skip的话就是原值
        self.flat = nn.ModuleList([nn.Linear(in_size, out_size)
                                   for in_size, out_size in
                                   zip(self.width[0:], self.width[1:])]) if flat else self.skip

    def forward(self, x, return_frequency=False):
        x = x.permute(0, -1, *self.X_dims - 1)
        x = torch.cat([x, F.gelu(self.convl(x))], dim=1)
        for i, (speconv, w, flat) in enumerate(zip(self.sp_convs, self.ws, self.flat)):
            x_1 = speconv(x)
            x_2 = w(x)
            x = flat(x.permute(0, *self.X_dims, 1)).permute(0, -1, *self.X_dims - 1) + F.gelu(x_1 + x_2)
            variables = [x, x_1, x_2]
            for i, var in enumerate(variables):
                print(f'--speconv_{i}: {var.shape},type: {var.dtype}')
        x_3 = x.permute(0, 2, 3, 1)
        x_4 = self.fc1(x_3)
        x_5 = F.gelu(x_4)
        x_6 = self.fc2(x_5)
        x_6 = x_6.permute(0, -1, *self.X_dims - 1)

        x_freq = self.T.inv(x_6, self.X_dims[1::])  # 频域表示（做了一次逆变换到频域）
        out = self.T(x_freq, self.X_dims[:0:-1])

        if self.Print:
            print('--------------- data shape ---------------\n')
            variables = [x, x_1, x_2, x_3, x_4, x_5, x_6, x_freq, out]
            for i, var in enumerate(variables):
                if i == 7:
                    print(f'--x_freq: {var.shape},type: {var.dtype}')
                elif i == 8:
                    print(f'--out: {var.shape},type: {var.dtype}')
                else:
                    print(f'--x_{i}: {var.shape},type: {var.dtype}')
            self.Print = False

        # 对于多维的问题，在最后的滤波环节去除时间维度的变换，只对空间维度进行滤波
        # 最后输出的形状是[b,nt,nx]
        out = out.permute(0, 2, 3, 1)
        if return_frequency:
            # 同时返回物理空间和频域表示
            x_freq_permuted = x_freq.permute(0, 2, 3, 1)  # 与out保持相同的维度顺序
            return out, x_freq_permuted
        else:
            return out


class SOLIII(nn.Module):
    '''
    对于exp3：
    model = Model(initial_step*2+2, modes, width, bandwidth, out_channels=2, dim = 2, triL=triL).to(device)
    in_channels=10*2+2=22 这个channel就是10个时间步*2个变量+x+y
    modes=24
    width=24
    bandwidth=1
    out_channels=2
    dim=2
    '''

    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=2, skip=True, triL=0, flat=True,
                 double_weights=True):
        super(SOLIII, self).__init__()

        modes = np.array([modes]) if isinstance(modes, int) else np.array(modes)
        bandwidth = np.array([bandwidth]) if isinstance(bandwidth, int) else np.array(bandwidth)
        triL = np.array([triL]) if isinstance(triL, int) else np.array(triL)
        # width = np.array([width])
        print('width:', width)

        self.modes = modes
        self.width = width
        self.triL = triL
        self.bandwidth = bandwidth
        self.T = T
        self.dim = dim
        self.X_dims = np.arange(-dim, 0)
        self.Print = True
        if dim == 1:
            convND = nn.Conv1d
        elif dim == 2:
            convND = nn.Conv2d
        elif dim == 3:
            convND = nn.Conv3d

        self.convl = PseudoSpectra2DII(T, dim, in_channels, width[0] - in_channels, modes[0], bandwidth[0], triL[0],
                                       double_weights)

        self.sp_convs = nn.ModuleList(
            [PseudoSpectra2DII(T, dim, in_size, out_size, modes, bandwidth, tril, double_weights)
             for in_size, out_size, modes, bandwidth, tril
             in zip(self.width[0:], self.width[1:], self.modes,
                    self.bandwidth, self.triL)])
        self.ws = nn.ModuleList([convND(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.width[0:], self.width[1:])])
        self.fc1 = nn.Linear(width[-1], 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()  # skip的话就是0，不skip的话就是原值
        self.flat = nn.ModuleList([nn.Linear(in_size, out_size)
                                   for in_size, out_size in
                                   zip(self.width[0:], self.width[1:])]) if flat else self.skip

    def forward(self, x):
        x = x.permute(0, -1, *self.X_dims - 1)
        x = torch.cat([x, F.gelu(self.convl(x))], dim=1)
        for i, (speconv, w, flat) in enumerate(zip(self.sp_convs, self.ws, self.flat)):
            x_1 = speconv(x)
            x_2 = w(x)
            x = flat(x.permute(0, *self.X_dims, 1)).permute(0, -1, *self.X_dims - 1) + F.gelu(x_1 + x_2)
            variables = [x, x_1, x_2]
            for i, var in enumerate(variables):
                print(f'--speconv_{i}: {var.shape},type: {var.dtype}')
        x_3 = x.permute(0, 2, 3, 1)
        x_4 = self.fc1(x_3)
        x_5 = F.gelu(x_4)
        x_6 = self.fc2(x_5)
        x_6 = x_6.permute(0, -1, *self.X_dims - 1)
        out = self.T(self.T.inv(x_6, self.X_dims[1::]), self.X_dims[:0:-1])
        if self.Print:
            print('--------------- data shape ---------------\n')
            variables = [x, x_1, x_2, x_3, x_4, x_5, x_6, out]
            for i, var in enumerate(variables):
                print(f'--x_{i}: {var.shape},type: {var.dtype}')
            self.Print = False
        # 对于多维的问题，在最后的滤波环节去除时间维度的变换，只对空间维度进行滤波
        # 最后输出的形状是[b,nt,nx]
        out = out.permute(0, 2, 3, 1).squeeze()
        return out


class PseudoSpectra2DII(nn.Module):
    '''
    实参数，用于非FFT变换
    '''

    def __init__(self, T, dim, in_channels, out_channels, modes, bandwidth=1, triL=0, double_weights=True):
        super(PseudoSpectra2DII, self).__init__()

        self.T = T
        self.double_weights = double_weights
        self.dim = dim
        self.in_channels = in_channels if isinstance(in_channels, (int, np.integer)) else in_channels[0]
        self.out_channels = out_channels if isinstance(out_channels, (int, np.integer)) else out_channels[0]
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-dim, 0)
        self.scale = 1 / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels * bandwidth.prod().item(), self.out_channels,
                                    self.modes.prod().item(),
                                    dtype=torch.float32))
        if double_weights:
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(self.in_channels * bandwidth.prod().item(), self.out_channels,
                                        self.modes.prod().item(),
                                        dtype=torch.float32))
            self.X_slices2 = [slice(None), slice(None), slice(-modes[0], None)] + [slice(freq) for freq in modes[1:]]
            self.pad_slices2 = [slice(None), slice(None), slice(-(modes + bandwidth - 1 - triL * 2)[0], None)] + [
                slice(freq) for freq in (modes + bandwidth - 1 - triL * 2)[1:]]
        self.unfold = torch.nn.Unfold(kernel_size=bandwidth, padding=triL)
        # 虽然这个操作看起来有些许复杂，但是最后得到的结果和直接reshape没啥差别。只是把xy的二位数据拉平罢辽
        self.X_slices1 = [slice(None), slice(None)] + [slice(freq) for freq in modes]
        self.pad_slices1 = [slice(None), slice(None)] + [slice(freq) for freq in modes + bandwidth - 1 - triL * 2]

    def quasi_diag_mul(self, input, weights):
        xpad = self.unfold(input)  # 这里unfold的目的究竟何在？？
        out = torch.einsum("bix, iox->box", xpad, weights)
        return out

    def forward(self, u):
        # 输入张量的szie是[20 3 130 130]
        batch_size = u.shape[0]
        b = self.T(u, self.X_dims[::-1])  # 先进行一次离散正弦变换 size: [20 3 130 130]
        out = torch.zeros(batch_size, self.out_channels, *u.shape[2:], device=u.device,
                          dtype=b.dtype)  # size: [20 29 130 130]
        out[self.X_slices1] = self.quasi_diag_mul(b[self.pad_slices1], self.weights1).reshape(
            batch_size, self.out_channels, *self.modes)  # size: [20 29 130 130]， 其中除[20,29,24,24]外均为0
        if self.double_weights:
            out[self.X_slices2] = self.quasi_diag_mul(b[self.pad_slices2], self.weights2).reshape(
                batch_size, self.out_channels, *self.modes)
        u = self.T.inv(out, self.X_dims)  # size: [20 29 130 130]
        return u


class PseudoSpectra1d(nn.Module):
    '''
    (T, width, width, modes, bandwidth, triL)
    T=DCT
    in_channels=2
    modes=20
    width=50
    bandwidth=4
    '''

    def __init__(self, T, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super().__init__()

        self.T = T
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-1, 0)

        scale = 1 / (in_channels * out_channels)  # 1/50*50
        self.weights = nn.Parameter(scale * torch.rand(modes, in_channels, out_channels, bandwidth))
        # self.unfold = torch.nn.Unfold(kernel_size=bandwidth,
        #                               padding=triL)

    def quasi_diag_mul(self, x, weights):
        #         print(f'x.shape:{x.shape}')
        #         print(f'weights.shape:{weights.shape}')
        xpad = x.unfold(-1, self.bandwidth, 1)
        #         print(f'xpad.shape:{xpad.shape}')
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        #  # u: (1100, 2, 4097)
        batch_size, _, Nx = u.shape
        b = self.T(u, self.X_dims)
        out = torch.zeros((batch_size, self.out_channels, Nx), device=u.device, dtype=u.dtype)  # 第一次是50-2=48
        out[..., :self.modes] = self.quasi_diag_mul(b[..., :self.modes + self.bandwidth - 1], self.weights)
        u = self.T.inv(out, self.X_dims)
        return u


class SOL1dII(nn.Module):
    # model = Model(2, modes, width, bandwidth, triL=triL).to(device).double()
    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=1, skip=True, triL=0):
        super(SOL1dII, self).__init__()
        modes = np.array([modes] * dim) if isinstance(modes, int) else np.array(modes)
        bandwidth = np.array([bandwidth] * dim) if isinstance(bandwidth, int) else np.array(bandwidth)
        triL = np.array([triL] * dim) if isinstance(triL, int) else np.array(triL)
        width = np.array([width] * dim) if isinstance(width, int) else np.array(width)
        print(modes)
        print(bandwidth)
        print(triL)
        print(width)

        self.modes = modes
        self.width = width
        self.triL = triL
        self.bandwidth = bandwidth
        self.T = T
        self.X_dims = [-1]
        self.Print = True
        self.convl = PseudoSpectra1d(T, in_channels, width[0] - in_channels, modes[0], bandwidth[0], triL[0])
        print('convl:', in_channels, width[0] - in_channels, modes[0], bandwidth[0], triL[0])
        self.sp_convs = nn.ModuleList([PseudoSpectra1d(T, in_size, out_size, modes, bandwidth, tril)
                                       for in_size, out_size, modes, bandwidth, tril
                                       in zip(self.width[:-1], self.width[1:], self.modes[1:],
                                              self.bandwidth[1:], self.triL[1:])])
        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.width[:-1], self.width[1:])])
        self.fc1 = nn.Linear(width[-1], 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()
        self.spectral_proj = SpectralProjection1D(T, out_channels, 30)  # 30 改成你的默认值

    def forward(self, x):
        # x: [b, nx, channel],最后一个维度的大小取决于输入的初始时间步长；
        x = x.permute(0, -1, 1)
        x = torch.cat([x, F.gelu(self.convl(x))], dim=1)
        #         print(f'after convl x.shape:{x.shape}')
        '''
        F.gelu(self.convl(x)): [b,width[0]-input_channel,nx]
        x:[b,input_channel,nx]
        x:[b,width[0],nx]
        '''
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x_1 = speconv(x)
            x_2 = w(x)
            x = self.skip(x) + F.gelu(x_1 + x_2)  # 如果width不是相同的，那么就需要用flat来调整维度进行skip连接。
            if self.Print:
                print(f'--speconv_{i}: {x.shape},type: {x.dtype}')
        # x:[b,width[-1],nx]
        x_3 = x.permute(0, 2, 1)  # [b,nx,width[-1]]
        x_4 = self.fc1(x_3)  # [b,nx,128]
        x_5 = F.gelu(x_4)  # [b,nx,128]
        x_6 = self.fc2(x_5)          # [b, nx, out_channels]
        x_7 = x_6.permute(0, 2, 1)  # [b,out_channels,nx]
        out = self.spectral_proj(x_7) # DCT → spectral weighting → IDCT
        out = out.permute(0, 2, 1)    # [b, nx, out_channels]
#         return out 
        if self.Print:
            print('--------------- data shape ---------------\n')
            variables = [x, x_1, x_2, x_3, x_4, x_5, x_6, x_7]
            for i, var in enumerate(variables):
                print(f'--x_{i}: {var.shape},type: {var.dtype}')
            self.Print = False
        return out


if __name__ == '__main__':
    from Burger.utils import *
    from functools import partial as PARTIAL
    from transforms import *
    import yaml
    from datetime import datetime

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        with open("informationII.yaml", encoding="utf-8") as stream:
            config = yaml.load(stream, yaml.FullLoader)
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试GB18030（兼容GBK）
        with open("informationII.yaml", encoding="gb18030") as stream:
            config = yaml.load(stream, yaml.FullLoader)

    _trans = PARTIAL(Wrapper, [dstII_SPFNO, dstII_SPFNO])
    _itrans = PARTIAL(Wrapper, [idstII_SPFNO, idstII_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOLIII, T)
    input_channel = 3  # config['model']['input_channel'] * config['data']['initial_step'] + 2
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)  # .to(torch.float32)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")


    def count_parameters(layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("\nTrainable parameters for each layer/module:")
    for name, layer in model.named_children():
        num_params = count_parameters(layer)
        print(f"{name}: {num_params} parameters")
    dummy_input = torch.randn(2, 65, 64, 3)
    output = model(dummy_input)
    print(f"\n最终输出: {output.shape}")
