'''
This project references the following open-source projects.
1. [SPFNO](https://github.com/liu-ziyuan-math/SPFNO) by Ziyuan Liu.
2. [spectral operator learning] (https://github.com/liu-ziyuan-math/spectral_operator_learning) by Ziyuan Liu.
3. [physics_informed FNO] (https://github.com/neuraloperator/physics_informed) by Zongyi Li.
4. [FNO] () by Zongyi Li.
5. [DCT] by
6. [FC-FNO] by Haydn Maust

'''
import functools
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


def dctII(u):
    '''
    这个也是计算DCT的一种方法，叫做“奇偶扩展法”
    根据gpt说，这个方法和上面那种通过奇延拓得到的是差不多的，这个更加精确一些？
    '''
    if not torch.is_tensor(u):
        u = torch.Tensor(u)
    Nx = u.shape[-1]
    v = torch.cat([u[..., ::2], u[..., 1::2].flip(dims=[-1])], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(Nx, dtype=u.dtype, device=u.device)
    W4 = torch.exp(-.5j * torch.pi * k / Nx)
    # print('dctII_SPFNO')
    return 2 * (V * W4).real / Nx


def idctII(a):
    if not torch.is_tensor(a):
        a = torch.Tensor(a)
    Nx = a.shape[-1]
    k = torch.arange(Nx, dtype=a.dtype, device=a.device)
    iW4 = 1 / torch.exp(-.5j * torch.pi * k / Nx)
    iW4[..., 0] /= 2
    V = torch.fft.ifft(a * iW4).real
    u = torch.zeros_like(V, dtype=a.dtype, device=a.device)
    u[..., ::2], u[..., 1::2] = V[..., :Nx - (Nx // 2)], V.flip(dims=[-1])[..., :Nx // 2]
    return u * Nx


class ZerosFilling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 返回与输入x相同形状的零张量
        return torch.zeros_like(x)


class SOL2D(nn.Module):
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

    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=2, skip=True, triL=0, flat=False,
                 double_weights=False):
        super(SOL2D, self).__init__()

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
        convND = nn.Conv2d
        self.convl = PseudoSpectra2D(T, dim, in_channels, width[0] - in_channels, modes[0], bandwidth[0], triL[0],
                                     double_weights)
        self.sp_convs = nn.ModuleList(
            [PseudoSpectra2D(T, dim, in_size, out_size, modes, bandwidth, tril, double_weights)
             for in_size, out_size, modes, bandwidth, tril
             in zip(self.width[0:], self.width[1:], self.modes,
                    self.bandwidth, self.triL)])
        self.ws = nn.ModuleList([convND(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.width[0:], self.width[1:])])
        self.fc1 = nn.Linear(width[-1], 128)
        self.fc2 = nn.Linear(128, out_channels)
        num_layers = len(self.width) - 1  # 线性层的数量

        if flat:
            # 创建线性层列表
            self.flat = nn.ModuleList([
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(self.width[0:], self.width[1:])
            ])
        else:
            # 创建与线性层数量相同的skip层列表
            if skip:
                self.flat = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
            else:
                self.flat = nn.ModuleList([ZerosFilling() for _ in range(num_layers)])

    # 在 forward 中添加调试
    def forward(self, x):
        x = x.permute(0, -1, *self.X_dims - 1)
        x = torch.cat([x, F.gelu(self.convl(x))], dim=1)
        for i, (speconv, w, flat) in enumerate(zip(self.sp_convs, self.ws, self.flat)):
            x_1 = speconv(x)
            x_2 = w(x)
            x_perm = x.permute(0, *self.X_dims, 1)
            x_flat = flat(x_perm)
            if x_flat.dim() != x_perm.dim():
                print(f"ERROR: flat changed dimensions! {x_perm.dim()} -> {x_flat.dim()}")
            x = x_flat.permute(0, -1, *self.X_dims - 1) + F.gelu(x_1 + x_2)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, -1, *self.X_dims - 1)
        x = x.permute(0, 2, 3, 1)
        return x


class PseudoSpectra2D(nn.Module):
    """
    实参数，用于非FFT变换
    bandwidth=[1,1] 时自动使用快速路径，其余情况使用原版
    """

    def __init__(self, T, dim, in_channels, out_channels, modes, bandwidth=1, triL=0, double_weights=True):
        super(PseudoSpectra2D, self).__init__()

        self.T = T
        self.double_weights = double_weights
        self.dim = dim
        self.in_channels = in_channels if isinstance(in_channels, (int, np.integer)) else in_channels[0]
        self.out_channels = out_channels if isinstance(out_channels, (int, np.integer)) else out_channels[0]
        self.modes = modes
        self.X_dims = np.arange(-dim, 0)
        self.scale = 1 / (self.in_channels * self.out_channels)

        # 标准化 bandwidth 为 np.ndarray
        if isinstance(bandwidth, (int, np.integer)):
            self.bandwidth = np.array([bandwidth, bandwidth])
        elif isinstance(bandwidth, (list, tuple)):
            self.bandwidth = np.array(bandwidth)
        else:
            self.bandwidth = bandwidth

        # 标准化 triL
        if isinstance(triL, (int, np.integer)):
            self.triL = triL
            triL_is_zero = (triL == 0)
        elif isinstance(triL, (list, tuple)):
            self.triL = np.array(triL)
            triL_is_zero = (self.triL == 0).all()
        elif isinstance(triL, np.ndarray):
            self.triL = triL
            triL_is_zero = (triL == 0).all()
        else:
            self.triL = triL
            triL_is_zero = False

        # 判断是否可以使用快速路径: bandwidth 全为1 且 triL 全为0
        bandwidth_is_one = (self.bandwidth == 1).all()
        self._use_fast = bandwidth_is_one and triL_is_zero

        if self._use_fast:
            self._init_fast()
        else:
            self._init_full()

    def _init_fast(self):
        """快速版本初始化: bandwidth=[1,1], triL=0"""
        # 确保 modes 是 2D 的
        if isinstance(self.modes, (int, np.integer)):
            modes_2d = np.array([self.modes, self.modes])
        elif len(self.modes.shape) == 0 or (len(self.modes) == 1):
            modes_2d = np.array([self.modes.item(), self.modes.item()])
        else:
            modes_2d = self.modes
        
        self.modes = modes_2d  # 更新为 2D
        
        self.weights = nn.Parameter(
            self.scale * torch.rand(
                self.in_channels,
                self.out_channels,
                self.modes.prod().item(),
                dtype=torch.float32
            )
        )
        self.low_freq_slice = [slice(None), slice(None)] + [slice(m) for m in self.modes]

    def _init_full(self):
        """原版初始化: 支持任意 bandwidth 和 triL"""
        # 确保 triL 是可用于 padding 的格式
        if isinstance(self.triL, np.ndarray):
            padding = tuple(self.triL.tolist())
        else:
            padding = self.triL

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                self.in_channels * self.bandwidth.prod().item(),
                self.out_channels,
                self.modes.prod().item(),
                dtype=torch.float32
            )
        )
        self.unfold = torch.nn.Unfold(
            kernel_size=tuple(self.bandwidth.tolist()),
            padding=padding
        )
        self.X_slices1 = [slice(None), slice(None)] + [slice(freq) for freq in self.modes]

        # 处理 triL 用于切片计算
        if isinstance(self.triL, np.ndarray):
            triL_for_slice = self.triL
        else:
            triL_for_slice = np.array([self.triL, self.triL])

        self.pad_slices1 = [slice(None), slice(None)] + [
            slice(freq) for freq in self.modes + self.bandwidth - 1 - triL_for_slice * 2
        ]

    def quasi_diag_mul(self, input, weights):
        """原版的准对角乘法"""
        xpad = self.unfold(input)
        out = torch.einsum("bix, iox->box", xpad, weights)
        return out

    def forward(self, u):
        if self._use_fast:
            return self._forward_fast(u)
        else:
            return self._forward_full(u)

    def _forward_fast(self, u):
        
        
        B = u.shape[0]
        b = self.T(u, self.X_dims[::-1])
       
        
        # 原代码
        b_low = b[self.low_freq_slice].reshape(B, self.in_channels, -1)
       
        b_low = b[self.low_freq_slice].reshape(B, self.in_channels, -1)
        out_low = torch.einsum("bim, iom -> bom", b_low, self.weights)

        out = torch.zeros(B, self.out_channels, *u.shape[2:], device=u.device, dtype=b.dtype)
        out[self.low_freq_slice] = out_low.reshape(B, self.out_channels, *self.modes)

        return self.T.inv(out, self.X_dims)

    def _forward_full(self, u):
        """原版前向: 支持任意 bandwidth 和 triL"""
        B = u.shape[0]

        b = self.T(u, self.X_dims[::-1])
        out = torch.zeros(B, self.out_channels, *u.shape[2:], device=u.device, dtype=b.dtype)
        out[self.X_slices1] = self.quasi_diag_mul(
            b[self.pad_slices1], self.weights1
        ).reshape(B, self.out_channels, *self.modes)

        return self.T.inv(out, self.X_dims)


class Transform:
    def __init__(self, fwd, inv):
        assert (type(fwd) == functools.partial and type(inv) == functools.partial)
        self.fwd = fwd
        self.inv = inv

    def __call__(self, *args, **kwargs):
        return self.fwd(*args, **kwargs)


class PseudoSpectra(nn.Module):
    def __init__(self, T, dim, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super(PseudoSpectra, self).__init__()

        self.T = T
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-dim, 0)

        # print([(l, 0) for l in triL])
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.rand(in_channels * bandwidth.prod().item(), out_channels,
                                                       modes.prod().item()))  # size：[3*1*1,29,24*24]
        self.unfold = torch.nn.Unfold(kernel_size=bandwidth, padding=triL)
        # 虽然这个操作看起来有些许复杂，但是最后得到的结果和直接reshape没啥差别。只是把xy的二位数据拉平罢辽
        self.X_slices = [slice(None), slice(None)] + [slice(freq) for freq in modes]
        self.pad_slices = [slice(None), slice(None)] + [slice(freq) for freq in modes + bandwidth - 1 - triL * 2]

    def quasi_diag_mul(self, input, weights):
        xpad = self.unfold(input)  # 这里unfold的目的究竟何在？？
        return torch.einsum("bix, iox->box", xpad, weights)

    def forward(self, u):
        # 输入张量的szie是[20 3 130 130]
        batch_size = u.shape[0]
        b = self.T(u, self.X_dims)  # 先进行一次离散正弦变换 size: [20 3 130 130]
        out = torch.zeros(batch_size, self.out_channels, *u.shape[2:], device=u.device,
                          dtype=u.dtype)  # size: [20 29 130 130]
        out[self.X_slices] = self.quasi_diag_mul(b[self.pad_slices], self.weights).reshape(
            batch_size, self.out_channels, *self.modes)  # size: [20 29 130 130]， 其中除[20,29,24,24]外均为0
        u = self.T.inv(out, self.X_dims)  # size: [20 29 130 130]
        return u





def Wrapper(func_list, u, dim):
    '''
    更换了原本的代码逻辑，原本的代码逻辑旨在对同一个维度的数据进行多次变换，这里修改为对各维度的数据进行对应的变换；
    具体的变换类型由func_list确定。这里如果输入的维度是[sample,channel,t,x,y,z]
    则  func_list 为 [fun_t, fun_x, fun_y, fun_z]
    tips,做逆变换时，需要保持与正变换对称的顺序；这里通过dim的顺序进行控制；
    2D例子：
    例如一个[nt,nx]的二维问题，输入的对于正变换，输入的func_list=[fft_fun, dctI_SPFNO], 对应的逆变换为[ifft_fun, idctI_SPFNO]
    u的input shape是[b, channel,nt, nx]
    dim=[-1,-2]
    那么会先对-1维度进行变换：
    —— d = -1
    —— func = func_list[-1] = dctI_SPFNO
    —— u = func(u) 对-1维度(x维度)做DCTI

    —— d = -2
    —— u = torch.transpose(u, d, -1), 把-2维度换到-1那里
    —— func = func_list[-2] = fft_fun
    —— u = func(u) 对现在的-1维度(t维度)做 fft
    —— u = torch.transpose(u, d, -1) 转变回原本维度
    '''
    if type(dim) == int:
        dim = [dim]  # dim = [-2,-1]
    total_dim = u.dim()  # 对于二维问题，d的维度一般是[sample,channel,nx,ny],total_dim=4
    for d in dim:
        if (d != total_dim - 1) and (d != -1):
            u = torch.transpose(u, d, -1)
        func = func_list[d]
        # print(func_list[d])
        u = func(u)
        if (d != total_dim - 1) and (d != -1):
            u = torch.transpose(u, d, -1)
    return u


def WrapperO(func_list, u, dim):
    '''
    传入的func_list：
    dim=-1
    '''
    # a wrapper to apply a list of function on given axises.
    # the func will be applied in turn.
    if type(dim) == int:
        dim = [dim]
    total_dim = u.dim()  # 得到的结果是u的维度，在1D的问题中，应该都是3
    for d in dim:
        if (d != total_dim - 1) and (d != -1):
            u = torch.transpose(u, d, -1)
            '''
            这个的作用就是依次将需要变换的维度放在最后一个维度，然后进行相应的谱变换；
            '''
        for func in func_list:
            u = func(u)
        '''
        以1D的例子来看，就是在最后一个维度使用选择的变换方法对输入进行变化。
        '''
        if (d != total_dim - 1) and (d != -1):
            u = torch.transpose(u, d, -1)
    return u


def reset_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    from Burger.utils import *
    from functools import partial as PARTIAL
    import yaml
    from datetime import datetime

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    _trans = PARTIAL(Wrapper, [dctII, dctII])
    _itrans = PARTIAL(Wrapper, [idctII, idctII])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOL2D, T)
    modes = [[30, 30], [30, 30], [30, 30], [30, 30], [30, 30]]
    width = [40, 40, 40, 40, 40]
    bandwidth = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    out_channels = 1
    dim = 2
    tril = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    input_channel = 5
    reset_seed(42)
    model_1 = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                    dim=dim, triL=tril, double_weights=False,
                    skip=True, flat=False).to(device)  # .to(torch.float32)

    _idctII = PARTIAL(WrapperO, [idctII])
    _dctII = PARTIAL(WrapperO, [dctII])
    DCT_II = Transform(_dctII, _idctII)
    CosNO_II = PARTIAL(SOL, DCT_II)
    modes = 30
    width = 40
    bandwidth = 1
    reset_seed(42)
    model_2 = CosNO_II(5, modes, width, bandwidth, out_channels=1, dim=2, triL=0).to(device)

    dummy_input = torch.randn(1, 64, 64, 5).to(device)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ========== 快速定位问题 ==========
    print("=" * 60)
    print("逐步对比，定位差异来源")
    print("=" * 60)

    # 2. 设置为 eval 模式
    model_1.eval()
    model_2.eval()

    final_out1 = model_1(dummy_input)
    final_out2 = model_2(dummy_input)
    print(f'final_out1.shape:{final_out1.shape},final_out2.shape:{final_out2.shape}')
    diff = (final_out1 - final_out2).abs().max().item()
    print(f"  最终输出: diff = {diff:.2e} {'✓' if diff < 1e-6 else '❌'}")
    if diff < 1e-5:
        print("\n✅ 两个模型等价!")
    else:
        print("\n❌ 两个模型有差异")
