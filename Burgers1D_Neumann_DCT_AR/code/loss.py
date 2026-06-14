import os, sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import torch.nn.functional as F
from transforms import *
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

def burger_residual(u):
    u = u.permute(0, 2, 1, 3).squeeze(-1)
    # u.shape inPINO_loss_1D:  torch.Size([20, 101, 257])
    #     print('u.shape inPINO_loss_1D: ',u.shape)
    # equation loss
    # time_loss1 = time.time()
    ux, Du = burgers_residual(u)
    # time_loss2 = time.time()
    # print(f'residual_for_burgers use time : {time_loss2 - time_loss1:.2f}s')
    # f = torch.zeros(Du.shape, device=u.device, dtype=torch.float32)
    # loss_f = F.mse_loss(Du, f).to(torch.float32)
    # init condition loss
    # ux_left = (u[..., 1] - u[..., 0]) / dx
    # # 右边界 ux(L) ≈ (u[-1] - u[-2]) / dx
    # ux_right = (u[..., -1] - u[..., -2]) / dx
    # # Neumann BC: ux = 0
    # loss_b = (torch.mean(ux_left ** 2) + torch.mean(ux_right ** 2)).to(torch.float32)
    return Du
class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def dx_dctII(x_n, use_Gaussian_filter=False):
    N = x_n.shape[-1]
    d_x_n_f = dctII_SPFNO(x_n)
    k = torch.linspace(0, N - 1, N, dtype=x_n.dtype, device=x_n.device)
    K = -1 * d_x_n_f * k

    if use_Gaussian_filter:
        sigma = N // 10  # 控制高频衰减的强度
        filter = torch.exp(-0.5 * (k / sigma) ** 2)  # Gaussian 滤波器
        K = K * filter
    K[..., 0:N - 1] = K[..., 1:N].clone()
    K[..., N - 1] = 0.
    d_x_n = idstII_SPFNO(K)
    return d_x_n


def dx_dstI(x_n, use_Gaussian_filter=False):
    N = x_n.shape[-1]
    d_x_n_f = dstI_SPFNO(x_n)
    k = torch.linspace(0, N - 1, N, dtype=x_n.dtype, device=x_n.device)
    K = d_x_n_f * k
    if use_Gaussian_filter:
        sigma = int(nx * use_Gaussian_filter)  # 控制高频衰减的强度
        filter = torch.exp(-0.5 * (k / sigma) ** 2)  # Gaussian 滤波器
        K = K * filter
    d_x_n = idctI_SPFNO(K)
    return d_x_n


def dx_dctI(x_n, use_Gaussian_filter=False):
    N = x_n.shape[-1]
    dxn = dctI_SPFNO(x_n)
    k = torch.linspace(0, N - 1, N, dtype=x_n.dtype, device=x_n.device)
    K = -1 / 2 * dxn * k / N
    if use_Gaussian_filter:
        sigma = int(nx * use_Gaussian_filter)  # 控制高频衰减的强度
        filter = torch.exp(-0.5 * (k / sigma) ** 2)  # Gaussian 滤波器
        K = K * filter
    d_x_n = dstI_SPFNO(K)
    return d_x_n


def dx_dstII(x_n, use_Gaussian_filter=False):
    N = x_n.shape[-1]
    d_x_n_f = dstII_SPFNO(x_n)
    k = torch.linspace(1, N, N, dtype=x_n.dtype, device=x_n.device)
    K = d_x_n_f * k

    if use_Gaussian_filter:
        sigma = int(nx * use_Gaussian_filter)  # 控制高频衰减的强度
        filter = (torch.exp(-0.5 * (k / sigma) ** 2)).to(torch.float32)  # Gaussian 滤波器
        K = K * filter
    K[1:N] = K[0:N - 1].clone()
    K[0] = 0.
    d_x_n = idctII_SPFNO(K)
    return d_x_n


def dx_fc(x_n, x_length, use_Gaussian_filter=False):
    '''
    L是求解区间长度
    :param x_n:
    :param L:
    :return:
    '''
    dx_fc = CONTINUATION_FUNC(x_n)
    w_h = torch.fft.rfft(dx_fc, dim=-1)
    # Wavenumbers in y-direction
    N = dx_fc.size()[-1]
    k_x = torch.arange(start=0, end=N // 2 + 1, step=1, dtype=x_n.dtype, device=x_n.device).reshape(1, 1,
                                                                                                    N // 2 + 1)
    wx_h = 1j * k_x * w_h * (2 * x_length / (x_length * (400 + CONTINUATION_GRIDPOINTS) / 400))
    # hann_window = torch.hann_window(int(N/2)+1, periodic=False)

    if use_Gaussian_filter:
        sigma = int(N * use_Gaussian_filter)  # 控制高频衰减的强度
        filter = np.exp(-0.5 * (k_x / sigma) ** 2)  # Gaussian 滤波器
        wx_h = wx_h * filter

    wx = torch.fft.irfft(wx_h, dim=-1, n=N)
    DX_arr = wx[..., :-CONTINUATION_GRIDPOINTS]

    return DX_arr


import torch


def compute_k(nx, k_max, device=None):
    if nx % 2 == 0:
        k = torch.cat((
            torch.arange(0, k_max, 1, device=device, dtype=torch.int32),
            torch.arange(-k_max, 0, 1, device=device, dtype=torch.int32)
        ), dim=0).reshape(1, 1, nx)
    else:
        k_pos = torch.arange(0, k_max + 1, device=device, dtype=torch.int32)
        k_neg = torch.arange(-k_max, 0, 1, device=device, dtype=torch.int32)
        k = torch.cat((k_pos, k_neg), dim=0).reshape(1, 1, nx)

    return k


# 使用示例
nx = 11  # 假设是奇数长度序列，长度为 11
k_max = nx // 2  # 最大频率索引为 nx//2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = compute_k(nx, k_max, device)
print(k)


def dx_fft(x_n, use_Gaussian_filter=False):
    '''
    使用这个时，需要/x_length,其余方法需要 *pi/x_length
    :param x_n:
    :param use_Gaussian_filter:
    :return:
    '''
    nx = x_n.shape[-1]
    u_h = torch.fft.fft(x_n, dim=-1)
    print('u_h.dtype:', u_h.dtype)
    # Wavenumbers in y-direction
    k_max = nx // 2
    k = compute_k(nx, k_max, x_n.device)
    ux_h = 2j * torch.pi * k * u_h
    if use_Gaussian_filter:
        sigma = int(nx * use_Gaussian_filter)  # 控制高频衰减的强度
        filter = torch.exp(-0.5 * (k / sigma) ** 2).to(torch.float32)  # Gaussian 滤波器
        ux_h = ux_h * filter
    ux = torch.fft.irfft(ux_h[:, :, :], dim=-1, n=nx)
    return ux


def dtI(x_n, time_lentgh):
    '''
    使用前向欧拉法求解
    :param x_n:
    :param D: 时间长度
    :return:
    '''
    nt = x_n.size(1)
    dt = time_lentgh / (nt - 1)
    xt = (x_n[:, 2:, :] - x_n[:, :-2, :]) / (2 * dt)
    return xt


def dtII(x_n, D):
    '''
    龙格库塔法
    :param x_n:
    :param D:
    :return:
    '''
    pass


def chebdiffD(n):
    if n == 0:
        return np.array([])

    # CGL nodes
    x = np.cos(np.pi * np.arange(n + 1) / n)

    # Coefficients
    c = np.ones(n + 1)
    c[0] = 2
    c[-1] = 2
    c *= (-1) ** np.arange(n + 1)

    # Compute differentiation matrix
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = (np.outer(c, 1 / c)) / (dX + np.eye(n + 1))  # Off-diagonal entries
    D -= np.diag(np.sum(D, axis=1))  # Diagonal entries
    D = torch.tensor(D, dtype=torch.float)
    return D


def DX_chebshev_physical_space(x_n):
    '''
    只适用于[-1,1]上的CGL点；

    :param x_n:
    :return:
    '''
    N = x_n.shape[-1]
    print('x_n.shape:', x_n.shape)
    D = chebdiffD(N - 1)
    print('D.shape:', D.shape)
    epsilon = alpha
    dx = torch.matmul(D, x_n.float())
    return dx


def burgers_residual(u, dt=0.01, dx_N=0.5, dxx_N=0.5, nu=0.01, x_length=2.0):
    if u.ndim == 2:
        u = np.expand_dims(u, axis=0)  # 在axis=0增加一维
    if not isinstance(u, torch.Tensor):
        u = torch.from_numpy(u).float()
    # nt = u.shape[1]
    N = u.shape[-1]

    k = torch.linspace(0, N - 1, N, dtype=u.dtype, device=u.device)
    C_dct = -1 / (2 * (N - 1))
    C_case = torch.pi / x_length
    u_f = dctI_SPFNO(u)

    dxn_f = u_f
    cutoff = N // dx_N
    dxn_f[..., k > cutoff] = 0

    K_first_order = k * C_dct * C_case
    ux = dstI_SPFNO(dxn_f * K_first_order)

    dxxn_f = u_f
    cutoff = N // dxx_N
    dxxn_f[..., k > cutoff] = 0
    K_second_order = k ** 2 * C_dct * C_case ** 2
    uxx = dctI_SPFNO(dxxn_f * K_second_order)
    # dt = time_lentgh / (nt - 1)

    ut = (-u[:, 4:, :] + 8 * u[:, 3:-1, :] - 8 * u[:, 1:-3, :] + u[:, :-4, :]
          ) / (12 * dt)
    Du = ut + (ux * u - nu * uxx)[:, 2:-2, :]
    return ux, Du


def burgers_residual_B(u, dx_N=0.5, dxx_N=0.5, nu=0.01, x_length=2.0):
    """
        计算 Burgers 方程残差: u_t + u * u_x - nu * u_xx = 0

        参数:
            u: [batch, nt, nx] 完整的场
            dt: 时间步长
            dx_N: 一阶导数截断比例
            dxx_N: 二阶导数截断比例
            nu: 粘性系数
            x_length: 空间域长度

        返回:
            ux: 空间一阶导数
            Du: PDE 残差 [batch, nt-1, nx]
        """
    if u.ndim == 2:
        u = np.expand_dims(u, axis=0)  # 在axis=0增加一维
    if not isinstance(u, torch.Tensor):
        u = torch.from_numpy(u).float()
    batch, nt, N = u.shape

    k = torch.linspace(0, N - 1, N, dtype=u.dtype, device=u.device)
    C_dct = -1 / (2 * (N - 1))
    C_case = torch.pi / x_length
    u_f = dctI_SPFNO(u)

    dxn_f = u_f
    cutoff = N // dx_N
    dxn_f[..., k > cutoff] = 0

    K_first_order = k * C_dct * C_case
    ux = dstI_SPFNO(dxn_f * K_first_order)  # [batch, nt, nx]

    dxxn_f = u_f
    cutoff = N // dxx_N
    dxxn_f[..., k > cutoff] = 0
    K_second_order = k ** 2 * C_dct * C_case ** 2
    uxx = dctI_SPFNO(dxxn_f * K_second_order)
    dt = 1.0 / (nt - 1)

    ut = (-u[:, 4:, :] + 8*u[:, 3:-1, :] - 8*u[:, 1:-3, :] + u[:, :-4, :]) / (12*dt)
    Du = ut + (ux * u - nu * uxx)[:, 2:-2, :]  # [batch, nt-1, nx]
    return ux, Du


def burgers_residual_2(u, f, dt=0.01, dx_N=0.5, dxx_N=0.5, nu=0.01, x_length=2.0):
    if u.ndim == 2:
        u = np.expand_dims(u, axis=0)  # 在axis=0增加一维
    if not isinstance(u, torch.Tensor):
        u = torch.from_numpy(u).float()
    # nt = u.shape[1]
    N = u.shape[-1]

    k = torch.linspace(0, N - 1, N, dtype=u.dtype, device=u.device)
    C_dct = -1 / (2 * (N - 1))
    C_case = torch.pi / x_length
    u_f = f

    dxn_f = u_f
    cutoff = N // dx_N
    dxn_f[..., k > cutoff] = 0

    K_first_order = k * C_dct * C_case
    ux = dstI_SPFNO(dxn_f * K_first_order)

    dxxn_f = u_f
    cutoff = N // dxx_N
    dxxn_f[..., k > cutoff] = 0
    K_second_order = k ** 2 * C_dct * C_case ** 2
    uxx = dctI_SPFNO(dxxn_f * K_second_order)
    # dt = time_lentgh / (nt - 1)

    ut = (-u[:, 4:, :] + 8 * u[:, 3:-1, :] - 8 * u[:, 1:-3, :] + u[:, :-4, :]
          ) / (12 * dt)
    Du = ut + (ux * u - nu * uxx)[:, 2:-2, :]
    return ux, Du


def residual_for_burgers(u, nu, x_length, time_lentgh):
    if u.ndim == 2:
        u = np.expand_dims(u, axis=0)  # 在axis=0增加一维
    if not isinstance(u, torch.Tensor):
        u = torch.from_numpy(u).float()
    nt = u.shape[1]
    ux = dx_dctI(u) * (torch.pi / x_length)
    uxx = dx_dstI(ux) * (torch.pi / x_length)
    dt = time_lentgh / (nt - 1)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux * u - nu * uxx)[:, 1:-1, :]
    return ux, Du


def loss_boundary_Dirichlet(u):
    u = u.squeeze()
    bc_1 = u[..., 0]
    bc_2 = u[..., -1]
    u_bc = torch.zeros_like(bc_2)
    loss_bc_1 = F.mse_loss(bc_1, u_bc)
    loss_bc_2 = F.mse_loss(bc_2, u_bc)
    loss_bc = (loss_bc_2 + loss_bc_1) / 2
    return loss_bc


def loss_boundary_Neumann(du):
    u = du.squeeze()
    bc_1 = u[..., 0]
    bc_2 = u[..., -1]
    u_bc = torch.zeros_like(bc_2, dtype=torch.float32)
    loss_bc_1 = F.mse_loss(bc_1, u_bc)
    loss_bc_2 = F.mse_loss(bc_2, u_bc)
    loss_bc = (loss_bc_2 + loss_bc_1) / 2
    return loss_bc


def loss_init(u, u0):
    init_u = u[:, 0:1, :].squeeze()
    #     print('init_u.shape:',init_u.shape)
    loss_u = F.mse_loss(init_u, u0)
    return loss_u


def PINO_loss_1DIII(u):
    """
    计算 Burgers 方程的 PINO loss

    参数:
        u: [batch, nx, nt] 完整的场（包含初始条件）
        x_length: 空间域长度 (默认 2.0，对应 [-1, 1])
        time_length: 时间域长度 (默认 1.0)

    返回:
        loss_f: PDE 残差损失
        loss_b: 边界条件损失
    """
    x_length = 2.0
    batch, nx, nt = u.shape

    # 计算网格间距
    dx = x_length / (nx - 1)

    # ========== PDE 残差损失 ==========
    u_transposed = u.permute(0, 2, 1)  # [batch, nt, nx]
    ux, Du = burgers_residual_B(u_transposed)
    f = torch.zeros(Du.shape, device=u.device, dtype=u.dtype)
    loss_f = F.mse_loss(Du, f)

    # ========== 边界条件损失 (Neumann BC: u_x = 0) ==========
    # 左边界 u_x(x=0)
    ux_left = (u[:, 1, :] - u[:, 0, :]) / dx  # [batch, nt]
    # 右边界 u_x(x=L)
    ux_right = (u[:, -1, :] - u[:, -2, :]) / dx  # [batch, nt]

    loss_b = torch.mean(ux_left ** 2) + torch.mean(ux_right ** 2)

    return Du, loss_f, loss_b


def PINO_loss_1D(u, u0, init_t, x_length=2, time_lentgh=1):
    u = u.permute(0, 2, 1, 3).squeeze(-1)
#     u.shape inPINO_loss_1D:  torch.Size([20, 101, 257])
#     print('u.shape inPINO_loss_1D: ',u.shape)
    # equation loss
    # time_loss1 = time.time()
    batch,  nt,nx = u.shape
    dx = x_length / (nx - 1)
    ux, Du = burgers_residual(u)
    # time_loss2 = time.time()
    # print(f'residual_for_burgers use time : {time_loss2 - time_loss1:.2f}s')
    f = torch.zeros(Du.shape, device=u.device, dtype=torch.float32)
    loss_f = F.mse_loss(Du, f).to(torch.float32)
    # init condition loss
    init_u = u[:, 0:init_t, ...]
    loss_i = F.mse_loss(init_u, u0).to(torch.float32)
    # boundary condition loss
    # 左边界 ux(0) ≈ (u[1] - u[0]) / dx
    ux_left = (u[..., 1] - u[..., 0]) / dx
    # 右边界 ux(L) ≈ (u[-1] - u[-2]) / dx
    ux_right = (u[..., -1] - u[..., -2]) / dx
    # Neumann BC: ux = 0
    loss_b = (torch.mean(ux_left ** 2) + torch.mean(ux_right ** 2)).to(torch.float32)
    return loss_i, loss_f, loss_b


def PINO_loss_1DII(u, u0, init_t, x_length, time_lentgh):
    '''
    use FDM DIFF
    :param u:
    :param u0:
    :param v:
    :param x_length:
    :param time_lentgh:
    :return:
    '''
    v = 0.1
    # print('u[:, :, 0, :]:', u[:, :, 0, :])
    batchsize = u.size(0)
    nt = u.size(2)
    nx = u.size(1)
    u = u.permute(0, 2, 1, 3).squeeze(-1)  # shape:[batch,nt,nx]
    # print('u[:, 0, :, :]:', u[:, 0, :])
    # print('u.shape:', u.shape)
    # equation loss
    # time_loss1 = time.time()
    # ux, Du = residual_for_burgers(u, v, x_length, time_lentgh)
    # time_loss1 = time.time()
    ux, Du = calculate_FDM(time_lentgh / (nt - 1), x_length / (nx - 1), u, v)
    # time_loss2 = time.time()
    # print(f'residual_for_burgers use time : {time_loss2 - time_loss1:.2f}s')
    # time_loss2 = time.time()
    # print(f'residual_for_burgers use time : {time_loss2 - time_loss1:.2f}s')
    f = torch.zeros(Du.shape, device=u.device, dtype=torch.float32)
    loss_f = F.mse_loss(Du, f).to(torch.float32)
    # init condition loss
    init_u = u[:, 0:init_t, ...]
    loss_i = F.mse_loss(init_u, u0).to(torch.float32)
    # boundary condition loss
    loss_b = loss_boundary_Neumann(ux).to(torch.float32)
    return loss_i, loss_f, loss_b


def residual_test(u, u0, v):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)
    u = u.reshape(batchsize, nt, nx)
    # equation loss
    Du, ux = residual_for_burgers(u, v, 1, 1)[:, :, :]
    Du, ux = calculate_FDM(u, v, 1, 1)[:, :, :]

    f = torch.zeros(Du.shape, device=u.device).to(torch.float64)
    loss_f = F.mse_loss(Du, f)
    # init condition loss
    loss_i = loss_init(u, u0)
    # boundary condition loss
    loss_b = loss_boundary_Neumann(ux)
    return loss_f


class Conv1dDerivative(torch.nn.Module):
    def __init__(self, DerFilter, resol, kernel_size, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = torch.nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                      1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = torch.nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        '''
        这里的self.resol一般是指比如dx ** 2这样子；
        :param input:
        :return:
        '''
        derivative = self.filter(input)
        return derivative / self.resol


def calculate_FDM(delta_t, delta_x, u, v):
    """

    :param t_axis:
    :param x_axis:
    :param u:
    :param v:
    :return:
    """
    # 定义时间偏导数
    dt = Conv1dDerivative(DerFilter=[[[-1, 0, 1]]],
                          resol=(delta_t * 2),
                          kernel_size=3,
                          name='partial_t').to(device)
    # 定义拉普拉斯算子，用于计算二阶导数 u_xx
    laplace = Conv1dDerivative(DerFilter=[[[-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12]]],
                               resol=(delta_x ** 2),
                               kernel_size=5,
                               name='laplace_operator').to(device)
    # 定义一阶空间导数，用于计算 u_x（五点差分法）
    dx = Conv1dDerivative(DerFilter=[[[-1 / 12, 8 / 12, 0, -8 / 12, 1 / 12]]],
                          resol=delta_x,
                          kernel_size=5,
                          name='partial_x').to(device)

    sample = u.shape[0]
    nt = u.shape[1]
    nx = u.shape[2]
    # 计算 u_xx
    u_xx_input = u.reshape(sample * nt, 1, nx)
    u_xx = laplace(u_xx_input)
    u_xx = u_xx.view(sample, nt, -1)
    u_xx = u_xx[:, 1:-1:]  # 去除边界，防止超出计算范围

    # 计算 u_x
    u_x_input = u.reshape(sample * nt, 1, nx)
    u_x = dx(u_x_input)
    u_x = u_x.view(sample, nt, -1)
    u_x = u_x[:, 1:-1, :]

    # 计算 u_t
    u_conv_for_t = u.permute(0, 2, 1)
    u_conv_for_t = u_conv_for_t.reshape(sample * nx, 1, nt)
    u_t = dt(u_conv_for_t)  # .reshape(sample, nx, -1)
    u_t = u_t.view(sample, nx, -1)
    u_t = u_t.permute(0, 2, 1)[:, :, 2:-2]

    # 获取有效部分的 u
    u = u[:, 1:-1, 2:-2]
    assert u.shape == u_x.shape == u_xx.shape == u_t.shape, "Shapes of tensors do not match!"
    # 计算 Du
    Du = u_t + (-u_x * u - v * u_xx)
    return -u_x, Du


def FDM_Burgers(u, v=0.01, D=1):
    '''
    说是那么多diff的方法实际上还是在输出中做文章啊诶
    '''
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt - 1)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    dx = 1 / (nx)

    u_h = torch.fft.fft(u, dim=2)
    # Wavenumbers in y-direction
    k_max = nx // 2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1, 1, nx)
    ux_h = 2j * np.pi * k_x * u_h
    uxx_h = 2j * np.pi * k_x * ux_h
    ux = torch.fft.irfft(ux_h[:, :, :k_max + 1], dim=2, n=nx)
    uxx = torch.fft.irfft(uxx_h[:, :, :k_max + 1], dim=2, n=nx)

    Du = ut + (ux * u - v * uxx)[:, 1:-1, :]
    return ux, Du


def u_exact(t, x):
    return np.exp(-pi ** 2 * t) * np.cos(pi * x)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # 参数定义
    alpha = 1
    pi = np.pi

    # 定义空间和时间网格
    nx, nt = 256, 101
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    print(dx, dt)
    # 计算 u 的数值
    u = np.zeros((nt, nx))
    for i in range(nt):
        for j in range(nx):
            u[i, j] = u_exact(t[i], x[j])
    u = torch.tensor(u)

    # 计算时间一阶导数 du/dt（使用前向差分）
    u_t_1 = (u[1:, :] - u[:-1, :]) / dt
    print('u_t_1.shape:', u_t_1.shape)
    u_t_2 = dtI(u.unsqueeze(0), 1)  # 中心差分法计算dt
    print('u_t_2.shape:', u_t_2.shape)

    # 计算空间二阶导数 d²u/dx²（使用中心差分）
    u_xx = np.zeros_like(u)
    u_xx[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dx ** 2)
    print('u_xx.shape:', u_xx.shape)

    u_x = dx_dctI(u.unsqueeze(0))
    print('u_x.shape:', u_x.shape)
    u_xx_2 = dx_fc(u_x, 1)
    print('u_xx_2.shape:', u_xx_2.shape)

    # 对齐时间和空间维度，方便计算残差
    u_t = u_t_1[:-1, 1:-1]  # 对齐时间维度
    u_xx = u_xx[1:-1, :]  # 对齐时间维度

    # 计算残差 R = du/dt - alpha * d²u/dx²
    residual = u_t - alpha * u_xx[:, 1:-1]

    # 绘制残差
    plt.imshow(u_xx[:, 1:-1], extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Residual')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Residual of Heat Equation with Homogeneous Neumann BCs')
    plt.show()

    residual = u_t_2.squeeze().cpu() - alpha * u_xx_2.squeeze()[1:-1, :].cpu()
    plt.imshow(u_xx_2.squeeze()[1:-1, :].cpu(), extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Residual')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Residual of Heat Equation with Homogeneous Neumann BCs')
    plt.show()



def _compute_ut(u, dt):
    """计算时间导数（混合差分格式）"""
    batch, nt, N = u.shape
    ut = torch.zeros(batch, nt - 1, N, device=u.device, dtype=u.dtype)
    
    # t=0: 一阶前向欧拉
    ut[:, 0, :] = (u[:, 1, :] - u[:, 0, :]) / dt
    
    # t=1: 二阶中心差分
    if nt > 2:
        ut[:, 1, :] = (u[:, 2, :] - u[:, 0, :]) / (2 * dt)
    
    # t=2 到 t=nt-3: 四阶中心差分
    if nt > 4:
        fourth_order = (-u[:, 4:, :] + 8 * u[:, 3:-1, :] - 8 * u[:, 1:-3, :] + u[:, :-4, :]) / (12 * dt)
        ut[:, 2:2 + fourth_order.shape[1], :] = fourth_order
    
    # t=nt-2, t=nt-1
    if nt > 3:
        ut[:, -2, :] = (u[:, -1, :] - u[:, -3, :]) / (2 * dt)
        ut[:, -1, :] = (u[:, -1, :] - u[:, -2, :]) / dt
    
    return ut


def compare_residuals(u, nu=0.01, x_length=2.0):
    """
    对比不同残差计算方法
    
    参数:
        u: [batch, nt, nx] 完整的场
    """
    Du_strong, Du_conserve, Du_weak, Du_strong_hat, Du_conserve_hat = burgers_residual_weak(
        u, nu=nu, x_length=x_length
    )
    
    # 计算各种统计量
    results = {
        'strong_form': {
            'L2': torch.sqrt(torch.mean(Du_strong ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_strong)).item(),
            'mean': torch.mean(torch.abs(Du_strong)).item(),
        },
        'conserve_form': {
            'L2': torch.sqrt(torch.mean(Du_conserve ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_conserve)).item(),
            'mean': torch.mean(torch.abs(Du_conserve)).item(),
        },
        'weak_form': {
            'L2': torch.sqrt(torch.mean(Du_weak ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_weak)).item(),
            'mean': torch.mean(torch.abs(Du_weak)).item(),
        },
    }
    
    return results, Du_strong, Du_conserve, Du_weak


def visualize_residual_comparison(filepath, n_samples=2, sub_t=1, sub_x=1, nu=0.01, save_dir=None):
    """
    可视化对比强形式、守恒形式、弱形式残差
    """
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # 读取数据
    with h5py.File(filepath, 'r') as h5_file:
        data_list = sorted(h5_file.keys())
    
    train_list = data_list[100:]
    selected_indices = np.random.choice(len(train_list), size=min(n_samples, len(train_list)), replace=False)
    selected_keys = [train_list[i] for i in selected_indices]
    
    # 创建大图: n_samples 行, 4 列
    fig = plt.figure(figsize=(20, 5 * n_samples))
    gs = GridSpec(n_samples, 4, figure=fig, hspace=0.3, wspace=0.25)
    
    for row, key in enumerate(selected_keys):
        # 读取数据
        with h5py.File(filepath, 'r') as h5_file:
            data = h5_file[key][:]
            data = data.astype('float32')
            data = torch.tensor(data, dtype=torch.float32)
        
        # 下采样
        data = data[::sub_t, ::sub_x]
        nt, nx = data.shape
        
        # 转换为 [1, nt, nx]
        u = data.unsqueeze(0)
        
        # 计算残差
        results, Du_strong, Du_conserve, Du_weak = compare_residuals(u, nu=nu)
        
        # 转换为 numpy
        u_np = data.numpy()
        Du_strong_np = Du_strong.squeeze(0).numpy()
        Du_conserve_np = Du_conserve.squeeze(0).numpy()
        Du_weak_np = Du_weak.squeeze(0).numpy()
        
        # 网格
        x_grid = np.linspace(-1, 1, nx)
        t_grid = np.linspace(0, 1, nt)
        t_grid_res = np.linspace(0, 1, nt - 1)
        T, X = np.meshgrid(t_grid, x_grid)
        T_res, X_res = np.meshgrid(t_grid_res, x_grid)
        
        # 列1: 解 u(x,t)
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.pcolormesh(T, X, u_np.T, shading='auto', cmap='viridis')
        ax1.set_title(f'Sample: {key}\nSolution u(x,t)', fontsize=10)
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 列2: 强形式残差
        ax2 = fig.add_subplot(gs[row, 1])
        vmax = max(np.abs(Du_strong_np).max(), 1e-10)
        im2 = ax2.pcolormesh(T_res, X_res, Du_strong_np.T, shading='auto', 
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax2.set_title(f'Strong Form\nL2={results["strong_form"]["L2"]:.2e}, Max={results["strong_form"]["Linf"]:.2e}', 
                     fontsize=10)
        ax2.set_xlabel('t')
        ax2.set_ylabel('x')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 列3: 守恒形式残差
        ax3 = fig.add_subplot(gs[row, 2])
        vmax = max(np.abs(Du_conserve_np).max(), 1e-10)
        im3 = ax3.pcolormesh(T_res, X_res, Du_conserve_np.T, shading='auto', 
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax3.set_title(f'Conservation Form\nL2={results["conserve_form"]["L2"]:.2e}, Max={results["conserve_form"]["Linf"]:.2e}', 
                     fontsize=10)
        ax3.set_xlabel('t')
        ax3.set_ylabel('x')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 列4: 弱形式残差（低频）
        ax4 = fig.add_subplot(gs[row, 3])
        vmax = max(np.abs(Du_weak_np).max(), 1e-10)
        im4 = ax4.pcolormesh(T_res, X_res, Du_weak_np.T, shading='auto', 
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax4.set_title(f'Weak Form (Low Freq)\nL2={results["weak_form"]["L2"]:.2e}, Max={results["weak_form"]["Linf"]:.2e}', 
                     fontsize=10)
        ax4.set_xlabel('t')
        ax4.set_ylabel('x')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    fig.suptitle('Residual Comparison: Strong vs Conservation vs Weak Form', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/residual_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/residual_comparison.pdf', bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    return results

def burgers_residual_weak_fixed(u, dx_N=0.5, dxx_N=0.5, nu=0.01, x_length=2.0, weak_cutoff=0.25):
    """
    计算 Burgers 方程的强形式、守恒形式、弱形式（低频）残差
    
    参数:
        u: [batch, nt, nx] 完整的场
        weak_cutoff: 弱形式保留的低频比例（默认 25%）
    
    返回:
        Du_strong: 强形式残差 [batch, nt-4, nx]
        Du_conserve: 守恒形式残差 [batch, nt-4, nx]
        Du_weak: 弱形式残差（只保留低频）[batch, nt-4, nx]
    """
    if u.ndim == 2:
        u = u.unsqueeze(0)
    if not isinstance(u, torch.Tensor):
        u = torch.from_numpy(u).float()
    
    batch, nt, N = u.shape
    dt = 1.0 / (nt - 1)
    
    k = torch.linspace(0, N - 1, N, dtype=u.dtype, device=u.device)
    C_dct = -1 / (2 * (N - 1))
    C_case = torch.pi / x_length
    K_first = k * C_dct * C_case
    K_second = k ** 2 * C_dct * C_case ** 2
    
    # ========== 频域变换 ==========
    u_f = dctI_SPFNO(u)
    
    # u_x（带截断）
    u_f_dx = u_f.clone()
    cutoff = int(N * dx_N) if dx_N < 1 else int(N / dx_N)
    u_f_dx[..., cutoff:] = 0
    ux = dstI_SPFNO(u_f_dx * K_first)
    
    # u_xx（带截断）
    u_f_dxx = u_f.clone()
    cutoff = int(N * dxx_N) if dxx_N < 1 else int(N / dxx_N)
    u_f_dxx[..., cutoff:] = 0
    uxx = dctI_SPFNO(u_f_dxx * K_second)
    
    # ========== 时间导数（全局四阶） ==========
    ut = (-u[:, 4:, :] + 8*u[:, 3:-1, :] - 8*u[:, 1:-3, :] + u[:, :-4, :]) / (12*dt)
    # ut: [batch, nt-4, nx]，对应 t_2 到 t_{nt-3}
    
    # 对齐到 t_2 到 t_{nt-3}
    u_mid = u[:, 2:-2, :]
    ux_mid = ux[:, 2:-2, :]
    uxx_mid = uxx[:, 2:-2, :]
    
    # ========== 强形式残差 ==========
    # u_t + u * u_x - ν * u_xx = 0
    Du_strong = ut + u_mid * ux_mid - nu * uxx_mid
    
    # ========== 守恒形式残差 ==========
    # u_t + (u²/2)_x - ν * u_xx = 0
    F = 0.5 * u ** 2
    F_f = dctI_SPFNO(F)
    F_f_dx = F_f.clone()
    cutoff = int(N * dx_N) if dx_N < 1 else int(N / dx_N)
    F_f_dx[..., cutoff:] = 0
    F_x = dstI_SPFNO(F_f_dx * K_first)
    F_x_mid = F_x[:, 2:-2, :]
    
    Du_conserve = ut + F_x_mid - nu * uxx_mid
    
    # ========== 弱形式残差（只保留低频） ==========
    # 将强形式残差变换到频域，只保留低频部分，再变换回物理域
    Du_strong_hat = dctI_SPFNO(Du_strong)  # 残差的 DCT
    
    # 只保留前 weak_cutoff 比例的模态
    cutoff_weak = int(N * weak_cutoff)
    Du_weak_hat = torch.zeros_like(Du_strong_hat)
    Du_weak_hat[..., :cutoff_weak] = Du_strong_hat[..., :cutoff_weak]
    
    # 逆变换回物理域
    Du_weak = dctI_SPFNO(Du_weak_hat)  # DCT-I 是自逆的
    
    return Du_strong, Du_conserve, Du_weak


def compare_residuals_fixed(u, nu=0.01, x_length=2.0):
    """对比三种残差"""
    Du_strong, Du_conserve, Du_weak = burgers_residual_weak_fixed(u, nu=nu, x_length=x_length)
    
    results = {
        'strong_form': {
            'L2': torch.sqrt(torch.mean(Du_strong ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_strong)).item(),
        },
        'conserve_form': {
            'L2': torch.sqrt(torch.mean(Du_conserve ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_conserve)).item(),
        },
        'weak_form': {
            'L2': torch.sqrt(torch.mean(Du_weak ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_weak)).item(),
        },
    }
    
    print("=" * 50)
    print(f"{'形式':<15} {'L2':<15} {'Max':<15}")
    print("=" * 50)
    print(f"{'强形式':<15} {results['strong_form']['L2']:<15.4e} {results['strong_form']['Linf']:<15.4e}")
    print(f"{'守恒形式':<15} {results['conserve_form']['L2']:<15.4e} {results['conserve_form']['Linf']:<15.4e}")
    print(f"{'弱形式(低频)':<15} {results['weak_form']['L2']:<15.4e} {results['weak_form']['Linf']:<15.4e}")
    print("=" * 50)
    
    return results, Du_strong, Du_conserve, Du_weak


import torch.fft

def low_pass_filter(x, cutoff_ratio=0.25):
    """
    对信号做低通滤波（只保留低频）
    x: [..., N]
    """
    N = x.shape[-1]
    cutoff = int(N * cutoff_ratio)
    
    # 使用 rfft（实数 FFT）
    x_fft = torch.fft.rfft(x, dim=-1)
    
    # 截断高频
    x_fft_filtered = x_fft.clone()
    x_fft_filtered[..., cutoff:] = 0
    
    # 逆变换
    x_filtered = torch.fft.irfft(x_fft_filtered, n=N, dim=-1)
    
    return x_filtered


def burgers_residual_weak_v2(u, dx_N=0.5, dxx_N=0.5, nu=0.01, x_length=2.0, weak_cutoff=0.25):
    """
    使用 FFT 低通滤波的弱形式残差
    """
    if u.ndim == 2:
        u = u.unsqueeze(0)
    if not isinstance(u, torch.Tensor):
        u = torch.from_numpy(u).float()
    
    batch, nt, N = u.shape
    dt = 1.0 / (nt - 1)
    
    k = torch.linspace(0, N - 1, N, dtype=u.dtype, device=u.device)
    C_dct = -1 / (2 * (N - 1))
    C_case = torch.pi / x_length
    K_first = k * C_dct * C_case
    K_second = k ** 2 * C_dct * C_case ** 2
    
    # 频域变换
    u_f = dctI_SPFNO(u)
    
    # u_x
    u_f_dx = u_f.clone()
    cutoff = int(N * dx_N) if dx_N < 1 else int(N / dx_N)
    u_f_dx[..., cutoff:] = 0
    ux = dstI_SPFNO(u_f_dx * K_first)
    
    # u_xx
    u_f_dxx = u_f.clone()
    cutoff = int(N * dxx_N) if dxx_N < 1 else int(N / dxx_N)
    u_f_dxx[..., cutoff:] = 0
    uxx = dctI_SPFNO(u_f_dxx * K_second)
    
    # 时间导数（全局四阶）
    ut = (-u[:, 4:, :] + 8*u[:, 3:-1, :] - 8*u[:, 1:-3, :] + u[:, :-4, :]) / (12*dt)
    
    # 对齐
    u_mid = u[:, 2:-2, :]
    ux_mid = ux[:, 2:-2, :]
    uxx_mid = uxx[:, 2:-2, :]
    
    # 强形式残差
    Du_strong = ut + u_mid * ux_mid - nu * uxx_mid
    
    # 守恒形式残差
    F = 0.5 * u ** 2
    F_f = dctI_SPFNO(F)
    F_f_dx = F_f.clone()
    cutoff = int(N * dx_N) if dx_N < 1 else int(N / dx_N)
    F_f_dx[..., cutoff:] = 0
    F_x = dstI_SPFNO(F_f_dx * K_first)
    F_x_mid = F_x[:, 2:-2, :]
    Du_conserve = ut + F_x_mid - nu * uxx_mid
    
    # 弱形式残差：对强形式残差做低通滤波
    Du_weak = low_pass_filter(Du_strong, cutoff_ratio=weak_cutoff)
    
    return Du_strong, Du_conserve, Du_weak


def compare_residuals_v2(u, nu=0.01, x_length=2.0):
    """对比三种残差"""
    Du_strong, Du_conserve, Du_weak = burgers_residual_weak_v2(u, nu=nu, x_length=x_length)
    
    results = {
        'strong_form': {
            'L2': torch.sqrt(torch.mean(Du_strong ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_strong)).item(),
        },
        'conserve_form': {
            'L2': torch.sqrt(torch.mean(Du_conserve ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_conserve)).item(),
        },
        'weak_form': {
            'L2': torch.sqrt(torch.mean(Du_weak ** 2)).item(),
            'Linf': torch.max(torch.abs(Du_weak)).item(),
        },
    }
    
    print("=" * 50)
    print(f"{'形式':<15} {'L2':<15} {'Max':<15}")
    print("=" * 50)
    print(f"{'强形式':<15} {results['strong_form']['L2']:<15.4e} {results['strong_form']['Linf']:<15.4e}")
    print(f"{'守恒形式':<15} {results['conserve_form']['L2']:<15.4e} {results['conserve_form']['Linf']:<15.4e}")
    print(f"{'弱形式(低频)':<15} {results['weak_form']['L2']:<15.4e} {results['weak_form']['Linf']:<15.4e}")
    print("=" * 50)
    
    return results, Du_strong, Du_conserve, Du_weak
def analyze_residual_spectrum(u, nu=0.01, x_length=2.0):
    """
    分析残差的频谱能量分布
    """
    if u.ndim == 2:
        u = u.unsqueeze(0)
    
    batch, nt, N = u.shape
    
    # 计算强形式残差
    Du_strong, Du_conserve, Du_weak = burgers_residual_weak_v2(u, nu=nu, x_length=x_length)
    
    # 对残差做 FFT
    Du_fft = torch.fft.rfft(Du_strong, dim=-1)
    power_spectrum = torch.abs(Du_fft) ** 2  # [batch, nt-4, N//2+1]
    
    # 平均功率谱
    avg_spectrum = power_spectrum.mean(dim=(0, 1))  # [N//2+1]
    
    # 累积能量
    total_energy = avg_spectrum.sum()
    cumulative_energy = torch.cumsum(avg_spectrum, dim=0) / total_energy
    
    # 找到 90%, 95%, 99% 能量对应的频率
    freq_90 = (cumulative_energy >= 0.90).nonzero()[0].item() if (cumulative_energy >= 0.90).any() else N//2
    freq_95 = (cumulative_energy >= 0.95).nonzero()[0].item() if (cumulative_energy >= 0.95).any() else N//2
    freq_99 = (cumulative_energy >= 0.99).nonzero()[0].item() if (cumulative_energy >= 0.99).any() else N//2
    
    print("=" * 60)
    print("残差频谱分析")
    print("=" * 60)
    print(f"总模态数: {N//2 + 1}")
    print(f"90% 能量所需模态: {freq_90} ({100*freq_90/(N//2+1):.1f}%)")
    print(f"95% 能量所需模态: {freq_95} ({100*freq_95/(N//2+1):.1f}%)")
    print(f"99% 能量所需模态: {freq_99} ({100*freq_99/(N//2+1):.1f}%)")
    print("=" * 60)
    
    # 绘图
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 功率谱
    ax1 = axes[0]
    ax1.semilogy(avg_spectrum.cpu().numpy())
    ax1.axvline(freq_90, color='g', linestyle='--', label=f'90%: mode {freq_90}')
    ax1.axvline(freq_95, color='orange', linestyle='--', label=f'95%: mode {freq_95}')
    ax1.axvline(freq_99, color='r', linestyle='--', label=f'99%: mode {freq_99}')
    ax1.set_xlabel('Mode')
    ax1.set_ylabel('Power (log)')
    ax1.set_title('Residual Power Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 累积能量
    ax2 = axes[1]
    ax2.plot(cumulative_energy.cpu().numpy())
    ax2.axhline(0.90, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(0.95, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(0.99, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(freq_90, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(freq_95, color='orange', linestyle='--', alpha=0.5)
    ax2.axvline(freq_99, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('Cumulative Energy')
    ax2.set_title('Cumulative Energy Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 不同截断比例的残差对比
    ax3 = axes[2]
    cutoffs = [0.1, 0.25, 0.5, 0.75, 1.0]
    l2_values = []
    for cutoff in cutoffs:
        Du_filtered = low_pass_filter(Du_strong, cutoff_ratio=cutoff)
        l2 = torch.sqrt(torch.mean(Du_filtered ** 2)).item()
        l2_values.append(l2)
    
    ax3.bar([f'{int(c*100)}%' for c in cutoffs], l2_values, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Frequency Cutoff')
    ax3.set_ylabel('Residual L2')
    ax3.set_title('Residual L2 vs Frequency Cutoff')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'avg_spectrum': avg_spectrum,
        'cumulative_energy': cumulative_energy,
        'freq_90': freq_90,
        'freq_95': freq_95,
        'freq_99': freq_99,
    }


def compare_cutoffs(u, nu=0.01, x_length=2.0):
    """
    对比不同截断比例的残差
    """
    if u.ndim == 2:
        u = u.unsqueeze(0)
    
    Du_strong, _, _ = burgers_residual_weak_v2(u, nu=nu, x_length=x_length, weak_cutoff=1.0)
    
    print("=" * 60)
    print(f"{'截断比例':<15} {'L2':<15} {'Max':<15} {'相对强形式':<15}")
    print("=" * 60)
    
    l2_full = torch.sqrt(torch.mean(Du_strong ** 2)).item()
    
    cutoffs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 1.00]
    for cutoff in cutoffs:
        Du_filtered = low_pass_filter(Du_strong, cutoff_ratio=cutoff)
        l2 = torch.sqrt(torch.mean(Du_filtered ** 2)).item()
        linf = torch.max(torch.abs(Du_filtered)).item()
        ratio = l2 / l2_full * 100
        print(f"{cutoff*100:>10.0f}%      {l2:<15.4e} {linf:<15.4e} {ratio:>10.1f}%")
    
    print("=" * 60)

    
def comprehensive_spectral_analysis(u, nu=0.01, x_length=2.0, save_dir=None, sample_name='sample'):
    """
    全面的频谱分析：解、强形式残差、弱形式残差（不同截断）
    
    参数:
        u: [batch, nt, nx] 或 [nt, nx] 完整的场
        nu: 粘性系数
        x_length: 空间域长度
        save_dir: 保存路径
        sample_name: 样本名称
    
    返回:
        分析结果字典
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    
    if u.ndim == 2:
        u = u.unsqueeze(0)
    if not isinstance(u, torch.Tensor):
        u = torch.from_numpy(u).float()
    
    batch, nt, nx = u.shape
    dt = 1.0 / (nt - 1)
    dx = x_length / (nx - 1)
    
    print("=" * 80)
    print(f"全面频谱分析（含弱形式）- {sample_name}")
    print("=" * 80)
    print(f"数据形状: [batch={batch}, nt={nt}, nx={nx}]")
    print(f"参数: nu={nu}, x_length={x_length}, dt={dt:.4f}, dx={dx:.4f}")
    print("=" * 80)
    
    # ==================== 1. 计算各种场和导数 ====================
    k = torch.linspace(0, nx - 1, nx, dtype=u.dtype, device=u.device)
    C_dct = -1 / (2 * (nx - 1))
    C_case = torch.pi / x_length
    K_first = k * C_dct * C_case
    K_second = k ** 2 * C_dct * C_case ** 2
    
    # DCT 变换
    u_dct = dctI_SPFNO(u)
    
    # 空间导数
    ux = dstI_SPFNO(u_dct * K_first)
    uxx = dctI_SPFNO(u_dct * K_second)
    
    # 时间导数 (全局四阶)
    ut = (-u[:, 4:, :] + 8*u[:, 3:-1, :] - 8*u[:, 1:-3, :] + u[:, :-4, :]) / (12*dt)
    
    # 对齐时间维度
    u_mid = u[:, 2:-2, :]
    ux_mid = ux[:, 2:-2, :]
    uxx_mid = uxx[:, 2:-2, :]
    
    # ==================== 2. 三种残差形式 ====================
    
    # 强形式残差: u_t + u*u_x - nu*u_xx
    Du_strong = ut + u_mid * ux_mid - nu * uxx_mid
    
    # 守恒形式残差: u_t + (u²/2)_x - nu*u_xx
    F = 0.5 * u ** 2
    F_dct = dctI_SPFNO(F)
    F_x = dstI_SPFNO(F_dct * K_first)
    F_x_mid = F_x[:, 2:-2, :]
    Du_conserve = ut + F_x_mid - nu * uxx_mid
    
    # 弱形式残差（不同截断比例）
    weak_cutoffs = [0.10, 0.20, 0.30, 0.40, 0.50]
    Du_weak_dict = {}
    for cutoff in weak_cutoffs:
        Du_weak_dict[cutoff] = low_pass_filter(Du_strong, cutoff_ratio=cutoff)
    
    # ==================== 3. FFT 频谱分析 ====================
    def get_spectrum(x):
        """计算平均功率谱和累积能量"""
        x_fft = torch.fft.rfft(x, dim=-1)
        power = (torch.abs(x_fft) ** 2).mean(dim=(0, 1))
        cumulative = torch.cumsum(power, dim=0) / (power.sum() + 1e-10)
        return power, cumulative
    
    # 各场的频谱
    u_power, u_cumsum = get_spectrum(u)
    ux_power, ux_cumsum = get_spectrum(ux)
    uxx_power, uxx_cumsum = get_spectrum(uxx)
    ut_power, ut_cumsum = get_spectrum(ut)
    Du_strong_power, Du_strong_cumsum = get_spectrum(Du_strong)
    Du_conserve_power, Du_conserve_cumsum = get_spectrum(Du_conserve)
    
    # 弱形式残差的频谱
    Du_weak_spectra = {}
    for cutoff in weak_cutoffs:
        power, cumsum = get_spectrum(Du_weak_dict[cutoff])
        Du_weak_spectra[cutoff] = {'power': power, 'cumsum': cumsum}
    
    n_freq = len(u_power)
    modes = np.arange(n_freq)
    
    # ==================== 4. 关键指标计算 ====================
    # 放大因子
    amp_strong = Du_strong_power / (u_power[:len(Du_strong_power)] + 1e-10)
    amp_conserve = Du_conserve_power / (u_power[:len(Du_conserve_power)] + 1e-10)
    
    # 弱形式的放大因子
    amp_weak_dict = {}
    for cutoff in weak_cutoffs:
        amp_weak_dict[cutoff] = Du_weak_spectra[cutoff]['power'] / (u_power[:len(Du_weak_spectra[cutoff]['power'])] + 1e-10)
    
    # 能量集中度
    def find_energy_threshold(cumsum, threshold=0.99):
        idx = (cumsum >= threshold).nonzero()
        return idx[0].item() if len(idx) > 0 else len(cumsum) - 1
    
    u_99 = find_energy_threshold(u_cumsum, 0.99)
    u_95 = find_energy_threshold(u_cumsum, 0.95)
    u_90 = find_energy_threshold(u_cumsum, 0.90)
    
    Du_99 = find_energy_threshold(Du_strong_cumsum, 0.99)
    Du_95 = find_energy_threshold(Du_strong_cumsum, 0.95)
    Du_90 = find_energy_threshold(Du_strong_cumsum, 0.90)
    
    # L2 范数
    u_l2 = torch.sqrt(torch.mean(u ** 2)).item()
    Du_strong_l2 = torch.sqrt(torch.mean(Du_strong ** 2)).item()
    Du_conserve_l2 = torch.sqrt(torch.mean(Du_conserve ** 2)).item()
    
    # 弱形式的 L2
    Du_weak_l2 = {}
    for cutoff in weak_cutoffs:
        Du_weak_l2[cutoff] = torch.sqrt(torch.mean(Du_weak_dict[cutoff] ** 2)).item()
    
    # 相对残差
    relative_residual_strong = Du_strong_l2 / (u_l2 + 1e-10)
    
    # 峰值位置
    u_peak = torch.argmax(u_power).item()
    Du_strong_peak = torch.argmax(Du_strong_power).item()
    amp_peak = torch.argmax(amp_strong).item()
    
    # Gibbs 判断
    has_gibbs = amp_strong.max() > 10
    gibbs_modes = (amp_strong > 10).nonzero().squeeze(-1) if has_gibbs else None
    
    # ==================== 5. 打印结果 ====================
    print("\n" + "=" * 80)
    print("【1. 基本统计】")
    print("=" * 80)
    print(f"  解 u:              L2={u_l2:.4e}, range=[{u.min():.4f}, {u.max():.4f}]")
    print(f"  强形式残差:        L2={Du_strong_l2:.4e}, Max={Du_strong.abs().max():.4e}")
    print(f"  守恒形式残差:      L2={Du_conserve_l2:.4e}, Max={Du_conserve.abs().max():.4e}")
    print(f"  相对残差(强形式):  {relative_residual_strong:.4e} ({relative_residual_strong*100:.4f}%)")
    
    print("\n" + "=" * 80)
    print("【2. 弱形式残差（不同截断）】")
    print("=" * 80)
    print(f"  {'截断比例':<12} {'L2':<15} {'相对强形式':<15} {'Max':<15}")
    print("-" * 57)
    for cutoff in weak_cutoffs:
        l2 = Du_weak_l2[cutoff]
        ratio = l2 / Du_strong_l2 * 100
        linf = Du_weak_dict[cutoff].abs().max().item()
        print(f"  {cutoff*100:>8.0f}%    {l2:<15.4e} {ratio:>10.1f}%      {linf:<15.4e}")
    print(f"  {'100% (强形式)':<12} {Du_strong_l2:<15.4e} {100.0:>10.1f}%      {Du_strong.abs().max().item():<15.4e}")
    
    print("\n" + "=" * 80)
    print("【3. 解 u 的能量分布】")
    print("=" * 80)
    print(f"  峰值位置:      mode {u_peak}")
    print(f"  90% 能量:      前 {u_90} 个模态 ({100*u_90/n_freq:.1f}%)")
    print(f"  95% 能量:      前 {u_95} 个模态 ({100*u_95/n_freq:.1f}%)")
    print(f"  99% 能量:      前 {u_99} 个模态 ({100*u_99/n_freq:.1f}%)")
    
    print("\n" + "=" * 80)
    print("【4. 强形式残差的能量分布】")
    print("=" * 80)
    print(f"  峰值位置:      mode {Du_strong_peak}")
    print(f"  90% 能量:      前 {Du_90} 个模态 ({100*Du_90/n_freq:.1f}%)")
    print(f"  95% 能量:      前 {Du_95} 个模态 ({100*Du_95/n_freq:.1f}%)")
    print(f"  99% 能量:      前 {Du_99} 个模态 ({100*Du_99/n_freq:.1f}%)")
    
    print("\n" + "=" * 80)
    print("【5. 放大因子分析 (残差/解)】")
    print("=" * 80)
    print(f"  强形式:")
    print(f"    最大放大因子:  {amp_strong.max():.4f}x (位于 mode {amp_peak})")
    print(f"    平均放大因子:  {amp_strong.mean():.4f}x")
    print(f"    放大因子 > 1:  {(amp_strong > 1).sum().item()} 个模态")
    print(f"    放大因子 > 10: {(amp_strong > 10).sum().item()} 个模态")
    
    print(f"\n  弱形式（不同截断）:")
    for cutoff in weak_cutoffs:
        max_amp = amp_weak_dict[cutoff].max().item()
        print(f"    {cutoff*100:>3.0f}% 截断: 最大放大因子 = {max_amp:.4f}x")
    
    print("\n" + "=" * 80)
    print("【6. Gibbs 伪影判断】")
    print("=" * 80)
    if has_gibbs:
        print(f"  ⚠️  检测到 Gibbs 伪影风险!")
        print(f"  问题区域: mode {gibbs_modes[0].item()} - {gibbs_modes[-1].item()}")
        recommended_cutoff = gibbs_modes[0].item() / n_freq
        print(f"  建议截断比例: < {recommended_cutoff*100:.0f}%")
    else:
        print(f"  ✓ 未检测到明显 Gibbs 伪影")
        print(f"  最大放大因子 {amp_strong.max():.2f}x < 10")
    
    # ==================== 6. 绘图 ====================
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Row 1: 解的时空图和频谱
    ax1 = fig.add_subplot(gs[0, 0])
    u_plot = u[0].cpu().numpy()
    im1 = ax1.imshow(u_plot.T, aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, 1, -1, 1])
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_title(f'Solution u(x,t)\nL2={u_l2:.4e}')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(modes, u_power.cpu().numpy(), 'b-', linewidth=2, label='|û|²')
    ax2.axvline(u_90, color='g', linestyle='--', alpha=0.7, label=f'90%: mode {u_90}')
    ax2.axvline(u_99, color='r', linestyle='--', alpha=0.7, label=f'99%: mode {u_99}')
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('Power (log)')
    ax2.set_title('Solution Power Spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(modes, u_cumsum.cpu().numpy(), 'b-', linewidth=2)
    ax3.axhline(0.90, color='g', linestyle='--', alpha=0.5)
    ax3.axhline(0.99, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Mode')
    ax3.set_ylabel('Cumulative Energy')
    ax3.set_title('Solution Cumulative Energy')
    ax3.grid(True, alpha=0.3)
    
    # Row 2: 强形式残差
    ax4 = fig.add_subplot(gs[1, 0])
    Du_plot = Du_strong[0].cpu().numpy()
    vmax = np.abs(Du_plot).max()
    im4 = ax4.imshow(Du_plot.T, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[0, 1, -1, 1], vmin=-vmax, vmax=vmax)
    ax4.set_xlabel('t')
    ax4.set_ylabel('x')
    ax4.set_title(f'Strong Form Residual\nL2={Du_strong_l2:.4e}')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogy(modes[:len(Du_strong_power)], Du_strong_power.cpu().numpy(), 'r-', linewidth=2, label='Strong |D̂u|²')
    ax5.semilogy(modes[:len(Du_conserve_power)], Du_conserve_power.cpu().numpy(), 'b--', linewidth=2, label='Conserve |D̂u|²')
    ax5.axvline(Du_90, color='g', linestyle='--', alpha=0.7, label=f'90%: mode {Du_90}')
    ax5.axvline(Du_99, color='orange', linestyle='--', alpha=0.7, label=f'99%: mode {Du_99}')
    ax5.set_xlabel('Mode')
    ax5.set_ylabel('Power (log)')
    ax5.set_title('Residual Power Spectrum')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(modes[:len(Du_strong_cumsum)], Du_strong_cumsum.cpu().numpy(), 'r-', linewidth=2, label='Strong')
    ax6.plot(modes[:len(Du_conserve_cumsum)], Du_conserve_cumsum.cpu().numpy(), 'b--', linewidth=2, label='Conserve')
    ax6.axhline(0.90, color='g', linestyle='--', alpha=0.5)
    ax6.axhline(0.99, color='orange', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Mode')
    ax6.set_ylabel('Cumulative Energy')
    ax6.set_title('Residual Cumulative Energy')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Row 3: 弱形式残差（不同截断）
    ax7 = fig.add_subplot(gs[2, 0])
    Du_weak_20 = Du_weak_dict[0.20][0].cpu().numpy()
    vmax = np.abs(Du_weak_20).max()
    im7 = ax7.imshow(Du_weak_20.T, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[0, 1, -1, 1], vmin=-vmax, vmax=vmax)
    ax7.set_xlabel('t')
    ax7.set_ylabel('x')
    ax7.set_title(f'Weak Form (20% cutoff)\nL2={Du_weak_l2[0.20]:.4e}')
    plt.colorbar(im7, ax=ax7)
    
    ax8 = fig.add_subplot(gs[2, 1])
    colors = plt.cm.viridis(np.linspace(0, 1, len(weak_cutoffs)))
    for i, cutoff in enumerate(weak_cutoffs):
        power = Du_weak_spectra[cutoff]['power'].cpu().numpy()
        ax8.semilogy(modes[:len(power)], power, color=colors[i], linewidth=2, 
                     label=f'{cutoff*100:.0f}%')
    ax8.semilogy(modes[:len(Du_strong_power)], Du_strong_power.cpu().numpy(), 'k--', 
                 linewidth=2, label='Strong (100%)')
    ax8.set_xlabel('Mode')
    ax8.set_ylabel('Power (log)')
    ax8.set_title('Weak Form Residual Spectra')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    cutoff_labels = [f'{c*100:.0f}%' for c in weak_cutoffs] + ['100%']
    l2_values = [Du_weak_l2[c] for c in weak_cutoffs] + [Du_strong_l2]
    bars = ax9.bar(cutoff_labels, l2_values, color='steelblue', edgecolor='black')
    bars[-1].set_color('darkred')  # 强形式用不同颜色
    ax9.set_xlabel('Frequency Cutoff')
    ax9.set_ylabel('Residual L2')
    ax9.set_title('Weak Form L2 vs Cutoff')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Row 4: 放大因子分析
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.semilogy(modes, u_power.cpu().numpy(), 'b-', linewidth=2, label='u')
    ax10.semilogy(modes, ux_power.cpu().numpy(), 'g-', linewidth=2, label='u_x')
    ax10.semilogy(modes, uxx_power.cpu().numpy(), 'm-', linewidth=2, label='u_xx')
    ax10.set_xlabel('Mode')
    ax10.set_ylabel('Power (log)')
    ax10.set_title('Derivatives Spectrum')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.semilogy(modes[:len(amp_strong)], amp_strong.cpu().numpy(), 'r-', linewidth=2, label='Strong')
    for i, cutoff in enumerate([0.20, 0.40]):
        ax11.semilogy(modes[:len(amp_weak_dict[cutoff])], amp_weak_dict[cutoff].cpu().numpy(), 
                      color=colors[weak_cutoffs.index(cutoff)], linewidth=2, label=f'Weak {cutoff*100:.0f}%')
    ax11.axhline(1, color='gray', linestyle='--', alpha=0.7, label='No amplification')
    ax11.axhline(10, color='orange', linestyle='--', alpha=0.7, label='Gibbs threshold')
    ax11.set_xlabel('Mode')
    ax11.set_ylabel('Amplification (log)')
    ax11.set_title(f'Amplification Factor\nStrong max={amp_strong.max():.2f}x')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(modes, u_cumsum.cpu().numpy(), 'b-', linewidth=2, label='Solution u')
    ax12.plot(modes[:len(Du_strong_cumsum)], Du_strong_cumsum.cpu().numpy(), 'r-', linewidth=2, label='Strong Du')
    ax12.plot(modes[:len(Du_weak_spectra[0.20]['cumsum'])], Du_weak_spectra[0.20]['cumsum'].cpu().numpy(), 
              'g--', linewidth=2, label='Weak 20%')
    ax12.set_xlabel('Mode')
    ax12.set_ylabel('Cumulative Energy')
    ax12.set_title('Energy Distribution Comparison')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    # Row 5: 对比总结
    ax13 = fig.add_subplot(gs[4, 0])
    # 不同形式的残差时空对比 (切片)
    t_slice = nt // 2  # 中间时刻
    x_grid = np.linspace(-1, 1, nx)
    ax13.plot(x_grid, Du_strong[0, t_slice//2, :].cpu().numpy(), 'r-', linewidth=2, label='Strong')
    ax13.plot(x_grid, Du_conserve[0, t_slice//2, :].cpu().numpy(), 'b--', linewidth=2, label='Conserve')
    ax13.plot(x_grid, Du_weak_dict[0.20][0, t_slice//2, :].cpu().numpy(), 'g-.', linewidth=2, label='Weak 20%')
    ax13.set_xlabel('x')
    ax13.set_ylabel('Residual')
    ax13.set_title(f'Residual Slice at t={t_slice//2}')
    ax13.legend()
    ax13.grid(True, alpha=0.3)
    
    ax14 = fig.add_subplot(gs[4, 1])
    # 残差 L2 随时间变化
    Du_strong_t = torch.sqrt(torch.mean(Du_strong ** 2, dim=-1)).squeeze(0).cpu().numpy()
    Du_weak_20_t = torch.sqrt(torch.mean(Du_weak_dict[0.20] ** 2, dim=-1)).squeeze(0).cpu().numpy()
    Du_weak_40_t = torch.sqrt(torch.mean(Du_weak_dict[0.40] ** 2, dim=-1)).squeeze(0).cpu().numpy()
    t_grid = np.linspace(0, 1, len(Du_strong_t))
    ax14.semilogy(t_grid, Du_strong_t, 'r-', linewidth=2, label='Strong')
    ax14.semilogy(t_grid, Du_weak_20_t, 'g--', linewidth=2, label='Weak 20%')
    ax14.semilogy(t_grid, Du_weak_40_t, 'm-.', linewidth=2, label='Weak 40%')
    ax14.set_xlabel('t')
    ax14.set_ylabel('L2 Residual (log)')
    ax14.set_title('Residual L2 vs Time')
    ax14.legend()
    ax14.grid(True, alpha=0.3)
    
    ax15 = fig.add_subplot(gs[4, 2])
    # 总结表格
    summary_text = f"""
    ╔══════════════════════════════════════════════════╗
    ║              分析结果总结                        ║
    ╠══════════════════════════════════════════════════╣
    ║  解 u:                                           ║
    ║    L2 = {u_l2:.4e}                            ║
    ║    99% 能量: 前 {u_99} 个模态 ({100*u_99/n_freq:.1f}%)              ║
    ╠══════════════════════════════════════════════════╣
    ║  强形式残差:                                     ║
    ║    L2 = {Du_strong_l2:.4e}                    ║
    ║    99% 能量: 前 {Du_99} 个模态 ({100*Du_99/n_freq:.1f}%)              ║
    ║    最大放大因子: {amp_strong.max():.2f}x                    ║
    ╠══════════════════════════════════════════════════╣
    ║  弱形式残差 (20% 截断):                          ║
    ║    L2 = {Du_weak_l2[0.20]:.4e}                    ║
    ║    相对强形式: {Du_weak_l2[0.20]/Du_strong_l2*100:.1f}%                       ║
    ╠══════════════════════════════════════════════════╣
    ║  Gibbs 伪影: {'是 ⚠️' if has_gibbs else '否 ✓'}                              ║
    ║  建议: {'使用弱形式 (截断<' + str(int(gibbs_modes[0].item()/n_freq*100)) + '%)' if has_gibbs else '可用强形式'}                     ║
    ╚══════════════════════════════════════════════════╝
    """
    ax15.text(0.05, 0.95, summary_text, transform=ax15.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax15.axis('off')
    ax15.set_title('Summary')
    
    plt.suptitle(f'Comprehensive Spectral Analysis (with Weak Form) - {sample_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{sample_name}_full_analysis.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/{sample_name}_full_analysis.pdf', bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    # ==================== 7. 最终建议 ====================
    print("\n" + "=" * 80)
    print("【最终建议】")
    print("=" * 80)
    
    if has_gibbs:
        recommended = gibbs_modes[0].item() / n_freq
        print(f"  1. ⚠️ 检测到 Gibbs 伪影")
        print(f"  2. 推荐使用弱形式残差，截断比例: {recommended*100:.0f}%")
        print(f"  3. 弱形式 L2 将降低至强形式的 {Du_weak_l2[min(weak_cutoffs, key=lambda x: abs(x-recommended))]/Du_strong_l2*100:.1f}%")
    else:
        # 找到能量跳跃点
        l2_ratios = [Du_weak_l2[c]/Du_strong_l2*100 for c in weak_cutoffs]
        jumps = [l2_ratios[i+1] - l2_ratios[i] for i in range(len(l2_ratios)-1)]
        
        if max(jumps) > 30:
            jump_idx = jumps.index(max(jumps))
            recommended = weak_cutoffs[jump_idx]
            print(f"  1. ✓ 无明显 Gibbs 伪影，但残差能量有跳跃")
            print(f"  2. 推荐截断比例: {recommended*100:.0f}%")
            print(f"  3. 可减少 {100-l2_ratios[jump_idx]:.1f}% 的残差能量（主要是高频噪声）")
        else:
            print(f"  1. ✓ 无 Gibbs 伪影，能量分布平缓")
            print(f"  2. 可直接使用强形式残差")
            print(f"  3. 如需滤波，建议截断 50% 以保守处理")
    
    print("=" * 80)
    
    # 返回结果
    return {
        'u_l2': u_l2,
        'Du_strong_l2': Du_strong_l2,
        'Du_conserve_l2': Du_conserve_l2,
        'Du_weak_l2': Du_weak_l2,
        'relative_residual': relative_residual_strong,
        'u_energy': {'90%': u_90, '95%': u_95, '99%': u_99},
        'Du_energy': {'90%': Du_90, '95%': Du_95, '99%': Du_99},
        'amplification': {
            'strong_max': amp_strong.max().item(),
            'strong_mean': amp_strong.mean().item(),
            'peak_mode': amp_peak,
            'weak': {c: amp_weak_dict[c].max().item() for c in weak_cutoffs}
        },
        'has_gibbs': has_gibbs,
        'spectra': {
            'u': u_power,
            'Du_strong': Du_strong_power,
            'Du_conserve': Du_conserve_power,
            'Du_weak': Du_weak_spectra,
            'amplification_strong': amp_strong,
            'amplification_weak': amp_weak_dict,
        },
        'residuals': {
            'strong': Du_strong,
            'conserve': Du_conserve,
            'weak': Du_weak_dict,
        }
    }