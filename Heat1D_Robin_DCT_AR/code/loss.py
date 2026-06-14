import numpy as np
import torch


def cheb(N):
    """
    计算切比雪夫求导矩阵和切比雪夫节点

    参数:
        N: 区间数

    返回:
        D: 切比雪夫求导矩阵 (N+1, N+1)
        x: 切比雪夫节点 [-1, 1]，从 +1 到 -1
    """
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    # 计算切比雪夫节点
    x = np.cos(np.pi * np.arange(N + 1) / N)
    # 计算权重系数
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c *= (-1.0) ** np.arange(N + 1)
    # 构造差商矩阵
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    # 计算切比雪夫求导矩阵
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    # 修正对角线元素，使每行和为 0
    np.fill_diagonal(D, D.diagonal() - np.sum(D, axis=1))
    return D, x


def compute_heat_residual(u, D2, nu=0.02):
    """
    使用预计算的 D² 矩阵计算残差

    参数:
        u: [batch, nx, nt, 1]
        D2_x: 二阶切比雪夫微分矩阵 [nx, nx]
        dt: 时间步长
        nu: 扩散系数
    """
    # print('u.shape:', u.shape)
    # print('D2_x.shape:', D2.shape)
    nt = u.shape[-2]
    dt = 1 / (u.shape[-2] - 1)
    # print('dt:', dt)
    u_squeezed = u.squeeze(-1)  # [batch, nx, nt]
    # 直接计算 u_xx
    # print(f'u_squeezed.type:{u_squeezed.dtype},D2.type:{D2.dtype}' )
    u_xx = torch.einsum('ij,bjt->bit', D2, u_squeezed)  # [batch, nx, nt]
    # 时间导数（中心差分）
    # u_t = (u_squeezed[:, :, 2:] - u_squeezed[:, :, :-2]) / (2 * dt)
    u_t = (-u_squeezed[:, :, 4:] + 8 * u_squeezed[:, :, 3:-1] - 8 * u_squeezed[:, :, 1:-3] + u_squeezed[:, :, :-4]) / (
            12 * dt)
    # u_t = torch.zeros_like(u_squeezed, dtype=torch.float64)

    # t=0: 1阶前向
    # u_t[:, :, 0] = (u_squeezed[:, :, 1] - u_squeezed[:, :, 0]) / dt
    #
    # # t=1: 2阶中心
    # u_t[:, :, 1] = (u_squeezed[:, :, 2] - u_squeezed[:, :, 0]) / (2 * dt)
    #
    # # t=2 到 t=nt-3: 4阶中心
    # u_t[:, :, 2:-2] = (-u_squeezed[:, :, 4:] + 8 * u_squeezed[:, :, 3:-1]
    #                    - 8 * u_squeezed[:, :, 1:-3] + u_squeezed[:, :, :-4]) / (12 * dt)
    #
    # # t=nt-2: 2阶中心
    # u_t[:, :, -2] = (u_squeezed[:, :, -1] - u_squeezed[:, :, -3]) / (2 * dt)
    #
    # # t=nt-1: 1阶后向
    # u_t[:, :, -1] = (u_squeezed[:, :, -1] - u_squeezed[:, :, -2]) / dt
    # 残差
    residual = u_t - nu * u_xx[:, :, 2:-2]
    # print(torch.mean(abs(residual[:,:,0])))
    # print(torch.mean(abs(residual[:, :, 1])))
    # print(torch.mean(abs(residual[:, :, 2:-2])))
    # print(torch.mean(abs(residual[:, :, -2])))
    # print(torch.mean(abs(residual[:, :, -1])))
    return residual.unsqueeze(-1)


def compute_heat_residualII(u, D2, u_t, init_t, nu=0.02):
    """
    使用预计算的 D² 矩阵计算残差

    参数:
        u: [batch, nx, nt, 1]
        D2_x: 二阶切比雪夫微分矩阵 [nx, nx]
        dt: 时间步长
        nu: 扩散系数
    """

    u_squeezed = u.squeeze(-1)  # [batch, nx, nt]
    u_xx = torch.einsum('ij,bjt->bit', D2, u_squeezed)  # [batch, nx, nt]
    # 时间导数（中心差分）
    residual = u_t - nu * u_xx[:, :, init_t:]
    return residual

# 使用 D² 矩阵
