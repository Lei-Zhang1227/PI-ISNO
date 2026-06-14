import torch
from scipy.fft import dct, idct, dst, idst


def compute_burgers2d_residual_scipy(u, nu=0.1, Lx=2.0, Ly=2.0, T=1.0):
    """
    scipy 批量版本 - 作为参考
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    dt = T / (Nt - 1)

    # 转 numpy
    u_np = u.permute(0, 3, 1, 2).numpy()  # [batch, Nt, Nx, Ny]

    # 时间导数
    u_t = np.zeros_like(u_np)
    u_t[:, 2:-2] = (-u_np[:, 4:] + 8 * u_np[:, 3:-1] - 8 * u_np[:, 1:-3] + u_np[:, :-4]) / (12 * dt)
    u_t[:, 0] = (-3 * u_np[:, 0] + 4 * u_np[:, 1] - u_np[:, 2]) / (2 * dt)
    u_t[:, 1] = (u_np[:, 2] - u_np[:, 0]) / (2 * dt)
    u_t[:, -2] = (u_np[:, -1] - u_np[:, -3]) / (2 * dt)
    u_t[:, -1] = (3 * u_np[:, -1] - 4 * u_np[:, -2] + u_np[:, -3]) / (2 * dt)

    # x 方向导数 (沿 axis=2)
    kx = np.arange(Nx) * (np.pi / Lx)

    u_hat_x = dct(u_np, type=2, axis=2, norm='ortho')

    # 一阶导: 乘 -k，左移，IDST
    du_hat_x = -u_hat_x * kx[np.newaxis, np.newaxis, :, np.newaxis]
    du_hat_x_shifted = np.zeros_like(du_hat_x)
    du_hat_x_shifted[:, :, :-1, :] = du_hat_x[:, :, 1:, :]
    u_x = idst(du_hat_x_shifted, type=2, axis=2, norm='ortho')

    # 二阶导: 乘 -k^2，IDCT
    d2u_hat_x = -u_hat_x * (kx ** 2)[np.newaxis, np.newaxis, :, np.newaxis]
    u_xx = idct(d2u_hat_x, type=2, axis=2, norm='ortho')

    # y 方向导数 (沿 axis=3)
    ky = np.arange(Ny) * (np.pi / Ly)

    u_hat_y = dct(u_np, type=2, axis=3, norm='ortho')

    # 一阶导
    du_hat_y = -u_hat_y * ky[np.newaxis, np.newaxis, np.newaxis, :]
    du_hat_y_shifted = np.zeros_like(du_hat_y)
    du_hat_y_shifted[:, :, :, :-1] = du_hat_y[:, :, :, 1:]
    u_y = idst(du_hat_y_shifted, type=2, axis=3, norm='ortho')

    # 二阶导
    d2u_hat_y = -u_hat_y * (ky ** 2)[np.newaxis, np.newaxis, np.newaxis, :]
    u_yy = idct(d2u_hat_y, type=2, axis=3, norm='ortho')

    # Laplacian
    lap_u = u_xx + u_yy

    # 残差
    residual = u_t + u_np * u_x + u_np * u_y - nu * lap_u

    # 去掉时间边界
    residual = residual[:, 2:-2, :, :]

    # 转回 torch
    residual = torch.from_numpy(residual).float()
    residual = residual.permute(0, 2, 3, 1)  # [batch, Nx, Ny, Nt-4]

    return residual.unsqueeze(-1)


# ========== 批量版本函数 (保持不变) ==========
def dctII_batch(u, axis=-1):
    """批量 DCT-II"""
    u = torch.moveaxis(u, axis, -1)
    Nx = u.shape[-1]

    v = torch.cat([u[..., ::2], u[..., 1::2].flip(dims=[-1])], dim=-1)
    V = torch.fft.fft(v, dim=-1)

    k = torch.arange(Nx, dtype=u.dtype, device=u.device)
    W4 = torch.exp(-0.5j * torch.pi * k / Nx)

    result = 2 * (V * W4).real / Nx
    return torch.moveaxis(result, -1, axis)


def idctII_batch(a, axis=-1):
    """批量 IDCT-II"""
    a = torch.moveaxis(a, axis, -1)
    Nx = a.shape[-1]

    k = torch.arange(Nx, dtype=a.dtype, device=a.device)
    iW4 = 1 / torch.exp(-0.5j * torch.pi * k / Nx)
    iW4_scaled = iW4.clone()
    iW4_scaled[0] = iW4_scaled[0] / 2

    V = torch.fft.ifft(a.to(torch.complex64) * iW4_scaled, dim=-1).real

    u = torch.zeros_like(V, dtype=a.dtype, device=a.device)
    u[..., ::2] = V[..., :Nx - (Nx // 2)]
    u[..., 1::2] = V.flip(dims=[-1])[..., :Nx // 2]

    return torch.moveaxis(u * Nx, -1, axis)


def idstII_batch(x, axis=-1):
    """批量 IDST-II"""
    x = torch.moveaxis(x, axis, -1)

    v = idctII_batch(x.flip(dims=[-1]), axis=-1)
    u = v.clone()
    u[..., 1::2] = -u[..., 1::2]

    return torch.moveaxis(u, -1, axis)


def compute_burgers2d_residual_batch(u, nu=0.1, Lx=2.0, Ly=2.0, T=1.0):
    """批量版本 - PyTorch"""
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype

    dt = T / (Nt - 1)

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # 时间导数
    u_t = torch.zeros_like(u)
    u_t[:, 2:-2] = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)
    # u_t[:, 0] = (-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) / (2 * dt)
    # u_t[:, 1] = (u[:, 2] - u[:, 0]) / (2 * dt)
    # u_t[:, -2] = (u[:, -1] - u[:, -3]) / (2 * dt)
    # u_t[:, -1] = (3 * u[:, -1] - 4 * u[:, -2] + u[:, -3]) / (2 * dt)

    # x 方向 (axis=2)
    kx = torch.arange(Nx, device=device, dtype=dtype)
    Cx = torch.pi / Lx

    u_hat_x = dctII_batch(u, axis=2)

    # 一阶导
    du_hat_x = -u_hat_x * kx.view(1, 1, -1, 1)
    du_hat_x_shifted = torch.zeros_like(du_hat_x)
    du_hat_x_shifted[:, :, :-1, :] = du_hat_x[:, :, 1:, :]
    u_x = idstII_batch(du_hat_x_shifted * Cx, axis=2)

    # 二阶导
    K2x = -(kx * Cx) ** 2
    u_xx = idctII_batch(u_hat_x * K2x.view(1, 1, -1, 1), axis=2)

    # y 方向 (axis=3)
    ky = torch.arange(Ny, device=device, dtype=dtype)
    Cy = torch.pi / Ly

    u_hat_y = dctII_batch(u, axis=3)

    # 一阶导
    du_hat_y = -u_hat_y * ky.view(1, 1, 1, -1)
    du_hat_y_shifted = torch.zeros_like(du_hat_y)
    du_hat_y_shifted[:, :, :, :-1] = du_hat_y[:, :, :, 1:]
    u_y = idstII_batch(du_hat_y_shifted * Cy, axis=3)

    # 二阶导
    K2y = -(ky * Cy) ** 2
    u_yy = idctII_batch(u_hat_y * K2y.view(1, 1, 1, -1), axis=3)

    lap_u = u_xx + u_yy
    residual = u_t + u * u_x + u * u_y - nu * lap_u
    residual = residual[:, 2:-2, :, :]
    residual = residual.permute(0, 2, 3, 1)

    return residual.unsqueeze(-1)


# ========== 1. BC Loss (齐次 Neumann 边界) ==========
def compute_neumann_bc_loss(u, Lx=2.0, Ly=2.0):
    """
    计算齐次 Neumann 边界条件损失: ∂u/∂n = 0

    输入:
        u: [batch, Nx, Ny, Nt, 1]
        Lx, Ly: 空间区域大小

    返回:
        bc_loss: 标量
    """
    if u.dim() == 5:
        u = u.squeeze(-1)  # [batch, Nx, Ny, Nt]

    batch, Nx, Ny, Nt = u.shape

    dx = Lx / Nx
    dy = Ly / Ny

    # x 方向边界: ∂u/∂x = 0 at x = -1, 1
    # 用二阶单侧差分
    # 左边界 (x = -1): (-3*u[0] + 4*u[1] - u[2]) / (2*dx) = 0
    # 右边界 (x = 1):  (3*u[-1] - 4*u[-2] + u[-3]) / (2*dx) = 0
    du_dx_left = (-3 * u[:, 0, :, :] + 4 * u[:, 1, :, :] - u[:, 2, :, :]) / (2 * dx)
    du_dx_right = (3 * u[:, -1, :, :] - 4 * u[:, -2, :, :] + u[:, -3, :, :]) / (2 * dx)

    # y 方向边界: ∂u/∂y = 0 at y = -1, 1
    du_dy_bottom = (-3 * u[:, :, 0, :] + 4 * u[:, :, 1, :] - u[:, :, 2, :]) / (2 * dy)
    du_dy_top = (3 * u[:, :, -1, :] - 4 * u[:, :, -2, :] + u[:, :, -3, :]) / (2 * dy)

    # BC loss
    bc_loss = (torch.mean(du_dx_left ** 2) + torch.mean(du_dx_right ** 2) +
               torch.mean(du_dy_bottom ** 2) + torch.mean(du_dy_top ** 2)) / 4

    return bc_loss


def compute_neumann_bc_loss_spectral(u, Lx=2.0, Ly=2.0):
    """
    用谱方法计算 Neumann BC loss (更精确)

    输入:
        u: [batch, Nx, Ny, Nt, 1]

    返回:
        bc_loss: 标量
    """
    if u.dim() == 5:
        u = u.squeeze(-1)  # [batch, Nx, Ny, Nt]

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # x 方向导数
    kx = torch.arange(Nx, device=device, dtype=dtype)
    Cx = torch.pi / Lx

    u_hat_x = dctII_batch(u, axis=2)
    du_hat_x = -u_hat_x * kx.view(1, 1, -1, 1)
    du_hat_x_shifted = torch.zeros_like(du_hat_x)
    du_hat_x_shifted[:, :, :-1, :] = du_hat_x[:, :, 1:, :]
    u_x = idstII_batch(du_hat_x_shifted * Cx, axis=2)

    # y 方向导数
    ky = torch.arange(Ny, device=device, dtype=dtype)
    Cy = torch.pi / Ly

    u_hat_y = dctII_batch(u, axis=3)
    du_hat_y = -u_hat_y * ky.view(1, 1, 1, -1)
    du_hat_y_shifted = torch.zeros_like(du_hat_y)
    du_hat_y_shifted[:, :, :, :-1] = du_hat_y[:, :, :, 1:]
    u_y = idstII_batch(du_hat_y_shifted * Cy, axis=3)

    # 边界处的导数值
    # x 边界
    du_dx_left = u_x[:, :, 0, :]  # [batch, Nt, Ny]
    du_dx_right = u_x[:, :, -1, :]  # [batch, Nt, Ny]

    # y 边界
    du_dy_bottom = u_y[:, :, :, 0]  # [batch, Nt, Nx]
    du_dy_top = u_y[:, :, :, -1]  # [batch, Nt, Nx]

    bc_loss = (torch.mean(du_dx_left ** 2) + torch.mean(du_dx_right ** 2) +
               torch.mean(du_dy_bottom ** 2) + torch.mean(du_dy_top ** 2)) / 4

    return bc_loss


# ========== 2. 有限差分版本 Residual ==========
def compute_burgers2d_residual_fd(u, nu=0.1, Lx=2.0, Ly=2.0, T=1.0):
    """
    有限差分版本计算 2D Burgers 残差

    输入:
        u: [batch, Nx, Ny, Nt, 1]

    返回:
        residual: [batch, Nx, Ny, Nt-4, 1]
    """
    if u.dim() == 5:
        u = u.squeeze(-1)  # [batch, Nx, Ny, Nt]

    batch, Nx, Ny, Nt = u.shape
    device = u.device

    dx = Lx / Nx
    dy = Ly / Ny
    dt = T / (Nt - 1)

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # ========== 时间导数 (四阶中心差分) ==========
    u_t = torch.zeros_like(u)
    u_t[:, 2:-2] = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)
    u_t[:, 0] = (-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) / (2 * dt)
    u_t[:, 1] = (u[:, 2] - u[:, 0]) / (2 * dt)
    u_t[:, -2] = (u[:, -1] - u[:, -3]) / (2 * dt)
    u_t[:, -1] = (3 * u[:, -1] - 4 * u[:, -2] + u[:, -3]) / (2 * dt)

    # ========== 空间导数 (四阶中心差分 + 边界处理) ==========

    # x 方向一阶导 u_x
    u_x = torch.zeros_like(u)
    # 内部点: 四阶中心差分
    u_x[:, :, 2:-2, :] = (-u[:, :, 4:, :] + 8 * u[:, :, 3:-1, :] - 8 * u[:, :, 1:-3, :] + u[:, :, :-4, :]) / (12 * dx)
    # 边界点: 二阶差分 (考虑 Neumann BC)
    u_x[:, :, 0, :] = (-3 * u[:, :, 0, :] + 4 * u[:, :, 1, :] - u[:, :, 2, :]) / (2 * dx)
    u_x[:, :, 1, :] = (u[:, :, 2, :] - u[:, :, 0, :]) / (2 * dx)
    u_x[:, :, -2, :] = (u[:, :, -1, :] - u[:, :, -3, :]) / (2 * dx)
    u_x[:, :, -1, :] = (3 * u[:, :, -1, :] - 4 * u[:, :, -2, :] + u[:, :, -3, :]) / (2 * dx)

    # y 方向一阶导 u_y
    u_y = torch.zeros_like(u)
    # 内部点
    u_y[:, :, :, 2:-2] = (-u[:, :, :, 4:] + 8 * u[:, :, :, 3:-1] - 8 * u[:, :, :, 1:-3] + u[:, :, :, :-4]) / (12 * dy)
    # 边界点
    u_y[:, :, :, 0] = (-3 * u[:, :, :, 0] + 4 * u[:, :, :, 1] - u[:, :, :, 2]) / (2 * dy)
    u_y[:, :, :, 1] = (u[:, :, :, 2] - u[:, :, :, 0]) / (2 * dy)
    u_y[:, :, :, -2] = (u[:, :, :, -1] - u[:, :, :, -3]) / (2 * dy)
    u_y[:, :, :, -1] = (3 * u[:, :, :, -1] - 4 * u[:, :, :, -2] + u[:, :, :, -3]) / (2 * dy)

    # x 方向二阶导 u_xx
    u_xx = torch.zeros_like(u)
    # 内部点: 四阶中心差分
    u_xx[:, :, 2:-2, :] = (-u[:, :, 4:, :] + 16 * u[:, :, 3:-1, :] - 30 * u[:, :, 2:-2, :] + 16 * u[:, :, 1:-3, :] - u[
        :, :, :-4, :]) / (12 * dx ** 2)
    # 边界点: 二阶差分 (使用 Neumann BC 的镜像延拓)
    # u[-1] = u[1], u[N] = u[N-2] (ghost points)
    u_xx[:, :, 0, :] = (2 * u[:, :, 1, :] - 2 * u[:, :, 0, :]) / (dx ** 2)  # 用 Neumann: u[-1]=u[1]
    u_xx[:, :, 1, :] = (u[:, :, 2, :] - 2 * u[:, :, 1, :] + u[:, :, 0, :]) / (dx ** 2)
    u_xx[:, :, -2, :] = (u[:, :, -1, :] - 2 * u[:, :, -2, :] + u[:, :, -3, :]) / (dx ** 2)
    u_xx[:, :, -1, :] = (2 * u[:, :, -2, :] - 2 * u[:, :, -1, :]) / (dx ** 2)  # 用 Neumann: u[N]=u[N-2]

    # y 方向二阶导 u_yy
    u_yy = torch.zeros_like(u)
    # 内部点
    u_yy[:, :, :, 2:-2] = (-u[:, :, :, 4:] + 16 * u[:, :, :, 3:-1] - 30 * u[:, :, :, 2:-2] + 16 * u[:, :, :, 1:-3] - u[
        :, :, :, :-4]) / (12 * dy ** 2)
    # 边界点
    u_yy[:, :, :, 0] = (2 * u[:, :, :, 1] - 2 * u[:, :, :, 0]) / (dy ** 2)
    u_yy[:, :, :, 1] = (u[:, :, :, 2] - 2 * u[:, :, :, 1] + u[:, :, :, 0]) / (dy ** 2)
    u_yy[:, :, :, -2] = (u[:, :, :, -1] - 2 * u[:, :, :, -2] + u[:, :, :, -3]) / (dy ** 2)
    u_yy[:, :, :, -1] = (2 * u[:, :, :, -2] - 2 * u[:, :, :, -1]) / (dy ** 2)

    # Laplacian
    lap_u = u_xx + u_yy

    # 残差
    residual = u_t + u * u_x + u * u_y - nu * lap_u

    # 去掉时间边界
    residual = residual[:, 2:-2, :, :]
    residual = residual.permute(0, 2, 3, 1)  # [batch, Nx, Ny, Nt-4]

    return residual.unsqueeze(-1)


def compute_neumann_bc_loss_halfgrid(u, Lx=2.0, Ly=2.0):
    """
    半网格上计算 Neumann BC Loss

    半网格点: x[j] = -1 + (j + 0.5) * dx, 不包含 x = -1, 1
    需要外推边界导数值

    输入:
        u: [batch, Nx, Ny, Nt, 1]

    返回:
        bc_loss: 标量
    """
    if u.dim() == 5:
        u = u.squeeze(-1)  # [batch, Nx, Ny, Nt]

    batch, Nx, Ny, Nt = u.shape

    dx = Lx / Nx
    dy = Ly / Ny

    # 半网格点位置:
    # x[0] = -1 + dx/2, x[1] = -1 + 3*dx/2, ...
    # 边界 x = -1 在 x[0] 左边 dx/2 处

    # x 方向边界导数 ∂u/∂x at x = -1, 1
    # 用二阶外推: 在边界处的导数 ≈ 用最近几个点拟合
    #
    # 对于 Neumann BC: ∂u/∂x = 0 at boundary
    # 等价于: u 在边界附近关于边界对称 (偶延拓)
    #
    # 近似方法: 用 (u[1] - u[0]) / dx 作为边界附近的导数
    # 如果满足 Neumann BC，这个值应该接近 0

    # 左边界 x = -1: 用 u[0], u[1] 估计
    # ∂u/∂x ≈ (u[1] - u[0]) / dx (应该 → 0)
    du_dx_left = (u[:, 1, :, :] - u[:, 0, :, :]) / dx

    # 右边界 x = 1: 用 u[-1], u[-2] 估计
    du_dx_right = (u[:, -1, :, :] - u[:, -2, :, :]) / dx

    # y 方向类似
    du_dy_bottom = (u[:, :, 1, :] - u[:, :, 0, :]) / dy
    du_dy_top = (u[:, :, -1, :] - u[:, :, -2, :]) / dy

    bc_loss = (torch.mean(du_dx_left ** 2) + torch.mean(du_dx_right ** 2) +
               torch.mean(du_dy_bottom ** 2) + torch.mean(du_dy_top ** 2)) / 4

    return bc_loss


def compute_neumann_bc_loss_spectral_halfgrid(u, Lx=2.0, Ly=2.0):
    """
    半网格 + 谱方法计算 Neumann BC Loss

    DCT-II 本身假设 Neumann BC，理论上边界导数 = 0
    这里计算实际边界导数值作为检验

    输入:
        u: [batch, Nx, Ny, Nt, 1]

    返回:
        bc_loss: 标量
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # 用谱方法计算导数
    kx = torch.arange(Nx, device=device, dtype=dtype)
    ky = torch.arange(Ny, device=device, dtype=dtype)
    Cx = torch.pi / Lx
    Cy = torch.pi / Ly

    # x 方向导数
    u_hat_x = dctII_batch(u, axis=2)
    du_hat_x = -u_hat_x * kx.view(1, 1, -1, 1)
    du_hat_x_shifted = torch.zeros_like(du_hat_x)
    du_hat_x_shifted[:, :, :-1, :] = du_hat_x[:, :, 1:, :]
    u_x = idstII_batch(du_hat_x_shifted * Cx, axis=2)

    # y 方向导数
    u_hat_y = dctII_batch(u, axis=3)
    du_hat_y = -u_hat_y * ky.view(1, 1, 1, -1)
    du_hat_y_shifted = torch.zeros_like(du_hat_y)
    du_hat_y_shifted[:, :, :, :-1] = du_hat_y[:, :, :, 1:]
    u_y = idstII_batch(du_hat_y_shifted * Cy, axis=3)

    # 外推到边界
    # 半网格最近点到边界距离 = dx/2
    # 线性外推: u_x(boundary) ≈ u_x[0] - (u_x[1] - u_x[0]) / 2
    #                        = 1.5 * u_x[0] - 0.5 * u_x[1]

    # x 边界
    du_dx_left = 1.5 * u_x[:, :, 0, :] - 0.5 * u_x[:, :, 1, :]
    du_dx_right = 1.5 * u_x[:, :, -1, :] - 0.5 * u_x[:, :, -2, :]

    # y 边界
    du_dy_bottom = 1.5 * u_y[:, :, :, 0] - 0.5 * u_y[:, :, :, 1]
    du_dy_top = 1.5 * u_y[:, :, :, -1] - 0.5 * u_y[:, :, :, -2]

    bc_loss = (torch.mean(du_dx_left ** 2) + torch.mean(du_dx_right ** 2) +
               torch.mean(du_dy_bottom ** 2) + torch.mean(du_dy_top ** 2)) / 4

    return bc_loss


def compute_neumann_bc_loss_dst(u, Lx=2.0, Ly=2.0):
    """
    用 DST 系数检验 Neumann BC

    理论: 如果 u 满足 Neumann BC，则 du/dx 在边界 = 0
    DCT(u) -> DST(du/dx)
    DST 系数的特定模式反映边界条件

    输入:
        u: [batch, Nx, Ny, Nt, 1]

    返回:
        bc_loss: 标量
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # 计算 u_x 的 DST 系数
    kx = torch.arange(Nx, device=device, dtype=dtype)
    Cx = torch.pi / Lx

    u_hat_x = dctII_batch(u, axis=2)
    du_hat_x = -u_hat_x * kx.view(1, 1, -1, 1) * Cx

    # 左移后就是 DST 系数
    dst_coef_x = torch.zeros_like(du_hat_x)
    dst_coef_x[:, :, :-1, :] = du_hat_x[:, :, 1:, :]

    # y 方向
    ky = torch.arange(Ny, device=device, dtype=dtype)
    Cy = torch.pi / Ly

    u_hat_y = dctII_batch(u, axis=3)
    du_hat_y = -u_hat_y * ky.view(1, 1, 1, -1) * Cy

    dst_coef_y = torch.zeros_like(du_hat_y)
    dst_coef_y[:, :, :, :-1] = du_hat_y[:, :, :, 1:]

    # Neumann BC 要求导数在边界 = 0
    # 等价于 DST 系数的加权和 = 0
    # 简化: 取高频部分的能量作为 BC 违反度

    # 高频系数 (后 1/4)
    high_freq_x = dst_coef_x[:, :, 3 * Nx // 4:, :]
    high_freq_y = dst_coef_y[:, :, :, 3 * Ny // 4:]

    bc_loss = torch.mean(high_freq_x ** 2) + torch.mean(high_freq_y ** 2)

    return bc_loss


def analyze_bc_methods(u, Lx=2.0, Ly=2.0):
    """
    分析三种 BC Loss 方法的物理含义
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype

    dx = Lx / Nx
    dy = Ly / Ny

    print("=" * 60)
    print("半网格 Neumann BC 分析")
    print("=" * 60)
    print(f"网格: Nx={Nx}, Ny={Ny}")
    print(f"dx={dx:.4f}, dy={dy:.4f}")
    print(f"第一个点: x[0] = {-1 + dx / 2:.4f} (距边界 {dx / 2:.4f})")
    print(f"最后一个点: x[-1] = {1 - dx / 2:.4f} (距边界 {dx / 2:.4f})")

    # ========== 方法1: 有限差分 ==========
    # 测量的是: 最近两个网格点之间的斜率
    du_dx_left_fd = (u[:, 1, :, :] - u[:, 0, :, :]) / dx
    du_dx_right_fd = (u[:, -1, :, :] - u[:, -2, :, :]) / dx

    print(f"\n方法1 (有限差分):")
    print(f"  测量: 边界附近两点斜率")
    print(f"  左边界 |du/dx| max: {torch.max(torch.abs(du_dx_left_fd)).item():.4e}")
    print(f"  右边界 |du/dx| max: {torch.max(torch.abs(du_dx_right_fd)).item():.4e}")

    # ========== 方法2: 谱方法 + 外推 ==========
    u_perm = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    kx = torch.arange(Nx, device=device, dtype=dtype)
    Cx = torch.pi / Lx

    u_hat_x = dctII_batch(u_perm, axis=2)
    du_hat_x = -u_hat_x * kx.view(1, 1, -1, 1)
    du_hat_x_shifted = torch.zeros_like(du_hat_x)
    du_hat_x_shifted[:, :, :-1, :] = du_hat_x[:, :, 1:, :]
    u_x = idstII_batch(du_hat_x_shifted * Cx, axis=2)

    # 外推到边界
    du_dx_left_spec = 1.5 * u_x[:, :, 0, :] - 0.5 * u_x[:, :, 1, :]
    du_dx_right_spec = 1.5 * u_x[:, :, -1, :] - 0.5 * u_x[:, :, -2, :]

    # 不外推，直接取最近点
    du_dx_left_direct = u_x[:, :, 0, :]
    du_dx_right_direct = u_x[:, :, -1, :]

    print(f"\n方法2 (谱方法):")
    print(f"  网格点处导数:")
    print(f"    左边界最近点 |du/dx| max: {torch.max(torch.abs(du_dx_left_direct)).item():.4e}")
    print(f"    右边界最近点 |du/dx| max: {torch.max(torch.abs(du_dx_right_direct)).item():.4e}")
    print(f"  线性外推到边界:")
    print(f"    左边界 |du/dx| max: {torch.max(torch.abs(du_dx_left_spec)).item():.4e}")
    print(f"    右边界 |du/dx| max: {torch.max(torch.abs(du_dx_right_spec)).item():.4e}")

    # ========== 方法3: DST 系数 ==========
    dst_coef_x = du_hat_x_shifted * Cx

    print(f"\n方法3 (DST 系数):")
    print(f"  DST 系数能量分布:")
    print(f"    低频 (0-25%):  {torch.mean(dst_coef_x[:, :, :Nx // 4, :] ** 2).item():.4e}")
    print(f"    中频 (25-75%): {torch.mean(dst_coef_x[:, :, Nx // 4:3 * Nx // 4, :] ** 2).item():.4e}")
    print(f"    高频 (75-100%): {torch.mean(dst_coef_x[:, :, 3 * Nx // 4:, :] ** 2).item():.4e}")


# ========== BC Loss ==========
def compute_neumann_bc_loss_recommended(u, Lx=2.0, Ly=2.0):
    """
    推荐的半网格 Neumann BC Loss

    直接用谱方法计算的导数在最近网格点处的值
    (不外推，因为外推会放大误差)

    输入:
        u: [batch, Nx, Ny, Nt, 1]

    返回:
        bc_loss: 标量
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # x 方向导数
    kx = torch.arange(Nx, device=device, dtype=dtype)
    Cx = torch.pi / Lx

    u_hat_x = dctII_batch(u, axis=2)
    du_hat_x = -u_hat_x * kx.view(1, 1, -1, 1)
    du_hat_x_shifted = torch.zeros_like(du_hat_x)
    du_hat_x_shifted[:, :, :-1, :] = du_hat_x[:, :, 1:, :]
    u_x = idstII_batch(du_hat_x_shifted * Cx, axis=2)

    # y 方向导数
    ky = torch.arange(Ny, device=device, dtype=dtype)
    Cy = torch.pi / Ly

    u_hat_y = dctII_batch(u, axis=3)
    du_hat_y = -u_hat_y * ky.view(1, 1, 1, -1)
    du_hat_y_shifted = torch.zeros_like(du_hat_y)
    du_hat_y_shifted[:, :, :, :-1] = du_hat_y[:, :, :, 1:]
    u_y = idstII_batch(du_hat_y_shifted * Cy, axis=3)

    # 最近网格点处的导数值 (不外推)
    # 如果满足 Neumann BC，这些值应该很小
    bc_x = torch.mean(u_x[:, :, 0, :] ** 2) + torch.mean(u_x[:, :, -1, :] ** 2)
    bc_y = torch.mean(u_y[:, :, :, 0] ** 2) + torch.mean(u_y[:, :, :, -1] ** 2)

    return (bc_x + bc_y) / 4


# ========== 测试 ==========
if __name__ == "__main__":
    import h5py
    import time

    filepath = r"../data/burgers2d_spectral.h5"

    with h5py.File(filepath, 'r') as f:
        U = f['0850']['data'][:]

    stride = 2
    U_down = U[::2, ::stride, ::stride, :]

    Nt, Nx, Ny, _ = U_down.shape
    U_torch = torch.from_numpy(U_down).float()
    U_torch = U_torch.permute(1, 2, 0, 3).unsqueeze(0)  # [1, Nx, Ny, Nt, 1]

    print(f"Input shape: {U_torch.shape}")

    # ========== BC Loss 测试 ==========
    print("\n" + "=" * 60)
    print("BC Loss 测试")
    print("=" * 60)

    bc_loss_fd = compute_neumann_bc_loss(U_torch)
    bc_loss_spectral = compute_neumann_bc_loss_spectral(U_torch)

    print(f"BC Loss (有限差分):  {bc_loss_fd.item():.4e}")
    print(f"BC Loss (谱方法):    {bc_loss_spectral.item():.4e}")

    # ========== Residual 对比 ==========
    print("\n" + "=" * 60)
    print("Residual 对比")
    print("=" * 60)

    res_spectral = compute_burgers2d_residual_batch(U_torch, nu=0.1, T=1.0)
    res_fd = compute_burgers2d_residual_fd(U_torch, nu=0.1, T=1.0)

    print(f"谱方法 MSE:    {torch.mean(res_spectral ** 2).item():.4e}")
    print(f"有限差分 MSE:  {torch.mean(res_fd ** 2).item():.4e}")
    print(f"差异:          {torch.max(torch.abs(res_spectral - res_fd)).item():.4e}")

    # ========== 速度测试 ==========
    print("\n" + "=" * 60)
    print("速度测试")
    print("=" * 60)

    n_runs = 20

    t0 = time.time()
    for _ in range(n_runs):
        _ = compute_burgers2d_residual_batch(U_torch, nu=0.1, T=1.0)
    t1 = time.time()
    print(f"谱方法 CPU:    {(t1 - t0) / n_runs * 1000:.2f} ms")

    t0 = time.time()
    for _ in range(n_runs):
        _ = compute_burgers2d_residual_fd(U_torch, nu=0.1, T=1.0)
    t1 = time.time()
    print(f"有限差分 CPU:  {(t1 - t0) / n_runs * 1000:.2f} ms")

    if torch.cuda.is_available():
        U_cuda = U_torch.cuda()

        # 预热
        for _ in range(5):
            _ = compute_burgers2d_residual_batch(U_cuda, nu=0.1, T=1.0)
            _ = compute_burgers2d_residual_fd(U_cuda, nu=0.1, T=1.0)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_runs):
            _ = compute_burgers2d_residual_batch(U_cuda, nu=0.1, T=1.0)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"谱方法 GPU:    {(t1 - t0) / n_runs * 1000:.2f} ms")

        t0 = time.time()
        for _ in range(n_runs):
            _ = compute_burgers2d_residual_fd(U_cuda, nu=0.1, T=1.0)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"有限差分 GPU:  {(t1 - t0) / n_runs * 1000:.2f} ms")

    # ========== 自动求导测试 ==========
    print("\n" + "=" * 60)
    print("自动求导测试")
    print("=" * 60)

    U_test = U_torch.clone().requires_grad_(True)

    res = compute_burgers2d_residual_fd(U_test, nu=0.1, T=1.0)
    loss = torch.mean(res ** 2)
    loss.backward()
    print(f"有限差分 Gradient: {U_test.grad is not None}")

    U_test2 = U_torch.clone().requires_grad_(True)
    bc_loss = compute_neumann_bc_loss(U_test2)
    bc_loss.backward()
    print(f"BC Loss Gradient:  {U_test2.grad is not None}")

    print("\n" + "=" * 60)
    print("半网格 Neumann BC Loss 对比")
    print("=" * 60)

    bc1 = compute_neumann_bc_loss_halfgrid(U_torch)
    bc2 = compute_neumann_bc_loss_spectral_halfgrid(U_torch)
    bc3 = compute_neumann_bc_loss_dst(U_torch)

    print(f"方法1 (有限差分外推):     {bc1.item():.4e}")
    print(f"方法2 (谱方法+线性外推):  {bc2.item():.4e}")
    print(f"方法3 (DST高频系数):      {bc3.item():.4e}")

    # 对比
    print("\n" + "=" * 60)
    print("BC Loss 对比")
    print("=" * 60)

    bc1 = compute_neumann_bc_loss_halfgrid(U_torch)
    bc2 = compute_neumann_bc_loss_spectral_halfgrid(U_torch)
    bc3 = compute_neumann_bc_loss_dst(U_torch)
    bc4 = compute_neumann_bc_loss_recommended(U_torch)

    print(f"方法1 (有限差分):      {bc1.item():.4e}")
    print(f"方法2 (谱+外推):       {bc2.item():.4e}")
    print(f"方法3 (DST高频):       {bc3.item():.4e}")
    print(f"方法4 (谱+最近点):     {bc4.item():.4e}  <-- 推荐")
