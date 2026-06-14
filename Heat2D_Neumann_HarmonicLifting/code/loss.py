"""
Loss functions for 2D Heat Equation with Non-homogeneous Neumann BC
Using DCT-I on grid INCLUDING endpoints (N+1 points, e.g. 129)

PDE:  ∂u/∂t = κ Δu
BC:   ∂u/∂x(±1) = ∓1,  ∂u/∂y(±1) = ±1
Lifting: u_b = -x²/2 + y²/2  (Δu_b = 0)

DCT-I basis: cos(kπj/N), j=0,...,N (includes endpoints)
  - Derivative at j=0 and j=N is exactly zero → homogeneous Neumann by construction
  - Eigenvalue for d²/dx²: -(kπ/L)²
"""

import torch
from scipy.fft import dct, idct
import numpy as np
import torch.nn.functional as F


# ================================================================
# PyTorch DCT-I / IDCT-I batch implementation via FFT
# ================================================================

def dctI_batch(u, axis=-1):
    """Batch DCT-I via FFT.

    For a sequence x[0], ..., x[N] (N+1 points):
      X[k] = x[0] + (-1)^k x[N] + 2 Σ_{n=1}^{N-1} x[n] cos(πkn/N)

    Implementation: symmetric extension to length 2N, then FFT.
    """
    u = torch.moveaxis(u, axis, -1)
    N1 = u.shape[-1]  # N+1 points
    N = N1 - 1

    # Symmetric extension: [x[0], x[1], ..., x[N], x[N-1], ..., x[1]]
    y = torch.cat([u, u[..., 1:N].flip(dims=[-1])], dim=-1)
    Y = torch.fft.rfft(y, dim=-1)
    # rfft returns N+1 complex values for input of length 2N
    result = Y.real

    return torch.moveaxis(result, -1, axis)


def idctI_batch(c, axis=-1):
    """Batch IDCT-I (inverse of dctI_batch).

    IDCT-I = DCT-I / (2N), since DCT-I is self-adjoint up to scaling.
    """
    c = torch.moveaxis(c, axis, -1)
    N1 = c.shape[-1]
    N = N1 - 1

    y = torch.cat([c, c[..., 1:N].flip(dims=[-1])], dim=-1)
    Y = torch.fft.rfft(y, dim=-1)
    result = Y.real / (2 * N)

    return torch.moveaxis(result, -1, axis)


def verify_dctI_pytorch():
    """Quick verification of PyTorch DCT-I implementation."""
    torch.manual_seed(42)
    x = torch.randn(3, 129, 129)

    # Round-trip
    c = dctI_batch(x, axis=-1)
    x_back = idctI_batch(c, axis=-1)
    err = torch.max(torch.abs(x - x_back)).item()
    print(f"DCT-I round-trip error (axis=-1): {err:.2e}")

    c2 = dctI_batch(x, axis=-2)
    x_back2 = idctI_batch(c2, axis=-2)
    err2 = torch.max(torch.abs(x - x_back2)).item()
    print(f"DCT-I round-trip error (axis=-2): {err2:.2e}")

    # 2D round-trip
    c2d = dctI_batch(dctI_batch(x, axis=-1), axis=-2)
    x_back2d = idctI_batch(idctI_batch(c2d, axis=-2), axis=-1)
    err2d = torch.max(torch.abs(x - x_back2d)).item()
    print(f"DCT-I 2D round-trip error: {err2d:.2e}")

    # Compare with scipy
    x_np = x[0].numpy()
    from scipy.fft import dctn, idctn
    c_scipy = dctn(x_np, type=1)
    c_torch = dctI_batch(dctI_batch(x[0:1], axis=-1), axis=-2)[0].numpy()
    err_scipy = np.max(np.abs(c_scipy - c_torch))
    print(f"PyTorch vs scipy DCT-I error: {err_scipy:.2e}")

    return err2d < 1e-6


# ================================================================
# PDE Residual: ∂u/∂t - κ Δu = 0  (DCT-I version)
# ================================================================

def compute_heat2d_residual_scipy(u, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
    """
    scipy 参考版本: 2D Heat 残差 (DCT-I)

    输入: u: [batch, Nx, Ny, Nt, 1]
    输出: residual: [batch, Nx, Ny, Nt-4, 1]
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    dt = T / (Nt - 1)

    u_np = u.permute(0, 3, 1, 2).numpy()  # [batch, Nt, Nx, Ny]

    # 时间导数 (四阶中心差分)
    u_t = np.zeros_like(u_np)
    u_t[:, 2:-2] = (-u_np[:, 4:] + 8 * u_np[:, 3:-1] - 8 * u_np[:, 1:-3] + u_np[:, :-4]) / (12 * dt)
    u_t[:, 0] = (-3 * u_np[:, 0] + 4 * u_np[:, 1] - u_np[:, 2]) / (2 * dt)
    u_t[:, 1] = (u_np[:, 2] - u_np[:, 0]) / (2 * dt)
    u_t[:, -2] = (u_np[:, -1] - u_np[:, -3]) / (2 * dt)
    u_t[:, -1] = (3 * u_np[:, -1] - 4 * u_np[:, -2] + u_np[:, -3]) / (2 * dt)

    # x 方向二阶导 (DCT-I)
    # Grid: Nx points on [-1,1], N = Nx-1 intervals
    # Eigenvalue: -(kπ/Lx)²
    from scipy.fft import dctn, idctn
    kx = np.arange(Nx) * (np.pi / Lx)

    # DCT-I along axis=2, then multiply eigenvalue, then IDCT-I
    u_hat_x = dctn(u_np, type=1, axes=[2])  # DCT-I along x only
    # Wait, dctn applies to all axes. Use dct for single axis.
    u_hat_x = dct(u_np, type=1, axis=2)
    d2u_hat_x = u_hat_x * (-(kx ** 2))[np.newaxis, np.newaxis, :, np.newaxis]
    u_xx = idct(d2u_hat_x, type=1, axis=2)

    # y 方向二阶导 (DCT-I)
    ky = np.arange(Ny) * (np.pi / Ly)
    u_hat_y = dct(u_np, type=1, axis=3)
    d2u_hat_y = u_hat_y * (-(ky ** 2))[np.newaxis, np.newaxis, np.newaxis, :]
    u_yy = idct(d2u_hat_y, type=1, axis=3)

    lap_u = u_xx + u_yy

    # 残差: ∂u/∂t - κ Δu
    residual = u_t - kappa * lap_u
    residual = residual[:, 2:-2, :, :]

    residual = torch.from_numpy(residual).float()
    residual = residual.permute(0, 2, 3, 1)  # [batch, Nx, Ny, Nt-4]
    return residual.unsqueeze(-1)


def compute_heat2d_residual_batch(u, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype
    dt = T / (Nt - 1)

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # lifting
    x = torch.linspace(-Lx / 2, Lx / 2, Nx, device=device, dtype=dtype)
    y = torch.linspace(-Ly / 2, Ly / 2, Ny, device=device, dtype=dtype)
    u_b = -0.5 * x.view(-1, 1) ** 2 + 0.5 * y.view(1, -1) ** 2
    u_h = u - u_b  # 广播自动扩展

    # 时间导数
    u_t = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)

    # ===== 核心优化：一次 2D DCT，一次 2D IDCT =====
    # 2D DCT-I: 先 x 后 y
    u_h_hat = dctI_batch(dctI_batch(u_h[:, 2:-2], axis=2), axis=3)

    # 拉普拉斯特征值 K2x + K2y（2D 合并）
    kx = torch.arange(Nx, device=device, dtype=dtype)
    ky = torch.arange(Ny, device=device, dtype=dtype)
    K2 = -(kx * torch.pi / Lx).view(1, 1, -1, 1) ** 2 \
         - (ky * torch.pi / Ly).view(1, 1, 1, -1) ** 2

    # 一次乘法 + 2D IDCT-I
    lap_u = idctI_batch(idctI_batch(u_h_hat * K2, axis=3), axis=2)

    residual = u_t - kappa * lap_u
    residual = residual.permute(0, 2, 3, 1)
    return residual.unsqueeze(-1)


# def compute_heat2d_residual_batch(u, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
#     if u.dim() == 5:
#         u = u.squeeze(-1)
#
#     batch, Nx, Ny, Nt = u.shape
#     device = u.device
#     dtype = u.dtype
#     dt = T / (Nt - 1)
#
#     u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]
#
#     # 构造 lifting u_b = -x²/2 + y²/2
#     x = torch.linspace(-Lx / 2, Lx / 2, Nx, device=device, dtype=dtype)
#     y = torch.linspace(-Ly / 2, Ly / 2, Ny, device=device, dtype=dtype)
#     X, Y = torch.meshgrid(x, y, indexing='ij')
#     u_b = -0.5 * X ** 2 + 0.5 * Y ** 2  # (Nx, Ny)
#
#     # u_h = u - u_b (满足齐次Neumann，可以安全地用DCT-I)
#     u_h = u - u_b.unsqueeze(0).unsqueeze(0)  # (batch, Nt, Nx, Ny)
#
#     # 时间导数 (对完整u，等价于对u_h因为∂u_b/∂t=0)
#     u_t = torch.zeros_like(u)
#     u_t[:, 2:-2] = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)
#
#     # Laplacian of u_h via DCT-I (u_h满足齐次Neumann)
#     kx = torch.arange(Nx, device=device, dtype=dtype)
#     K2x = -(kx * torch.pi / Lx) ** 2
#
#     ky = torch.arange(Ny, device=device, dtype=dtype)
#     K2y = -(ky * torch.pi / Ly) ** 2
#
#     u_h_hat_x = dctI_batch(u_h, axis=2)
#     u_h_xx = idctI_batch(u_h_hat_x * K2x.view(1, 1, -1, 1), axis=2)
#
#     u_h_hat_y = dctI_batch(u_h, axis=3)
#     u_h_yy = idctI_batch(u_h_hat_y * K2y.view(1, 1, 1, -1), axis=3)
#
#     # Δu = Δu_h + Δu_b = Δu_h + 0
#     lap_u = u_h_xx + u_h_yy
#
#     residual = u_t - kappa * lap_u
#     residual = residual[:, 2:-2, :, :]
#     residual = residual.permute(0, 2, 3, 1)
#     return residual.unsqueeze(-1)


# def compute_heat2d_residual_fd(u, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
#     """
#     有限差分版本: 2D Heat 残差
#
#     输入: u: [batch, Nx, Ny, Nt, 1]
#     输出: residual: [batch, Nx, Ny, Nt-4, 1]
#     """
#     if u.dim() == 5:
#         u = u.squeeze(-1)
#
#     batch, Nx, Ny, Nt = u.shape
#     device = u.device
#     dx = Lx / (Nx - 1)
#     dy = Ly / (Ny - 1)
#     dt = T / (Nt - 1)
#
#     u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]
#
#     # 时间导数
#     u_t = torch.zeros_like(u)
#     u_t[:, 2:-2] = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)
#     u_t[:, 0] = (-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) / (2 * dt)
#     u_t[:, 1] = (u[:, 2] - u[:, 0]) / (2 * dt)
#     u_t[:, -2] = (u[:, -1] - u[:, -3]) / (2 * dt)
#     u_t[:, -1] = (3 * u[:, -1] - 4 * u[:, -2] + u[:, -3]) / (2 * dt)
#
#     # x 方向二阶导
#     u_xx = torch.zeros_like(u)
#     u_xx[:, :, 2:-2, :] = (-u[:, :, 4:, :] + 16 * u[:, :, 3:-1, :] - 30 * u[:, :, 2:-2, :] +
#                            16 * u[:, :, 1:-3, :] - u[:, :, :-4, :]) / (12 * dx ** 2)
#     # 边界: Neumann镜像
#     u_xx[:, :, 0, :] = (2 * u[:, :, 1, :] - 2 * u[:, :, 0, :] - 2 * dx) / (dx ** 2)
#     u_xx[:, :, -1, :] = (2 * u[:, :, -2, :] - 2 * u[:, :, -1, :] - 2 * dx) / (dx ** 2)
#     # u_xx[:, :, 0, :] = (2 * u[:, :, 1, :] - 2 * u[:, :, 0, :]) / (dx ** 2)
#     u_xx[:, :, 1, :] = (u[:, :, 2, :] - 2 * u[:, :, 1, :] + u[:, :, 0, :]) / (dx ** 2)
#     u_xx[:, :, -2, :] = (u[:, :, -1, :] - 2 * u[:, :, -2, :] + u[:, :, -3, :]) / (dx ** 2)
#     # u_xx[:, :, -1, :] = (2 * u[:, :, -2, :] - 2 * u[:, :, -1, :]) / (dx ** 2)
#
#
#     # y 方向二阶导
#     u_yy = torch.zeros_like(u)
#     u_yy[:, :, :, 2:-2] = (-u[:, :, :, 4:] + 16 * u[:, :, :, 3:-1] - 30 * u[:, :, :, 2:-2] +
#                            16 * u[:, :, :, 1:-3] - u[:, :, :, :-4]) / (12 * dy ** 2)
#     # u_yy[:, :, :, 0] = (2 * u[:, :, :, 1] - 2 * u[:, :, :, 0]) / (dy ** 2)
#     u_yy[:, :, :, 1] = (u[:, :, :, 2] - 2 * u[:, :, :, 1] + u[:, :, :, 0]) / (dy ** 2)
#     u_yy[:, :, :, -2] = (u[:, :, :, -1] - 2 * u[:, :, :, -2] + u[:, :, :, -3]) / (dy ** 2)
#     # u_yy[:, :, :, -1] = (2 * u[:, :, :, -2] - 2 * u[:, :, :, -1]) / (dy ** 2)
#     u_yy[:, :, :, 0] = (2 * u[:, :, :, 1] - 2 * u[:, :, :, 0] + 2 * dy) / (dy ** 2)
#     u_yy[:, :, :, -1] = (2 * u[:, :, :, -2] - 2 * u[:, :, :, -1] + 2 * dy) / (dy ** 2)
#
#     lap_u = u_xx + u_yy
#     residual = u_t - kappa * lap_u
#     residual = residual[:, 2:-2, :, :]
#     residual = residual.permute(0, 2, 3, 1)
#
#     return residual.unsqueeze(-1)
def compute_heat2d_residual_fd(u, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = T / (Nt - 1)

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # ============ 时间导数: conv1d ============
    kt = torch.tensor([1, -8, 0, 8, -1], device=device, dtype=dtype) / (12 * dt)
    u_t = F.conv1d(
        u.permute(0, 2, 3, 1).reshape(-1, 1, Nt),
        kt.view(1, 1, -1)
    ).reshape(batch, Nx, Ny, Nt - 4).permute(0, 3, 1, 2)

    u_mid = u[:, 2:-2, :, :]  # [batch, Nt-4, Nx, Ny]

    # ============ x方向: 内部conv2d + 边界切片 ============
    kx = torch.tensor([[-1, 16, -30, 16, -1]], device=device, dtype=dtype).view(1, 1, 5, 1) / (12 * dx ** 2)
    u_flat = u_mid.reshape(-1, 1, Nx, Ny)  # [batch*(Nt-4), 1, Nx, Ny]
    u_xx_inner = F.conv2d(u_flat, kx).reshape(batch, -1, Nx - 4, Ny)  # 内部 [2:-2]

    u_xx = torch.zeros_like(u_mid)
    u_xx[:, :, 2:-2, :] = u_xx_inner
    u_xx[:, :, 0, :] = (2 * u_mid[:, :, 1, :] - 2 * u_mid[:, :, 0, :] - 2 * dx) / (dx ** 2)
    u_xx[:, :, 1, :] = (u_mid[:, :, 2, :] - 2 * u_mid[:, :, 1, :] + u_mid[:, :, 0, :]) / (dx ** 2)
    u_xx[:, :, -2, :] = (u_mid[:, :, -1, :] - 2 * u_mid[:, :, -2, :] + u_mid[:, :, -3, :]) / (dx ** 2)
    u_xx[:, :, -1, :] = (2 * u_mid[:, :, -2, :] - 2 * u_mid[:, :, -1, :] - 2 * dx) / (dx ** 2)

    # ============ y方向: 内部conv2d + 边界切片 ============
    ky = torch.tensor([[-1, 16, -30, 16, -1]], device=device, dtype=dtype).view(1, 1, 1, 5) / (12 * dy ** 2)
    u_yy_inner = F.conv2d(u_flat, ky).reshape(batch, -1, Nx, Ny - 4)

    u_yy = torch.zeros_like(u_mid)
    u_yy[:, :, :, 2:-2] = u_yy_inner
    u_yy[:, :, :, 0] = (2 * u_mid[:, :, :, 1] - 2 * u_mid[:, :, :, 0] + 2 * dy) / (dy ** 2)
    u_yy[:, :, :, 1] = (u_mid[:, :, :, 2] - 2 * u_mid[:, :, :, 1] + u_mid[:, :, :, 0]) / (dy ** 2)
    u_yy[:, :, :, -2] = (u_mid[:, :, :, -1] - 2 * u_mid[:, :, :, -2] + u_mid[:, :, :, -3]) / (dy ** 2)
    u_yy[:, :, :, -1] = (2 * u_mid[:, :, :, -2] - 2 * u_mid[:, :, :, -1] + 2 * dy) / (dy ** 2)

    residual = u_t - kappa * (u_xx + u_yy)
    residual = residual.permute(0, 2, 3, 1)
    return residual.unsqueeze(-1)


def compute_neumann_bc_loss(u, Lx=2.0, Ly=2.0):
    """
    非齐次 Neumann BC Loss (有限差分, 端点网格)

    Target:
      ∂u/∂x(-1,y) = 1,   ∂u/∂x(1,y) = -1
      ∂u/∂y(x,-1) = -1,  ∂u/∂y(x,1) = 1

    输入: u: [batch, Nx, Ny, Nt, 1]
    输出: bc_loss: 标量
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # x 方向 (二阶单侧FD)
    du_dx_left = (-3 * u[:, 0, :, :] + 4 * u[:, 1, :, :] - u[:, 2, :, :]) / (2 * dx)
    du_dx_right = (3 * u[:, -1, :, :] - 4 * u[:, -2, :, :] + u[:, -3, :, :]) / (2 * dx)

    # y 方向
    du_dy_bottom = (-3 * u[:, :, 0, :] + 4 * u[:, :, 1, :] - u[:, :, 2, :]) / (2 * dy)
    du_dy_top = (3 * u[:, :, -1, :] - 4 * u[:, :, -2, :] + u[:, :, -3, :]) / (2 * dy)

    # 非齐次目标
    bc_loss = (torch.mean((du_dx_left - 1.0) ** 2) +
               torch.mean((du_dx_right + 1.0) ** 2) +
               torch.mean((du_dy_bottom + 1.0) ** 2) +
               torch.mean((du_dy_top - 1.0) ** 2)) / 4

    return bc_loss


def compute_heat2d_residual_batch_fft(u, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
    """
    FFT版本: 2D Heat 残差
    假设周期性边界，不做lifting，不保证Neumann BC

    输入: u: [batch, Nx, Ny, Nt, 1]
    输出: residual: [batch, Nx, Ny, Nt-4, 1]
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype
    dt = T / (Nt - 1)

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # 时间导数 (和DCT版一样)
    u_t = torch.zeros_like(u)
    u_t[:, 2:-2] = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)

    # FFT 拉普拉斯: 特征值 -(2πk/L)²
    # fftfreq 返回 [0, 1, 2, ..., N/2, -N/2+1, ..., -1] / N
    # 乘以 2π/dx 得到波数
    kx = torch.fft.fftfreq(Nx, d=Lx / Nx, device=device, dtype=dtype) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, d=Ly / Ny, device=device, dtype=dtype) * 2 * torch.pi

    K2x = -(kx ** 2)  # [Nx]
    K2y = -(ky ** 2)  # [Ny]

    # x方向
    u_hat_x = torch.fft.fft(u, dim=2)
    u_xx = torch.fft.ifft(u_hat_x * K2x.view(1, 1, -1, 1), dim=2).real

    # y方向
    u_hat_y = torch.fft.fft(u, dim=3)
    u_yy = torch.fft.ifft(u_hat_y * K2y.view(1, 1, 1, -1), dim=3).real

    lap_u = u_xx + u_yy

    residual = u_t - kappa * lap_u
    residual = residual[:, 2:-2, :, :]
    residual = residual.permute(0, 2, 3, 1)
    return residual.unsqueeze(-1)


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    import h5py
    import time

    # Verify DCT-I implementation
    print("=" * 60)
    print("DCT-I PyTorch 实现验证")
    print("=" * 60)
    ok = verify_dctI_pytorch()
    if not ok:
        print("ERROR: DCT-I verification failed!")
        exit(1)
    print("DCT-I verification passed!\n")

    # Load data
    filepath = r"./data/heat2d_neumann.h5"  # 修改为你的路径

    with h5py.File(filepath, 'r') as f:
        key = list(f.keys())[1]
        U = f[key]['data'][:101, ...]
        t_grid = f[key]['grid']['t'][:101]

    Nt, Nx, Ny, _ = U.shape
    T = float(t_grid[-1])
    print('T:', T)
    print('U.shape:', U.shape)

    # [batch, Nx, Ny, Nt, 1]
    U_torch = torch.from_numpy(U).float()
    U_torch = U_torch.permute(1, 2, 0, 3).unsqueeze(0)

    print(f"Input shape: {U_torch.shape}")
    print(f"T = {T}, Nt = {Nt}, Nx = {Nx}, Ny = {Ny}")

    # # ========== BC Loss ==========
    # print("\n" + "=" * 60)
    # print("BC Loss 测试 (非齐次 Neumann)")
    # print("=" * 60)
    #
    # bc_loss_fd = compute_neumann_bc_loss(U_torch)
    # print(f"BC Loss (有限差分):  {bc_loss_fd.item():.4e}")
    # print("(应该很小——数据精确满足BC)")

    # ========== Residual ==========
    print("\n" + "=" * 60)
    print("PDE Residual 测试: ∂u/∂t - κΔu = 0 (DCT-I)")
    print("=" * 60)

    res_batch = compute_heat2d_residual_batch(U_torch, kappa=0.02, T=T)
    # res_fd = compute_heat2d_residual_fd(U_torch, kappa=0.02, T=T)
    # res_scipy = compute_heat2d_residual_scipy(U_torch, kappa=0.02, T=T)

    print(f"DCT-I PyTorch MSE: {torch.mean(torch.abs(res_batch)).item():.4e}")

    # ========== FFT vs DCT 对比 ==========
    print("\n" + "=" * 60)
    print("FFT vs DCT 残差对比")
    print("=" * 60)

    res_dct = compute_heat2d_residual_batch(U_torch, kappa=0.02, T=T)
    res_fft = compute_heat2d_residual_batch_fft(U_torch, kappa=0.02, T=T)

    print(f"DCT 残差 MAE: {torch.mean(torch.abs(res_dct)).item():.4e}")
    print(f"FFT 残差 MAE: {torch.mean(torch.abs(res_fft)).item():.4e}")
    print(f"DCT 残差 MAX: {torch.max(torch.abs(res_dct)).item():.4e}")
    print(f"FFT 残差 MAX: {torch.max(torch.abs(res_fft)).item():.4e}")

    # 误差空间分布
    res_fft_sq = res_fft.squeeze().abs()  # [Nx, Ny, Nt-4]
    res_fft_mean = res_fft_sq.mean(dim=-1)  # [Nx, Ny] 时间平均

    print(f"\n内部区域 (去掉边界5个点):")
    interior = res_fft_sq[5:-5, 5:-5, :]
    print(f"  MAE: {interior.mean().item():.4e}")
    print(f"  MAX: {interior.max().item():.4e}")

    print(f"边界区域 (最外5个点):")
    boundary_vals = torch.cat([
        res_fft_sq[:5, :, :].reshape(-1),
        res_fft_sq[-5:, :, :].reshape(-1),
        res_fft_sq[:, :5, :].reshape(-1),
        res_fft_sq[:, -5:, :].reshape(-1),
    ])
    print(f"  MAE: {boundary_vals.mean().item():.4e}")
    print(f"  MAX: {boundary_vals.max().item():.4e}")

    res_fd = compute_heat2d_residual_fd(U_torch, kappa=0.02, T=T)
    print(f"FDM 残差 MAE: {torch.mean(torch.abs(res_fd)).item():.4e}")
    print(f"FDM 残差 MAX: {torch.max(torch.abs(res_fd)).item():.4e}")

    res_fd_abs = res_fd.squeeze().abs()
    interior_fd = res_fd_abs[1:-1, 1:-1, :]
    print(f"FDM 内部 MAE: {interior_fd.mean().item():.4e}")
    print(f"FDM 内部 MAX: {interior_fd.max().item():.4e}")

    # ========== BC Loss 测试 ==========
    print("\n" + "=" * 60)
    print("BC Loss 测试 (非齐次 Neumann)")
    print("=" * 60)

    bc_loss = compute_neumann_bc_loss(U_torch)
    print(f"精确解 BC Loss: {bc_loss.item():.4e}  (应该很小)")

    # 加噪声对比
    U_noisy = U_torch + 0.1 * torch.randn_like(U_torch)
    bc_loss_noisy = compute_neumann_bc_loss(U_noisy)
    print(f"加噪声 BC Loss: {bc_loss_noisy.item():.4e}  (应该明显变大)")

    # print(f"有限差分 MSE:      {torch.mean(res_fd ** 2).item():.4e}")
    # print(f"DCT-I scipy MSE:   {torch.mean(res_scipy ** 2).item():.4e}")
    # print(f"PyTorch-scipy 差异: {torch.max(torch.abs(res_batch - res_scipy)).item():.4e}")
    # print(f"PyTorch-FD 差异:    {torch.max(torch.abs(res_batch - res_fd)).item():.4e}")
    # print("(MSE 应该很小——数据是精确解)")

    # ========== 速度测试 ==========
    # print("\n" + "=" * 60)
    # print("速度测试")
    # print("=" * 60)
    #
    # n_runs = 20
    #
    # t0 = time.time()
    # for _ in range(n_runs):
    #     _ = compute_heat2d_residual_batch(U_torch, kappa=0.02, T=T)
    # t1 = time.time()
    # print(f"DCT-I PyTorch CPU: {(t1 - t0) / n_runs * 1000:.2f} ms")
    #
    # t0 = time.time()
    # for _ in range(n_runs):
    #     _ = compute_heat2d_residual_fd(U_torch, kappa=0.02, T=T)
    # t1 = time.time()
    # print(f"有限差分 CPU:      {(t1 - t0) / n_runs * 1000:.2f} ms")
    #
    # if torch.cuda.is_available():
    #     U_cuda = U_torch.cuda()
    #     for _ in range(5):
    #         _ = compute_heat2d_residual_batch(U_cuda, kappa=0.02, T=T)
    #     torch.cuda.synchronize()
    #
    #     t0 = time.time()
    #     for _ in range(n_runs):
    #         _ = compute_heat2d_residual_batch(U_cuda, kappa=0.02, T=T)
    #     torch.cuda.synchronize()
    #     t1 = time.time()
    #     print(f"DCT-I PyTorch GPU: {(t1 - t0) / n_runs * 1000:.2f} ms")

    # ========== 自动求导测试 ==========
    # print("\n" + "=" * 60)
    # print("自动求导测试")
    # print("=" * 60)
    #
    # U_test = U_torch.clone().requires_grad_(True)
    # res = compute_heat2d_residual_batch(U_test, kappa=0.02, T=T)
    # loss = torch.mean(res ** 2)
    # loss.backward()
    # print(f"Residual gradient OK: {U_test.grad is not None}")
    #
    # U_test2 = U_torch.clone().requires_grad_(True)
    # bc = compute_neumann_bc_loss(U_test2)
    # bc.backward()
    # print(f"BC Loss gradient OK:  {U_test2.grad is not None}")
