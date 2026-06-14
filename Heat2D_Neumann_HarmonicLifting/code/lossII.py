"""
Loss functions for 2D Heat Equation with Per-Sample Non-homogeneous Neumann BC
Using DCT-I on grid INCLUDING endpoints (N+1 points, e.g. 129)

PDE:  ∂u/∂t = κ Δu
BC:   ∂u/∂x(-Lx/2) = a,  ∂u/∂x(+Lx/2) = b
      ∂u/∂y(-Ly/2) = c,  ∂u/∂y(+Ly/2) = d
      where (a, b, c, d) vary per sample, with (b-a)+(d-c)=0

Parametric harmonic lifting:
  u_b(x,y) = α x² + β y² + γ x + δ y
  α = (b-a)/4,  β = (d-c)/4,  γ = (a+b)/2,  δ = (c+d)/2
  Δu_b = 2α + 2β = 0  (by compatibility condition)

DCT-I basis: cos(kπj/N), j=0,...,N (includes endpoints)
  - Derivative at j=0 and j=N is exactly zero → homogeneous Neumann by construction
  - Eigenvalue for d²/dx²: -(kπ/L)²
"""

import torch
from scipy.fft import dct, idct
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
from pathlib import Path
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
# Per-sample lifting construction
# ================================================================

def build_lifting_batch(bc_params, x, y):
    """Construct per-sample harmonic lifting from BC parameters.

    Args:
        bc_params: [batch, 4] tensor of (a, b, c, d) per sample
        x: [Nx] tensor of x coordinates
        y: [Ny] tensor of y coordinates

    Returns:
        u_b: [batch, 1, Nx, Ny] lifting field (broadcastable over time)
    """
    a = bc_params[:, 0]  # [batch]
    b = bc_params[:, 1]
    c = bc_params[:, 2]
    d = bc_params[:, 3]

    alpha = (b - a) / 4.0  # [batch]
    beta = (d - c) / 4.0
    gamma = (a + b) / 2.0
    delta = (c + d) / 2.0

    # x: [Nx], y: [Ny] -> X²: [1,1,Nx,1], Y²: [1,1,1,Ny]
    X2 = (x ** 2).view(1, 1, -1, 1)   # [1, 1, Nx, 1]
    Y2 = (y ** 2).view(1, 1, 1, -1)   # [1, 1, 1, Ny]
    Xv = x.view(1, 1, -1, 1)          # [1, 1, Nx, 1]
    Yv = y.view(1, 1, 1, -1)          # [1, 1, 1, Ny]

    # coeffs: [batch] -> [batch, 1, 1, 1]
    alpha = alpha.view(-1, 1, 1, 1)
    beta = beta.view(-1, 1, 1, 1)
    gamma = gamma.view(-1, 1, 1, 1)
    delta = delta.view(-1, 1, 1, 1)

    u_b = alpha * X2 + beta * Y2 + gamma * Xv + delta * Yv  # [batch, 1, Nx, Ny]
    return u_b


# ================================================================
# PDE Residual: ∂u/∂t - κ Δu = 0  (DCT-I version, per-sample BC)
# ================================================================

def compute_heat2d_residual_batch(u, bc_params, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
    """
    2D Heat PDE residual via DCT-I spectral Laplacian, with per-sample lifting.

    输入:
        u: [batch, Nx, Ny, Nt, 1]  物理解 u = u_h + u_b
        bc_params: [batch, 4]  每个 sample 的 (a, b, c, d)
    输出:
        residual: [batch, Nx, Ny, Nt-4, 1]
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype
    dt = T / (Nt - 1)

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # Per-sample lifting: u_b [batch, 1, Nx, Ny]
    x = torch.linspace(-Lx / 2, Lx / 2, Nx, device=device, dtype=dtype)
    y = torch.linspace(-Ly / 2, Ly / 2, Ny, device=device, dtype=dtype)
    u_b = build_lifting_batch(bc_params.to(device=device, dtype=dtype), x, y)

    u_h = u - u_b  # [batch, Nt, Nx, Ny] - [batch, 1, Nx, Ny] -> broadcast

    # 时间导数 (四阶中心差分, 对 u 即可, ∂u_b/∂t = 0)
    u_t = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)

    # ===== 2D DCT-I spectral Laplacian on u_h =====
    u_h_hat = dctI_batch(dctI_batch(u_h[:, 2:-2], axis=2), axis=3)

    kx = torch.arange(Nx, device=device, dtype=dtype)
    ky = torch.arange(Ny, device=device, dtype=dtype)
    K2 = -(kx * torch.pi / Lx).view(1, 1, -1, 1) ** 2 \
         - (ky * torch.pi / Ly).view(1, 1, 1, -1) ** 2

    lap_u = idctI_batch(idctI_batch(u_h_hat * K2, axis=3), axis=2)

    residual = u_t - kappa * lap_u
    residual = residual.permute(0, 2, 3, 1)
    return residual.unsqueeze(-1)


def compute_heat2d_residual_scipy(u, bc_params, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
    """
    scipy 参考版本: 2D Heat 残差 (DCT-I), per-sample BC

    输入:
        u: [batch, Nx, Ny, Nt, 1]
        bc_params: [batch, 4]
    输出:
        residual: [batch, Nx, Ny, Nt-4, 1]
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    dt = T / (Nt - 1)

    u_np = u.permute(0, 3, 1, 2).numpy()  # [batch, Nt, Nx, Ny]

    # Per-sample lifting
    x_np = np.linspace(-Lx / 2, Lx / 2, Nx)
    y_np = np.linspace(-Ly / 2, Ly / 2, Ny)
    bc_np = bc_params.numpy()  # [batch, 4]

    u_h_np = np.zeros_like(u_np)
    for i in range(batch):
        a, b, c, d = bc_np[i]
        alpha = (b - a) / 4.0
        beta = (d - c) / 4.0
        gamma = (a + b) / 2.0
        delta = (c + d) / 2.0
        u_b_i = (alpha * x_np[:, None] ** 2 + beta * y_np[None, :] ** 2
                 + gamma * x_np[:, None] + delta * y_np[None, :])
        u_h_np[i] = u_np[i] - u_b_i[None, :, :]

    # 时间导数
    u_t = np.zeros_like(u_np)
    u_t[:, 2:-2] = (-u_np[:, 4:] + 8 * u_np[:, 3:-1] - 8 * u_np[:, 1:-3] + u_np[:, :-4]) / (12 * dt)
    u_t[:, 0] = (-3 * u_np[:, 0] + 4 * u_np[:, 1] - u_np[:, 2]) / (2 * dt)
    u_t[:, 1] = (u_np[:, 2] - u_np[:, 0]) / (2 * dt)
    u_t[:, -2] = (u_np[:, -1] - u_np[:, -3]) / (2 * dt)
    u_t[:, -1] = (3 * u_np[:, -1] - 4 * u_np[:, -2] + u_np[:, -3]) / (2 * dt)

    # Spectral Laplacian on u_h
    kx = np.arange(Nx) * (np.pi / Lx)
    ky_arr = np.arange(Ny) * (np.pi / Ly)

    u_hat_x = dct(u_h_np, type=1, axis=2)
    d2u_hat_x = u_hat_x * (-(kx ** 2))[np.newaxis, np.newaxis, :, np.newaxis]
    u_xx = idct(d2u_hat_x, type=1, axis=2)

    u_hat_y = dct(u_h_np, type=1, axis=3)
    d2u_hat_y = u_hat_y * (-(ky_arr ** 2))[np.newaxis, np.newaxis, np.newaxis, :]
    u_yy = idct(d2u_hat_y, type=1, axis=3)

    lap_u = u_xx + u_yy
    residual = u_t - kappa * lap_u
    residual = residual[:, 2:-2, :, :]

    residual = torch.from_numpy(residual).float()
    residual = residual.permute(0, 2, 3, 1)
    return residual.unsqueeze(-1)


def compute_heat2d_residual_fd(u, bc_params, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
    """
    有限差分版本: 2D Heat 残差, per-sample BC

    FD 边界处使用 Neumann 镜像, 镜像值依赖 per-sample BC.

    输入:
        u: [batch, Nx, Ny, Nt, 1]
        bc_params: [batch, 4]  每个 sample 的 (a, b, c, d)
    输出:
        residual: [batch, Nx, Ny, Nt-4, 1]
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = T / (Nt - 1)

    bc_params = bc_params.to(device=device, dtype=dtype)
    a = bc_params[:, 0].view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
    b = bc_params[:, 1].view(-1, 1, 1, 1)
    c = bc_params[:, 2].view(-1, 1, 1, 1)
    d = bc_params[:, 3].view(-1, 1, 1, 1)

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # 时间导数: conv1d
    kt = torch.tensor([1, -8, 0, 8, -1], device=device, dtype=dtype) / (12 * dt)
    u_t = F.conv1d(
        u.permute(0, 2, 3, 1).reshape(-1, 1, Nt),
        kt.view(1, 1, -1)
    ).reshape(batch, Nx, Ny, Nt - 4).permute(0, 3, 1, 2)

    u_mid = u[:, 2:-2, :, :]  # [batch, Nt-4, Nx, Ny]

    # x方向: 内部 conv2d + 边界 Neumann 镜像
    kx_kernel = torch.tensor([[-1, 16, -30, 16, -1]], device=device, dtype=dtype).view(1, 1, 5, 1) / (12 * dx ** 2)
    u_flat = u_mid.reshape(-1, 1, Nx, Ny)
    u_xx_inner = F.conv2d(u_flat, kx_kernel).reshape(batch, -1, Nx - 4, Ny)

    u_xx = torch.zeros_like(u_mid)
    u_xx[:, :, 2:-2, :] = u_xx_inner
    # 边界: ∂u/∂x(-Lx/2) = a  ->  u[-1] ≈ u[1] - 2*dx*a
    u_xx[:, :, 0, :] = (2 * u_mid[:, :, 1, :] - 2 * u_mid[:, :, 0, :] - 2 * dx * a.squeeze(-1)) / (dx ** 2)
    u_xx[:, :, 1, :] = (u_mid[:, :, 2, :] - 2 * u_mid[:, :, 1, :] + u_mid[:, :, 0, :]) / (dx ** 2)
    u_xx[:, :, -2, :] = (u_mid[:, :, -1, :] - 2 * u_mid[:, :, -2, :] + u_mid[:, :, -3, :]) / (dx ** 2)
    # ∂u/∂x(+Lx/2) = b  ->  u[N+1] ≈ u[N-1] + 2*dx*b
    u_xx[:, :, -1, :] = (2 * u_mid[:, :, -2, :] - 2 * u_mid[:, :, -1, :] + 2 * dx * b.squeeze(-1)) / (dx ** 2)

    # y方向: 内部 conv2d + 边界 Neumann 镜像
    ky_kernel = torch.tensor([[-1, 16, -30, 16, -1]], device=device, dtype=dtype).view(1, 1, 1, 5) / (12 * dy ** 2)
    u_yy_inner = F.conv2d(u_flat, ky_kernel).reshape(batch, -1, Nx, Ny - 4)

    u_yy = torch.zeros_like(u_mid)
    u_yy[:, :, :, 2:-2] = u_yy_inner
    # ∂u/∂y(-Ly/2) = c
    u_yy[:, :, :, 0] = (2 * u_mid[:, :, :, 1] - 2 * u_mid[:, :, :, 0] - 2 * dy * c.squeeze(-1)) / (dy ** 2)
    u_yy[:, :, :, 1] = (u_mid[:, :, :, 2] - 2 * u_mid[:, :, :, 1] + u_mid[:, :, :, 0]) / (dy ** 2)
    u_yy[:, :, :, -2] = (u_mid[:, :, :, -1] - 2 * u_mid[:, :, :, -2] + u_mid[:, :, :, -3]) / (dy ** 2)
    # ∂u/∂y(+Ly/2) = d
    u_yy[:, :, :, -1] = (2 * u_mid[:, :, :, -2] - 2 * u_mid[:, :, :, -1] + 2 * dy * d.squeeze(-1)) / (dy ** 2)

    residual = u_t - kappa * (u_xx + u_yy)
    residual = residual.permute(0, 2, 3, 1)
    return residual.unsqueeze(-1)


def compute_neumann_bc_loss(u, bc_params, Lx=2.0, Ly=2.0):
    """
    非齐次 Neumann BC Loss (有限差分, 端点网格), per-sample BC targets.

    输入:
        u: [batch, Nx, Ny, Nt, 1]
        bc_params: [batch, 4]  每个 sample 的 (a, b, c, d)
    输出:
        bc_loss: 标量
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    bc_params = bc_params.to(device=device, dtype=dtype)
    a = bc_params[:, 0].view(-1, 1, 1)  # [batch, 1, 1]
    b = bc_params[:, 1].view(-1, 1, 1)
    c = bc_params[:, 2].view(-1, 1, 1)
    d = bc_params[:, 3].view(-1, 1, 1)

    # x 方向 (二阶单侧FD)
    du_dx_left = (-3 * u[:, 0, :, :] + 4 * u[:, 1, :, :] - u[:, 2, :, :]) / (2 * dx)
    du_dx_right = (3 * u[:, -1, :, :] - 4 * u[:, -2, :, :] + u[:, -3, :, :]) / (2 * dx)

    # y 方向
    du_dy_bottom = (-3 * u[:, :, 0, :] + 4 * u[:, :, 1, :] - u[:, :, 2, :]) / (2 * dy)
    du_dy_top = (3 * u[:, :, -1, :] - 4 * u[:, :, -2, :] + u[:, :, -3, :]) / (2 * dy)

    bc_loss = (torch.mean((du_dx_left - a) ** 2) +
               torch.mean((du_dx_right - b) ** 2) +
               torch.mean((du_dy_bottom - c) ** 2) +
               torch.mean((du_dy_top - d) ** 2)) / 4

    return bc_loss


def compute_heat2d_residual_batch_fft(u, kappa=0.02, Lx=2.0, Ly=2.0, T=5.0):
    """
    FFT版本: 2D Heat 残差 (不依赖 BC, 假设周期性)
    保留原接口，不需要 bc_params.

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

    u_t = torch.zeros_like(u)
    u_t[:, 2:-2] = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)

    kx = torch.fft.fftfreq(Nx, d=Lx / Nx, device=device, dtype=dtype) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, d=Ly / Ny, device=device, dtype=dtype) * 2 * torch.pi

    K2x = -(kx ** 2)
    K2y = -(ky ** 2)

    u_hat_x = torch.fft.fft(u, dim=2)
    u_xx = torch.fft.ifft(u_hat_x * K2x.view(1, 1, -1, 1), dim=2).real

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

    # Load data (兼容新旧数据集)
    filepath = rstr(CASE_ROOT / 'data' / 'heat2d_change_neumann_1100.h5')

    with h5py.File(filepath, 'r') as f:
        key = list(k for k in f.keys() if k.isdigit())[0]
        U = f[key]['data'][:101, ...]
        t_grid = f[key]['grid']['t'][:101]

        # 读取 BC
        if 'bc' in f[key]:
            a = float(f[key]['bc']['a'][()])
            b = float(f[key]['bc']['b'][()])
            c = float(f[key]['bc']['c'][()])
            d = float(f[key]['bc']['d'][()])
            print(f"Per-sample BC: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")
        else:
            a, b, c, d = 1.0, -1.0, -1.0, 1.0
            print("Fixed BC: a=1, b=-1, c=-1, d=1")

    bc_params = torch.tensor([[a, b, c, d]], dtype=torch.float32)

    Nt, Nx, Ny, _ = U.shape
    T = float(t_grid[-1])
    print('T:', T)
    print('U.shape:', U.shape)

    # [batch, Nx, Ny, Nt, 1]
    U_torch = torch.from_numpy(U).float()
    U_torch = U_torch.permute(1, 2, 0, 3).unsqueeze(0)

    print(f"Input shape: {U_torch.shape}")
    print(f"T = {T}, Nt = {Nt}, Nx = {Nx}, Ny = {Ny}")

    # ========== Residual ==========
    print("\n" + "=" * 60)
    print("PDE Residual 测试: ∂u/∂t - κΔu = 0 (DCT-I, per-sample BC)")
    print("=" * 60)

    res_batch = compute_heat2d_residual_batch(U_torch, bc_params, kappa=0.02, T=T)

    print(f"DCT-I PyTorch MAE: {torch.mean(torch.abs(res_batch)).item():.4e}")

    # ========== FFT vs DCT 对比 ==========
    print("\n" + "=" * 60)
    print("FFT vs DCT 残差对比")
    print("=" * 60)

    res_dct = compute_heat2d_residual_batch(U_torch, bc_params, kappa=0.02, T=T)
    res_fft = compute_heat2d_residual_batch_fft(U_torch, kappa=0.02, T=T)

    print(f"DCT 残差 MAE: {torch.mean(torch.abs(res_dct)).item():.4e}")
    print(f"FFT 残差 MAE: {torch.mean(torch.abs(res_fft)).item():.4e}")
    print(f"DCT 残差 MAX: {torch.max(torch.abs(res_dct)).item():.4e}")
    print(f"FFT 残差 MAX: {torch.max(torch.abs(res_fft)).item():.4e}")

    res_fd = compute_heat2d_residual_fd(U_torch, bc_params, kappa=0.02, T=T)
    print(f"FDM 残差 MAE: {torch.mean(torch.abs(res_fd)).item():.4e}")
    print(f"FDM 残差 MAX: {torch.max(torch.abs(res_fd)).item():.4e}")

    # ========== BC Loss 测试 ==========
    print("\n" + "=" * 60)
    print("BC Loss 测试 (per-sample 非齐次 Neumann)")
    print("=" * 60)

    bc_loss = compute_neumann_bc_loss(U_torch, bc_params)
    print(f"精确解 BC Loss: {bc_loss.item():.4e}  (应该很小)")

    U_noisy = U_torch + 0.1 * torch.randn_like(U_torch)
    bc_loss_noisy = compute_neumann_bc_loss(U_noisy, bc_params)
    print(f"加噪声 BC Loss: {bc_loss_noisy.item():.4e}  (应该明显变大)")