"""
Loss functions for 2D Allen-Cahn Equation with Per-Sample Non-homogeneous Neumann BC
Using DCT-I on grid INCLUDING endpoints (N+1 points, e.g. 129)

PDE:  ∂u/∂t = ε Δu + u - u³
BC:   ∂u/∂x(-Lx/2) = a,  ∂u/∂x(+Lx/2) = b
      ∂u/∂y(-Ly/2) = c,  ∂u/∂y(+Ly/2) = d
      where (a, b, c, d) vary per sample, with (b-a)+(d-c)=0

Parametric harmonic lifting:
  u_b(x,y) = α x² + β y² + γ x + δ y
  α = (b-a)/4,  β = (d-c)/4,  γ = (a+b)/2,  δ = (c+d)/2
  Δu_b = 2α + 2β = 0  (by compatibility condition)

Residual: R = ∂u/∂t - ε Δu - u + u³
  - ∂u/∂t: 4th-order central FD in time
  - Δu = Δu_h (since Δu_b = 0), computed via DCT-I spectral method
  - u - u³: pointwise in physical space
"""

import torch
import numpy as np
import torch.nn.functional as F


# ================================================================
# PyTorch DCT-I / IDCT-I batch implementation via FFT
# ================================================================

def dctI_batch(u, axis=-1):
    """Batch DCT-I via FFT (symmetric extension)."""
    u = torch.moveaxis(u, axis, -1)
    N1 = u.shape[-1]
    N = N1 - 1
    y = torch.cat([u, u[..., 1:N].flip(dims=[-1])], dim=-1)
    Y = torch.fft.rfft(y, dim=-1)
    result = Y.real
    return torch.moveaxis(result, -1, axis)


def idctI_batch(c, axis=-1):
    """Batch IDCT-I (inverse of dctI_batch)."""
    c = torch.moveaxis(c, axis, -1)
    N1 = c.shape[-1]
    N = N1 - 1
    y = torch.cat([c, c[..., 1:N].flip(dims=[-1])], dim=-1)
    Y = torch.fft.rfft(y, dim=-1)
    result = Y.real / (2 * N)
    return torch.moveaxis(result, -1, axis)


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
        u_b: [batch, 1, Nx, Ny] lifting field
    """
    a = bc_params[:, 0]
    b = bc_params[:, 1]
    c = bc_params[:, 2]
    d = bc_params[:, 3]

    alpha = (b - a) / 4.0
    beta = (d - c) / 4.0
    gamma = (a + b) / 2.0
    delta = (c + d) / 2.0

    X2 = (x ** 2).view(1, 1, -1, 1)
    Y2 = (y ** 2).view(1, 1, 1, -1)
    Xv = x.view(1, 1, -1, 1)
    Yv = y.view(1, 1, 1, -1)

    alpha = alpha.view(-1, 1, 1, 1)
    beta = beta.view(-1, 1, 1, 1)
    gamma = gamma.view(-1, 1, 1, 1)
    delta = delta.view(-1, 1, 1, 1)

    u_b = alpha * X2 + beta * Y2 + gamma * Xv + delta * Yv
    return u_b


# ================================================================
# PDE Residual: ∂u/∂t - ε Δu - u + u³ = 0  (DCT-I version)
# ================================================================

def compute_ac2d_residual_batch(u, bc_params, epsilon=0.05, Lx=2.0, Ly=2.0, T=5.0):
    """
    2D Allen-Cahn PDE residual via DCT-I spectral Laplacian, per-sample BC.

    R = ∂u/∂t - ε Δu - u + u³

    输入:
        u: [batch, Nx, Ny, Nt, 1]  物理解 u = u_h + u_b
        bc_params: [batch, 4]  每个 sample 的 (a, b, c, d)
        epsilon: interface thickness parameter
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

    # ---- Per-sample lifting ----
    x = torch.linspace(-Lx / 2, Lx / 2, Nx, device=device, dtype=dtype)
    y = torch.linspace(-Ly / 2, Ly / 2, Ny, device=device, dtype=dtype)
    u_b = build_lifting_batch(bc_params.to(device=device, dtype=dtype), x, y)  # [batch, 1, Nx, Ny]

    u_h = u - u_b  # [batch, Nt, Nx, Ny]

    # ---- 时间导数 (四阶中心差分) ----
    u_t = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)

    # ---- 谱拉普拉斯 Δu = Δu_h (since Δu_b = 0) ----
    u_h_hat = dctI_batch(dctI_batch(u_h[:, 2:-2], axis=2), axis=3)

    kx = torch.arange(Nx, device=device, dtype=dtype)
    ky = torch.arange(Ny, device=device, dtype=dtype)
    K2 = -(kx * torch.pi / Lx).view(1, 1, -1, 1) ** 2 \
         - (ky * torch.pi / Ly).view(1, 1, 1, -1) ** 2

    lap_u = idctI_batch(idctI_batch(u_h_hat * K2, axis=3), axis=2)

    # ---- 反应项 (物理空间逐点计算) ----
    u_mid = u[:, 2:-2]  # [batch, Nt-4, Nx, Ny]
    reaction = u_mid - u_mid ** 3  # u - u³

    # ---- 残差: R = ∂u/∂t - ε Δu - (u - u³) ----
    residual = u_t - epsilon * lap_u - reaction

    residual = residual.permute(0, 2, 3, 1)  # [batch, Nx, Ny, Nt-4]
    return residual.unsqueeze(-1)


# ================================================================
# FD Residual (baseline, for comparison)
# ================================================================

def compute_ac2d_residual_fd(u, bc_params, epsilon=0.05, Lx=2.0, Ly=2.0, T=5.0):
    """
    有限差分版本: 2D Allen-Cahn 残差, per-sample BC.

    R = ∂u/∂t - ε Δu - u + u³

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
    a = bc_params[:, 0].view(-1, 1, 1, 1)
    b = bc_params[:, 1].view(-1, 1, 1, 1)
    c = bc_params[:, 2].view(-1, 1, 1, 1)
    d = bc_params[:, 3].view(-1, 1, 1, 1)

    u = u.permute(0, 3, 1, 2)  # [batch, Nt, Nx, Ny]

    # ---- 时间导数: conv1d ----
    kt = torch.tensor([1, -8, 0, 8, -1], device=device, dtype=dtype) / (12 * dt)
    u_t = F.conv1d(
        u.permute(0, 2, 3, 1).reshape(-1, 1, Nt),
        kt.view(1, 1, -1)
    ).reshape(batch, Nx, Ny, Nt - 4).permute(0, 3, 1, 2)  # [batch, Nt-4, Nx, Ny]

    u_mid = u[:, 2:-2, :, :]  # [batch, Nt-4, Nx, Ny]

    # ---- x方向: 内部 conv2d + 边界 Neumann 镜像 ----
    kx_kernel = torch.tensor([[-1, 16, -30, 16, -1]], device=device, dtype=dtype).view(1, 1, 5, 1) / (12 * dx ** 2)
    u_flat = u_mid.reshape(-1, 1, Nx, Ny)
    u_xx_inner = F.conv2d(u_flat, kx_kernel).reshape(batch, -1, Nx - 4, Ny)

    u_xx = torch.zeros_like(u_mid)
    u_xx[:, :, 2:-2, :] = u_xx_inner
    u_xx[:, :, 0, :] = (2 * u_mid[:, :, 1, :] - 2 * u_mid[:, :, 0, :] - 2 * dx * a.squeeze(-1)) / (dx ** 2)
    u_xx[:, :, 1, :] = (u_mid[:, :, 2, :] - 2 * u_mid[:, :, 1, :] + u_mid[:, :, 0, :]) / (dx ** 2)
    u_xx[:, :, -2, :] = (u_mid[:, :, -1, :] - 2 * u_mid[:, :, -2, :] + u_mid[:, :, -3, :]) / (dx ** 2)
    u_xx[:, :, -1, :] = (2 * u_mid[:, :, -2, :] - 2 * u_mid[:, :, -1, :] + 2 * dx * b.squeeze(-1)) / (dx ** 2)

    # ---- y方向: 内部 conv2d + 边界 Neumann 镜像 ----
    ky_kernel = torch.tensor([[-1, 16, -30, 16, -1]], device=device, dtype=dtype).view(1, 1, 1, 5) / (12 * dy ** 2)
    u_yy_inner = F.conv2d(u_flat, ky_kernel).reshape(batch, -1, Nx, Ny - 4)

    u_yy = torch.zeros_like(u_mid)
    u_yy[:, :, :, 2:-2] = u_yy_inner
    u_yy[:, :, :, 0] = (2 * u_mid[:, :, :, 1] - 2 * u_mid[:, :, :, 0] - 2 * dy * c.squeeze(-1)) / (dy ** 2)
    u_yy[:, :, :, 1] = (u_mid[:, :, :, 2] - 2 * u_mid[:, :, :, 1] + u_mid[:, :, :, 0]) / (dy ** 2)
    u_yy[:, :, :, -2] = (u_mid[:, :, :, -1] - 2 * u_mid[:, :, :, -2] + u_mid[:, :, :, -3]) / (dy ** 2)
    u_yy[:, :, :, -1] = (2 * u_mid[:, :, :, -2] - 2 * u_mid[:, :, :, -1] + 2 * dy * d.squeeze(-1)) / (dy ** 2)

    # ---- 反应项 ----
    reaction = u_mid - u_mid ** 3

    # ---- 残差 ----
    residual = u_t - epsilon * (u_xx + u_yy) - reaction
    residual = residual.permute(0, 2, 3, 1)
    return residual.unsqueeze(-1)


# ================================================================
# FFT Residual (baseline, assumes periodicity)
# ================================================================

def compute_ac2d_residual_fft(u, epsilon=0.05, Lx=2.0, Ly=2.0, T=5.0):
    """
    FFT版本: 2D Allen-Cahn 残差 (不依赖 BC, 假设周期性).

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

    # 时间导数
    u_t = (-u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, :-4]) / (12 * dt)

    # FFT Laplacian
    kx = torch.fft.fftfreq(Nx, d=Lx / Nx, device=device, dtype=dtype) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, d=Ly / Ny, device=device, dtype=dtype) * 2 * torch.pi

    K2x = -(kx ** 2)
    K2y = -(ky ** 2)

    u_mid = u[:, 2:-2]
    u_hat_x = torch.fft.fft(u_mid, dim=2)
    u_xx = torch.fft.ifft(u_hat_x * K2x.view(1, 1, -1, 1), dim=2).real

    u_hat_y = torch.fft.fft(u_mid, dim=3)
    u_yy = torch.fft.ifft(u_hat_y * K2y.view(1, 1, 1, -1), dim=3).real

    lap_u = u_xx + u_yy

    # 反应项
    reaction = u_mid - u_mid ** 3

    residual = u_t - epsilon * lap_u - reaction
    residual = residual.permute(0, 2, 3, 1)
    return residual.unsqueeze(-1)


# ================================================================
# Neumann BC Loss
# ================================================================

def compute_neumann_bc_loss(u, bc_params, Lx=2.0, Ly=2.0):
    """
    非齐次 Neumann BC Loss (有限差分), per-sample BC targets.

    输入:
        u: [batch, Nx, Ny, Nt, 1]
        bc_params: [batch, 4]
    输出:
        bc_loss: scalar
    """
    if u.dim() == 5:
        u = u.squeeze(-1)

    batch, Nx, Ny, Nt = u.shape
    device = u.device
    dtype = u.dtype
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    bc_params = bc_params.to(device=device, dtype=dtype)
    a = bc_params[:, 0].view(-1, 1, 1)
    b = bc_params[:, 1].view(-1, 1, 1)
    c = bc_params[:, 2].view(-1, 1, 1)
    d = bc_params[:, 3].view(-1, 1, 1)

    du_dx_left = (-3 * u[:, 0, :, :] + 4 * u[:, 1, :, :] - u[:, 2, :, :]) / (2 * dx)
    du_dx_right = (3 * u[:, -1, :, :] - 4 * u[:, -2, :, :] + u[:, -3, :, :]) / (2 * dx)
    du_dy_bottom = (-3 * u[:, :, 0, :] + 4 * u[:, :, 1, :] - u[:, :, 2, :]) / (2 * dy)
    du_dy_top = (3 * u[:, :, -1, :] - 4 * u[:, :, -2, :] + u[:, :, -3, :]) / (2 * dy)

    bc_loss = (torch.mean((du_dx_left - a) ** 2) +
               torch.mean((du_dx_right - b) ** 2) +
               torch.mean((du_dy_bottom - c) ** 2) +
               torch.mean((du_dy_top - d) ** 2)) / 4

    return bc_loss


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    import h5py

    print("=" * 60)
    print("Allen-Cahn 2D Loss Function Test")
    print("=" * 60)

    # Load test data (use the test file from test_allen_cahn.py or generate_allen_cahn2d.py)
    filepath = "./ac2d_test.h5"
    import os
    if not os.path.exists(filepath):
        print(f"[INFO] {filepath} not found, using synthetic data for test")
        # Synthetic: random u, random bc
        torch.manual_seed(42)
        batch, Nx, Ny, Nt = 2, 65, 65, 51
        u = torch.randn(batch, Nx, Ny, Nt, 1)
        bc_params = torch.tensor([[0.3, -0.2, 0.1, 0.6],
                                   [-0.1, 0.4, 0.2, -0.3]])
        T_val = 5.0
    else:
        with h5py.File(filepath, 'r') as f:
            key = sorted([k for k in f.keys() if k.isdigit()])[0]
            U = f[key]['data'][:101, ...]
            t_grid = f[key]['grid']['t'][:101]
            a = float(f[key]['bc']['a'][()])
            b = float(f[key]['bc']['b'][()])
            c = float(f[key]['bc']['c'][()])
            d = float(f[key]['bc']['d'][()])

        bc_params = torch.tensor([[a, b, c, d]])
        T_val = float(t_grid[-1])
        u = torch.from_numpy(U).float().permute(1, 2, 0, 3).unsqueeze(0)
        batch, Nx, Ny, Nt = 1, u.shape[1], u.shape[2], u.shape[3]

    print(f"Input shape: {u.shape}, T={T_val}")
    print(f"BC: {bc_params}")

    # DCT residual
    res_dct = compute_ac2d_residual_batch(u, bc_params, epsilon=0.05, T=T_val)
    print(f"\nDCT-I residual MAE: {torch.mean(torch.abs(res_dct)).item():.4e}")
    print(f"DCT-I residual MAX: {torch.max(torch.abs(res_dct)).item():.4e}")

    # FFT residual
    res_fft = compute_ac2d_residual_fft(u, epsilon=0.05, T=T_val)
    print(f"\nFFT residual MAE:   {torch.mean(torch.abs(res_fft)).item():.4e}")
    print(f"FFT residual MAX:   {torch.max(torch.abs(res_fft)).item():.4e}")

    # FD residual
    res_fd = compute_ac2d_residual_fd(u, bc_params, epsilon=0.05, T=T_val)
    print(f"\nFD residual MAE:    {torch.mean(torch.abs(res_fd)).item():.4e}")
    print(f"FD residual MAX:    {torch.max(torch.abs(res_fd)).item():.4e}")

    # BC loss
    bc_loss = compute_neumann_bc_loss(u, bc_params)
    print(f"\nBC loss: {bc_loss.item():.4e}")

    # Autograd test
    print("\n" + "=" * 60)
    print("Autograd test")
    print("=" * 60)
    u_test = u[:1].clone().requires_grad_(True)
    bc_test = bc_params[:1]
    res = compute_ac2d_residual_batch(u_test, bc_test, epsilon=0.05, T=T_val)
    loss = (res ** 2).mean()
    loss.backward()
    print(f"DCT residual grad OK: {u_test.grad is not None}")
    print(f"  grad norm: {u_test.grad.norm().item():.4e}")