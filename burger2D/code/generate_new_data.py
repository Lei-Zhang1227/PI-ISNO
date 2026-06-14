import numpy as np
import matplotlib.pyplot as plt


def generate_grf_neumann_with_check(nx, ny, Lx=2.0, Ly=2.0, alpha=16, tau=16,
                                    n_modes=50, scale=None, seed=None):
    """
    生成GRF并解析计算导数验证Neumann BC
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 预计算系数
    kx_arr = np.arange(n_modes)
    ky_arr = np.arange(n_modes)
    Kx, Ky = np.meshgrid(kx_arr, ky_arr, indexing='ij')

    lambda_k = (Kx * np.pi / Lx) ** 2 + (Ky * np.pi / Ly) ** 2
    eigenvalue = tau - lambda_k
    std = np.sqrt(alpha) / np.abs(eigenvalue)
    std[np.abs(eigenvalue) < 1e-10] = 0

    coef = std * np.random.randn(n_modes, n_modes)

    # 生成 u0
    u0 = np.zeros((nx, ny))
    # 解析计算 du/dx 和 du/dy
    du_dx = np.zeros((nx, ny))
    du_dy = np.zeros((nx, ny))

    for kx in range(n_modes):
        for ky in range(n_modes):
            c = coef[kx, ky]

            cos_x = np.cos(kx * np.pi * X / Lx)
            cos_y = np.cos(ky * np.pi * Y / Ly)
            sin_x = np.sin(kx * np.pi * X / Lx)
            sin_y = np.sin(ky * np.pi * Y / Ly)

            u0 += c * cos_x * cos_y
            du_dx += c * (-kx * np.pi / Lx) * sin_x * cos_y
            du_dy += c * (-ky * np.pi / Ly) * cos_x * sin_y

    if scale is not None:
        factor = scale / np.max(np.abs(u0))
        u0 *= factor
        du_dx *= factor
        du_dy *= factor

    return u0, du_dx, du_dy


# ============ 测试 ============
nx, ny = 129, 129
Lx, Ly = 2.0, 2.0

u0, du_dx, du_dy = generate_grf_neumann_with_check(
    nx, ny, Lx, Ly, alpha=16, tau=16, n_modes=50, scale=0.5, seed=42
)

print(f"u0 幅值范围: [{u0.min():.3f}, {u0.max():.3f}]")

# 解析导数在边界处的值（应该精确为0）
print("\n===== 解析导数检验 (应为机器精度 ~1e-15) =====")
print(f"  左边界 du/dx (x=0):   max = {np.max(np.abs(du_dx[0, :])):.2e}")
print(f"  右边界 du/dx (x=L):   max = {np.max(np.abs(du_dx[-1, :])):.2e}")
print(f"  下边界 du/dy (y=0):   max = {np.max(np.abs(du_dy[:, 0])):.2e}")
print(f"  上边界 du/dy (y=L):   max = {np.max(np.abs(du_dy[:, -1])):.2e}")

# 对比：数值差分（有截断误差）
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

du_dx_num_left = (-3 * u0[0, :] + 4 * u0[1, :] - u0[2, :]) / (2 * dx)
du_dx_num_right = (3 * u0[-1, :] - 4 * u0[-2, :] + u0[-3, :]) / (2 * dx)

print("\n===== 数值差分检验 (有截断误差 O(h²)) =====")
print(f"  左边界 du/dx (数值):  max = {np.max(np.abs(du_dx_num_left)):.2e}")
print(f"  右边界 du/dx (数值):  max = {np.max(np.abs(du_dx_num_right)):.2e}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

x_plot = np.linspace(-1, 1, nx)
y_plot = np.linspace(-1, 1, ny)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot, indexing='ij')

im0 = axes[0].contourf(X_plot, Y_plot, u0, levels=50, cmap='RdBu_r')
axes[0].set_title('$u_0$')
axes[0].set_aspect('equal')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(X_plot, Y_plot, du_dx, levels=50, cmap='RdBu_r')
axes[1].set_title('$\\partial u / \\partial x$ (解析)')
axes[1].set_aspect('equal')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].contourf(X_plot, Y_plot, du_dy, levels=50, cmap='RdBu_r')
axes[2].set_title('$\\partial u / \\partial y$ (解析)')
axes[2].set_aspect('equal')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('grf_neumann_check.png', dpi=150)
plt.show()

