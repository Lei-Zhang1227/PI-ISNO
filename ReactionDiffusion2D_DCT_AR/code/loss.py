import torch.nn.functional as F
from datasets import *

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReactionDiffusionResidual(nn.Module):
    """
    预编译的反应扩散方程残差计算器
    """

    def __init__(self, dx=0.015625, dt=5 / 100, du=0.001, dv=0.005, k=0.005, dtype=torch.float32):
        super().__init__()

        # 物理参数
        self.dx = dx
        self.dt = dt
        self.du = du
        self.dv = dv
        self.k = k

        # 预定义拉普拉斯卷积核（register_buffer 不参与训练，但会随模型移动到正确设备）
        laplace_kernel = torch.tensor([[[[0., 1., 0.],
                                         [1., -4., 1.],
                                         [0., 1., 0.]]]], dtype=dtype)
        self.register_buffer('laplace_kernel', laplace_kernel)

        # 预计算常数
        self.register_buffer('dx2_inv', torch.tensor(1.0 / (dx ** 2), dtype=dtype))
        self.register_buffer('dt2_inv', torch.tensor(1.0 / (2 * dt), dtype=dtype))

    def forward(self, yy):
        """
        参数:
            yy: [batch, nx, ny, nt, 2]
        返回:
            f_u, f_v: [batch, nx-2, ny-2, nt-2]
        """
        batch, nx, ny, nt, _ = yy.shape

        u = yy[..., 0]  # [batch, nx, ny, nt]
        v = yy[..., 1]

        # ============ 空间拉普拉斯 ============
        # 合并 u, v 一起卷积，减少 kernel launch 次数
        uv_for_conv = torch.stack([u, v], dim=1)  # [batch, 2, nx, ny, nt]
        uv_for_conv = uv_for_conv.permute(0, 4, 1, 2, 3).reshape(batch * nt * 2, 1, nx, ny)

        # 单次卷积计算 u 和 v 的拉普拉斯
        laplace_uv = F.conv2d(uv_for_conv, self.laplace_kernel, padding=0)  # [batch*nt*2, 1, nx-2, ny-2]
        laplace_uv = laplace_uv.view(batch, nt, 2, nx - 2, ny - 2)

        # 分离并调整维度
        laplace_u = laplace_uv[:, 1:-1, 0].permute(0, 2, 3, 1) * self.dx2_inv  # [batch, nx-2, ny-2, nt-2]
        laplace_v = laplace_uv[:, 1:-1, 1].permute(0, 2, 3, 1) * self.dx2_inv

        # ============ 时间导数 ============
        u_inner = u[:, 1:-1, 1:-1, :]
        v_inner = v[:, 1:-1, 1:-1, :]

        u_t = (u_inner[..., 2:] - u_inner[..., :-2]) * self.dt2_inv
        v_t = (v_inner[..., 2:] - v_inner[..., :-2]) * self.dt2_inv

        # ============ 反应项 ============
        u_mid = u_inner[..., 1:-1]
        v_mid = v_inner[..., 1:-1]

        # 反应项可以用 fused 操作
        Ru = u_mid - u_mid.pow(3) - self.k - v_mid
        Rv = u_mid - v_mid

        # ============ 残差 ============
        f_u = u_t - self.du * laplace_u - Ru
        f_v = v_t - self.dv * laplace_v - Rv

        return f_u, f_v


if __name__ == '__main__':
    import h5py
    import numpy as np
    from loss import *
    import matplotlib.pyplot as pltimport
    import numpy as np
    import matplotlib.pyplot as plt

    # 使用

    filepath = 'F:\data\OLdata/2D_diff-react_NA_NA.h5'
    test_data = FNODatasetMult(file_path=filepath,
                               initial_step=10,
                               sub_x=1,
                               sub_t=1,
                               if_test=False,
                               )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)
    residual_calculator = ReactionDiffusionResidual(
        dx=0.015625, dt=5 / 100, du=0.001, dv=0.005, k=0.005,
        dtype=torch.float32
    ).to(device)

    # 训练循环中
    for xx, yy, grid in test_loader:
        print(xx.shape)
        print(yy.shape)
        pred_permuted = yy.to(device)
        print('output_permuted.shape:', pred_permuted.shape)
        dt = 5 / 100
        dx = grid[0, 1, 0, 0] - grid[0, 0, 0, 0]
        print(f'dx is {dx}')
        f_u, f_v = residual_calculator(yy)
        print('residual.shape:', f_u.shape)
        print(torch.mean(abs(f_u[..., 8:])))
        print(torch.mean(abs(f_v[..., 8:])))

        print(torch.mean(abs(f_u[..., 5:])))
        print(torch.mean(abs(f_v[..., 5:])))

        print(torch.mean(abs(f_u[..., 3:])))
        print(torch.mean(abs(f_v[..., 3:])))

        print(torch.mean(abs(f_u)))
        print(torch.mean(abs(f_v)))
