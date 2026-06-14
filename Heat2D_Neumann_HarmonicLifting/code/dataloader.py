from torch.utils.data import Dataset
import torch
import h5py
import numpy as np
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]


class FNODatasetMult(Dataset):
    """
    通用 PDE 数据集

    数据格式: [t, x, y, v] = [101, 128, 128, 2]

    返回:
        xx: [x, y, initial_step, v] 初始条件
        yy: [x, y, t, v] 完整时间序列
        grid: [x, y, 2] 空间网格 (meshgrid of x, y)
    """

    def __init__(self,
                 file_path=str(CASE_ROOT / 'data' / 'heat2d_neumann_1100.h5'),
                 initial_step=5,
                 full_step=101,
                 sub_x=2,
                 sub_t=2,
                 if_test=False,
                 ):
        self.file_path = file_path
        self.initial_step = initial_step
        self.full_step = full_step
        self.sub_x = sub_x
        self.sub_t = sub_t

        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
            # 过滤掉非数据键
            self.data_list = [k for k in data_list if k not in ['grid', 'params']]

        # test_idx = int(len(self.data_list) * (1 - 0.1))
        if if_test:
            self.data_list = self.data_list[:100]
        else:
            self.data_list = self.data_list[100:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # 读取数据 [t, x, y, v] = [101, 128, 128, 2]
            data = torch.tensor(np.array(seed_group["data"], dtype='f'), dtype=torch.float)

            # 读取空间网格
            x_grid = torch.tensor(np.array(seed_group["grid"]["x"], dtype='f'), dtype=torch.float)
            y_grid = torch.tensor(np.array(seed_group["grid"]["y"], dtype='f'), dtype=torch.float)

        # 转换为 [x, y, t, v]
        data = data.permute(1, 2, 0, 3)  # [t, x, y, v] -> [x, y, t, v]
        data = data[..., :self.full_step, :]
        # 下采样
        data = data[::self.sub_x, ::self.sub_x, ::self.sub_t, :]
        x_grid = x_grid[::self.sub_x]
        y_grid = y_grid[::self.sub_x]

        # 分离初始条件和完整序列
        xx = data[:, :, :self.initial_step, :]  # [x, y, initial_step, v]
        yy = data  # [x, y, t, v]

        # 构建空间网格 meshgrid
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        grid = torch.stack([X, Y], dim=-1)  # [x, y, 2]

        return xx, yy, grid


class FNODatasetMult_A(Dataset):
    """
    通用 PDE 数据集

    数据格式: [t, x, y, v] = [101, 128, 128, 2]

    返回:
        xx: [x, y, initial_step, v] 初始条件
        yy: [x, y, t, v] 完整时间序列
        grid: [x, y, 2] 空间网格 (meshgrid of x, y)
    """

    def __init__(self,
                 file_path=str(CASE_ROOT / 'data' / 'heat2d_neumann_1100.h5'),
                 initial_step=1,
                 out_step=5,
                 sub_x=1,
                 sub_t=1,
                 if_test=False,
                 ):
        self.file_path = file_path
        self.initial_step = initial_step
        self.out_step = out_step
        self.sub_x = sub_x
        self.sub_t = sub_t

        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
            # 过滤掉非数据键
            self.data_list = [k for k in data_list if k not in ['grid', 'params']]

        test_idx = int(len(self.data_list) * (1 - 0.1))
        if if_test:
            self.data_list = self.data_list[test_idx:]
        else:
            self.data_list = self.data_list[:test_idx]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # 读取数据 [t, x, y, v] = [101, 128, 128, 2]
            data = torch.tensor(np.array(seed_group["data"], dtype='f'), dtype=torch.float)

        # 转换为 [x, y, t, v]
        data = data.permute(1, 2, 0, 3)  # [t, x, y, v] -> [x, y, t, v]

        # 下采样
        data = data[::self.sub_x, ::self.sub_x, ::self.sub_t, :]
        # 分离初始条件和完整序列
        xx = data[:, :, :self.initial_step, :]  # [x, y, initial_step, v]
        nx, ny = xx.shape[0], xx.shape[1]
        xx = xx.reshape(nx, ny, -1)  # [nx, ny, initial_step*2]
        # yy: [x, y, out_step, 2] 从 initial_step 开始取 out_step 个时间步
        yy = data[:, :, self.initial_step:self.initial_step + self.out_step, :]
        return xx, yy


def plot_residual_at_times(residual, time_indices=[0, 1, 2, 3, 9, 19, 39, 50], save_path=None):
    """
    绘制指定时间点的残差图

    参数:
        residual: [batch, Nx, Ny, Nt, 1] 或 [1, 64, 64, 51, 1]
        time_indices: 要绘制的时间索引列表 (0-indexed)
        save_path: 保存路径
    """
    # 处理维度
    res = residual.squeeze()  # 去掉 batch 和最后一维
    if res.ndim == 4:
        res = res[0]  # 取第一个 batch

    # res: [Nx, Ny, Nt] = [64, 64, 51]

    # 转 numpy
    if hasattr(res, 'cpu'):
        res = res.cpu().numpy()

    Nx, Ny, Nt = res.shape

    # 过滤有效索引
    valid_indices = [t for t in time_indices if 0 <= t < Nt]
    n_plots = len(valid_indices)

    # 计算子图布局
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, t_idx in enumerate(valid_indices):
        ax = axes[i]
        res_t = res[:, :, t_idx]  # [Nx, Ny]

        # 统计量
        res_max = np.max(np.abs(res_t))
        res_mean = np.mean(np.abs(res_t))

        # 每个子图独立色标
        vmax = res_max
        vmin = -vmax

        im = ax.imshow(res_t.T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='RdBu_r', vmin=vmin, vmax=vmax)

        ax.set_title(f't_idx={t_idx + 1}\nmax={res_max:.2e}, mean={res_mean:.2e}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 隐藏多余子图
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    # 计算整体统计
    total_mean = np.mean(np.abs(res))
    total_max = np.max(np.abs(res))

    plt.suptitle(f'PDE Residual (shape: {Nx}×{Ny}×{Nt})\n'
                 f'Overall: mean={total_mean:.2e}, max={total_max:.2e}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == '__main__':
    import h5py
    import numpy as np
    from loss import *
    import matplotlib.pyplot as pltimport
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    # 使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filepath = r"./data/heat2d_neumann.h5"
    test_data = FNODatasetMult(file_path=filepath,
                               initial_step=5,
                               sub_x=2,
                               sub_t=2,
                               if_test=False,
                               )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)
    for xx, yy, grid in test_loader:
        print(xx.shape)
        print(yy.shape)
        print(grid.shape)
        pred = yy.to(device)
        print(f'pred.shape:{pred.shape}')

        if device.type == 'cuda':
            _ = compute_heat2d_residual_batch(pred, kappa=0.02, T=1)
            torch.cuda.synchronize()

            # 计时
        n_runs = 20
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_runs):
            residual = compute_heat2d_residual_batch(pred, kappa=0.02, T=1)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        t1 = time.perf_counter()
        avg_ms = (t1 - t0) / n_runs * 1000
        print(f"数据形状: {pred.shape}")
        print(f"设备: {device}")
        print(f"平均耗时: {avg_ms:.2f} ms ({n_runs} 次)")

        time_indices = [0, 1, 2, 3, 9, 19, 39, 46]
        plot_residual_at_times(residual, time_indices=time_indices, save_path='./residual_times.png')

        # ========== 绘制解场的时间演化 ==========
        # yy: [batch, Nx, Ny, Nt, v]
        sol = yy[0, :, :, :, 0].cpu().numpy()  # 取第一个batch、第一个变量 [Nx, Ny, Nt]
        Nx, Ny, Nt = sol.shape

        # 从grid获取坐标范围
        x_coords = grid[0, :, 0, 0].cpu().numpy()
        y_coords = grid[0, 0, :, 1].cpu().numpy()
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        # 选择要绘制的时间步
        plot_times = [0, 1, 4, 10, 20, 30, 40, Nt - 1]
        plot_times = [t for t in plot_times if t < Nt]
        n_plots = len(plot_times)
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        # 全局色标范围
        vmin, vmax = sol.min(), sol.max()

        for i, t_idx in enumerate(plot_times):
            ax = axes[i]
            field = sol[:, :, t_idx]
            im = ax.imshow(field.T, origin='lower', extent=extent,
                           cmap='RdBu_r', vmin=vmin, vmax=vmax)
            label = f'IC (t_idx={t_idx})' if t_idx < xx.shape[3] else f't_idx={t_idx}'
            ax.set_title(label, fontsize=12)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, fraction=0.046)

        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Solution Field Evolution (var=0, shape: {Nx}×{Ny}×{Nt})', fontsize=14)
        plt.tight_layout()
        plt.savefig('./solution_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 如果有第二个变量，也画一张
        if yy.shape[-1] > 1:
            sol2 = yy[0, :, :, :, 1].cpu().numpy()
            vmin2, vmax2 = sol2.min(), sol2.max()

            fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes2 = axes2.flatten()

            for i, t_idx in enumerate(plot_times):
                ax = axes2[i]
                field = sol2[:, :, t_idx]
                im = ax.imshow(field.T, origin='lower', extent=extent,
                               cmap='RdBu_r', vmin=vmin2, vmax=vmax2)
                label = f'IC (t_idx={t_idx})' if t_idx < xx.shape[3] else f't_idx={t_idx}'
                ax.set_title(label, fontsize=12)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                plt.colorbar(im, ax=ax, fraction=0.046)

            for i in range(n_plots, len(axes2)):
                axes2[i].axis('off')

            plt.suptitle(f'Solution Field Evolution (var=1, shape: {Nx}×{Ny}×{Nt})', fontsize=14)
            plt.tight_layout()
            plt.savefig('./solution_evolution_var1.png', dpi=150, bbox_inches='tight')
            plt.show()

        break
