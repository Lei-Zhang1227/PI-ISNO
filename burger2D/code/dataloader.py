from torch.utils.data import Dataset
import torch
import h5py
import numpy as np


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
                 file_path='../data/burgers2d_spectral.h5',
                 initial_step=5,
                 sub_x=1,
                 sub_t=2,
                 if_test=False,
                 ):
        self.file_path = file_path
        self.initial_step = initial_step
        self.sub_x = sub_x
        self.sub_t = sub_t

        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
            # 过滤掉非数据键
            self.data_list = [k for k in data_list if k not in ['grid', 'params','meta']]

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
                 file_path='../data/burgers2d_spectral.h5',
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

    # 使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filepath = '../data/burgers2d_spectral.h5'
    test_data = FNODatasetMult(file_path=filepath,
                               initial_step=5,
                               sub_x=1,
                               sub_t=2,
                               if_test=False,
                               )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

    # 训练循环中
    for xx, yy, grid in test_loader:
        print(xx.shape)
        print(yy.shape)
        print(grid.shape)
        pred = yy.to(device)
        print(f'pred.shape:{pred.shape}')
        # pred_permuted = pred.permute(0, 3, 1, 2,4)
        # print('pred_permuted.shape:', pred_permuted.shape)
        residual = compute_burgers2d_residual_batch(pred, nu=0.1, Lx=2.0, Ly=2.0)[:, 1:-1, 1:-1, 1:]
        results = {
            'pde_residual_l2': torch.sqrt(torch.mean(residual ** 2)),
            'pde_residual_mse:': torch.mean(residual ** 2),
            'pde_residual_max': torch.max(torch.abs(residual)),
            'pde_residual_mean': torch.mean(torch.abs(residual)),
        }
        print(results)

        time_indices = [0, 1, 2, 3, 9, 19, 39, 46]
        plot_residual_at_times(residual, time_indices=time_indices, save_path='./residual_times.png')
        break
