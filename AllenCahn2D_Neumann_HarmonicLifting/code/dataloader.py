from torch.utils.data import Dataset
import torch
import h5py
import numpy as np


class FNODatasetMult(Dataset):
    """
    2D Heat Equation 数据集 (支持 per-sample 随机非齐次 Neumann BC)

    数据格式: [t, x, y, v] = [501, 129, 129, 1]

    返回:
        xx: [x, y, initial_step, v] 初始条件
        yy: [x, y, t, v] 完整时间序列
        grid: [x, y, 2] 空间网格 (meshgrid of x, y)
        bc_params: [4] 边界条件参数 (a, b, c, d)
    """

    def __init__(self,
                 file_path='/data/zhanglei/BurgersEquationII/burgers2d_spectral.h5',
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
            self.data_list = [k for k in data_list if k not in ['grid', 'params']]

            # 检测是否有 per-sample BC
            first_key = self.data_list[0]
            self.has_bc = 'bc' in h5_file[first_key]

        if if_test:
            self.data_list = self.data_list[:100]
        else:
            self.data_list = self.data_list[100:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            data = torch.tensor(np.array(seed_group["data"], dtype='f'), dtype=torch.float)

            x_grid = torch.tensor(np.array(seed_group["grid"]["x"], dtype='f'), dtype=torch.float)
            y_grid = torch.tensor(np.array(seed_group["grid"]["y"], dtype='f'), dtype=torch.float)

            # 读取 per-sample BC 参数
            if self.has_bc:
                a = float(seed_group['bc']['a'][()])
                b = float(seed_group['bc']['b'][()])
                c = float(seed_group['bc']['c'][()])
                d = float(seed_group['bc']['d'][()])
                bc_params = torch.tensor([a, b, c, d], dtype=torch.float)
            else:
                # 兼容旧数据集: 固定 BC (a=1, b=-1, c=-1, d=1)
                bc_params = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float)

        # [t, x, y, v] -> [x, y, t, v]
        data = data.permute(1, 2, 0, 3)
#         print('data.shape:',data.shape)
        data = data[..., :self.full_step, :]
#         print('data.shape after t slip:',data.shape)
        # 下采样
        data = data[::self.sub_x, ::self.sub_x, ::self.sub_t, :]
        x_grid = x_grid[::self.sub_x]
        y_grid = y_grid[::self.sub_x]

        xx = data[:, :, :self.initial_step, :]
        yy = data

        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        grid = torch.stack([X, Y], dim=-1)

        return xx, yy, grid, bc_params


class FNODatasetMult_test(Dataset):
    """
    测试用数据集 (支持 per-sample 随机非齐次 Neumann BC)
    """

    def __init__(self,
                 file_path='/data/zhanglei/BurgersEquationII/burgers2d_spectral.h5',
                 initial_step=5,
                 full_step=501,
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
            self.data_list = [k for k in data_list if k not in ['grid', 'params']]

            first_key = self.data_list[0]
            self.has_bc = 'bc' in h5_file[first_key]

        if if_test:
            self.data_list = self.data_list[:100]
        else:
            self.data_list = self.data_list[100:200]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            data = torch.tensor(np.array(seed_group["data"], dtype='f'), dtype=torch.float)

            x_grid = torch.tensor(np.array(seed_group["grid"]["x"], dtype='f'), dtype=torch.float)
            y_grid = torch.tensor(np.array(seed_group["grid"]["y"], dtype='f'), dtype=torch.float)

            if self.has_bc:
                a = float(seed_group['bc']['a'][()])
                b = float(seed_group['bc']['b'][()])
                c = float(seed_group['bc']['c'][()])
                d = float(seed_group['bc']['d'][()])
                bc_params = torch.tensor([a, b, c, d], dtype=torch.float)
            else:
                bc_params = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float)

        data = data.permute(1, 2, 0, 3)
        data = data[..., :self.full_step, :]
        data = data[::self.sub_x, ::self.sub_x, ::self.sub_t, :]
        x_grid = x_grid[::self.sub_x]
        y_grid = y_grid[::self.sub_x]

        xx = data[:, :, :self.initial_step, :]
        yy = data

        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        grid = torch.stack([X, Y], dim=-1)

        return xx, yy, grid, bc_params


class FNODatasetMult_A(Dataset):
    """
    自回归训练数据集 (支持 per-sample 随机非齐次 Neumann BC)
    """

    def __init__(self,
                 file_path='/data/zhanglei/BurgersEquationII/burgers2d_spectral.h5',
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
            self.data_list = [k for k in data_list if k not in ['grid', 'params']]

            first_key = self.data_list[0]
            self.has_bc = 'bc' in h5_file[first_key]

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

            data = torch.tensor(np.array(seed_group["data"], dtype='f'), dtype=torch.float)

            if self.has_bc:
                a = float(seed_group['bc']['a'][()])
                b = float(seed_group['bc']['b'][()])
                c = float(seed_group['bc']['c'][()])
                d = float(seed_group['bc']['d'][()])
                bc_params = torch.tensor([a, b, c, d], dtype=torch.float)
            else:
                bc_params = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float)

        data = data.permute(1, 2, 0, 3)
        data = data[::self.sub_x, ::self.sub_x, ::self.sub_t, :]

        xx = data[:, :, :self.initial_step, :]
        nx, ny = xx.shape[0], xx.shape[1]
        xx = xx.reshape(nx, ny, -1)
        yy = data[:, :, self.initial_step:self.initial_step + self.out_step, :]

        return xx, yy, bc_params