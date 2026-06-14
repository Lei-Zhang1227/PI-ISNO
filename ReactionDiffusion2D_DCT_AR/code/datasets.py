from torch.utils.data import Dataset
import h5py
import numpy as np
import torch


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
                 file_path='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                 initial_step=10,
                 sub_x=1,
                 sub_t=1,
                 if_test=False,
                 ):
        self.file_path = file_path
        self.initial_step = initial_step
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
                 file_path='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                 initial_step=1,
                 out_step=9,
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


class FNODatasetExtend(Dataset):
    """
    Extend 预测数据集：同时加载短期和长期数据集

    Long 数据集是从 short 测试集生成的：
    - long 的 sample_0000 ~ sample_0099 对应 short 的 seen 部分 (short[-200:-100])
    - long 的 sample_0100 ~ sample_0199 对应 short 的 unseen 部分 (short[-100:])
    """

    def __init__(self,
                 short_file='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                 long_file='/data/zhanglei/BurgersEquationII/2D_diff-react_t5_t15_101.h5',
                 initial_step=10,
                 sub_x=1,
                 sub_t=1,
                 mode='unseen',
                 ):
        self.short_file = short_file
        self.long_file = long_file
        self.initial_step = initial_step
        self.sub_x = sub_x
        self.sub_t = sub_t

        # 读取 short 数据集的 key 列表
        with h5py.File(self.short_file, 'r') as h5_file:
            all_keys_short = sorted([k for k in h5_file.keys() if k not in ['grid', 'params']])

        # long 数据集: sample_0000 ~ sample_0199，对应 short[-200:]
        # seen: sample_0000 ~ sample_0099 <-> short[-200:-100]
        # unseen: sample_0100 ~ sample_0199 <-> short[-100:]
        if mode == 'unseen':
            short_keys = all_keys_short[-100:]  # short 的最后 100 个
            long_indices = range(100, 200)  # long 的 sample_0100 ~ sample_0199
        elif mode == 'seen':
            short_keys = all_keys_short[-200:-100]  # short 的倒数 200~100
            long_indices = range(0, 100)  # long 的 sample_0000 ~ sample_0099
        else:
            raise ValueError(f"mode 必须是 'unseen' 或 'seen', 得到 {mode}")

        # 建立 (short_key, long_key) 对应列表
        self.data_list = [(sk, f'sample_{li:04d}') for sk, li in zip(short_keys, long_indices)]

        print(f"Extend数据集 ({mode}): {len(self.data_list)} 样本")
        print(f"  Short key 范围: {self.data_list[0][0]} ~ {self.data_list[-1][0]}")
        print(f"  Long key 范围: {self.data_list[0][1]} ~ {self.data_list[-1][1]}")

        # 验证衔接
        self._verify_connection()

    def _verify_connection(self, n_check=3):
        print("验证数据集衔接...")
        for i in range(min(n_check, len(self.data_list))):
            short_key, long_key = self.data_list[i]
            with h5py.File(self.short_file, 'r') as h5_short:
                data_short = np.array(h5_short[short_key]["data"], dtype='f')
            with h5py.File(self.long_file, 'r') as h5_long:
                data_long = np.array(h5_long[long_key]["data"], dtype='f')

            print(f"  short shape: {data_short.shape}, long shape: {data_long.shape}")

            # short: [t, x, y, 2], long: [x, y, t, 2] ?
            # 根据实际 shape 调整索引
            if data_long.shape[2] > data_long.shape[0]:  # long 是 [x, y, t, 2]
                diff = np.abs(data_short[-1] - data_long[:, :, 0, :]).max()
            else:  # long 是 [t, x, y, 2]
                diff = np.abs(data_short[-1] - data_long[0]).max()

            print(f"  {short_key} <-> {long_key}: max diff = {diff:.2e}")
            assert diff < 1e-5, f"样本 {short_key} <-> {long_key} 衔接不一致! diff={diff}"
        print("衔接验证通过 ✓")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        short_key, long_key = self.data_list[idx]

        with h5py.File(self.short_file, 'r') as h5_file:
            seed_group = h5_file[short_key]
            data_short = torch.tensor(np.array(seed_group["data"], dtype='f'), dtype=torch.float)  # [t, x, y, 2]
            x_grid = torch.tensor(np.array(seed_group["grid"]["x"], dtype='f'), dtype=torch.float)
            y_grid = torch.tensor(np.array(seed_group["grid"]["y"], dtype='f'), dtype=torch.float)

        with h5py.File(self.long_file, 'r') as h5_file:
            data_long = torch.tensor(np.array(h5_file[long_key]["data"], dtype='f'), dtype=torch.float)  # [x, y, t, 2]

        # short: [t, x, y, 2] -> [x, y, t, 2]
        data_short = data_short.permute(1, 2, 0, 3)

        # long 已经是 [x, y, t, 2]，不需要 permute

        # 下采样
        data_short = data_short[::self.sub_x, ::self.sub_x, ::self.sub_t, :]
        data_long = data_long[::self.sub_x, ::self.sub_x, ::self.sub_t, :]
        x_grid = x_grid[::self.sub_x]
        y_grid = y_grid[::self.sub_x]

        # 初始条件
        xx = data_short[:, :, :self.initial_step, :]

        # 构建空间网格
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        grid = torch.stack([X, Y], dim=-1)

        return xx, data_short, data_long, grid


class FNODatasetMult_B(Dataset):
    """
    通用 PDE 数据集

    数据格式: [t, x, y, v] = [101, 128, 128, 2]

    返回:
        xx: [x, y, initial_step, v] 初始条件
        yy: [x, y, t, v] 完整时间序列
        grid: [x, y, 2] 空间网格 (meshgrid of x, y)
    """

    def __init__(self,
                 file_path='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                 initial_step=10,
                 sub_x=1,
                 sub_t=1,
                 if_test=False,
                 ):
        self.file_path = file_path
        self.initial_step = initial_step
        self.sub_x = sub_x
        self.sub_t = sub_t

        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
            # 过滤掉非数据键
            self.data_list = [k for k in data_list if k not in ['grid', 'params']]

        if if_test:
            self.data_list = self.data_list[100:]
        else:
            self.data_list = self.data_list[:100]

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