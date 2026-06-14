import scipy.io
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from loss import *
import h5py


class h5DatasetFor1DHeat_2D(Dataset):
    '''
    2D 时空数据集
    输入形状为 [nx, nt, channel]，channel = initial_step + 2 (u0, x, t)
    输出形状为 [nx, nt, 1]
    '''

    def __init__(self, filepath, initial_step=1, sub_t=1, sub_x=1, if_test=False):
        self.sub_t = sub_t
        self.sub_x = sub_x
        self.file_path = filepath
        self.initial_step = initial_step

        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
            data_list = [k for k in data_list if k.startswith('sample')]

            self.x_grid_full = torch.tensor(h5_file['grid/x'][:].flatten(), dtype=torch.float32)
            self.t_grid_full = torch.tensor(h5_file['grid/t'][:].flatten(), dtype=torch.float32)

        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            data = torch.tensor(seed_group['data'][:], dtype=torch.float32)  # [nt, nx] 原始

            # 转置为 [nx, nt]
            data = data.T  # [nx, nt]

            # 下采样
            data = data[::self.sub_x, ::self.sub_t]  # [nx_sub, nt_sub]
            nx, nt = data.shape

            # 网格下采样
            x_grid = self.x_grid_full[::self.sub_x]  # [nx_sub]
            t_grid = self.t_grid_full[::self.sub_t]  # [nt_sub]

            # 输出: [nx, nt, 1]
            ys = data.unsqueeze(-1)

            # 扩展网格到 [nx, nt, 1]
            gridx_expanded = x_grid.view(-1, 1, 1).expand(-1, nt, -1)  # [nx, nt, 1]
            gridt_expanded = t_grid.view(1, -1, 1).expand(nx, -1, -1)  # [nx, nt, 1]

            # 初始条件: 取 t=0 时刻的数据
            initial_data = data[:, 0:self.initial_step]  # [nx, initial_step]
            initial_expanded = initial_data.unsqueeze(1).expand(-1, nt, -1)  # [nx, nt, initial_step]

            # 拼接输入: [nx, nt, initial_step + 2]
            xs = torch.cat([initial_expanded, gridx_expanded, gridt_expanded], dim=-1)

            name = self.data_list[idx]

        return xs, ys, name


class h5DatasetFor1DHeat(Dataset):
    '''
    用于自回归训练的数据集
    1. 读取的文件类型是.h5， 用key来标注sample，从0000到0999一共1000个，具体数据格式见数据文件夹里的info
    2. 这个类给出了初始条件、完整数据和网格数据; 这里的初始条件的步数是通过initial_step来控制的，也就是用initial_step个时间步的数据来预测后后面时刻的值；
    3. 和I不同的是，用于自回归模型的数据集不再在nt上延伸复制，也就不是构造为二维的数据。
    4. 此时不再考虑时间维度；
    5. 返回的两个数据Xs:输入数据，形状为[nx,initial_step*channel],ys:[nt,nx]
    '''

    def __init__(self, filepath,
                 initial_step=10,
                 sub_t=1,
                 sub_x=1,
                 if_test=False
                 ):
        # Define path to files
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())[2:]
            # print(data_list)
            self.x_grid = torch.tensor(h5_file['grid/x'][:], dtype=torch.float64).reshape(513, 1)  # [nx]
            # print(self.x_grid)
        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        就是一个一个的取呗，所以grid也不需要repeat
        '''
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            # data dim = [513,401]
            # print("sample_0001 的 keys:", list(seed_group.keys()))
            data = seed_group['data'][:]
            # 类型转换
            data = torch.tensor(data, dtype=torch.float64)  # data.shape: torch.Size([401, 513])
            # print('data.shape:', data.shape)
            #             print(f'1.data.shape:{data.shape}')
            ys = data[::self.sub_t, ::self.sub_x].permute(1, 0).unsqueeze(-1)  # [nx, nt,1]
            #             print(f'ys.shape:{ys.shape}')
            data = data.unsqueeze(-1)  # [nt, nx,1]
            #             print(f'2.data.shape:{data.shape}')
            data = data[::self.sub_t, ::self.sub_x]
            #             print(f'3.data.shape:{data.shape}')
            grid = self.x_grid[::self.sub_x, :]  # [nx,1]
            Xs = data[0:self.initial_step, :, :]  # [self.initial_ste,nx,1]
            #             print(f'Xs.shape:{ys.shape}')
            xs = Xs.permute(1, 0, 2)  # [nx, self.initial_step,1]
            name = self.data_list[idx]
        return xs, ys, grid, name


class h5DatasetFor1DHeat_uni(Dataset):
    '''
    用于自回归训练的数据集
    1. 读取的文件类型是.h5， 用key来标注sample，从0000到0999一共1000个，具体数据格式见数据文件夹里的info
    2. 这个类给出了初始条件、完整数据和网格数据; 这里的初始条件的步数是通过initial_step来控制的，也就是用initial_step个时间步的数据来预测后后面时刻的值；
    3. 和I不同的是，用于自回归模型的数据集不再在nt上延伸复制，也就不是构造为二维的数据。
    4. 此时不再考虑时间维度；
    5. 返回的两个数据Xs:输入数据，形状为[nx,initial_step*channel],ys:[nt,nx]
    '''

    def __init__(self, filepath,
                 initial_step=10,
                 sub_t=1,
                 sub_x=1,
                 if_test=False
                 ):
        # Define path to files
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())[2:]
            # print(data_list)
            self.x_grid = torch.tensor(h5_file['grid/x'][:], dtype=torch.float64).reshape(512, 1)  # [nx]
            # print(self.x_grid)
        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        就是一个一个的取呗，所以grid也不需要repeat
        '''
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            # data dim = [513,401]
            # print("sample_0001 的 keys:", list(seed_group.keys()))
            data = seed_group['data'][:]
            # 类型转换
            data = torch.tensor(data, dtype=torch.float64)  # data.shape: torch.Size([401, 513])
            # print('data.shape:', data.shape)
            #             print(f'1.data.shape:{data.shape}')
            ys = data[::self.sub_t, ::self.sub_x].permute(1, 0).unsqueeze(-1)  # [nx, nt,1]
            #             print(f'ys.shape:{ys.shape}')
            data = data.unsqueeze(-1)  # [nt, nx,1]
            #             print(f'2.data.shape:{data.shape}')
            data = data[::self.sub_t, ::self.sub_x]
            #             print(f'3.data.shape:{data.shape}')
            grid = self.x_grid[::self.sub_x, :]  # [nx,1]
            Xs = data[0:self.initial_step, :, :]  # [self.initial_ste,nx,1]
            #             print(f'Xs.shape:{ys.shape}')
            xs = Xs.permute(1, 0, 2)  # [nx, self.initial_step,1]
            name = self.data_list[idx]
        return xs, ys, grid, name


class h5DatasetFor1DHeat_A(Dataset):
    '''
    用于自回归训练的数据集

    返回:
        xx: [nx, 2] 初始条件(第一个时间步) + 位置坐标 (u0, x)
        yy: [nx, n_output_steps] 除初始条件外前n_output_steps个时间步的值
        grid: [nx, 1] 空间网格
        name: 样本名称
    '''

    def __init__(self, filepath,
                 sub_t=1,
                 sub_x=1,
                 n_output_steps=4, initial_step=1, if_test=False,  # 输出时间步数
                 ):
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x
        self.n_output_steps = n_output_steps
        self.initial_step = initial_step

        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())[2:]
            self.x_grid = torch.tensor(h5_file['grid/x'][:], dtype=torch.float32).reshape(-1, 1)  # [nx, 1]

        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            # data dim = [nx, nt] = [513, 401]
            data = torch.tensor(seed_group['data'][:], dtype=torch.float64)

            # 下采样: [nx, nt] -> [nx_sub, nt_sub]
        data = data[::self.sub_x, ::self.sub_t]  # 注意顺序：先 x 后 t
        grid = self.x_grid[::self.sub_x]  # [nx_sub, 1]

        # 输入: 第一个时间步 + 位置坐标
        u0 = data[:, 0:1]  # [nx, 1] 取第一个时间步
        xx = torch.cat([u0, grid], dim=-1)  # [nx, 2]

        # 输出: 除初始条件外的前 n_output_steps 个时间步
        yy = data[:, 1:1 + self.n_output_steps]  # [nx, n_output_steps]

        name = self.data_list[idx]

        return xx, yy, name


class h5DatasetFor1DHeat_TwoStage(Dataset):
    '''
    用于两阶段模型评估的数据集

    返回:
        xx: [nx, 2] 初始条件(第一个时间步) + 位置坐标 (u0, x)，用于Model A输入
        yy: [nx, nt, 1] 完整时间序列，用于评估
        grid: [nx, 1] 空间网格
        name: 样本名称
    '''

    def __init__(self, filepath,
                 sub_t=1,
                 sub_x=1,
                 if_test=False,
                 ):
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x

        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())[2:]
            self.x_grid = torch.tensor(h5_file['grid/x'][:], dtype=torch.float64).reshape(-1, 1)  # [nx, 1]

        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            data = torch.tensor(seed_group['data'][:], dtype=torch.float64)

        # 下采样: [nx, nt] -> [nx_sub, nt_sub]
        data = data[::self.sub_x, ::self.sub_t]
        grid = self.x_grid[::self.sub_x]  # [nx_sub, 1]

        # Model A 输入: 第一个时间步 + 位置坐标
        u0 = data[:, 0:1]  # [nx, 1]
        xx = torch.cat([u0, grid], dim=-1)  # [nx, 2]

        # 完整序列用于评估
        yy = data.unsqueeze(-1)  # [nx, nt, 1]

        name = self.data_list[idx]

        return xx, yy, grid, name


class h5DatasetFor1DHeat_extend(Dataset):
    '''
    用于自回归训练的数据集
    1. 读取的文件类型是.h5， 用key来标注sample，从0000到0999一共1000个，具体数据格式见数据文件夹里的info
    2. 这个类给出了初始条件、完整数据和网格数据; 这里的初始条件的步数是通过initial_step来控制的，也就是用initial_step个时间步的数据来预测后后面时刻的值；
    3. 和I不同的是，用于自回归模型的数据集不再在nt上延伸复制，也就不是构造为二维的数据。
    4. 此时不再考虑时间维度；
    5. 返回的两个数据Xs:输入数据，形状为[nx,initial_step*channel],ys:[nt,nx]
    '''

    def __init__(self, filepath,
                 initial_step=10,
                 sub_t=1,
                 sub_x=1,
                 unseen=True
                 ):
        # Define path to files
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())[2:]
            # print(data_list)
            self.x_grid = torch.tensor(h5_file['grid/x'][:], dtype=torch.float64).reshape(513, 1)  # [nx]
            # print(self.x_grid)
        if unseen:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:200])
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        就是一个一个的取呗，所以grid也不需要repeat
        '''
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            # data dim = [513,401]
            # print("sample_0001 的 keys:", list(seed_group.keys()))
            data = seed_group['data'][:]
            # 类型转换
            data = torch.tensor(data, dtype=torch.float64)  # data.shape: torch.Size([401, 513])
            # print('data.shape:', data.shape)
            #             print(f'1.data.shape:{data.shape}')
            ys = data[::self.sub_t, ::self.sub_x].permute(1, 0).unsqueeze(-1)  # [nx, nt,1]
            #             print(f'ys.shape:{ys.shape}')
            data = data.unsqueeze(-1)  # [nt, nx,1]
            #             print(f'2.data.shape:{data.shape}')
            data = data[::self.sub_t, ::self.sub_x]
            #             print(f'3.data.shape:{data.shape}')
            grid = self.x_grid[::self.sub_x, :]  # [nx,1]
            Xs = data[0:self.initial_step, :, :]  # [self.initial_ste,nx,1]
            #             print(f'Xs.shape:{ys.shape}')
            xs = Xs.permute(1, 0, 2)  # [nx, self.initial_step,1]
            name = self.data_list[idx]
        return xs, ys, grid, name


def cheb_diff_matrix(N, domain=[-1, 1]):
    """
    生成切比雪夫一阶微分矩阵

    参数:
        N: 节点数
        domain: [a, b] 定义域

    返回:
        D: [N, N] 微分矩阵
    """
    k = torch.arange(N, dtype=torch.float32)
    x = torch.cos(torch.pi * k / (N - 1))  # 标准 CGL 节点 [1, -1]

    c = torch.ones(N, dtype=torch.float32)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** k)

    X = x.unsqueeze(1).expand(N, N)
    dX = X - X.T
    D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + torch.eye(N, dtype=torch.float32))
    D = D - torch.diag(D.sum(dim=1))

    # 缩放到 [a, b]
    # 标准节点 x ∈ [1, -1] 映射到 t ∈ [a, b]: t = (a+b)/2 - (b-a)/2 * x
    # dt/dx = -(b-a)/2
    # df/dt = df/dx * dx/dt = df/dx * (-2/(b-a))
    a, b = domain
    D = D * (-2 / (b - a))

    return D


def get_heat_matrices(nx, nt, x_domain=[-1, 1], t_domain=[0, 1], device='cpu'):
    """
    获取热方程所需的微分矩阵
    """
    # 空间微分 [-1, 1]
    Dx = cheb_diff_matrix(nx, domain=x_domain).to(device)
    D2_x = Dx @ Dx

    # 时间微分 [0, 1]
    Dt = cheb_diff_matrix(nt, domain=t_domain).to(device)

    return D2_x, Dt


def compute_heat_residual(u, D2_x, Dt, nu=0.02):
    """
    计算热方程残差: u_t - nu * u_xx

    参数:
        u: [batch, nx, nt, 1]
        D2_x: 空间二阶微分矩阵 [nx, nx]
        Dt: 时间一阶微分矩阵 [nt, nt]
        nu: 扩散系数

    返回:
        residual: [batch, nx, nt, 1]
    """
    u_squeezed = u.squeeze(-1)  # [batch, nx, nt]

    # u_xx: [batch, nx, nt]
    u_xx = torch.einsum('ij,bjt->bit', D2_x, u_squeezed)

    # u_t: [batch, nx, nt]
    u_t = torch.einsum('ij,bxj->bxi', Dt, u_squeezed)

    # 残差
    residual = u_t - nu * u_xx

    return residual.unsqueeze(-1)


if __name__ == '__main__':
    import h5py
    import numpy as np
    from loss import *
    import matplotlib.pyplot as pltimport
    import numpy as np
    import matplotlib.pyplot as plt

    # 使用

    nx = 65
    nt = 51
    D2_x, Dt = get_heat_matrices(nx, nt, x_domain=[-1, 1], t_domain=[0, 1])
    filepath = 'F:\data\OLdata\heat1D_robin_highprec.h5'
    dataset = h5DatasetFor1DHeat_2D('F:\data\OLdata\heat1D_robin_cgl.h5', initial_step=1, sub_t=8, sub_x=8)
    # xs, ys, name = dataset[0]
    # print(f"xs: {xs.shape}")  # 应该是 [nx, nt, 3]
    # print(f"ys: {ys.shape}")
    # with h5py.File(filepath, 'r') as h5_file:
    #     data_list = sorted(h5_file.keys())
    #
    # train_data = h5DatasetFor1DHeat(filepath,
    #                                 sub_x=4,
    #                                 sub_t=1,
    #                                 initial_step=1)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for xx, yy, name in train_loader:
        print(xx.shape)
        print(yy.shape)
        residual = compute_heat_residual_2d(yy, D2_x, Dt, nu=0.02)
        print('residual.shape:', residual.shape)
        print(torch.mean(abs(residual[:, 2:-2, 1:, :])))
        continue

        result = verify_robin_bc(yy, nu=0.02)
        res_plot = abs(residual[0, :, :, 0].detach().cpu().numpy())

        plt.figure(figsize=(10, 6))
        im = plt.imshow(res_plot, aspect='auto', origin='lower',
                        vmin=0, vmax=1.0)  # vmax 设置为最大值
        plt.colorbar(im, label='|Residual|')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title(f'|Residual| (max={res_plot.max():.2e}, mean={res_plot.mean():.2e})')
        plt.show()

        # 计算每个时间步的平均绝对残差
        # residual shape: [batch, nx, nt, 1] → 对 batch 和 nx 维度取平均
        res_abs = abs(residual[:, :, :, 0].detach().cpu().numpy())  # [batch, nx, nt]
        mean_per_t = res_abs.mean(axis=(0, 1))  # [nt]
        print(mean_per_t[:5])
        # 绘制柱状图
        plt.figure(figsize=(12, 5))
        t_steps = np.arange(len(mean_per_t))
        plt.bar(t_steps, mean_per_t, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Mean |Residual|')
        plt.title(f'Mean |Residual| per Time Step (total max={mean_per_t.max():.2e})')
        plt.xticks(t_steps[::5])  # 每隔5个显示一个刻度
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
