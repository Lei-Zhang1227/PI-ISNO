import scipy.io
import numpy as np
import os
import h5py
from pyDOE import lhs
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class h5DatasetFor1DBurgers_muti(Dataset):
    def __init__(self, filepath, initial_step=10, sub_t=1, sub_x=1,
                 if_test=False, resolution_df=None, label_num=400):
        self.sub_t = sub_t
        self.file_path = filepath
        self.default_sub_x = sub_x  # 默认下采样率
        self.initial_step = initial_step

        # 加载数据列表
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = [k for k in h5_file.keys() if k.isdigit()]
            data_list = sorted(data_list, key=int)  # 对筛选后的列表排序

        # 划分训练/测试集
        if if_test:
            self.data_list = np.array(data_list[:100])
            self.label_list = set() 
        else:
            self.data_list = np.array(data_list[100:])
            self.label_list = set(data_list[100:label_num])

        # 关联分辨率信息
        #         self.resolution_df = resolution_df  # 包含 'sample_id' 和 'recommended_res'
        self.resolution_map = dict(zip(
            resolution_df['sample_id'],
            resolution_df['recommended_res']
        ))

        # 定义推荐分辨率到 sub_x 的映射
        self.resolution_to_subx = {
            129: 4,  # 高分辨率，下采样率大
            257: 2,
            517: 1  # 低分辨率，下采样率小（或不下采样）
        }

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            #             print(idx)
            seed_group = h5_file[self.data_list[idx]]
            data = seed_group[:].astype('float32')
            data = torch.tensor(data, dtype=torch.float32)
            # 动态确定 sub_x
            name = self.data_list[idx]
            recommended_res = self.resolution_map[name]
            sub_x = self.resolution_to_subx.get(recommended_res, self.default_sub_x)
            # 应用动态 sub_x
            ys = data[::self.sub_t, ::sub_x].permute(1, 0).unsqueeze(-1)
            data = data.unsqueeze(-1)
            gridx = torch.linspace(-1, 1, data.shape[-2], dtype=torch.float).reshape(data.shape[-2], 1)
            data = data[::self.sub_t, ::sub_x]
            grid = gridx[::sub_x, :]
            Xs = data[0:self.initial_step, :]
            xs = Xs.permute(1, 0, 2)
            if self.data_list[idx] in self.label_list:
                label = 1.0
            else:
                label = 0.0
        return xs, ys, grid, recommended_res, label


class h5DatasetFor1DBurgers_muti_2D(Dataset):
    '''
    目前对于变分辨率对x和t使用相同的分辨率；
    输入形状为[nt,nx,nx,channel]
    label形状为[nt,nx,1]
    '''

    def __init__(self, filepath, initial_step=10, sub_t=1, sub_x=1,
                 if_test=False, resolution_df=None):
        self.sub_t = sub_t
        self.file_path = filepath
        self.default_sub_x = sub_x  # 默认下采样率
        self.initial_step = initial_step

        # 加载数据列表
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        # 划分训练/测试集
        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])

        # 关联分辨率信息
        #         self.resolution_df = resolution_df  # 包含 'sample_id' 和 'recommended_res'
        self.resolution_map = dict(zip(
            resolution_df['sample_id'],
            resolution_df['recommended_res']
        ))
        self.gridx = torch.tensor(np.linspace(-1, 1, 513), dtype=torch.float).reshape(1, 513, 1)
        self.gridt = torch.tensor(np.linspace(0, 1, 201), dtype=torch.float).reshape(201, 1, 1)

        # 定义推荐分辨率到 sub_x 的映射
        self.resolution_to_subx = {
            129: 4,  # 高分辨率，下采样率大
            257: 2,
            517: 1,  # 低分辨率，下采样率小（或不下采样）
        }

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            #             print(idx)
            seed_group = h5_file[self.data_list[idx]]
            data = seed_group[:].astype('float32')
            data = torch.tensor(data, dtype=torch.float32)
            # 动态确定 sub_x
            name = self.data_list[idx]
            recommended_res = self.resolution_map[name]
            sub_x = self.resolution_to_subx.get(recommended_res, self.default_sub_x)
            sub_t = sub_x
            # 应用动态 sub_x
            ys = data[::sub_t, ::sub_x].unsqueeze(-1)  # 形状: [nt_sub, nx_sub, 1]
            gridx = self.gridx[:, ::sub_x, :]  # 形状: [1, nx_sub, 1]
            gridt = self.gridt[::sub_t, :, :]  # 形状: [nt_sub, 1, 1]
            gridx_expanded = gridx.expand(ys.shape[0], -1, -1)  # 形状: [nt_sub, nx_sub, 1]
            gridt_expanded = gridt.expand(-1, ys.shape[1], -1)  # 形状: [nt_sub, nx_sub, 1]
            initial_data = ys[0:self.initial_step, :, :]  # [initial_step, nx_sub,1]
            # 调整维度顺序：从 [initial_step, n_x] 到 [1, n_x, initial_step]
            initial_data = initial_data.permute(2, 1, 0)  # [1, nx_sub, initial_step]
            # 扩展到与网格相同的时空维度
            Xs = initial_data.expand(ys.shape[0], -1, -1)  # [nt_sub, nx_sub, initial_step]
            print('Xs.shape,gridx_expanded.shape,gridt_expanded.shape:', Xs.shape, gridx_expanded.shape,
                  gridt_expanded.shape)
            input = torch.cat([Xs, gridx_expanded, gridt_expanded], dim=-1)  # [n_t, n_x, initial_step + 2]
        return input, ys, recommended_res


class h5DatasetFor1DBurgers_fix_2D(Dataset):
    '''
    fix的分辨率，指定nt,nx
    输入形状为[nt,nx,nx,channel]
    label形状为[nt,nx,1]
    '''

    def __init__(self, filepath, initial_step=10, sub_t=1, sub_x=1,
                 if_test=False):
        self.sub_t = sub_t
        self.sub_x = sub_x
        self.file_path = filepath
        self.initial_step = initial_step
        # 加载数据列表
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
        # 划分训练/测试集
        if if_test:
            self.data_list = np.array(data_list[6:10])
        else:
            self.data_list = np.array(data_list[100:])
        self.gridx = torch.tensor(np.linspace(-1, 1, 513), dtype=torch.float).reshape(1, 513, 1)
        self.gridt = torch.tensor(np.linspace(0, 1, 201), dtype=torch.float).reshape(201, 1, 1)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            data = seed_group[:].astype('float32')
            data = torch.tensor(data, dtype=torch.float32)
            # 应用动态 sub_x
            ys = data[::self.sub_t, ::self.sub_x].unsqueeze(-1)  # 形状: [nt_sub, nx_sub, 1]
            gridx = self.gridx[:, ::self.sub_x, :]  # 形状: [1, nx_sub, 1]
            gridt = self.gridt[::self.sub_t, :, :]  # 形状: [nt_sub, 1, 1]
            gridx_expanded = gridx.expand(ys.shape[0], -1, -1)  # 形状: [nt_sub, nx_sub, 1]
            gridt_expanded = gridt.expand(-1, ys.shape[1], -1)  # 形状: [nt_sub, nx_sub, 1]
            initial_data = ys[0:self.initial_step, :, :]  # [initial_step, n_x]
            # 调整维度顺序：从 [initial_step, n_x] 到 [1, n_x, initial_step]
            initial_data = initial_data.permute(2, 1, 0)  # [1, nx_sub, initial_step]
            # 扩展到与网格相同的时空维度
            Xs = initial_data.expand(ys.shape[0], -1, -1)  # [nt_sub, nx_sub, initial_step]
            xs = torch.cat([Xs, gridx_expanded, gridt_expanded], dim=-1)  # [n_t, n_x, initial_step + 2]
            name = self.data_list[idx]
        return xs, ys, name


class ResolutionWeightedBatchSampler(Sampler):
    def __init__(self, resolution_groups, batch_size, group_weights=None, shuffle=True):
        """
        Args:
            resolution_groups: dict {res: [indices]}，按分辨率分组的数据索引
            batch_size: 每个批次的样本数
            group_weights: dict {res: weight}，控制组的样本扩展倍数（如 {"129": 2} 表示129组的样本扩展为2倍）
            shuffle: 是否打乱全局批次顺序
        """
        self.resolution_groups = resolution_groups
        self.batch_size = batch_size
        #         print("First key in resolution_groups:", next(iter(self.resolution_groups.keys())))
        self.group_weights = group_weights or {int(res): 1 for res in resolution_groups}  # 默认权重1
        # print('self.group_weights:', self.group_weights)
        self.shuffle = shuffle

    def __iter__(self):
        # 1. 为每个组生成扩展后的样本列表
        random.seed(0)
        all_batches = []
        for res, indices in self.resolution_groups.items():
            # print('res, indices in ResolutionWeightedBatchSampler(Sampler):')
            # print(res, indices)
            # print('self.group_weights:', self.group_weights)

            n_repeats = self.group_weights[str(res)]  # 该组的样本扩展倍数

            # 扩展样本列表（随机排列并拼接）
            extended_indices = []
            for _ in range(n_repeats):
                shuffled = indices.copy()
                random.shuffle(shuffled)
                extended_indices.extend(shuffled)

            # 按 batch_size 生成批次
            for i in range(0, len(extended_indices), self.batch_size):
                batch = extended_indices[i:i + self.batch_size]
                if len(batch) < self.batch_size:
                    # 不足时随机重复补齐
                    batch += random.choices(extended_indices, k=self.batch_size - len(batch))
                all_batches.append((res, batch))  # 记录组别用于验证
                # print('res, batch')
                # print(res, batch)

        # 2. 打乱所有批次的全局顺序
        if self.shuffle:
            random.shuffle(all_batches)
        #         print('all_batches"',all_batches)

        # 3. 返回批次索引（忽略组别信息）
        for _, batch in all_batches:
            #             print("BatchSampler yields indices:", batch)
            yield batch

    def __len__(self):
        # 总批次数量 = sum(ceil(组样本数 * 组权重 / batch_size))
        total = 0
        for res, indices in self.resolution_groups.items():
            n_samples = len(indices) * self.group_weights[str(res)]
            total += (n_samples + self.batch_size - 1) // self.batch_size
        return total


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class BurgersLoader(Dataset):
    '''
    一次性得到所有时间步的值而非自回归；
    所以burgerinput最终的形状是[sample, n_t, n_x, 3], 其中channel部分是uxt的叠加； u是指初始条件；
    output的形状是[sample, n_t, n_x]
    '''

    def __init__(self, filepath,
                 initial_step=10,
                 sub_t=1,
                 sub_x=1,
                 if_test=False, test_ratio=0.1
                 ):
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x

        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
            seed_group = h5_file[data_list[0]]
            data_0 = seed_group[:]
            # 类型转换
            data_0 = data_0.astype('float32')
            data_0 = torch.tensor(data_0)
        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])
        self.initial_step = initial_step
        # 提前构建 gridx 和 gridt
        self.gridx = torch.tensor(np.linspace(-1, 1, data_0.shape[-2]), dtype=torch.float).reshape(1, data_0.shape[-2],
                                                                                                   1)
        self.gridt = torch.tensor(np.linspace(0, 1, data_0.shape[-3]), dtype=torch.float).reshape(data_0.shape[-3], 1,
                                                                                                  1)

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        '''
        当数据NX=512,TX=101,共读取800个例子作为训练数据时：
        1. Xs.shape: torch.Size([800, 512])
        1. ys.shape: torch.Size([800, 101, 512])
        gridx,gridt: torch.Size([1, 1, 512]) torch.Size([1, 101, 1])
        2. Xs.shape: torch.Size([800, 101, 512])
        3. Xs.shape: torch.Size([800, 101, 512, 3])
        dataset.shape: torch.Size([800, 101, 512, 3]) torch.Size([800, 101, 512])
        '''
        Xs = self.x_data[start:start + n_sample]
        ys = self.y_data[start:start + n_sample]

        # 使用提前构建的 gridx 和 gridt
        Xs = Xs.reshape(n_sample, 1, self.s).expand(-1, self.T, -1)  # [sample, nt, nx]
        Xs = torch.stack([Xs, self.gridx.expand(n_sample, self.T, -1), self.gridt.expand(n_sample, -1, self.s)], dim=3)

        # 创建 TensorDataset
        dataset = torch.utils.data.TensorDataset(Xs, ys)

        # 创建 DataLoader，设置 num_workers 和 pin_memory
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=64,  # 可以根据实际情况调整 num_workers 的值
            pin_memory=True  # 加速数据从 CPU 到 GPU 的传输
        )

        return loader


class h5DatasetFor1DBurgers(Dataset):
    '''
    1. 读取的文件类型是.h5， 用key来标注sample，从0000到0999一共1000个，具体数据格式见数据文件夹里的info
    2. 这个类给出了初始条件、完整数据和网格数据; 这里的初始条件的步数是通过initial_step来控制的，也就是用initial_step个时间步的数据来预测后后面时刻的值；
    3. 最后return的东西有三个：
        data[..., ::self.t_step, :][..., :self.initial_step, :]：前initial_step个时间步的数据
        data[..., ::self.t_step, :]：所有时间步的数据
        grid：网格数据,如果不进行下采样的话就是[128,128]
        在使用torch.utils.data.DataLoader指定sample加载后，得到的数据格式分别为：
        xx: [sample，128，128，initial_step，2]
        yy: [sample，128，128，101，2]
        grid: [sample,128,128]
    Nx = 128, Ny = 128, and Nt = 101. 所以就是用前10个时间步预测后91个时间步？
    '''

    def __init__(self, filepath,
                 initial_step=10,
                 sub_t=1,
                 sub_x=1,
                 if_test=False, test_ratio=0.1
                 ):
        # Define path to files
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x

        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])
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
            #             print('self.data_list[idx]:',self.data_list[idx])
            # data dim = [t, x1, ..., xd, v] [201,4096]
            data = seed_group[:]
            # 类型转换
            data = data.astype('float32')
            data = torch.tensor(data)
            ys = data[::self.sub_t, ::self.sub_x]
            data = data.unsqueeze(-1)
            gridx = torch.tensor(np.linspace(-1, 1, data.shape[-2]), dtype=torch.float).reshape(1, data.shape[-2], 1)
            gridt = torch.tensor(np.linspace(0, 1, data.shape[-3]), dtype=torch.float).reshape(data.shape[-3], 1, 1)
            data = data[::self.sub_t, ::self.sub_x, :]
            gridx = gridx[:, ::self.sub_x, :]
            gridt = gridt[:: self.sub_t, :, :]
            nx = data.shape[1]
            nt = data.shape[0]
            Xs = data[0:self.initial_step, ...]
            Xs = Xs.reshape(1, nx, self.initial_step).expand(nt, -1, self.initial_step)  # [nt, nx,self.initial_step]
            xs = torch.cat([Xs, gridx.expand(nt, -1, 1), gridt.expand(-1, nx, 1)], dim=-1)
        return xs, ys


class h5DatasetFor1DBurgersII(Dataset):
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
                 if_test=False, test_ratio=0.1
                 ):
        # Define path to files
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = [k for k in h5_file.keys() if k.isdigit()]
            data_list = sorted(data_list)  # 对筛选后的列表排序
        test_idx = int(len(data_list) * (1 - test_ratio))
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
            # data dim = [t, x1, ..., xd, v] [201,4096]
            data = seed_group[:]
            # 类型转换
            data = data.astype('float32')
            data = torch.tensor(data, dtype=torch.float32)
            #             print(f'1.data.shape:{data.shape}')
            ys = data[::self.sub_t, ::self.sub_x].permute(1, 0).unsqueeze(-1)  # [nx, nt,1]
            #             print(f'ys.shape:{ys.shape}')
            data = data.unsqueeze(-1)  # [nt, nx,1]
            #             print(f'2.data.shape:{data.shape}')
            gridx = torch.tensor(np.linspace(-1, 1, data.shape[-2]), dtype=torch.float).reshape(data.shape[-2], 1)
            data = data[::self.sub_t, ::self.sub_x]
            #             print(f'3.data.shape:{data.shape}')
            grid = gridx[::self.sub_x, :]  # [nx,1]
            Xs = data[0:self.initial_step, :, :]  # [self.initial_ste,nx,1]
            #             print(f'Xs.shape:{ys.shape}')
            xs = Xs.permute(1, 0, 2)  # [nx, self.initial_step,1]
            name = self.data_list[idx]
        return xs, ys, grid, name


class h5DatasetFor1DBurgersII_extend(Dataset):
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
                 unseen=False
                 ):
        # Define path to files
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = [k for k in h5_file.keys() if k.isdigit()]
            data_list = sorted(data_list)#, key=int)
        if unseen:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:200])
#         print(self.data_list)
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
            # data dim = [t, x1, ..., xd, v] [201,4096]
            data = seed_group[:]
            # 类型转换
            data = data.astype('float32')
            data = torch.tensor(data, dtype=torch.float32)
#             print(f'1.data.shape:{data.shape}')
            ys = data[::self.sub_t, ::self.sub_x].permute(1, 0).unsqueeze(-1)  # [nx, nt,1]
            #             print(f'ys.shape:{ys.shape}')
            data = data.unsqueeze(-1)  # [nt, nx,1]
            #             print(f'2.data.shape:{data.shape}')
            gridx = torch.tensor(np.linspace(-1, 1, data.shape[-2]), dtype=torch.float).reshape(data.shape[-2], 1)
            data = data[::self.sub_t, ::self.sub_x]
            #             print(f'3.data.shape:{data.shape}')
            grid = gridx[::self.sub_x, :]  # [nx,1]
            Xs = data[0:self.initial_step, :, :]  # [self.initial_ste,nx,1]
            #             print(f'Xs.shape:{ys.shape}')
            xs = Xs.permute(1, 0, 2)  # [nx, self.initial_step,1]
            name = self.data_list[idx]
        return xs, ys, grid, name



class h5DatasetFor1DBurgersII_extend_train(Dataset):
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
                 if_test=False,
                 end_time=101
                 ):
        # Define path to files
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())
        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])
        self.initial_step = initial_step
        self.end_time = end_time

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        就是一个一个的取呗，所以grid也不需要repeat
        '''
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            # data dim = [t, x1, ..., xd, v] [201,4096]
            data = seed_group[:]
            # 类型转换
            data = data.astype('float32')
            data = torch.tensor(data, dtype=torch.float32)
            data = data[:self.end_time, :]
            #             print(f'1.data.shape:{data.shape}')
            ys = data[::self.sub_t, ::self.sub_x].permute(1, 0).unsqueeze(-1)  # [nx, nt,1]
            #             print(f'ys.shape:{ys.shape}')
            data = data.unsqueeze(-1)  # [nt, nx,1]
            #             print(f'2.data.shape:{data.shape}')
            gridx = torch.tensor(np.linspace(-1, 1, data.shape[-2]), dtype=torch.float).reshape(data.shape[-2], 1)
            data = data[::self.sub_t, ::self.sub_x]
            #             print(f'3.data.shape:{data.shape}')
            grid = gridx[::self.sub_x, :]  # [nx,1]
            Xs = data[0:self.initial_step, :, :]  # [self.initial_ste,nx,1]
            #             print(f'Xs.shape:{ys.shape}')
            xs = Xs.permute(1, 0, 2)  # [nx, self.initial_step,1]
            name = self.data_list[idx]
        return xs, ys, grid, name

class h5DatasetFor1DBurgers_A(Dataset):
    '''
    用于 Burgers 方程 A model 的数据集

    返回:
        xx: [nx, 2] 初始条件(第一个时间步) + 位置坐标 (u0, x)
        yy: [nx, n_output_steps] 除初始条件外前n_output_steps个时间步的值
        grid: [nx, 1] 空间网格
        name: 样本名称
    '''

    def __init__(self, filepath,
                 sub_t=1,
                 sub_x=1,
                 n_output_steps=9,
                 initial_step=1,
                 if_test=False,
                 ):
        self.file_path = filepath
        self.sub_t = sub_t
        self.sub_x = sub_x
        self.n_output_steps = n_output_steps
        self.initial_step = initial_step

        # 加载数据列表
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = [k for k in h5_file.keys() if k.isdigit()]
            data_list = sorted(data_list)
            print('len(data_list):',len(data_list))

        # 构建空间网格 (原始数据是 513 个点，范围 [-1, 1])
        x_grid_full = torch.tensor(np.linspace(-1, 1, 513), dtype=torch.float32)
        self.x_grid = x_grid_full[::self.sub_x].reshape(-1, 1)  # [nx_sub, 1]

        # 划分训练/测试集
        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            # data dim = [nt, nx] = [201, 513]
            data = torch.tensor(seed_group[:].astype('float32'), dtype=torch.float32)

        # 下采样: [nt, nx] -> [nt_sub, nx_sub]
#         print(data.shape)
        data = data[::self.sub_t, ::self.sub_x]

        # 转置为 [nx, nt]
        data = data.T  # [nx_sub, nt_sub]

        grid = self.x_grid  # [nx_sub, 1]

        # 输入: 第一个时间步 + 位置坐标
        u0 = data[:, 0:1]  # [nx, 1] 取第一个时间步
        xx = torch.cat([u0, grid], dim=-1)  # [nx, 2]

        # 输出: 除初始条件外的前 n_output_steps 个时间步
        yy = data[:, 1:1 + self.n_output_steps]  # [nx, n_output_steps]

        name = self.data_list[idx]

        return xx, yy, grid, name


class h5DatasetFor1DBurgers_TwoStage(Dataset):
    '''
    用于 Burgers 方程 A model 的数据集

    返回:
        xx: [nx, 2] 初始条件(第一个时间步) + 位置坐标 (u0, x)
        yy: [nx, n_output_steps] 除初始条件外前n_output_steps个时间步的值
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

        # 加载数据列表
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = [k for k in h5_file.keys() if k.isdigit()]
            data_list = sorted(data_list)

        # 构建空间网格 (原始数据是 513 个点，范围 [-1, 1])
        x_grid_full = torch.tensor(np.linspace(-1, 1, 513), dtype=torch.float32)
        self.x_grid = x_grid_full[::self.sub_x].reshape(-1, 1)  # [nx_sub, 1]

        # 划分训练/测试集
        if if_test:
            self.data_list = np.array(data_list[:100])
        else:
            self.data_list = np.array(data_list[100:])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
            # data dim = [nt, nx] = [201, 513]
            data = torch.tensor(seed_group[:].astype('float32'), dtype=torch.float32)

        # 下采样: [nt, nx] -> [nt_sub, nx_sub]
        data = data[::self.sub_t, ::self.sub_x]

        # 转置为 [nx, nt]
        data = data.T  # [nx_sub, nt_sub]

        grid = self.x_grid  # [nx_sub, 1]

        # 输入: 第一个时间步 + 位置坐标
        u0 = data[:, 0:1]  # [nx, 1] 取第一个时间步
        xx = torch.cat([u0, grid], dim=-1)  # [nx, 2]

        # 输出: 除初始条件外的前 n_output_steps 个时间步
        yy = data.unsqueeze(-1)  # [nx, n_t,1]

        name = self.data_list[idx]

        return xx, yy, grid, name


class ResolutionWeightedBatchSamplerII(Sampler):
    """
    层次训练
    """

    def __init__(self, resolution_groups, batch_size, group_weights=None, shuffle=True):
        self.resolution_groups = resolution_groups
        self.batch_size = batch_size
        self.group_weights = group_weights or {res: 1 for res in resolution_groups}
        self.shuffle = shuffle
        self.active_resolutions = list(resolution_groups.keys())  # 默认启用所有分辨率

    def set_active_resolutions(self, resolutions: list):
        """动态设置当前激活的分辨率（训练过程中调用）"""
        self.active_resolutions = resolutions

    def __iter__(self):
        # 只处理激活的分辨率
        #         print('active_groups:', self.active_resolutions)
        #         print('self.resolution_groups:',self.resolution_groups)
        active_groups = {res: self.resolution_groups[int(res)] for res in self.active_resolutions}
        all_batches = []

        for res, indices in active_groups.items():
            n_repeats = self.group_weights.get(res, 1)
            extended_indices = []
            for _ in range(n_repeats):
                shuffled = indices.copy()
                if self.shuffle:
                    random.shuffle(shuffled)
                extended_indices.extend(shuffled)

            for i in range(0, len(extended_indices), self.batch_size):
                batch = extended_indices[i:i + self.batch_size]
                if len(batch) < self.batch_size:
                    batch += random.choices(extended_indices, k=self.batch_size - len(batch))
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)
        #         print('len(all_batches):',len(all_batches))
        yield from all_batches

    def __len__(self):
        total = 0
        active_groups = {res: self.resolution_groups[int(res)]
                         for res in self.active_resolutions}  # 注意这里过滤了active

        for res, indices in active_groups.items():
            n_samples = len(indices) * self.group_weights.get(res, 1)
            total += (n_samples + self.batch_size - 1) // self.batch_size
        return total


if __name__ == '__main__':
    import h5py
    import numpy as np
    from loss import residual_for_burgers, calculate_FDM

    filepath = '/data/zhanglei/BurgersEquation/burgers_neumann.h5'
    with h5py.File(filepath, 'r') as h5_file:
        data_list = sorted(h5_file.keys())
        print("Data list:", data_list)  # 确认是否有有效数据
    train_data =  h5DatasetFor1DBurgerscII_extend_train(filepath,
                                       sub_x=4,
                                       sub_t=1,
                                       initial_step=1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, )
    for xx, yy in train_loader:
        print(xx.shape)
        print(yy.shape)
        yy = yy.unsqueeze(1)
        ux, Du = burger_residual(yy)
        A_mean = np.mean(np.abs(Du.numpy()))
        print(f'dct+fft:{A_mean}')
