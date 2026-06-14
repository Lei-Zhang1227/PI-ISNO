"""
	TEST1: Data做下采样；然后添加PDE约束；
今天晚上先进行这个实验；
1. 第一次尝试、卒于失败的lr设计；loss没降下来。
2. 加载pretrain模型，使用test data数据进行实例微调；
"""
import json

from NOs_dict.models import CosNO_II as Model
import os
from torch.utils.data import Dataset, DataLoader
from utilities import *
import h5py
from Adam import Adam
from timeit import default_timer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser setup
import argparse


class FNODatasetMult(Dataset):
    '''
    没有下载原始数据，但是大概能看出来数据格式，根据PDEBench的提示，给出的数据格式应该是:[samples,t,x,y,channel],以及一个grid矩阵，应该是[x,y]
    这个类给出了初始条件、完整数据和网格数据
    也就是输入的前10个时间步的数据，输出是所有的（包括前10个和后不知道多少个的数据）
    data[..., ::self.t_step, :][..., :self.initial_step, :] 这里对数据进行了permute，将时间维度转移到了倒数第二维度；
    等等，initial_step=10,self.t_step=1
    Nx = 128, Ny = 128, and Nt = 101. 所以就是用前10个时间步预测后91个时间步？
    '''

    def __init__(self, filename,
                 initial_step=10,
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False, test_ratio=0.1
                 ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        self.file_path = filename
        self.t_step = reduced_resolution_t

        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])

        # Time steps used as initial conditions
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

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape) - 1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

            # Extract spatial dimension of data
            dim = len(data.shape) - 2

            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(seed_group["grid"]["x"], dtype='f')
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                grid = torch.stack((X, Y), axis=-1)
            elif dim == 3:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                z = np.array(seed_group["grid"]["z"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z)
                grid = torch.stack((X, Y, Z), axis=-1)

        return data[..., ::self.t_step, :][..., :self.initial_step, :], data[..., ::self.t_step, :], grid


lap_2d_op = [[[[0, 0, -1 / 12, 0, 0],
               [0, 0, 4 / 3, 0, 0],
               [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
               [0, 0, 4 / 3, 0, 0],
               [0, 0, -1 / 12, 0, 0]]]]


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=5, name=''):
        super(Conv2dDerivative, self).__init__()
        self.deno = deno
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)
        # 固定的导数算子
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()
        self.deno = deno
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)
        # 固定的导数算子
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class loss_generator(nn.Module):
    ''' 用于物理损失计算 '''

    def __init__(self, dt=(1.0 / 2), dx=(1.0 / 100)):
        super(loss_generator, self).__init__()
        self.dx = dx

        # 空间导数算子，转到 device
        self.laplace = Conv2dDerivative(
            DerFilter=lap_2d_op,
            deno=(dx ** 2),
            kernel_size=5,
            name='laplace_operator').to(device)

        # 时间导数算子，转到 device
        self.dt = Conv1dDerivative(
            DerFilter=[[[-1, 1, 0]]],
            deno=(dt * 1),
            kernel_size=3,
            name='partial_t').to(device)

    def get_phy_Loss(self, output):
        '''
        计算物理残差，输入 shape 为 [time, channel, height, width]
        '''
        # 空间导数
        laplace_u = self.laplace(output[0:-2, 0:1, :, :])
        laplace_v = self.laplace(output[0:-2, 1:2, :, :])

        # 时间导数 - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height, width, channel, time]
        u_conv1d = u_conv1d.reshape(lenx * leny, 1, lent)
        u_t = self.dt(u_conv1d)  # 时间步减少2
        u_t = u_t.reshape(leny, lenx, 1, lent - 2)
        u_t = u_t.permute(3, 2, 0, 1)  # [time-2, channel, height, width]

        # 时间导数 - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)
        v_conv1d = v_conv1d.reshape(lenx * leny, 1, lent)
        v_t = self.dt(v_conv1d)
        v_t = v_t.reshape(leny, lenx, 1, lent - 2)
        v_t = v_t.permute(3, 2, 0, 1)

        # 对应区域
        u = output[0:-2, 0:1, 2:-2, 2:-2]
        v = output[0:-2, 1:2, 2:-2, 2:-2]

        # 保证形状一致
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # Gray-Scott 模型参数
        Du = 0.001
        Dv = 0.005
        k = 0.005

        f_u = (Du * laplace_u + u - (u ** 3) - k - v - u_t)
        f_v = (Dv * laplace_v + u - v - v_t)
        return f_u, f_v

    def get_phy_LossII(self, output):
        '''
        计算物理残差，输入 shape 为 [sample, time, channel, height, width]
        '''
        # 获取输入形状
        num_samples = output.shape[0]  # 样本数
        time_steps = output.shape[1]  # 时间步数
        channels = output.shape[2]  # 通道数
        height = output.shape[3]  # 高度
        width = output.shape[4]  # 宽度

        # 调整输入形状以适应卷积操作
        # 将 sample 和 time 维度合并为 batch 维度
        output_reshaped = output.reshape(num_samples * time_steps, channels, height,
                                         width)  # [sample * time, channel, height, width]

        # 空间导数
        laplace_u = self.laplace(output_reshaped[:, 0:1, :, :])  # [sample * time, 1, height, width]
        laplace_v = self.laplace(output_reshaped[:, 1:2, :, :])  # [sample * time, 1, height, width]

        # 恢复 sample 和 time 维度
        laplace_u = laplace_u.reshape(num_samples, time_steps, 1, height - 4,
                                      width - 4)  # [sample, time, 1, height, width]
        laplace_v = laplace_v.reshape(num_samples, time_steps, 1, height - 4,
                                      width - 4)  # [sample, time, 1, height, width]
        laplace_u = laplace_u[:, 0:-2, :, :, :]
        laplace_v = laplace_v[:, 0:-2, :, :, :]
        # 时间导数 - u
        u = output[:, :, 0:1, 2:-2, 2:-2]  # [sample, time, 1, height-4, width-4]
        u_conv1d = u.permute(0, 3, 4, 2, 1)  # [sample, height-4, width-4, 1, time]
        u_conv1d = u_conv1d.reshape(num_samples * (height - 4) * (width - 4), 1,
                                    time_steps)  # [sample * (height-4) * (width-4), 1, time]
        u_t = self.dt(u_conv1d)  # [sample * (height-4) * (width-4), 1, time-2]
        u_t = u_t.reshape(num_samples, height - 4, width - 4, 1,
                          time_steps - 2)  # [sample, height-4, width-4, 1, time-2]
        u_t = u_t.permute(0, 4, 3, 1, 2)  # [sample, time-2, 1, height-4, width-4]

        # 时间导数 - v
        v = output[:, :, 1:2, 2:-2, 2:-2]  # [sample, time, 1, height-4, width-4]
        v_conv1d = v.permute(0, 3, 4, 2, 1)  # [sample, height-4, width-4, 1, time]
        v_conv1d = v_conv1d.reshape(num_samples * (height - 4) * (width - 4), 1,
                                    time_steps)  # [sample * (height-4) * (width-4), 1, time]
        v_t = self.dt(v_conv1d)  # [sample * (height-4) * (width-4), 1, time-2]
        v_t = v_t.reshape(num_samples, height - 4, width - 4, 1,
                          time_steps - 2)  # [sample, height-4, width-4, 1, time-2]
        v_t = v_t.permute(0, 4, 3, 1, 2)  # [sample, time-2, 1, height-4, width-4]

        # 提取对应区域
        u = output[:, 0:-2, 0:1, 2:-2, 2:-2]  # [sample, time-2, 1, height-4, width-4]
        v = output[:, 0:-2, 1:2, 2:-2, 2:-2]  # [sample, time-2, 1, height-4, width-4]

        # 保证形状一致
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # Gray-Scott 模型参数
        Du = 0.001
        Dv = 0.005
        k = 0.005

        # 计算物理残差
        f_u = (Du * laplace_u + u - (u ** 3) - k - v - u_t)  # [sample, time-2, 1, height-4, width-4]
        f_v = (Dv * laplace_v + u - v - v_t)  # [sample, time-2, 1, height-4, width-4]

        return f_u, f_v


def residual_stitasticII(residuals):
    """
    计算每个样本和每个时间切片的残差统计信息，并找出高残差点最多的样本和时间切片。

    参数:
        residuals: 形状为 [sample, t, 1, x, y] 的残差数据，可以是 torch.Tensor 或 numpy.ndarray

    输出:
        打印每个样本和每个时间切片的最大值、均值、标准差、阈值、高残差点数量及占比，
        并返回一个包含各样本和各时刻统计信息的列表。
    """
    # 如果是 torch.Tensor，则先转成 numpy 数组（确保在 CPU 上）
    if isinstance(residuals, torch.Tensor):
        residuals = residuals.cpu().detach().numpy()
    residuals = np.abs(residuals)

    sample_dim = residuals.shape[0]
    t_dim = residuals.shape[1]
    stats_list = []

    for sample in range(sample_dim):
        for t in range(t_dim):
            slice_t = residuals[sample, t, 0]  # shape: [x, y]
            max_value = np.max(slice_t)
            mean_value = np.mean(slice_t)
            std_value = np.std(slice_t)
            # 设定阈值：均值加上2倍标准差
            threshold = mean_value + 2 * std_value
            num_large = np.sum(slice_t > threshold)
            total_points = slice_t.size
            percentage_large = num_large / total_points * 100

            stats = {
                "sample_index": sample,
                "time_index": t,
                "max": max_value,
                "mean": mean_value,
                "std": std_value,
                "threshold": threshold,
                "num_large": num_large,
                "percentage_large": percentage_large
            }
            stats_list.append(stats)

            print(f"Sample {sample}, t = {t}: max = {max_value:.3f}, mean = {mean_value:.3f}, std = {std_value:.3f}, "
                  f"threshold = {threshold:.3f}, num_large = {num_large}, percentage_large = {percentage_large:.2f}%")

    # 找出高残差点最多的样本和时间切片
    max_high_sample_t = max(stats_list, key=lambda s: s["num_large"])
    print(
        f"\n高残差点最多的样本和时间切片: Sample {max_high_sample_t['sample_index']}, t = {max_high_sample_t['time_index']}")

    # 计算全局最大残差值及其位置
    max_value = np.max(residuals)
    max_index = np.unravel_index(np.argmax(residuals), residuals.shape)
    print("全局最大残差值：", max_value)
    print("全局最大残差位置（索引）：", max_index)

    # 计算全局平均残差和标准差
    mean_value = np.mean(residuals)
    std_value = np.std(residuals)
    print("全局平均残差：", mean_value)
    print("全局标准差：", std_value)

    # 统计全局较大残差的数量与占比
    threshold = mean_value + 10 * std_value
    num_large = np.sum(residuals > threshold)
    total_points = residuals.size
    percentage_large = num_large / total_points * 100

    print("全局残差大于 {:.3f} 的点数：{}，占比：{:.2f}%".format(threshold, num_large, percentage_large))

    return stats_list


def loss_gen(output, loss_func):
    '''计算物理损失'''
    # 周期性边界条件的 padding
    # output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
    # output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_LossII(output)
    # residual_stitasticII(f_u)
    # residual_stitasticII(f_v)
    loss = mse_loss(f_u[:, 10:, :, :, :], torch.zeros_like(f_u[:, 10:, :, :, :]).to(device)) + \
           mse_loss(f_v[:, 10:, :, :, :], torch.zeros_like(f_v[:, 10:, :, :, :]).to(device))
    return f_u, f_v, loss


def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning')
    parser.add_argument('--data-dict', default='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5', type=str,
                        help='dataset folder')
    parser.add_argument('--model-path',
                        default='./sp-diff-react1-modes24-width24-bw1-triL0-step-init_t10-sub_t1.pkl', type=str,
                        help='path to the saved model')
    parser.add_argument('--data-path', default='data/', type=str, help='path for data-dict')
    parser.add_argument('--initial-step', default=10, type=int, help='initial time steps')
    return parser.parse_args()


args = get_args()


# Model loading function with dynamic parameter extraction
def load_model_and_params(model_path):
    # 自动检测设备，若无GPU则映射到CPU
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=map_location)

    # 打印模型文件内容
    # print("\n===== Loaded Checkpoint Contents =====")
    # for key, value in checkpoint.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{key}: Tensor of shape {value.shape}")
    #     elif isinstance(value, list):
    #         print(f"{key}: List of length {len(value)}")
    #     else:
    #         print(f"{key}: {value}")
    # print("====================================\n")

    # 提取模型参数
    model_params = {
        'batch_size': checkpoint.get('batch_size', 5),
        'learning_rate': checkpoint.get('learning_rate', 1e-3),
        'width': checkpoint.get('width', 24),
        'modes': checkpoint.get('modes', 24),
        'sub': checkpoint.get('sub', 1),
        'weight_decay': checkpoint.get('weight_decay', 1e-4),
        'epochs': checkpoint.get('epochs', None)
    }

    # 初始化模型并加载权重
    model = Model(args.initial_step * 2 + 2, model_params['modes'], model_params['width'], 1, out_channels=2, dim=2,
                  triL=0).to(device)
    model.load_state_dict(checkpoint['model'])

    # 打印模型结构
    print("\n===== Model Architecture =====")
    print(model)
    print("====================================\n")

    # 打印每层的参数数量
    print("===== Parameters Per Layer =====")
    layer_param_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            layer_param_counts[name] = param_count
            print(f"{name}: {param_count} parameters")
    print("====================================\n")

    # 计算总参数量
    total_params = sum(layer_param_counts.values())
    print(f"Total Parameters: {total_params}")
    print("====================================\n")

    print(f"Model loaded from {model_path} with parameters: {model_params}")
    return model, model_params


def define_model():
    model = Model(10 * 2 + 2, 24, 24, 1, out_channels=2, dim=2, triL=0).to(device)
    return model


# Testing function
def test_model(model, initial_step):
    data_name = 'diff-react'
    # Define dataset and dataloader dynamically
    val_data = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                              initial_step=initial_step,
                              reduced_resolution=1,
                              reduced_resolution_t=1,
                              reduced_batch=5,
                              if_test=False)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, num_workers=128,
                                             shuffle=False)

    test_size = val_data.data_list.shape[0]
    ntest = test_size
    print('size-of-val:', test_size)
    training_type = 'autoregressive'
    t_train = (101 - 1) // 1 + 1
    myloss = LpLoss(size_average=False)
    loss_fn = myloss
    model.eval()
    test_err = torch.tensor([])
    with torch.no_grad():
        for xx, yy, grid in val_loader:
            val_l2_step = 0
            val_l2_full = 0
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)
            outp_shape = inp_shape[:-1] + [1, -1]
            loss = 0
            xx, yy, grid = xx.to(device), yy.to(device), grid.to(device)

            if training_type in ['autoregressive']:
                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    y = yy[..., t:t + 1, :]
                    # im = model(inp, grid)
                    im = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
                    _batch = im.size(0)
                    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                    pred = torch.cat((pred, im), -2)

                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                print('loss.item():', loss.item())
                val_l2_step += loss.item()
                _batch = yy.size(0)
                _pred = pred[..., initial_step:t_train, :]
                _yy = yy[..., initial_step:t_train, :]
                val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                print(val_l2_full)
                test_err = torch.cat([test_err,
                                      torch.tensor([val_l2_full])],
                                     dim=0)

    print('test_l2', test_err.sum().item() / test_size)
    print('test_l2 min-max:', test_err.min().item(), test_err.max().item())
    print('pred.shape:', pred.shape)
    print('yy.shape:', yy.shape)
    # 返回的是最后一个batch中的结果
    #  pred.shape: torch.Size([5, 128, 128, 101, 2])
    #  yy.shape: torch.Size([5, 128, 128, 101, 2])
    return pred, yy


def load_samplesII(file_path, sample_keys):
    data_list = []
    with h5py.File(file_path, 'r') as f:
        for key in sample_keys:
            data = f[f'{key}/data'][:]
            data_list.append(data)
        x = f[f'{sample_keys[0]}/grid/x'][:]
        y = f[f'{sample_keys[0]}/grid/y'][:]
        t = f[f'{sample_keys[0]}/grid/t'][:]

    # 沿 sample 维度堆叠
    data_stacked = np.stack(data_list, axis=0)  # [N, time, height, width, channel]
    return data_stacked, x, y, t


def calculate_loss(output):
    '''
    从模型预测得来的矩阵形状为: pred.shape: torch.Size([batch, x, y, t, channel])
    这里矩阵的形状应为：torch_tensor_permuted.shape: torch.Size([batch, t, channel, x, y])
    :param output:
    :return:
    '''
    file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5'
    sample_key = ['0000', '0001', '0002']
    data, x, y, t = load_samplesII(file_path, sample_key)
    # print('u.shape, v.shape, x.shape, y.shape, t.shape:', u.shape, v.shape, x.shape, y.shape, t.shape)
    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = 0.05  # 小时间步
    output_permuted = output.permute(0, 3, 4, 1, 2)
    print('output_permuted.shape:', output_permuted.shape)
    loss_func = loss_generator(dt, dx).to(device)
    f_u, f_v, loss_phy = loss_gen(output_permuted, loss_func)
    print("Physics Loss:", loss_phy)


# Visualization function
def visualize_results(pred, yy, initial_step):
    Nx = Ny = 128
    nx = np.linspace(-1, 1, Nx)
    ny = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(nx, ny)

    plt.figure(figsize=(12, 6))
    for i in range(2):
        plt.subplot(2, 2, i + 1)
        plt.pcolor(X, Y, yy[0, ..., initial_step, i].cpu(), cmap="jet")
        plt.title(f'True Channel {i + 1}')
        plt.colorbar()

        plt.subplot(2, 2, i + 3)
        plt.pcolor(X, Y, pred[0, ..., initial_step, i].cpu(), cmap="jet")
        plt.title(f'Predicted Channel {i + 1}')
        plt.colorbar()

    plt.show()


def train_model(model, epochs, weight_data, weight_phy, Test=True):
    data_name = 'diff-react'
    initial_step = 10
    mse_loss = nn.MSELoss()
    # Define dataset and dataloader dynamically
    result_PATH = "./TEST_3_B_model_checkpoint.pkl"
    train_data = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                                initial_step=initial_step,
                                reduced_resolution=1,
                                reduced_resolution_t=1,
                                reduced_batch=5)
    val_data = FNODatasetMult(filename='/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5',
                              initial_step=initial_step,
                              reduced_resolution=1,
                              reduced_resolution_t=1,
                              reduced_batch=5,
                              if_test=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, num_workers=126, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, num_workers=126, shuffle=False)

    file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5'
    sample_key = ['0000', '0001', '0002']
    data, x, y, t = load_samplesII(file_path, sample_key)
    # print('u.shape, v.shape, x.shape, y.shape, t.shape:', u.shape, v.shape, x.shape, y.shape, t.shape)
    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = 0.05  # 小时间步

    train_size, test_size = train_data.data_list.shape[0], val_data.data_list.shape[0]
    ntrain, ntest = train_size, test_size
    training_type = 'autoregressive'
    t_train = (101 - 1) // 1 + 1
    myloss = LpLoss(size_average=False)
    loss_fn = myloss
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)

    model.train()
    test_err = torch.tensor([])
    loss_list = []

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss_phy = []
        train_loss_data = []
        b = 0
        if Test:
            val_l2_step = 0
            val_l2_full = 0
            with torch.no_grad():
                for xx, yy, grid in val_loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)
                    outp_shape = inp_shape[:-1] + [1, -1]

                    if training_type in ['autoregressive']:
                        pred = yy[..., :initial_step, :]
                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)

                        for t in range(initial_step, yy.shape[-2]):
                            inp = xx.reshape(inp_shape)
                            y = yy[..., t:t + 1, :]
                            # im = model(inp, grid)
                            im = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
                            _batch = im.size(0)
                            loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                            pred = torch.cat((pred, im), -2)

                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]
                        val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                        pred_permuted = pred.permute(0, 3, 4, 1, 2)
                        # print('output_permuted.shape:', pred_permuted.shape)
                        loss_func = loss_generator(dt, dx).to(device)
                        f_u, f_v, loss_phy = loss_gen(pred_permuted, loss_func)
            test_l2 = val_l2_full / ntest
            # train_list.append(train_l2)
            loss_list.append(test_l2)
            print(f'ep:{ep}, test_l2: {test_l2}, test_pde_mse:{loss_phy.item()}')

        for xx, yy, grid in val_loader:
            # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # yy: target tensor [b, x1, ..., xd, t, v]
            # grid: meshgrid [b, x1, ..., xd, dims]
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)

            # Initialize the prediction tensor
            pred = yy[..., :initial_step, :]
            # Extract shape of the input tensor for reshaping (i.e. stacking the
            # time and channels dimension together)
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)
            outp_shape = inp_shape[:-1] + [1, -1]

            if training_type in ['autoregressive']:
                # Autoregressive loop
                for t in range(initial_step, t_train):
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = xx.reshape(inp_shape)


                    # Model run
                    im = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)

                    # Loss calculation
                    _batch = im.size(0)
                    # loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)

                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                # train_l2_step += loss.item()
                _batch = yy.size(0)
                _pred = pred[..., initial_step + 1:t_train, :]

                # l2_full = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                # train_l2_full += l2_full.item()
                # loss_data = mse_loss(_yy[:, ::2, ::2, :, :], _pred[:, ::2, ::2, :, :]).to(device)
                # 姑且先实验一下再空间维度稀释；
                # print('loss_data', loss_data)
                pred_permuted = pred.permute(0, 3, 4, 1, 2)
                # print('output_permuted.shape:', pred_permuted.shape)
                loss_func = loss_generator(dt, dx).to(device)
                f_u, f_v, loss_phy = loss_gen(pred_permuted, loss_func)
                # print("Physics Loss:", loss_phy)
                loss = weight_phy * loss_phy# + weight_data * loss_data
                train_loss_phy.append(loss_phy.item())
                # train_loss_data.append(loss_data.item())
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'eposh:{ep}/batch{b}--loss_phy:{loss_phy.item()},lr:{lr}')
                b += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        # train_l2 = train_l2_full / ntrain

    if epochs >= 10:
        torch.save({'model': model.state_dict()}, result_PATH)
        if epochs % 10 == 0:
            with open('TEST_3_A_train_loss.json', 'w') as f:
                json.dump({'train_loss_phy': train_loss_phy, 'train_loss_data': train_loss_data}, f)


# Main execution
if __name__ == '__main__':
    model, model_params = load_model_and_params(args.model_path)
    # pred, yy = test_model(model, args.initial_step)
    # calculate_loss(pred)
    # visualize_results(pred, yy, args.initial_step)
    train_model(model, epochs=200, weight_data=1, weight_phy=1, Test=True)
