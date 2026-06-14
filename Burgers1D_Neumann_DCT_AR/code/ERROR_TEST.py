'''
11/7:测试residual出错原因
'''
import sys
import os
import subprocess
import psutil

# 获取当前脚本所在目录的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 添加上一级目录到 sys.path
sys.path.insert(0, parent_dir)
from argparse import ArgumentParser
import yaml
from functools import partial as PARTIAL
from Burger.model import SOLII
from Burger.datasets import BurgersLoader, h5DatasetFor1DBurgers
from Burger.loss import *
import tqdm
from utils import *
import time
import shutil
from tqdm import tqdm
from datetime import datetime
from Burger.plot import *


def run(config):
    ################################################################
    # prepare
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{datetime.now()}--- set divice: {device} ---')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f'{datetime.now()} --- set seed ---')
    ################################################################
    # dataloader
    ################################################################
    data_config = config['data']
    batch_size = config['train']['batchsize']
    v = 0.01
    train_data = h5DatasetFor1DBurgers(data_config['datapath'],
                                       sub_x=data_config['sub_x'],
                                       sub_t=data_config['sub_t'],
                                       initial_step=data_config['initial_step'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                               num_workers=128, pin_memory=True)
    test_data = h5DatasetFor1DBurgers(data_config['datapath'],
                                      sub_x=data_config['sub_x'],
                                      sub_t=data_config['sub_t'],
                                      initial_step=data_config['initial_step'], if_test=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=128, pin_memory=True)

    #     train_size, test_size = train_data.data_list.shape[0], test_data.data_list.shape[0]
    #     print('size-of-train/val:', train_size, test_size)
    print(
        f'{datetime.now()} --- set dataset，batch size: {batch_size}, Train loader lens：{len(train_loader)}, Test loader lens：{len(test_loader)}')
    ################################################################
    # location
    ################################################################
    first_dic = f"./{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    shutil.copy('./information.yaml', f"./{config['prepare']['project']}")
    os.chdir(first_dic)
    print(f'{datetime.now()} --- set save dir ---')
    i=0
    for u, y in train_loader:
        u, y = u.to(device, non_blocking=True), y.to(device, non_blocking=True)  # 确保数据在相同设备上
        print(y.shape)
        print('y.dtype:',y.dtype)
#         print('key:',key)
        x = u[:, :, :, 1]  # 提取出空间位置 (所有样本, 所有时间步, 所有空间位置)
        t = u[:, :, :, 2]  # 提取出时间位置 (所有样本, 所有时间步, 所有空间位置)

        # 假设 x 和 t 对所有样本都是相同的，可以从第一个样本计算 delta_x 和 delta_t
        x_sample = x[0, 0, :]  # 取出第一个样本的第一个时间步的所有空间位置
        t_sample = t[0, :, 0]  # 取出第一个样本的所有时间步的第一个空间位置

        # 计算 delta_x 和 delta_t
        delta_x = x_sample[1] - x_sample[0]  # 假设空间均匀分布
        delta_t = t_sample[1] - t_sample[0]  # 假设时间均匀分布

        delta_x2 = x_sample[2] - x_sample[1]  # 假设空间均匀分布
        delta_t2 = t_sample[2] - t_sample[1]  # 假设时间均匀分布
        file_path = '/data/zhanglei/BurgersEquation/burgers_neumann.h5'

#         with h5py.File(file_path, 'r') as f:
#             # 获取所有的数据集键
#             # 读取特定的数据集，例如 'result_0'
#             if key[0] in f:
#                 print('success')
#                 result_0 = f['0'][:]
#         data_diff = torch.mean(torch.abs(torch.tensor(result_0,device=device,dtype =y.dtype )-y))
#         print('data_diff:',data_diff)
        uxI,DuI = residual_for_burgers(y, v=0.01, x_length=2.0, time_lentgh=1.0)
        A_mean = torch.mean(torch.abs(DuI))
        print(f'residual_for_burgers:{A_mean}')
        ux,Du = calculate_FDM(delta_t, delta_x, y, v)
        B_mean = torch.mean(torch.abs(Du))
        print(f'calculate_FDM:{B_mean}')
#         for i in range(2):
        uxp, Dup = ux[0:1, ...].cpu(),  Du[0:1, ...].cpu()
        uxIp, DuIp =  uxI[0:1, ...].cpu(),  DuI[0:1, ...].cpu()
#         print('ut.shape:',ut.shape)
#         plot_solutions(utp, f'ut_{i}')
#         plot_solutions(utIp, f'utI_{i}')

#         plot_solutions(uxp, f'ux_{i}')
#         plot_solutions(uxIp, f'uxI_{i}')

#         plot_solutions(uxxp, f'uxx_{i}')
#         plot_solutions(uxxIp, f'uxxI_{i}')

        plot_solutions(Dup, f'DU_{i}')
        plot_solutions(DuIp, f'DUI_{i}')
        i=i+1


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='./information.yaml', help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(config)
