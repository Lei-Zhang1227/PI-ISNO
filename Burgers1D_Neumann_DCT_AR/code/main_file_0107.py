"""
10/30:
对时间和空间维度都进行降维缩放；
11/5:
添加了test部分；
11/6 存档；
1. 统一分辨率的data loss和residual loss
"""
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
from Burger.utils import *
import time
import shutil
from tqdm import tqdm
from datetime import datetime
import torch
from Burger.NOs_dict.models import CosNO2d as Model


def get_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 2):.2f} MB")
    else:
        print("CUDA is not available.")


def get_process_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Process Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")


def run(config, args):
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
    print('------------------2-------------------')
    get_gpu_memory()
    get_process_memory()
    ################################################################
    # dataloader for .h5
    ################################################################
    data_config = config['data']
    batch_size = config['train']['batchsize']
    v = 0.01
    train_data = h5DatasetFor1DBurgers(data_config['datapath'],
                                       sub_x=data_config['sub_x'],
                                       sub_t=data_config['sub_t'],
                                       initial_step=data_config['initial_step'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
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
    print( first_dic)
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    if args.pretrain is not None:
        second_dic = f"./{config['prepare']['project']+config['prepare']['children']}"
        if not os.path.exists(second_dic):
            os.makedirs(second_dic)
        shutil.copy(f'./information.yaml', f"./{config['prepare']['project']+config['prepare']['children']}/information.yaml")
    else:
        shutil.copy('./information.yaml', f"./{config['prepare']['project']}")
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    ################################################################
    # model
    ################################################################
    # 定义使用的离散变换
    _trans = PARTIAL(Wrapper, [fft_fun, dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [ifft_fun, idctI_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOLII, T)
    input_channel = config['model']['input_channel'] + config['data']['initial_step'] - 1
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)  # .to(torch.float32)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")

    #     def count_parameters(layer):
    #         return sum(p.numel() for p in layer.parameters() if p.requires_grad)
    #     modes=16
    #     width=24
    #     bandwidth=2
    #     triL=0
    #     model = Model(3, modes, width, bandwidth, out_channels=1, triL=triL).to(device)
    #     for param in model.parameters():
    #         print(param.dtype)
    #     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print(f"Total trainable parameters: {total_params}")
    #     print("\nTrainable parameters for each layer/module:")
    #     for name, layer in model.named_children():
    #         num_params = count_parameters(layer)
    #         print(f"{name}: {num_params} parameters")
    ################################################################
    # 定义优化器
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                 lr=config['train']['base_lr'])
    now_lr = config['train']['base_lr']
    if config['train']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config['train']['milestones'],
                                                         gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['train']['gamma'],
                                                               threshold=1e-2, patience=config['train']['patience'],
                                                               verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['patience'],
                                                    gamma=config['train']['gamma'])

    print(f'{datetime.now()} --- set optimizer, scheduler ---')
    ################################################################
    # load model
    ################################################################
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        if args.load_lr is not True:
            pass
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
            scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率调度器状态

        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']

        # 获取 epoch
        epoch = checkpoint['epoch']
        print(f'模型【{args.pretrain}】已加载')
    print("当前所在目录:", os.getcwd())
#     print(f"{datetime.now()} --- set save dir :{config['prepare']['project']+config['prepare']['children']} ---")
    ################################################################
    # loss
    ################################################################
    print(f'{datetime.now()} --- training start ---')
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        # 从 checkpoint 中提取模型、优化器和其他状态
        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        os.chdir(f".{config['prepare']['children']}")
    else:
        loss_list = []
        test_loss_list = []
        lr_list = []
    
    model_save_record = []
    test_Loss_data = 100.0
    pbar = range(config['train']['epochs'])
    if args.pretrain is not None:
        pbar = range(epoch, epoch + config['train']['epochs'])
    rx = int(config['data']['data_sub_x'] / config['data']['sub_x'])
    rt = int(config['data']['data_sub_t'] / config['data']['sub_t'])
    if config['train']['use_tqdm'] == 0:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    for e in pbar:
        Loss_f = 0.0
        Loss_init = 0.0
        Loss_data = 0.0
        Loss_all = 0.0
        time00 = time.time()
        for x, y in train_loader:
            #             time0 = time.time()
            #             print(f'进入循环cost: {time0 - time00:.2f} S')
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # 确保数据在相同设备上
#             print('y.shape:',y.shape)
            #             torch.cuda.synchronize()
            #             time1 = time.time()
            #             print(f'读取数据cost: {time1 - time0:.2f} S')
            optimizer.zero_grad()
            out = model(x)
            #             torch.cuda.synchronize()
            #             time2 = time.time()
            #             print(f'模型推理: {time2 - time1:.2f} S')
            out = out.reshape(y.shape)
            if config['train']['loss_mode'] == 'both':
                out_data = out[:, ::rt, ::rx]
                y_data = y[:, ::rt, ::rx]
#                 print('y_data.shape:',y_data.shape)
                loss_data = F.mse_loss(out_data, y_data).to(torch.float32)
                #                 torch.cuda.synchronize()
                #                 time21 = time.time()
                #                 print(f'data loss: {time21 - time2:.2f} S')
                x_length, time_lentgh = config['data']['x_length'], config['data']['t_length']
                
                loss_init, loss_f, loss_b = PINO_loss_1DII(out, x[:, 0, :, 0], v, x_length, time_lentgh)
            #                 torch.cuda.synchronize()
            #                 time23 = time.time()
            #                 print(f'residual loss: {time23 - time21:.2f} S')
            elif config['train']['loss_mode'] == 'data':
                loss_init, loss_f = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                print(f'out.shape:{out.shape}')
                out_data = out[:, ::rx, ::rt]
                y_data = y[:, ::rx, ::rt]
                loss_data = F.mse_loss(out_data, y_data)
            else:
                x_length, time_lentgh = config['data']['x_length'], config['data']['t_length']
                loss_init, loss_f, loss_b = PINO_loss_1DII(out, x[:, 0, :, 0], v, x_length, time_lentgh)
                loss_data = torch.tensor(0.0, device=device)
            total_loss = loss_init * ic_weight + loss_f * f_weight + loss_data * data_weight
            #             torch.cuda.synchronize()
            #             time3 = time.time()
            #             print(f'all loss: {time3 - time2:.2f} S')
            assert not torch.isnan(total_loss).any(), "NaN in loss"
            total_loss.backward()
            #             torch.cuda.synchronize()
            #             time4 = time.time()
            #             print(f'backward: {time4 - time3:.2f} S')
            optimizer.step()
            Loss_data += loss_data
            Loss_init += loss_init
            Loss_f += loss_f
            Loss_all += total_loss
        #             time5 = time.time()
        #             print(f'all: {time5 - time0:.2f} S')
        Loss_data /= len(train_loader)
        Loss_init /= len(train_loader)
        Loss_f /= len(train_loader)
        Loss_all /= len(train_loader)
        if config['train']['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(Loss_all)
        else:
            scheduler.step()
        pbar.set_description(
            (
                f"epoch: {e + 1}, loss: {Loss_all.item():.5e}, loss_data: {Loss_data.item():.5e}, loss_PDE: {Loss_f.item():.5e},"
                f",loss_init: {Loss_init.item():.5e}, loss_test: {test_Loss_data:.5e}, now_lr: {now_lr:.2e}"))

        if e % config['train']['verbose_interval'] == 0:
            model.eval()
            now_lr = optimizer.state_dict()['param_groups'][0]['lr']  # 当前学习率查看
            loss_item = [e, Loss_all, Loss_init, Loss_f, Loss_data]
            loss_list.append(loss_item)
            lr_list.append(now_lr)
            # model test
            test_Loss_f = 0.0
            test_Loss_init = 0.0
            test_Loss_data = 0.0
            test_Loss_b = 0.0
            for x, y in test_loader:
                with torch.no_grad():
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # 确保数据在相同设备上
                    out = model(x)
                    out = out.reshape(y.shape)
                    test_loss_data = myloss(out, y)
                    x_length, time_lentgh = config['data']['x_length'], config['data']['t_length']
                    test_loss_init, test_loss_f, test_loss_b = PINO_loss_1DII(out, x[:, 0, :, 0], v, x_length,
                                                                              time_lentgh)
                    test_Loss_data += test_loss_data
                    test_Loss_init += test_loss_init
                    test_Loss_f += test_loss_f
                    test_Loss_b += test_loss_b
            test_Loss_data /= len(test_loader)
            test_Loss_init /= len(test_loader)
            test_Loss_f /= len(test_loader)
            test_Loss_b /= len(test_loader)
            test_loss_item = [e, test_Loss_init, test_Loss_f, test_Loss_data, test_Loss_b]
            test_loss_list.append(test_loss_item)
            save_checkpoint(model, e, optimizer, scheduler, loss_list,
                            lr_list, test_loss_list, filename=f'checkpoint')
            if e % config['train']['check_epochs'] == 0:
                save_checkpoint(model, e, optimizer, scheduler, loss_list,
                                lr_list, test_loss_list, filename=f'checkpoint-{e}')
            model.train()
    #         print('------------------10-------------------')
    #         get_gpu_memory()
    #         get_process_memory()
    print(f'{datetime.now()} --- training succeed ---')


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='./information.yaml', help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
    parser.add_argument('--load_lr', action='store_true', help='pretrain model path')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(config, args)
