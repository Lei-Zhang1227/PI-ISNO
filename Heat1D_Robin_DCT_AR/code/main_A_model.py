"""
heat equation with Chebshev transform in CGL points
model A:
从初始条件映射到前五个时间步的纯数据驱动model
"""
import os, sys
from pathlib import Path
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))
from argparse import ArgumentParser
import yaml
import tqdm
import shutil
from tqdm import tqdm
from model import *

CASE_ROOT = Path(__file__).resolve().parents[1]

def resolve_case_path(path_like):
    path = Path(path_like)
    return str(path if path.is_absolute() else CASE_ROOT / path)

from dataset import *
from loss import *
from utils import *
from datetime import datetime
import pandas as pd
import pickle
import math
import torch
import numpy as np


def run(config, args):
    # region prepare
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(config['prepare']['seed'])
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # endregion
    # region dataloader
    n_output_steps = 4
    data_config = config['data']
    batch_size = config['train']['batchsize']
    train_data = h5DatasetFor1DHeat_A(resolve_case_path(data_config['datapath']),
                                      sub_x=data_config['sub_x'],
                                      sub_t=data_config['sub_t'], n_output_steps=n_output_steps,
                                      initial_step=1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    test_data = h5DatasetFor1DHeat_A(resolve_case_path(data_config['datapath']),
                                     sub_x=data_config['sub_x'],
                                     sub_t=data_config['sub_t'], n_output_steps=n_output_steps,
                                     initial_step=1, if_test=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=False,
                                              num_workers=2, pin_memory=True)

    train_size, test_size = train_data.data_list.shape[0], test_data.data_list.shape[0]
    ntrain, ntest = train_size, test_size
    print('size-of-train/val:', train_size, test_size)
    # endregion
    # region location
    ################################################################
    first_dic = resolve_case_path(config['prepare']['project'])
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    if args.pretrain is not None:
        src = args.config_path
        base = os.path.basename(src)
        name, ext = os.path.splitext(base)

        # 目标默认路径（同名）
        dst = os.path.join(first_dic, base)

        # 如果源和目标是同一个文件，自动加 -1 / -2 ...
        if os.path.abspath(src) == os.path.abspath(dst):
            i = 1
            while True:
                new_name = f"{name}-retrain--{i}{ext}"
                new_dst = os.path.join(first_dic, new_name)
                if not os.path.exists(new_dst):
                    dst = new_dst
                    break
                i += 1

        # 执行复制
        shutil.copy(src, dst)
    else:
        shutil.copy(args.config_path, first_dic)
    # os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir: {first_dic} ---")
    # endregion
    # region model
    ################################################################
    # 这里有所不同的是，不再将这个一维含时的问题视为二维问题
    degree = config['model']['degree']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    in_channel = 2
    FLOAT = config['model']['float']
    FLOAT_CONFIG = {
        32: {'model_class': SOL_heat_32, 'dtype': torch.float32},
        64: {'model_class': SOL_heat, 'dtype': torch.float64},
    }
    model_config = FLOAT_CONFIG[FLOAT]
    model = model_config['model_class'](in_channel, degree, width, bandwidth, n_output_steps).to(device)
    dtype = model_config['dtype']
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- set model {model_config['model_class']}. Total trainable parameters: {total_params}")

    def count_parameters(layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("\nTrainable parameters for each layer/module:")
    for name, layer in model.named_children():
        num_params = count_parameters(layer)
        print(f"{name}: {num_params} parameters")
    # endregion
    # region optimizer
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                 lr=config['train']['base_lr'])
    now_lr = config['train']['base_lr']

    scheduler_name = config['train']['scheduler']
    if config['train']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config['train']['milestones'],
                                                         gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['train']['gamma'],
                                                               threshold=1e-2, patience=config['train']['patience'],
                                                               verbose=True)
    elif config['train']['scheduler'] == 'cosine_schedule_with_warmup':
        epoch_set = config['train']['epochs']
        bfe = math.ceil(train_size / batch_size)
        step = bfe * epoch_set
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=step * config['train']['cosine_schedul'], num_training_steps=step
        )
        cosine_schedul = config['train']['cosine_schedul']
        print(
            f'cosine_schedule_with_warmup, warm epoch is {int(cosine_schedul * epoch_set)}, total steps is {step}')
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['patience'],
                                                    gamma=config['train']['gamma'])

    # 定义梯度裁剪控制器
    clipper = RobustAdaptiveGradientClipper()
    print(
        f'{datetime.now()} --- set optimizer,lr is {now_lr}, scheduler:{scheduler_name},---')
    # endregion
    # region load model
    ################################################################
    if args.pretrain is not None:
        checkpoint = torch.load(resolve_case_path(args.pretrain))
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        if config['train']['retrain_load_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
            scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率调度器状态
        else:
            pass
        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        grad = checkpoint['grad']
        # 获取 epoch
        epoch = checkpoint['epoch']
        best_error = loss_list[-1][0]
        print(f'模型【{args.pretrain}】已加载,当前训练loss为：{best_error}')
    else:
        loss_list = []
        test_loss_list = []
        lr_list = []
        grad = []
        best_error = 100.0
    # endregion
    # region information_write
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[Experiment Log | {timestamp}]\n")
        f.write(f"├── Case: 1D Heat equation with Robin BCs\n")
        f.write(
            f"├── Model: Discrete Chebshev Transform (DCT) for spatial dimension modeling, and Autoregressive (AR) process for temporal dimension modeling\n")
        f.write(f"\n├─ Model Configuration\n")
        f.write(f"├── Model Class: SQL1D\n")
        # 核心参数（树形结构）
        if args.pretrain is not None:
            f.write(f"├── Loaded model: {args.pretrain}\n")
            f.write(f"├── Current loss: {best_error:.4e}\n")
        else:
            f.write(f"├── Initialized new model\n")
        f.write(f"├── Architecture Parameters:\n")
        f.write(f"│   ├── input_channels: {in_channel}\n")
        f.write(f"│   ├── modes: {degree}\n")
        f.write(f"│   ├── width: {width}\n")
        f.write(f"│   ├── bandwidth: {3}\n")
        f.write(f"│   ├── output_channels: {config['model']['output_channel']}\n")
        f.write(f"│   ├── dim: {config['model']['dim']}\n")
        f.write(f"│   └── triL: {config['model']['triL']}\n")
        # 设备信息
        f.write(f"├── Device: {next(model.parameters()).device}\n")
        f.write(f"├── Model Hyparameters: {total_params:,}\n")
        f.write(f"├── Trainable Parameters: {total_params:,}\n")  # 使用千位分隔符
        f.write(f"├── Layer Details:\n")
        for name, layer in model.named_children():
            num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            f.write(f"│   ├── {name}: {num_params:,} parameters\n")
        f.write(f"└── Note: Model architecture recorded\n")
        f.write(f"\n├─ Data Configuration\n")
        f.write(f"├── Data: {resolve_case_path(data_config['datapath'])}\n")
        f.write(f"├── Train/Test Size: {ntrain:,}/{ntest:,} samples\n")
        f.write(f"├── Batch Size: {batch_size} (train), 1 (test)\n")
        f.write(f"├── Spatial Subsampling: sub_x={int((data_config['nx'] - 1) / data_config['sub_x']) + 1}\n")
        f.write(f"├── Temporal Subsampling: sub_t={int((data_config['nt'] - 1) / data_config['sub_t']) + 1}\n")
        f.write(f"├── Initial Steps: {data_config['initial_step']}\n")
        f.write(f"│   └── k: ν={0.02}\n")
        f.write(f"└── Note: Data recorded\n")
        f.write("-" * 60 + "\n")  # 分隔线
        f.close()
    # endregion
    # region train
    ################################################################
    nx = int((data_config['nx'] - 1) / data_config['sub_x']) + 1
    print('nx:', nx)
    model_save_record = [[0, 100]]
    myloss = LpLoss(size_average=True)
    myloss_test = LpLoss(size_average=False)
    if args.pretrain is not None:
        ebar = trange(epoch, epoch + config['train']['epochs'], desc="Epoch")
        pre_epoch = epoch
    else:
        ebar = trange(config['train']['epochs'], desc="Epoch")
        pre_epoch = 0
    rx = int(config['data']['data_sub_x'] / config['data']['sub_x'])
    print(f'rx:{rx}')
    desc = DescStr()
    time_0 = time.time()
    time_old = time.time()
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n├─ Training Start! | {timestamp}] \n")
        f.close()
    model.eval()
    # val_l2_step = 0
    val_l2_full = 0
    with torch.no_grad():
        for xx, yy, _ in test_loader:
            print(f'xx.shape{xx.shape},yy.shape:{yy.shape}')
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            pred = model(xx)
            assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
            _batch = yy.size(0)
            val_l2_full += myloss_test(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
        test_l2 = val_l2_full / ntest
        print(f'epoch:{pre_epoch}, test_l2: {test_l2}')
    model.train()
    First_data = True
    avg_epoch_loss = 10000
    for e in ebar:
        epoch_loss = 0
        epoch_physics_loss = 0
        epoch_grad_norms = []

        for xx, yy, _ in train_loader:
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xx)
            assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"

            _batch = yy.size(0)
            out_data = pred.reshape(_batch, -1)
            y_data = yy.reshape(_batch, -1)
            loss_data = myloss(out_data, y_data)

            # 检查NaN/Inf
            assert torch.isfinite(loss_data).all(), f"Invalid loss: {loss_data.item()}"

            # 反向传播
            loss_data.backward()
            grad_norm = clipper.step(model)
            optimizer.step()

            # Scheduler（batch级别）
            if scheduler_name == 'cosine_schedule_with_warmup':
                scheduler.step()

            # 记录
            current_lr = optimizer.param_groups[0]['lr']
            loss_list.append([loss_data.item(), 0, 0, e])
            epoch_grad_norms.append(grad_norm)
            epoch_loss += loss_data.item()

        # Epoch结束
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_grad_norm = np.mean(epoch_grad_norms)
        grad.append([e, avg_epoch_loss, avg_grad_norm])
        lr_list.append([current_lr, 0, e])

        # 更新 epoch 进度条描述
        ebar.set_description(
            f"Epoch {e + 1}: Loss: {avg_epoch_loss:.4e}, Test L2: {test_l2:.4e}, lr: {current_lr:.2e}"
        )
        # ============ Epoch 结束后更新 lr ============
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(avg_epoch_loss)
        elif scheduler_name == 'cosine_schedule_with_warmup':
            pass  # cosine scheduler 在每个 batch 更新，不在这里
        else:
            scheduler.step()
        # 保存最佳模型
        if best_error > avg_epoch_loss:
            best_error = avg_epoch_loss
            model_save_record.append([e, avg_epoch_loss])
            save_checkpoint(model, e, optimizer, scheduler, loss_list,
                            test_loss_list, lr_list, model_save_record, grad,
                            filename=f'{first_dic}/checkpoint-best')
        if e % config['train']['verbose_interval'] == 0:
            model.eval()
            val_l2_full = 0
            with torch.no_grad():
                for xx, yy, _ in test_loader:
                    xx = xx.to(device, dtype=dtype, non_blocking=True)
                    yy = yy.to(device, dtype=dtype, non_blocking=True)
                    pred = model(xx)
                    assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
                    _batch = yy.size(0)
                    val_l2_full += myloss_test(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                test_l2 = val_l2_full / ntest
                test_loss_list.append([test_l2, e])
                if e % config['train']['check_epochs'] == 0:
                    save_checkpoint(model, e, optimizer, scheduler, loss_list, test_loss_list,
                                    lr_list, model_save_record, grad, filename=f'{first_dic}/checkpoint-{e}')
                    if e % 100 == 0:
                        time_elapsed = time.time() - time_0
                        hours = int(time_elapsed // 3600)
                        minutes = int((time_elapsed % 3600) // 60)
                        time_elapsed_100 = time.time() - time_old
                        # 转换为小时和分钟
                        hours_100 = int(time_elapsed_100 // 3600)
                        minutes_100 = int((time_elapsed_100 % 3600) // 60)
                        with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
                            f.write(f"├── Test l2 error in epoch {e}: {test_l2:.4e}\n")
                            f.write(f"│   └── Costed Time: {hours}h {minutes}m\n")
                            f.write(f"│   └── Costed Time per 100 epoch: {hours_100}h {minutes_100}m\n")
                            f.write(f"│   └── Current Best epoch: {model_save_record[-1][0]}m\n")
                            f.write(f"│       └── Batch loss: {model_save_record[-1][1]:.4e}\n")
                            f.close()
                        time_old = time.time()
            model.train()
    print(f'{datetime.now()} --- training succeed ---')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_elapsed = time.time() - time_0
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        f.write(f"├── training succeed | [{timestamp}]\n")
        f.write(f"│   └── Costed Time: {hours}h {minutes}m\n")
        f.write("-" * 60 + "\n")  # 分隔线
        f.close()
    # endregion


def test(config, args):
    """
    一个全面的评测函数，可以再三个不同分辨率上计算模型的l2误差
    :param config:
    :param args:
    :return:
    """
    # region prepare
    ################################################################
    # prepare
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # endregion
    # region dataloader
    ################################################################
    # dataloader
    ################################################################
    data_config = config['data']
    origin_nx = data_config['nx']
    initial_step = data_config['initial_step']
    n_output_steps = 4
    test_data_1x = h5DatasetFor1DHeat_A(resolve_case_path(data_config['datapath']),
                                        sub_x=1,
                                        sub_t=data_config['sub_t'], n_output_steps=n_output_steps,
                                        initial_step=1, if_test=True)
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)

    test_data_1_2x = h5DatasetFor1DHeat_A(resolve_case_path(data_config['datapath']),
                                          sub_x=2,
                                          sub_t=data_config['sub_t'], n_output_steps=n_output_steps,
                                          initial_step=1, if_test=True)
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_4x = h5DatasetFor1DHeat_A(resolve_case_path(data_config['datapath']),
                                          sub_x=4,
                                          sub_t=data_config['sub_t'], n_output_steps=n_output_steps,
                                          initial_step=1, if_test=True)
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_8x = h5DatasetFor1DHeat_A(resolve_case_path(data_config['datapath']),
                                          sub_x=8,
                                          sub_t=data_config['sub_t'], n_output_steps=n_output_steps,
                                          initial_step=1, if_test=True)
    test_loader_1_8x = torch.utils.data.DataLoader(test_data_1_8x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_16x = h5DatasetFor1DHeat_A(resolve_case_path(data_config['datapath']),
                                           sub_x=16,
                                           sub_t=data_config['sub_t'], n_output_steps=n_output_steps,
                                           initial_step=1, if_test=True)
    test_loader_1_16x = torch.utils.data.DataLoader(test_data_1_16x, batch_size=1, shuffle=False,
                                                    num_workers=0, pin_memory=True)

    test_size = test_data_1_16x.data_list.shape[0]
    ntest = test_size
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    first_dic = resolve_case_path(config['prepare']['project'])
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    n_output_steps = 4
    degree = config['model']['degree']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    in_channel = 2
    FLOAT = config['model']['float']
    FLOAT_CONFIG = {
        32: {'model_class': SOL_heat_32, 'dtype': torch.float32},
        64: {'model_class': SOL_heat, 'dtype': torch.float64},
    }
    model_config = FLOAT_CONFIG[FLOAT]
    model = model_config['model_class'](in_channel, degree, width, bandwidth, n_output_steps).to(device)
    dtype = model_config['dtype']
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- set model {model_config['model_class']}. Total trainable parameters: {total_params}")

    def count_parameters(layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("\nTrainable parameters for each layer/module:")
    for name, layer in model.named_children():
        num_params = count_parameters(layer)
        print(f"{name}: {num_params} parameters")
    ################################################################
    # load model
    ################################################################
    if args.pretrain is not None:
        model_name = args.pretrain[:-8]
        model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
        checkpoint = torch.load(resolve_case_path(args.pretrain))
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']  # checkpoint['loss_for_train']
        test_loss_list = checkpoint.get('test_loss_list')  # checkpoint.get('test_loss_list')
        grad_array = checkpoint['grad']
        if test_loss_list is None:
            print("[Warning] Checkpoint does not contain 'test_loss_list'.")
            test_loss_list = []
        lr_list = checkpoint.get('lr_list')
        if lr_list is None:
            print("[Warning] Checkpoint does not contain 'lr_list'.")
            lr_list = []

        # 获取 epoch
        epoch = checkpoint.get('epoch', checkpoint.get('epochs', None))
        if epoch is None:
            print("[Warning] Checkpoint does not contain 'epoch' or 'epochs'.")
            epoch = 0
        print(f"{datetime.now()} --- model【{args.pretrain}】has been loaded, {epoch} epochs has trained")
    else:
        print(os.getcwd())
        checkpoint = torch.load(os.path.join(first_dic, 'checkpoint-best.pth.tar'))
        grad_array = checkpoint['grad']
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        print('len(loss_list0,len(test_loss_list),len(lr_list)):', len(loss_list), len(test_loss_list), len(lr_list))
        # 获取 epoch
        epoch = checkpoint['epoch']
        model_name = 'checkpoint-best'
        print(f"{datetime.now()} --- current best model has been loaded, {epoch} epochs has trained")

    # endregion
    plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array,
                               f'{first_dic}/loss_carve_for_{model_name}')
    # region evaluate
    init_t = int(data_config['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    test_loaders = {
        f'test_{origin_nx}': test_loader_1x,
        f'test_{int((origin_nx - 1) / 2) + 1}': test_loader_1_2x,
        f'test_{int((origin_nx - 1) / 4) + 1}': test_loader_1_4x,
        f'test_{int((origin_nx - 1) / 8) + 1}': test_loader_1_8x,
        #         f'test_{int((origin_nx - 1) / 16) + 1}': test_loader_1_16x,
    }
    nx_test = [513, 257, 129, 65]
    results = []
    errors_for_talk_all = {}
    visualize_results_all = []
    i = 0
    for name, test_loader in test_loaders.items():
        errors_for_talk = []
        model.eval()  # 将模型设置为评估模式
        first = True
        with torch.no_grad():
            print(f'dataset:{name}')
            b = 0
            for xx, yy, _ in test_loader:
                # print(f'xx.shape{xx.shape},yy.shape:{yy.shape}')
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                pred = model(xx)
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
                _batch = yy.size(0)
                l2_error = myloss(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                errors_for_talk.append([l2_error, int(b)])
                b += 1
            error_records_sorted = sorted(errors_for_talk, key=lambda x: x[0])
            n_samples = len(error_records_sorted)

            selected_indices = set(
                [r[-1] for r in error_records_sorted[:3]] +  # best 3
                [r[-1] for r in error_records_sorted[-3:]] +  # worst 3
                [r[-1] for r in error_records_sorted[n_samples // 2 - 1: n_samples // 2 + 2]]  # mid 3
            )
            print('selected_indices:', selected_indices)
            test_iter = iter(test_loader)
            visualize_results = []
            for b in tqdm(range(len(test_loader))):
                xx, yy, sample_name = next(test_iter)
                if b not in selected_indices:
                    continue
                print(f'b is {b},dataset is {name}')
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                pred = model(xx)
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"  # pred.shape： [1,nx,101,1]
                _batch = yy.size(0)
                l2_error = error_records_sorted[[r[-1] for r in error_records_sorted].index(b)][0]
                if b in [r[-1] for r in error_records_sorted[:3]]:
                    category = 'best'
                elif b in [r[-1] for r in error_records_sorted[-3:]]:
                    category = 'worst'
                else:
                    category = 'mid'
                print('pred.squeeze().shape:', pred.squeeze().shape)

                visualize_results.append({
                    'Dataset': name,
                    'index': b,
                    'category': category,
                    'l2_error': l2_error,
                    'pred': pred.squeeze().cpu().numpy(),
                    'yy': yy.squeeze().cpu().numpy(),
                })
                print('len(visualize_results):', len(visualize_results))
            visualize_results_all.append(visualize_results)

        i += 1
        # 计算均值和标准差
        errors_for_talk = np.array(errors_for_talk)
        mean_l2_error = np.mean(errors_for_talk[:, 0])
        std_l2_error = np.std(errors_for_talk[:, 0])
        max_l2_error = np.max(errors_for_talk[:, 0])
        min_l2_error = np.min(errors_for_talk[:, 0])
        # 保存结果
        results.append({
            'Dataset': name,
            'Mean Relative L2 Error': mean_l2_error,
            'Std Relative L2 Error': std_l2_error,
            'Max Relative L2 Error': max_l2_error,
            'Min Relative L2 Error': min_l2_error, })
        errors_for_talk_all[name] = errors_for_talk
    # 将结果转换为 DataFrame 并保存为 CSV 文件
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)
    print('len(visualize_results_all):', len(visualize_results_all))
    with open('visualize_results.pkl', 'wb') as f:  # 注意 'wb' 二进制写入
        pickle.dump(visualize_results, f)
    with open('error_for_talk_all.pkl', 'wb') as f:  # 注意 'wb' 二进制写入
        pickle.dump(errors_for_talk_all, f)
    print("测试结果已保存到 test_results.csv/error_for_talk_all.pkl,可视化数据已保存到visualize_results_results.pkl")
    # endregion
    # region visualize
    ################################################################
    # visualize_results
    ################################################################
    # 示例：可视化第 idx 个样本的时间演化场
    for visualize_results in visualize_results_all:
        case = visualize_results[0]['Dataset']
        plot_visualization_results_data(visualize_results, save_dir=f'./figures_{case}')
    print('可视化完成~')
    # endregion
    # # region error talk
    # '''
    # 这里出图使用的是最大分辨率，计算loss用的也是最大分辨率；
    # '''
    # traing_nx = data_config['sub_x']
    # training_data = f'test_{int((origin_nx - 1) / traing_nx) + 1}'
    # print('training_data for error talk:', training_data)
    # talk_data = errors_for_talk_all[training_data]
    # sample_ids = talk_data[:, 3]
    #
    # # 绘制样本误差分布图
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    #
    # # 主纵轴：data_loss（对数坐标）
    # ax1.scatter(sample_ids, talk_data[:, 0], color='#0B5873', label='data_loss', alpha=0.7, s=10)
    # ax1.set_xlabel('Sample ID')
    # ax1.set_ylabel('data_loss', color='#0B5873')
    # # ax1.set_yscale('log')  # 设置主纵轴为对数坐标
    # ax1.tick_params(axis='y', labelcolor='#0B5873')
    #
    # # 副纵轴：pde_residual 和 ref_residual（对数坐标）
    # ax2 = ax1.twinx()
    # ax2.scatter(sample_ids, talk_data[:, 1], color='#8A0011', label='pde_residual', marker='x', s=10)
    # ax2.scatter(sample_ids, talk_data[:, 2], color='#107A38', label='ref_residual', marker='^', s=10)
    # ax2.set_ylabel('Residuals', color='#8A0011')
    # # ax2.set_yscale('log')  # 设置副纵轴为对数坐标
    # ax2.tick_params(axis='y', labelcolor='#8A0011')
    #
    # # 添加纵向网格线（基于主横轴）
    # ax1.grid(True, axis='x', linestyle='--', alpha=0.5)  # 纵向虚线网格
    # ax1.grid(True, axis='y', linestyle=':', alpha=0.5)  # 横向点线网格（对数坐标）
    #
    # # 图例和标题
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    # plt.title('Data Loss vs. PDE/Ref Residuals (Log Scale)')
    # plt.savefig('error_talk.png', dpi=300, bbox_inches='tight', transparent=True)
    #
    # # error talk
    # mean_loss = np.mean(talk_data[:, 0])
    # std_loss = np.std(talk_data[:, 0])
    # threshold = mean_loss + 1 * std_loss
    #
    # high_loss_samples = talk_data[talk_data[:, 0] > threshold]
    # print(f"存在高data_loss样本{len(high_loss_samples)}个：\n{high_loss_samples}")
    #
    # # 计算残差与data_loss的相关系数
    # corr_pde = np.corrcoef(talk_data[:, 0], talk_data[:, 1])[0, 1]  # data_loss vs pde_residual
    # corr_ref = np.corrcoef(talk_data[:, 0], talk_data[:, 2])[0, 1]  # data_loss vs ref_residual
    #
    # print(f"data_loss 与 pde_residual 的相关系数: {corr_pde:.3f}")
    # print(f"data_loss 与 ref_residual 的相关系数: {corr_ref:.3f}")
    #
    # high_loss_ids = high_loss_samples[:, 3].astype(int)
    # print("高data_loss的样本编号：", high_loss_ids)
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
    #     f.write(f"├── High Loss Analysis | [{timestamp}]\n")
    #     f.write(f"│   ├── High data_loss samples: {len(high_loss_samples)} cases\n")
    #     f.write(f"│   ├── Correlation coefficients:\n")
    #     f.write(f"│   │   ├── data_loss vs pde_residual: {corr_pde:.3f}\n")
    #     f.write(f"│   │   └── data_loss vs ref_residual: {corr_ref:.3f}\n")
    #     f.write(f"│    └── High loss sample IDs: {high_loss_ids.tolist()}\n")
    #     f.write(f"│\n")
    #     f.write(f"│   L2 metrics:\n")
    #     f.close()
    #
    # talk_visualize_results = []
    #
    # model.eval()  # 将模型设置为评估模式
    # loader = test_loaders[training_data]
    # with torch.no_grad():
    #     test_iter = iter(loader)
    #     for b in tqdm(range(len(loader))):
    #         xx, yy, grid, name = next(test_iter)
    #         if b in high_loss_ids:
    #             #                 print('xx.shape:',xx.shape)
    #             xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
    #                                                                                                        non_blocking=True)  # 确保数据在相同设备上
    #             inp_shape = list(xx.shape)
    #             inp_shape = inp_shape[:-2]
    #             inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
    #             outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
    #             pred = torch.empty(yy.shape, device=xx.device)
    #             pred[..., 0:init_t, :] = yy[..., 0:init_t, :]
    #             gridt = torch.tensor(np.linspace(0, 1, t_train), dtype=torch.float32, device=xx.device).reshape(
    #                 t_train, 1)
    #             for t in range(init_t, t_train):
    #                 current_time = gridt[t:t + 1, :]
    #                 inp = xx.reshape(inp_shape)
    #                 current_time = current_time.view(1, 1, 1).expand(xx.size(0), xx.size(1), 1)  # 扩展为[batch, nx, 1]
    #                 delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
    #                 last_step = xx[..., -1:, :]
    #                 out = last_step + delta
    #                 pred[..., t:t + 1, :] = out
    #                 xx = torch.cat((xx[..., 1:, :], out), dim=-2)
    #             assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
    #             # print('type(Du_yy):', Du_yy.dtype)
    #             Du_pred = compute_heat_residual(pred, D2).squeeze(-1)
    #             Du_yy = compute_heat_residual(yy, D2).squeeze(-1)
    #             mean_residual_pred = torch.mean(torch.abs(Du_pred))
    #             mean_residual_yy = torch.mean(torch.abs(Du_yy))
    #             # print(f'mean_residual_pred:{mean_residual_pred},mean_residual_yy:{mean_residual_yy}')
    #             _yy = yy[..., init_t + 1:t_train, :]  # if t_train is not -1
    #             _pred = pred[..., init_t + 1:t_train, :]
    #             _batch = yy.size(0)
    #             l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
    #             # print(' evaluarion: mean_residual_pred.shape:', mean_residual_pred.shape, type(mean_residual_pred))
    #             talk_visualize_results.append({
    #                 'index': name,
    #                 'yy': yy.squeeze().cpu().numpy(),
    #                 'pred': pred.squeeze().cpu().numpy(),
    #                 'Du_pred': torch.abs(Du_pred).squeeze().mT.cpu().numpy(),
    #                 'Du_yy': torch.abs(Du_yy).squeeze().mT.cpu().numpy(),
    #                 'Du_pred_mean': mean_residual_pred.item(),
    #                 'Du_yy_mean': mean_residual_yy.item(),
    #                 'l2_error': l2_error,
    #             })
    # for demo in talk_visualize_results:
    #     index = demo['index']
    #     Du_yy_mean = demo['Du_yy_mean']
    #     Du_yy = demo['Du_yy']
    #     Du_pred_mean = demo['Du_pred_mean']
    #     Du_pred = demo['Du_pred']
    #     pred_val = demo['pred']
    #     true_val = demo['yy']
    #     l2 = demo['l2_error']
    #     plt_name = f'Talk_{index}'
    #     # print('type(l2):', type(l2), type(Du_pred_mean), type(Du_yy_mean))
    #     with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
    #         f.write(
    #             f"│    └── key: {index}, Data L2 error: {l2:.3e}, Du_pred: {Du_pred_mean:.3e}, Du_yy: {Du_yy_mean:.3e}\n")
    #         f.close()
    #     print('type(l2): l2, Du_pred_mean, Du_yy_mean')
    #     print('type(l2):', index, l2, Du_pred_mean, Du_yy_mean)
    #     plot_solution_with_Du(true_val, pred_val, plt_name, Du_pred, Du_pred_mean, Du_yy, Du_yy_mean, l2,
    #                           save_svg=False,
    #                           Aspect_Ratio=1 / 1.4)
    #
    # # 分析
    #
    # # endregion


# endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='./yaml/information.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
    parser.add_argument('--load_lr', action='store_true', help='pretrain model path')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        try:
            with open(config_file, encoding="utf-8") as stream:
                config = yaml.load(stream, yaml.FullLoader)
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试GB18030（兼容GBK）
            with open(config_file, encoding="gb18030") as stream:
                config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(config, args)
        test(config, args)
    else:
        test(config, args)
