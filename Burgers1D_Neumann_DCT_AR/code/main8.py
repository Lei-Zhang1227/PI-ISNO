"""
24/10/22：
模型：
输入某一时刻的值，得到下一时刻的值；
训练：
自回归训练
25/01/14
1.由于在模型中没有添加flat，所以在info中width要保持相同的数，不然会报错；
2.由于是向量自回归，时间步的大小最好固定；
25/06/10
转眼半年过去，居然还在纠结这玩意
25/7/18
添加学习率预热
添加学习率余弦decay
# 添加参数空间预热
25/7/29
优化了存档逻辑；
添加了自适应的梯度裁剪；
优化了绘图逻辑；
存档不动 Burger/output/only_PDE/E/训练使用；
添加了时间维度课程学习的调度器；
将乱七八糟的模块移到了utils；
在lr的第二维度添加了t_train;
25/8/15
添加了跨分辨率的模型训练dataloader
25/8/21
修正了不知道啥时候突然变回原来的iter错误；
25/9/7
二维变换模型
"""
import os, sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))

from argparse import ArgumentParser
import yaml
from functools import partial as PARTIAL
import tqdm
import shutil
from tqdm import tqdm
from Burger.model import SOLII
from Burger.datasets import *
from Burger.loss import *
from Burger.utils import *
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pickle
import math
import torch
import numpy as np
from torch.utils.data import DataLoader


def run(config):
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
    data_config = config['data']
    batch_size = config['train']['batchsize']
    v = 0.01
    if data_config['loader'] == 'FixedRes':
        train_data = h5DatasetFor1DBurgers_fix_2D(data_config['datapath'],
                                                  sub_x=data_config['sub_x'],
                                                  sub_t=data_config['sub_t'],
                                                  initial_step=data_config['initial_step'])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=2, pin_memory=True)
        test_data = h5DatasetFor1DBurgers_fix_2D(data_config['datapath'],
                                                 sub_x=data_config['sub_x'],
                                                 sub_t=data_config['sub_t'],
                                                 initial_step=data_config['initial_step'], if_test=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                                  num_workers=2, pin_memory=True)
    else:
        file_path = r'/code/Burger/resolution_dfy.pkl'
        with open(file_path, 'rb') as f:  # 注意使用 'rb' 模式（二进制读取）
            resolution_df = pickle.load(f)
        train_data = h5DatasetFor1DBurgers_muti_2D(
            filepath=data_config['datapath'],
            initial_step=data_config['initial_step'],
            sub_t=1,
            sub_x=1,  # 默认值（会被动态覆盖）
            resolution_df=resolution_df
        )
        test_data = h5DatasetFor1DBurgers_muti_2D(
            filepath=data_config['datapath'],
            initial_step=data_config['initial_step'],
            sub_t=1,
            sub_x=1,  # 默认值（会被动态覆盖）
            resolution_df=resolution_df,
            if_test=True
        )
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                                  num_workers=1, pin_memory=True)
        resolution_groups = {}
        for idx, name in enumerate(train_data.data_list):
            res = resolution_df.loc[resolution_df["sample_id"] == name, "recommended_res"].values[0]
            if res not in resolution_groups:
                resolution_groups[res] = []
            resolution_groups[res].append(idx)
        group_weights = {"129": 1, "257": 1, "517": 1}  # 129组的样本扩展为2倍

        if data_config['loader'] == 'MixedRes':
            batch_sampler = ResolutionWeightedBatchSampler(
                resolution_groups=resolution_groups,
                batch_size=batch_size,
                group_weights=group_weights,
                shuffle=True
            )
            train_loader = DataLoader(
                train_data,
                batch_sampler=batch_sampler,
                num_workers=2
            )
        if data_config['loader'] == 'ProgRes':
            batch_sampler = ResolutionWeightedBatchSamplerII(
                resolution_groups=resolution_groups,
                batch_size=batch_size,
                group_weights=group_weights,
                shuffle=True
            )
            # 创建 DataLoader
            phases = config['data']["phases"]
            phase_manager = ResolutionPhaseManager(phases)
            train_loader = DataLoader(
                train_data,
                batch_sampler=batch_sampler,
                num_workers=2
            )
            sum_setps = 0
            for epoch in range(config['train']['epochs']):
                for phase in phases:
                    if epoch == phase["start_epoch"]:
                        print(f"\n[Epoch {epoch}] Switching to resolutions: {phase['resolutions']}")
                        batch_sampler.set_active_resolutions(phase["resolutions"])
                        print(len(train_loader))
                        break  # 找到匹配后立即退出
                sum_setps += len(train_loader)
        print(f'{datetime.now()} --- set multi resolution dataloader ---')

    train_size, test_size = train_data.data_list.shape[0], test_data.data_list.shape[0]
    ntrain, ntest = train_size, test_size
    print('size-of-train/val:', train_size, test_size)
    # endregion
    # region location
    ################################################################
    first_dic = f"/code/Burger{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    if args.pretrain is not None:
        shutil.copy(f'./information.yaml', f"/code/Burger{config['prepare']['project']}")
    else:
        shutil.copy('./information.yaml', f"/code/Burger{config['prepare']['project']}")
    # os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :/code/Burger{config['prepare']['project']} ---")
    # endregion
    # region log
    log_file = "process_log.txt"
    log_dic = f"/output/logs/"
    if not os.path.exists(log_dic):
        os.makedirs(log_dic)
    # writer = SummaryWriter(log_dir=log_dic)  #
    # endregion
    # region model
    ################################################################
    # 这里有所不同的是，不再将这个一维含时的问题视为二维问题
    _trans = PARTIAL(Wrapper, [fft_fun, dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [ifft_fun, idctI_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOLII, T)
    input_channel = config['model']['input_channel'] * config['data']['initial_step'] + 2
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)  # .to(torch.float32)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")

    def count_parameters(layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("\nTrainable parameters for each layer/module:")
    for name, layer in model.named_children():
        num_params = count_parameters(layer)
        print(f"{name}: {num_params} parameters")
    # endregion
    # region 定义优化器
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                 lr=config['train']['base_lr'])
    now_lr = config['train']['base_lr']
    # 定义学习率调整器
    scheduler_name = config['train']['scheduler']
    if config['train']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config['train']['milestones'],
                                                         gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['train']['gamma'],
                                                               threshold=1e-1, patience=config['train']['patience'],
                                                               verbose=True)
    elif config['train']['scheduler'] == 'cosine_schedule_with_warmup':
        epoch_set = config['train']['epochs']
        bfe = math.ceil(train_size / batch_size)
        step = bfe * epoch_set if data_config['loader'] != 'ProgRes' else sum_setps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=step * config['train']['cosine_schedul'], num_training_steps=step
        )
        cosine_schedul = config['train']['cosine_schedul']
        print(f'cosine_schedule_with_warmup, warm epoch is {int(cosine_schedul * epoch_set)}')
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['patience'],
                                                    gamma=config['train']['gamma'])
    # 定义梯度裁剪控制器
    clipper = RobustAdaptiveGradientClipper()
    # 定义课程学习调度器
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    init_t = int(config['train']['init_t'])
    use_curriculum = config['train']['curriculum']
    if use_curriculum:
        curriculum = CurriculumScheduler(
            max_t_train=t_train,
            min_steps=init_t + 10,
            warmup_epochs=config['train']['curriculum_para'][0],
            rollback_prob=config['train']['curriculum_para'][1]
        )

    print(f'{datetime.now()} --- set optimizer,lr is {now_lr}, scheduler:{scheduler_name} ---')
    # endregion
    # region load model
    ################################################################
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
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
    # region information write
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[Experiment Log | {timestamp}]\n")
        f.write(f"├── Case: 1D Burger's equation with Neumann BCs\n")
        f.write(
            f"├── Model: Discrete Cosine Transform (DCT) for spatial dimension modeling, and Autoregressive (AR) process for temporal dimension modeling\n")
        f.write(f"\n├─ Model Configuration\n")
        f.write(f"├── Model Class: SQL1D\n")
        f.write(f"├── Transform Components:\n")
        f.write(f"│   ├── Forward Transform: dctI\n")
        f.write(f"│   └── Inverse Transform: idctI\n")
        # 核心参数（树形结构）
        if args.pretrain is not None:
            f.write(f"├── Loaded model: {args.pretrain}\n")
            f.write(f"├── Current loss: {best_error:.4e}\n")
        else:
            f.write(f"├── Initialized new model\n")
        f.write(f"├── Architecture Parameters:\n")
        f.write(f"│   ├── input_channels: {input_channel}\n")
        f.write(f"│   ├── modes: {config['model']['modes']}\n")
        f.write(f"│   ├── width: {config['model']['width']}\n")
        f.write(f"│   ├── bandwidth: {config['model']['bandwidth']}\n")
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
        f.write(f"├── Data: {data_config['datapath']}\n")
        f.write(f"├── Train/Test Size: {ntrain:,}/{ntest:,} samples\n")
        f.write(f"├── Batch Size: {batch_size} (train), 1 (test)\n")
        f.write(f"├── Spatial Subsampling: sub_x={int((data_config['nx'] - 1) / data_config['sub_x']) + 1}\n")
        f.write(f"├── Temporal Subsampling: sub_t={int((data_config['nt'] - 1) / data_config['sub_t']) + 1}\n")
        f.write(f"├── Initial Steps: {data_config['initial_step']}\n")
        f.write(f"│   └── Viscosity: ν={v:.4f}\n")
        f.write(f"└── Note: Data recorded\n")
        f.write("-" * 60 + "\n")  # 分隔线
        f.close()
    # endregiion
    # region train
    ################################################################
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    init_t = int(config['train']['init_t'])
    model_save_record = [[0, 100]]
    myloss = LpLoss(size_average=config['train']['loss_size_average'])
    if args.pretrain is not None:
        ebar = trange(epoch, epoch + config['train']['epochs'], desc="Epoch")
        pre_epoch = epoch
    else:
        ebar = trange(config['train']['epochs'], desc="Epoch")
        pre_epoch = 0
    rx = int(config['data']['data_sub_x'] / config['data']['sub_x'])
    rt = int(config['data']['data_sub_t'] / config['data']['sub_t'])
    print(f'rx:{rx},rt:{rt}')
    x_length, time_lentgh = config['data']['x_length'], config['data']['t_length']
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
        # loss = 0.0
        for xx, yy, _ in test_loader:
            xx, yy = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True)  # 确保数据在相同设备上
            #             print('xx.shape:', xx.shape)
            pred = model(xx)  # xx: [sample,nt,nx,channel]; yy/pred:[sample,nt,nx,1]
            assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
            _batch = yy.size(0)
            _pred = pred[:, init_t:, ...]
            _yy = yy[:, init_t:, ...]
            val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
        test_l2 = val_l2_full / ntest
        print(f'epoch:{pre_epoch}, test_l2: {test_l2}')
    model.train()
    for e in ebar:
        # 内层进度条
        epoch_loss = 0
        if data_config['loader'] == 'ProgRes':
            current_resolutions = phase_manager.get_resolutions(e)

            def resolutions_equal(a, b):
                return sorted(a) == sorted(b)

            if not hasattr(batch_sampler, 'last_resolutions'):
                # 第一次运行
                print(f"\n[Epoch {e}] Initial resolutions set: {current_resolutions}")
                batch_sampler.last_resolutions = current_resolutions.copy()  # 保存副本
            elif not resolutions_equal(batch_sampler.last_resolutions, current_resolutions):
                # 分辨率变化
                print(f"\n[Epoch {e}] Resolution changed: "
                      f"{batch_sampler.last_resolutions} → {current_resolutions}")
                batch_sampler.last_resolutions = current_resolutions.copy()
            batch_sampler.set_active_resolutions(current_resolutions)
        train_iter = iter(train_loader)
        batch_loss_data = 0.0
        batch_loss_f = 0.0
        batch_loss_T = 0.0
        for b in trange(len(train_loader), file=desc, desc="batch"):
            """
            x:[b, nt_sub, nx_sub, initial_step + 2]
            y:[b, nt_sub, nx_sub, 1]
            """
            xx, yy, resolution = next(train_iter)
            xx, yy = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True)
            #             print('yy.shape:', yy.shape)
            #             print('xx.shape:', xx.shape)
            optimizer.zero_grad()
            init_x = yy[:, 0:init_t, ...].squeeze()
            pred = model(xx)
            assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
            pred = pred.permute(0, 2, 1, 3)  # [b,nx_sub,nt_sub 1]
            #             print('pred.shape:', pred.shape)
            if config['train']['loss_mode'] == 'both':
                out_data = pred[:, ::rt, ::rx, :]
                y_data = yy[:, ::rt, ::rx, :]
                loss_data = F.mse_loss(out_data, y_data).to(torch.float32)
                loss_fn = PINO_loss_1D if config['train']['residual_mode'] == 'Spectral' else PINO_loss_1DII
                loss_init, loss_f, loss_b = loss_fn(pred, init_x, v, x_length, time_lentgh)
                total_loss = (loss_init * ic_weight + loss_f * f_weight + loss_data * data_weight).to(torch.float32)
                total_loss.backward()
                grad_norm = clipper.step(model)  # 计算梯度范数
                grad.append([e, loss_data.item(), grad_norm])
                assert not torch.isnan(total_loss).any(), "NaN in loss"
            elif config['train']['loss_mode'] == 'data':

                loss_init, loss_f = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                out_data = pred[:, ::rx, ::rt, :]
                y_data = yy.permute(0, 2, 1, 3)[:, ::rx, ::rt, :]
                # print('pred.shape,yy.shape,out_data.shape,y_data.shape:', pred.shape, yy.shape, out_data.shape,
                #       y_data.shape)
                loss_data = myloss(out_data, y_data).to(torch.float32)
                # total_loss = (loss_init * ic_weight + loss_f * f_weight + loss_data * data_weight).to(torch.float32)
                total_loss = loss_data
                total_loss.backward()
                grad_norm = clipper.step(model)  # 计算梯度范数
                grad.append([e, loss_data.item(), grad_norm])
                assert not torch.isnan(total_loss).any(), "NaN in loss"
            elif config['train']['loss_mode'] == 'data_after_pde':
                out_data = pred[:, ::rt, ::rx, :]
                y_data = yy[:, ::rt, ::rx, :]
                loss_data = F.l1_loss(out_data, y_data, reduction='mean').to(torch.float32)
                loss_fn = PINO_loss_1D if config['train']['residual_mode'] == 'Spectral' else PINO_loss_1DII
                loss_init, loss_f, loss_b = loss_fn(pred, init_x, v, x_length, time_lentgh)
                total_loss = (loss_init * ic_weight + loss_f * f_weight + loss_data * data_weight).to(torch.float32)
                # print(loss_data, loss_init, loss_f, loss_b)
                total_loss.backward()
            else:
                loss_fn = PINO_loss_1D if config['train']['residual_mode'] == 'Spectral' else PINO_loss_1DII
                loss_init, loss_f, loss_b = loss_fn(pred, init_x, init_t, x_length, time_lentgh)
                out_data = pred[:, ::rx, ::rt, :]
                y_data = yy.permute(0, 2, 1, 3)[:, ::rx, ::rt, :]
                loss_data = F.mse_loss(out_data, y_data).to(torch.float32)
                total_loss = loss_f + loss_init * ic_weight
                total_loss.backward()
                grad_norm = clipper.step(model)  # 计算梯度范数
                grad.append([e, loss_data.item(), grad_norm])
                assert not torch.isnan(total_loss).any(), "NaN in loss"
            optimizer.step()
            if config['train']['scheduler'] == 'cosine_schedule_with_warmup':
                # print('scheduler step for cosine')
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            loss_list.append([loss_data.item(), loss_f.item(), total_loss.item(), loss_init.item(), e])
            new_desc = f"Epoch {e + 1}: {desc.read(b)},Loss_init：{loss_init.item():.4e},Loss_data: {loss_data.item():.4e},Loss_Train: {total_loss.item():.4e},Loss_phy: {loss_f.item():.4e},lr:{current_lr}, Grad Norm Threshold:{grad_norm:.2f}"
            ebar.set_description(new_desc)
            batch_loss_data += loss_data.item()
            batch_loss_f += loss_f.item()
            batch_loss_T += total_loss.item()
        batch_loss_data = batch_loss_data / (ntrain / batch_size)
        batch_loss_f = batch_loss_f / (ntrain / batch_size)
        batch_loss_T = batch_loss_T / (ntrain / batch_size)
        if config['train']['scheduler'] == 'Step':
            # print('scheduler step for step')
            scheduler.step()
        if config['train']['scheduler'] == 'ReduceLROnPlateau':
            # print('scheduler step for rlop')
            scheduler.step(batch_loss_T)
        lr_list.append([current_lr, e])
        # scheduler.step()
        if best_error > batch_loss_data:
            best_error = batch_loss_data
            model_save_record.append([e, batch_loss_data])
            save_checkpoint(model, e, optimizer, scheduler, loss_list,
                            test_loss_list, lr_list, model_save_record, grad, filename=f'{first_dic}/checkpoint-best')
        if e % config['train']['verbose_interval'] == 0:
            model.eval()
            val_l2_full = 0
            with torch.no_grad():
                # loss = 0.0
                for xx, yy, _ in test_loader:
                    xx, yy = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True)  # 确保数据在相同设备上
                    pred = model(xx)  # xx: [sample,nt,nx,channel]; yy/pred:[sample,nt,nx,1]
                    assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
                    _batch = yy.size(0)
                    _pred = pred[:, init_t:, ...]
                    _yy = yy[:, init_t:, ...]
                    val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                test_l2 = val_l2_full / ntest
                test_loss_list.append([test_l2, e])
                print(f'epoch:{e}, test_l2: {test_l2},batch_loss_T:{batch_loss_T}')
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
                    # grad_array = np.array(grad)
                    # 保存为 .npy 文件
                    # np.save(f'{first_dic}/gradient_logs-{e}.npy', grad_array)
                model.train()
    # writer.close()
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

    test_data_1x = h5DatasetFor1DBurgersII(data_config['datapath'],
                                           sub_x=1,
                                           sub_t=1,
                                           initial_step=data_config['initial_step'], if_test=True)
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)

    test_data_1_2x = h5DatasetFor1DBurgersII(data_config['datapath'],
                                             sub_x=2,
                                             sub_t=2,
                                             initial_step=data_config['initial_step'], if_test=True)
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_4x = h5DatasetFor1DBurgersII(data_config['datapath'],
                                             sub_x=4,
                                             sub_t=4,
                                             initial_step=data_config['initial_step'], if_test=True)
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_8x = h5DatasetFor1DBurgersII(data_config['datapath'],
                                             sub_x=8,
                                             sub_t=8,
                                             initial_step=data_config['initial_step'], if_test=True)
    test_loader_1_8x = torch.utils.data.DataLoader(test_data_1_8x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_16x = h5DatasetFor1DBurgersII(data_config['datapath'],
                                              sub_x=16,
                                              sub_t=16,
                                              initial_step=data_config['initial_step'], if_test=True)
    test_loader_1_16x = torch.utils.data.DataLoader(test_data_1_16x, batch_size=1, shuffle=False,
                                                    num_workers=0, pin_memory=True)

    test_size = test_data_1_16x.data_list.shape[0]
    ntest = test_size
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    first_dic = f"/code/Burger/{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    _trans = PARTIAL(Wrapper, [fft_fun, dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [ifft_fun, idctI_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOLII, T)
    input_channel = config['model']['input_channel'] * config['data']['initial_step'] + 2
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")

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
        checkpoint = torch.load(args.pretrain)
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
        checkpoint = torch.load('checkpoint-best.pth.tar')
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
    # region loss carve
    # batchsize = config['train']['batchsize']
    # bfe = math.ceil(990 / batchsize)
    # plot_loss_with_analysis(loss_list, lr_list, bfe, f'{first_dic}/loss_carve_for_{model_name}')
    plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array,
                               f'{first_dic}/loss_carve_for_{model_name}')
    print('损失曲线绘制完成~')
    # endregion
    # region evaluate
    init_t = int(config['train']['init_t'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    test_loaders = {
        f'test_{origin_nx}': test_loader_1x,
        f'test_{int((origin_nx - 1) / 2) + 1}': test_loader_1_2x,
        f'test_{int((origin_nx - 1) / 4) + 1}': test_loader_1_4x,
        f'test_{int((origin_nx - 1) / 8) + 1}': test_loader_1_8x,
        f'test_{int((origin_nx - 1) / 16) + 1}': test_loader_1_16x,
    }
    results = []
    errors_for_talk_all = {}
    visualize_results = []
    for name, test_loader in test_loaders.items():
        errors_for_talk = []
        model.eval()  # 将模型设置为评估模式
        first = True
        with torch.no_grad():
            test_iter = iter(test_loader)
            for b in tqdm(range(len(test_loader))):
                xx, yy, *_ = next(test_iter)
                xx, yy = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True)  # 确保数据在相同设备上
                #             print('xx.shape:', xx.shape)
                pred = model(xx)  # xx: [sample,nt,nx,channel]; yy/pred:[sample,nt,nx,1]
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
                _batch = yy.size(0)
                _pred = pred[:, init_t:, ...]
                _yy = yy[:, init_t:, ...]
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"  # pred.shape： [1,nx,101,1]
                if b in [0, int(ntest / 4), int(ntest / 2), ntest]:
                    visualize_results.append({
                        'Dataset': name,
                        'index': b,
                        'pred': pred.squeeze().cpu().numpy(),
                        'yy': yy.squeeze().cpu().numpy()
                    })
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                # relative_l2_errors.append(l2_error)
                # 计算pred残差和output残差
                _, Du_pred = burgers_residual(pred.permute(0, 2, 1, 3).squeeze(-1))
                _, Du_yy = burgers_residual(yy.permute(0, 2, 1, 3).squeeze(-1))
                mean_residual_pred = torch.mean(torch.abs(Du_pred)).item()
                mean_residual_yy = torch.mean(torch.abs(Du_yy)).item()
                errors_for_talk.append([l2_error, mean_residual_pred, mean_residual_yy, int(b)])

        # 计算均值和标准差
        errors_for_talk = np.array(errors_for_talk)
        mean_l2_error = np.mean(errors_for_talk[:, 0])
        std_l2_error = np.std(errors_for_talk[:, 0])
        max_l2_error = np.max(errors_for_talk[:, 0])
        min_l2_error = np.min(errors_for_talk[:, 0])

        mean_PDE_error = np.mean(errors_for_talk[:, 1])
        std_PDE_error = np.std(errors_for_talk[:, 1])
        max_PDE_error = np.max(errors_for_talk[:, 1])
        min_PDE_error = np.min(errors_for_talk[:, 1])

        mean_PDE_error_yy = np.mean(errors_for_talk[:, 2])
        std_PDE_error_yy = np.std(errors_for_talk[:, 2])
        max_PDE_error_yy = np.max(errors_for_talk[:, 2])
        min_PDE_error_yy = np.min(errors_for_talk[:, 2])
        # 保存结果
        results.append({
            'Dataset': name,
            'Mean Relative L2 Error': mean_l2_error,
            'Std Relative L2 Error': std_l2_error,
            'Max Relative L2 Error': max_l2_error,
            'Min Relative L2 Error': min_l2_error,
            'Mean PDE Error': mean_PDE_error,
            'Std PDE Error': std_PDE_error,
            'Max PDE Error': max_PDE_error,
            'Min PDE Error': min_PDE_error,
            'YY Mean PDE Error': mean_PDE_error_yy,
            'YY Std PDE Error': std_PDE_error_yy,
            'YY Max PDE Error': max_PDE_error_yy,
            'YY Min PDE Error': min_PDE_error_yy,
        })
        errors_for_talk_all[name] = errors_for_talk
    # 将结果转换为 DataFrame 并保存为 CSV 文件
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)
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
    for demo in visualize_results:
        name = demo['Dataset']
        index = demo['index']
        true_val = demo['yy']
        pred_val = demo['pred']
        plt_name = f'{name}-{index}'
        plot_solution_with_error(true_val, pred_val, plt_name, save_svg=False, Aspect_Ratio=1 / 1.4)
    print('可视化完成~')
    # endregion
    # region error talk
    '''
    这里出图使用的是最大分辨率，计算loss用的也是最大分辨率；
    '''
    traing_nx = data_config['sub_x']
    training_data = f'test_{int((origin_nx - 1) / traing_nx) + 1}'
    print('training_data for error talk:', training_data)
    talk_data = errors_for_talk_all[training_data]
    sample_ids = talk_data[:, 3]

    # 绘制样本误差分布图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 主纵轴：data_loss（对数坐标）
    ax1.scatter(sample_ids, talk_data[:, 0], color='#0B5873', label='data_loss', alpha=0.7, s=10)
    ax1.set_xlabel('Sample ID')
    ax1.set_ylabel('data_loss', color='#0B5873')
    # ax1.set_yscale('log')  # 设置主纵轴为对数坐标
    ax1.tick_params(axis='y', labelcolor='#0B5873')

    # 副纵轴：pde_residual 和 ref_residual（对数坐标）
    ax2 = ax1.twinx()
    ax2.scatter(sample_ids, talk_data[:, 1], color='#8A0011', label='pde_residual', marker='x', s=10)
    ax2.scatter(sample_ids, talk_data[:, 2], color='#107A38', label='ref_residual', marker='^', s=10)
    ax2.set_ylabel('Residuals', color='#8A0011')
    # ax2.set_yscale('log')  # 设置副纵轴为对数坐标
    ax2.tick_params(axis='y', labelcolor='#8A0011')

    # 添加纵向网格线（基于主横轴）
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)  # 纵向虚线网格
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)  # 横向点线网格（对数坐标）

    # 图例和标题
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Data Loss vs. PDE/Ref Residuals (Log Scale)')
    plt.savefig('error_talk.png', dpi=300, bbox_inches='tight', transparent=True)

    # error talk
    mean_loss = np.mean(talk_data[:, 0])
    std_loss = np.std(talk_data[:, 0])
    threshold = mean_loss + 1 * std_loss

    high_loss_samples = talk_data[talk_data[:, 0] > threshold]
    print(f"存在高data_loss样本{len(high_loss_samples)}个：\n{high_loss_samples}")

    # 计算残差与data_loss的相关系数
    corr_pde = np.corrcoef(talk_data[:, 0], talk_data[:, 1])[0, 1]  # data_loss vs pde_residual
    corr_ref = np.corrcoef(talk_data[:, 0], talk_data[:, 2])[0, 1]  # data_loss vs ref_residual

    print(f"data_loss 与 pde_residual 的相关系数: {corr_pde:.3f}")
    print(f"data_loss 与 ref_residual 的相关系数: {corr_ref:.3f}")

    high_loss_ids = high_loss_samples[:, 3].astype(int)
    print("高data_loss的样本编号：", high_loss_ids)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        f.write(f"├── High Loss Analysis | [{timestamp}]\n")
        f.write(f"│   ├── High data_loss samples: {len(high_loss_samples)} cases\n")
        f.write(f"│   ├── Correlation coefficients:\n")
        f.write(f"│   │   ├── data_loss vs pde_residual: {corr_pde:.3f}\n")
        f.write(f"│   │   └── data_loss vs ref_residual: {corr_ref:.3f}\n")
        f.write(f"│    └── High loss sample IDs: {high_loss_ids.tolist()}\n")
        f.write(f"│\n")
        f.write(f"│   L2 metrics:\n")
        f.close()

    talk_visualize_results = []

    model.eval()  # 将模型设置为评估模式
    loader = test_loaders[training_data]
    with torch.no_grad():
        test_iter = iter(loader)
        for b in tqdm(range(len(loader))):
            xx, yy, grid, name = next(test_iter)
            if b in high_loss_ids:
                #                 print('xx.shape:',xx.shape)
                xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                           non_blocking=True)  # 确保数据在相同设备上
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
                outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
                pred = torch.empty(yy.shape, device=xx.device)
                pred[..., 0:init_t, :] = yy[..., 0:init_t, :]
                gridt = torch.tensor(np.linspace(0, 1, t_train), dtype=torch.float32, device=xx.device).reshape(
                    t_train, 1)
                for t in range(init_t, t_train):
                    current_time = gridt[t:t + 1, :]
                    inp = xx.reshape(inp_shape)
                    current_time = current_time.view(1, 1, 1).expand(xx.size(0), xx.size(1), 1)  # 扩展为[batch, nx, 1]
                    out = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                    pred[..., t:t + 1, :] = out
                    xx = torch.cat((xx[..., 1:, :], out), dim=-2)
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
                # print('type(Du_yy):', Du_yy.dtype)
                ux_pred, Du_pred = burgers_residual(pred.permute(0, 2, 1, 3).squeeze(-1))
                ux_pred, Du_yy = burgers_residual(yy.permute(0, 2, 1, 3).squeeze(-1))
                mean_residual_pred = torch.mean(torch.abs(Du_pred))
                mean_residual_yy = torch.mean(torch.abs(Du_yy))
                # print(f'mean_residual_pred:{mean_residual_pred},mean_residual_yy:{mean_residual_yy}')
                _yy = yy[..., init_t + 1:t_train, :]  # if t_train is not -1
                _pred = pred[..., init_t + 1:t_train, :]
                _batch = yy.size(0)
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                # print(' evaluarion: mean_residual_pred.shape:', mean_residual_pred.shape, type(mean_residual_pred))
                talk_visualize_results.append({
                    'index': name,
                    'yy': yy.squeeze().cpu().numpy(),
                    'pred': pred.squeeze().cpu().numpy(),
                    'Du_pred': torch.abs(Du_pred).squeeze().mT.cpu().numpy(),
                    'Du_yy': torch.abs(Du_yy).squeeze().mT.cpu().numpy(),
                    'Du_pred_mean': mean_residual_pred.item(),
                    'Du_yy_mean': mean_residual_yy.item(),
                    'l2_error': l2_error,
                })
    for demo in talk_visualize_results:
        index = demo['index']
        Du_yy_mean = demo['Du_yy_mean']
        Du_yy = demo['Du_yy']
        Du_pred_mean = demo['Du_pred_mean']
        Du_pred = demo['Du_pred']
        pred_val = demo['pred']
        true_val = demo['yy']
        l2 = demo['l2_error']
        plt_name = f'Talk_{index}'
        # print('type(l2):', type(l2), type(Du_pred_mean), type(Du_yy_mean))
        with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
            f.write(
                f"│    └── key: {index}, Data L2 error: {l2:.3e}, Du_pred: {Du_pred_mean:.3e}, Du_yy: {Du_yy_mean:.3e}\n")
            f.close()
        print('type(l2): l2, Du_pred_mean, Du_yy_mean')
        print('type(l2):', index, l2, Du_pred_mean, Du_yy_mean)
        plot_solution_with_Du(true_val, pred_val, plt_name, Du_pred, Du_pred_mean, Du_yy, Du_yy_mean, l2,
                              save_svg=False,
                              Aspect_Ratio=1 / 1.4)

    # 分析

    # endregion


def test2(config, args):
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
    test_data = h5DatasetFor1DBurgersII(data_config['datapath'],
                                        sub_x=data_config['sub_x'],
                                        sub_t=data_config['sub_t'],
                                        initial_step=data_config['initial_step'], if_test=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                              num_workers=2, pin_memory=True)

    test_size = test_data.data_list.shape[0]
    ntest = test_size
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    first_dic = f"/code/Burger/{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    _trans = PARTIAL(Wrapper, [dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [idctI_SPFNO])
    T = Transform(_trans, _itrans)
    Model = PARTIAL(SOLII, T)
    input_channel = config['model']['input_channel'] * config['data']['initial_step'] + 2
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)  # .to(torch.float32)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")

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
        checkpoint = torch.load(args.pretrain)
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
        checkpoint = torch.load('checkpoint-best.pth.tar')
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
    # region evaluate
    init_t = int(config['train']['init_t'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    results = []
    with torch.no_grad():
        test_iter = iter(test_loader)
        for b in tqdm(range(len(test_loader))):
            xx, yy, grid, resolution = next(test_iter)
            #                 print('xx.shape:',xx.shape)
            xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                       non_blocking=True)  # 确保数据在相同设备上

            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
            outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
            pred = torch.empty(yy.shape, device=xx.device)
            pred[..., 0:init_t, :] = yy[..., 0:init_t, :]
            gridt = torch.tensor(np.linspace(0, 1, t_train), dtype=torch.float32, device=xx.device).reshape(
                t_train, 1)
            for t in range(init_t, t_train):
                # print("t:", t)
                current_time = gridt[t:t + 1, :]
                inp = xx.reshape(inp_shape)
                current_time = current_time.view(1, 1, 1).expand(xx.size(0), xx.size(1), 1)  # 扩展为[batch, nx, 1]
                out = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                pred[..., t:t + 1, :] = out
                xx = torch.cat((xx[..., 1:, :], out), dim=-2)
            assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"  # pred.shape： [1,nx,101,1]
            _yy = yy[..., init_t + 1:t_train, :]  # if t_train is not -1
            _pred = pred[..., init_t + 1:t_train, :]
            _batch = yy.size(0)
            l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
            # relative_l2_errors.append(l2_error)
            # 计算pred残差和output残差
            _, Du_pred = burgers_residual(pred.permute(0, 2, 1, 3).squeeze(-1))
            _, Du_yy = burgers_residual(yy.permute(0, 2, 1, 3).squeeze(-1))
            mean_residual_pred = torch.mean(torch.abs(Du_pred)).item()
            mean_residual_yy = torch.mean(torch.abs(Du_yy)).item()
            results.append([resolution, l2_error, mean_residual_pred, mean_residual_yy])
    # 转换为DataFrame
    df = pd.DataFrame(results, columns=['resolution', 'l2_error', 'mean_residual_pred', 'mean_residual_yy'])
    # 按resolution分组并计算均值
    grouped = df.groupby('resolution').mean().reset_index()
    # 输出结果
    print(grouped)
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n├─ data analysis for multi resolution! | {timestamp}] \n")
        f.write("resolution\tl2_error\tmean_residual_pred\tmean_residual_yy\n")
        # 写入数据（保留4位小数）
        grouped.to_string(f, header=False, float_format="%.4f", sep='\t')
        f.close()
    # 计算均值和标准差


# endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='./informationII.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
    parser.add_argument('--load_lr', action='store_true', help='pretrain model path')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(config)
        # test(config, args)
    elif args.mode == 'test':
        test(config, args)
    elif args.mode == 'test2':
        test2(config, args)
