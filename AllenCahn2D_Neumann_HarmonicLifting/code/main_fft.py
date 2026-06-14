"""
DCT-I for 2D Allen-Cahn Equation
u_t = ε Δu + u - u³ with non-homogeneous Neumann BC

Key differences from PME:
  - Operates in u space directly (no pressure variable transformation)
  - Uses delta prediction: u^{t+1} = u^t + δ
  - BC info encoded as u_b field (1 spatial field) since reaction term depends on u_b
  - input_channel = initial_step + 2 + 1  (u_window + grid + u_b_field)
"""
import os
import sys
import re
import math
import pandas as pd
from argparse import ArgumentParser
import yaml
import shutil
from dataloader import *
from loss import *  # ← AC: 使用 Allen-Cahn loss
from utils import *
from datetime import datetime
from model import *
import h5py
import torch
from tqdm import tqdm, trange
import pickle
from functools import partial as PARTIAL
import numpy as np
import time
import torch._dynamo

torch._dynamo.config.suppress_errors = True


# ================================================================
# AC 自回归预测
# ================================================================

def ac_autoregressive_predict(model, xx, grid, bc_params,
                              init_t, t_end, device, pre_mode='delta', dtype=torch.float32):
    """
    Allen-Cahn delta-mode autoregressive prediction in u space.

    模型输入: [u_window, grid, u_b_field] → 预测 δ
    更新: u^{t+1} = u^t + δ

    u_b 场直接编码了 BC 信息，因为反应项 f(u_h + u_b) 逐点依赖 u_b。

    Args:
        model: 神经算子
        xx: [batch, Nx, Ny, initial_step, 1] 初始窗口 (u 空间)
        yy_shape: target shape for allocation
        grid: [batch, Nx, Ny, 2]
        bc_params: [batch, 4] 每个 sample 的 (a, b, c, d)
        init_t: 初始步数
        t_end: 预测终止步
        device, dtype: 设备和类型

    Returns:
        pred: [batch, Nx, Ny, t_end, 1] 预测结果 (u 空间)
    """
    batch, Nx, Ny = xx.shape[0], xx.shape[1], xx.shape[2]

    # 从 grid 提取坐标，构造 u_b 场: [batch, Nx, Ny, 1]
    x_coord = grid[0, :, 0, 0]  # [Nx]
    y_coord = grid[0, 0, :, 1]  # [Ny]
    u_b = build_lifting_batch(bc_params, x_coord, y_coord)  # [batch, 1, Nx, Ny]
    u_b_field = u_b.permute(0, 2, 3, 1)  # [batch, Nx, Ny, 1]

    # 初始化
    u_window = xx.squeeze(-1)  # [batch, Nx, Ny, init_step]
    pred = torch.empty(batch, Nx, Ny, t_end, 1, device=device, dtype=dtype)
    for i in range(init_t):
        pred[..., i, :] = xx[..., i, :]

    for t in range(init_t, t_end):
        # 拼接: [u_window, grid, u_b_field] → [batch, Nx, Ny, init_step + 2 + 1]
        inp = torch.cat([u_window, grid, u_b_field], dim=-1)
        delta = model(inp)  # [batch, Nx, Ny, 1]

        # Delta update
        u_current = u_window[..., -1:]  # [batch, Nx, Ny, 1]
        if pre_mode == 'delta':
            u_new = u_current + delta  # [batch, Nx, Ny, 1]
        else:
            u_new = delta

        pred[..., t:t + 1, :] = u_new.unsqueeze(-1)

        # 滑窗更新
        u_window = torch.cat([u_window[..., 1:], u_new.squeeze(-1).unsqueeze(-1)], dim=-1)

    return pred


# ================================================================
# Training
# ================================================================

def run(config, args):
    # region prepare
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    initial_step = config['train']['initial_step']
    filepath = data_config['datapath']
    sub_x = data_config['sub_x']
    sub_t = data_config['sub_t']

    # ← AC: 读取物理参数
    epsilon = config['data'].get('epsilon', 0.05)
    T_total = config['data'].get('T', 1.0)
    print(f"Allen-Cahn params: epsilon={epsilon}, T={T_total}")

    train_data = FNODatasetMult(file_path=filepath,
                                initial_step=initial_step,
                                sub_x=sub_x,
                                sub_t=sub_t,
                                )
    test_data = FNODatasetMult(file_path=filepath,
                               initial_step=initial_step,
                               sub_x=sub_x,
                               sub_t=sub_t,
                               if_test=True,
                               )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=3, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=3, shuffle=False)
    train_size, test_size = len(train_data), len(test_data)
    ntrain, ntest = train_size, test_size
    print(
        f'{datetime.now()} --- set dataset, batch size: {batch_size}, '
        f'Train: {train_size}, Test: {test_size}')
    # endregion

    # region location
    first_dic = f"/code/AC2D{config['prepare']['project']}"  # ← AC
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    if args.pretrain is not None:
        src = args.config_path
        base = os.path.basename(src)
        name, ext = os.path.splitext(base)
        dst = os.path.join(first_dic, base)
        if os.path.abspath(src) == os.path.abspath(dst):
            i = 1
            while True:
                new_name = f"{name}-retrain--{i}{ext}"
                new_dst = os.path.join(first_dic, new_name)
                if not os.path.exists(new_dst):
                    dst = new_dst
                    break
                i += 1
        shutil.copy(src, dst)
    else:
        shutil.copy(args.config_path, first_dic)
    print(f"{datetime.now()} --- set save dir: {first_dic} ---")
    # endregion

    # region model
    _trans = PARTIAL(Wrapper, [fft_forward, fft_forward])
    _itrans = PARTIAL(Wrapper, [fft_inverse, fft_inverse])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOL2D_FFT, T)
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    out_channels = config['model']['output_channel']
    dim = config['model']['dim']
    tril = config['model']['triL']
    input_channel = initial_step + 2 + 1
    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    # endregion

    # region optimizer
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                 lr=config['train']['base_lr'])
    base_lr = config['train']['base_lr']
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    init_t = data_config['initial_step']

    use_curriculum = config['train']['curriculum']
    use_lr_schedule_in_curriculum = False

    if use_curriculum:
        curriculum = CausalCurriculumScheduler(
            max_t_train=t_train,
            min_steps=init_t + config['train']['curriculum_para'][0],
            warmup_epochs=config['train']['curriculum_para'][1],
            rollback_prob=config['train']['curriculum_para'][2],
            adaptive_gate=config['train'].get('adaptive_gate', False),
            loss_plateau_patience=config['train'].get('loss_plateau_patience', 12),
            loss_plateau_threshold=config['train'].get('loss_plateau_threshold', 0.001),
            force_expand_patience=config['train'].get('force_expand_patience', 30),
            force_patience_early_ratio=config['train'].get('force_patience_early_ratio', 1.5),
            force_patience_late_ratio=config['train'].get('force_patience_late_ratio', 0.5),
            use_causal_weights=config['train'].get('use_causal_weights', False),
            epsilon_start=config['train'].get('epsilon_start', 1.0),
            epsilon_end=config['train'].get('epsilon_end', 0.1),
            use_lr_schedule=config['train'].get('use_lr_schedule', False),
            lr_boost=config['train'].get('lr_boost', 5.0),
            lr_warmup_epochs=config['train'].get('lr_warmup_epochs', 5),
            lr_scheduler_patience=config['train'].get('lr_scheduler_patience', 5),
            lr_scheduler_factor=config['train'].get('gamma', 0.5),
            lr_min_ratio=config['train'].get('lr_min_ratio', 0.1),
            log_file=f'{first_dic}/Experiment_record.txt',
        )
        if config['train'].get('use_lr_schedule', False):
            curriculum.init_lr_scheduler(optimizer)
            use_lr_schedule_in_curriculum = True
            scheduler_name = 'P in C'
            scheduler = curriculum.lr_scheduler
        else:
            use_lr_schedule_in_curriculum = False

    if not use_lr_schedule_in_curriculum:
        scheduler_name = config['train']['scheduler']
        if config['train']['scheduler'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=config['train']['milestones'],
                                                             gamma=config['train']['gamma'])
        elif config['train']['scheduler'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['train']['gamma'],
                                                                   threshold=1e-2,
                                                                   patience=config['train']['patience'],
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

    clipper = RobustAdaptiveGradientClipperV2(
        initial_max_norm=500.0, window_size=60, trim_k=5, multiplier=5)
    print(
        f'{datetime.now()} --- set optimizer, lr={base_lr}, scheduler:{scheduler_name}, curriculum:{use_curriculum}---')
    warmup_epochs = config['train']['warmup_epochs']
    warmup_lr = config['train']['warmup_lr']
    # endregion

    # region load model
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['model'])
        if config['train']['retrain_load_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print('load optimizer and scheduler')
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        grad = checkpoint['grad']
        epoch = checkpoint['epoch']
        best_error = loss_list[-1][0]
        print(f'模型【{args.pretrain}】已加载, 当前训练loss为: {best_error}')
    else:
        loss_list = []
        test_loss_list = []
        lr_list = []
        grad = []
        best_error = 100.0
    # endregion

    # region information
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'=' * 60}\n")
        f.write(f"[Experiment Log | {timestamp}]\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"\n├─ Problem: 2D Allen-Cahn  u_t = ε Δu + u - u³\n")  # ← AC
        f.write(f"│   ├── epsilon: {epsilon}\n")
        f.write(f"│   ├── T: {T_total}\n")
        f.write(f"│   ├── Data Path: {filepath}\n")
        f.write(f"│   └── Sub-sampling: spatial={sub_x}, temporal={sub_t}\n")
        f.write(f"\n├─ Model: SOL2D + DCT-I\n")
        f.write(f"│   ├── Input Channels: {input_channel} (u:{initial_step} + grid:2 + u_b:1)\n")
        f.write(f"│   ├── Modes: {modes}, Width: {width}\n")
        f.write(f"│   └── Total Parameters: {total_params:,}\n")
        f.write(f"\n├─ Data: train={ntrain}, test={ntest}, batch={batch_size}\n")
        f.write(f"│   └── t_train={t_train}, init_t={init_t}\n")
        f.write(f"\n├─ Optimizer: Adam, lr={base_lr}, scheduler={scheduler_name}\n")
        f.write(f"└─ Device: {device}, Seed: {config['prepare']['seed']}\n")
        f.write(f"\n{'=' * 60}\n")
    # endregion

    # region train
    pre_mode = config['train'].get('pre_mode', 'delta')
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    bc_weight = config['train']['bc_loss']
    init_t = data_config['initial_step']
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    print(f't_train is {t_train}')
    nx = int((data_config['nx'] - 1) / data_config['sub_x']) + 1
    print('nx:', nx)
    model_save_record = [[0, 100]]
    myloss = LpLoss(size_average=True)

    if args.pretrain is not None:
        ebar = trange(epoch, epoch + config['train']['epochs'], desc="Epoch")
        pre_epoch = epoch
    else:
        ebar = trange(config['train']['epochs'], desc="Epoch")
        pre_epoch = 0

    rx = int(config['data']['data_sub_x'] / config['data']['sub_x'])
    rt = int(config['data']['data_sub_t'] / config['data']['sub_t'])
    print(f'rx:{rx}, rt:{rt}')
    desc = DescStr()
    time_0 = time.time()
    time_old = time.time()
    dtype = torch.float32
    residual_mode = config['train'].get('residual_mode', 'mse')

    # ===== 初始 eval =====
    model.eval()
    val_l2_full = 0
    with torch.no_grad():
        for xx, yy, grid, bc_params in test_loader:  # ← AC
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)  # ← AC

            pred = ac_autoregressive_predict(  # ← AC
                model, xx, grid, bc_params, init_t, t_train, device, pre_mode, dtype)
            pred[..., :initial_step, :] = yy[..., :initial_step, :]

            _batch = yy.size(0)
            _pred = pred[..., init_t:, :]
            _yy = yy[..., init_t:, :]
            val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
        test_l2 = val_l2_full * batch_size / ntest
        print(f'epoch:{pre_epoch}, test_l2: {test_l2}')
    model.train()

    First_data = True
    prev_loss = 10000

    for e in ebar:
        epoch_loss = 0
        epoch_physics_loss = 0
        epoch_grad_norms = []

        current_t_train = curriculum.update(e - pre_epoch, current_loss=prev_loss) if use_curriculum else t_train
        train_iter = iter(train_loader)

        for b in trange(len(train_loader), file=desc, desc="batch"):
            xx, yy, grid, bc_params = next(train_iter)  # ← AC
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)  # ← AC

            optimizer.zero_grad()

            if not torch.isfinite(xx).all():
                print(f"[Error] NaN in input xx at batch {b}")
                continue

            yy = yy[..., :current_t_train, :]

            # ← AC: 自回归预测 (delta mode in u space)
            pred = ac_autoregressive_predict(
                model, xx, grid, bc_params, init_t, current_t_train, device, pre_mode, dtype)
            pred[..., 0:init_t, :] = yy[..., 0:init_t, :]

            assert pred.shape == yy.shape, f"Shape mismatch: {pred.shape} != {yy.shape}"

            _batch = yy.size(0)
            out_data = pred[:, ::rx, ::rx, ::rt, :].reshape(_batch, -1)
            y_data = yy[:, ::rx, ::rx, ::rt, :].reshape(_batch, -1)
            loss_data = myloss(out_data, y_data)

            if First_data:
                print('out_data.shape, y_data.shape:', pred[:, ::rx, ::rx, ::rt, :].shape,
                      yy[:, ::rx, ::rx, ::rt, :].shape)
                First_data = False

            # ===== Loss =====
            loss_f = torch.tensor(0.0, device=device)
            bc_loss = torch.tensor(0.0, device=device)
            if warmup_epochs and e < warmup_epochs:
                last_input = yy[..., init_t - 1:init_t, :]
                target = last_input.expand(-1, -1, -1, current_t_train - init_t, -1)
                pred_part = pred[..., init_t:current_t_train, :]
                loss_pretrain = ((pred_part - target) ** 2).mean()
                total_loss = loss_pretrain
                epoch_physics_loss += 0.0
                if b == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr[e]
                    print(f"[Warmup {e}/{warmup_epochs}] Pretrain loss: {loss_pretrain.item():.4e}")
            else:
                if config['train']['loss_mode'] != 'data':
                    # ← AC: Allen-Cahn DCT residual
                    residual = compute_ac2d_residual_fd(pred, bc_params, epsilon=epsilon, T=T_total)
                    bc_loss = compute_neumann_bc_loss(pred, bc_params)
                    if residual_mode == 'mae':
                        loss_f = residual.abs().mean()
                    else:
                        loss_f = (residual ** 2).mean()
                    epoch_physics_loss += loss_f.item()

                if config['train']['loss_mode'] == 'both':
                    total_loss = loss_f * f_weight + loss_data * data_weight + bc_weight * bc_loss
                elif config['train']['loss_mode'] == 'data':
                    total_loss = loss_data
                else:
                    total_loss = loss_f * f_weight + bc_weight * bc_loss
            if not torch.isfinite(total_loss).all():
                print(f"[Warning] Invalid loss at epoch {e}, batch {b}, skipping...")
                break

            total_loss.backward()
            clip_info = clipper.step(model)
            optimizer.step()

            if not use_lr_schedule_in_curriculum and scheduler_name == 'cosine_schedule_with_warmup':
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            loss_list.append([loss_data.item(), loss_f.item(), bc_loss.item(), total_loss.item(), e])
            epoch_grad_norms.append(clip_info['grad_norm_after'])
            epoch_loss += total_loss.item()

        new_desc = (
            f"Epoch {e + 1} | "
            f"Loss_total: {total_loss.item():.4e}, Test L2: {test_l2:.4e}, "
            f"Loss_data: {loss_data.item():.4e}, Loss_phy: {loss_f.item():.4e}, "
            f"t_train: Loss_BC: {bc_loss.item():.4e},"
            f"lr: {current_lr:.2e}, Grad: {clip_info['grad_norm_after']:.2f}")
        ebar.set_description(new_desc)

        prev_loss = epoch_physics_loss / len(train_loader)
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_grad_norm = np.mean(epoch_grad_norms)
        grad.append([e, avg_epoch_loss, avg_grad_norm])
        lr_list.append([current_lr, current_t_train, e])

        if use_lr_schedule_in_curriculum:
            curriculum.step_lr_scheduler(avg_epoch_loss)
        elif scheduler is not None:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(avg_epoch_loss)
            elif scheduler_name == 'cosine_schedule_with_warmup':
                pass
            else:
                scheduler.step()

        if best_error > avg_epoch_loss:
            best_error = avg_epoch_loss
            model_save_record.append([e, avg_epoch_loss])
            save_checkpoint(model, e, optimizer, scheduler, loss_list,
                            test_loss_list, lr_list, model_save_record, grad,
                            filename=f'{first_dic}/checkpoint-best')
        save_checkpoint(model, e, optimizer, scheduler, loss_list,
                        test_loss_list, lr_list, model_save_record, grad,
                        filename=f'{first_dic}/checkpoint_newst')

        if e % config['train']['verbose_interval'] == 0:
            model.eval()
            val_l2_full = 0
            with torch.no_grad():
                for xx, yy, grid, bc_params in test_loader:  # ← AC
                    xx = xx.to(device, dtype=dtype, non_blocking=True)
                    yy = yy.to(device, dtype=dtype, non_blocking=True)
                    grid = grid.to(device, dtype=dtype, non_blocking=True)
                    bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

                    pred = ac_autoregressive_predict(
                        model, xx, grid, bc_params, init_t, t_train, device, pre_mode, dtype)
                    pred[..., :initial_step, :] = yy[..., :initial_step, :]

                    _batch = yy.size(0)
                    _pred = pred[..., init_t:, :]
                    _yy = yy[..., init_t:, :]
                    val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                test_l2 = val_l2_full * batch_size / ntest
                print(f'epoch:{e}, test_l2: {test_l2}')
                test_loss_list.append([test_l2, e])
                if e % config['train']['check_epochs'] == 0:
                    save_checkpoint(model, e, optimizer, scheduler, loss_list, test_loss_list,
                                    lr_list, model_save_record, grad, filename=f'{first_dic}/checkpoint-{e}')
                    time_elapsed = time.time() - time_0
                    hours = int(time_elapsed // 3600)
                    minutes = int((time_elapsed % 3600) // 60)
                    time_elapsed_100 = time.time() - time_old
                    hours_100 = int(time_elapsed_100 // 3600)
                    minutes_100 = int((time_elapsed_100 % 3600) // 60)
                    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
                        f.write(f"├── Test l2 error in epoch {e}: {test_l2:.4e}\n")
                        f.write(f"│   └── Costed Time: {hours}h {minutes}m\n")
                        f.write(f"│   └── Per 100 epoch: {hours_100}h {minutes_100}m\n")
                        f.write(f"│   └── Best epoch: {model_save_record[-1][0]}\n")
                        f.write(f"│       └── loss: {model_save_record[-1][1]:.4e}\n")
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
        f.write("-" * 60 + "\n")
    # endregion


# ================================================================
# Testing
# ================================================================

def test(config, args):
    # region prepare
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # endregion

    # region dataloader
    data_config = config['data']
    origin_nx = data_config['nx']
    initial_step = config['train']['initial_step']
    filepath = data_config['datapath']
    sub_t = data_config['sub_t']

    epsilon = config['data'].get('epsilon', 0.05)
    T_total = config['data'].get('T', 5.0)

    test_data_1x = FNODatasetMult(file_path=filepath, initial_step=initial_step, sub_x=1, sub_t=sub_t, if_test=True)
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False, num_workers=0,
                                                 pin_memory=True)

    test_data_1_2x = FNODatasetMult(file_path=filepath, initial_step=initial_step, sub_x=2, sub_t=sub_t, if_test=True)
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False, num_workers=0,
                                                   pin_memory=True)

    test_data_1_4x = FNODatasetMult(file_path=filepath, initial_step=initial_step, sub_x=4, sub_t=sub_t, if_test=True)
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False, num_workers=0,
                                                   pin_memory=True)

    ntest = len(test_data_1x)
    # endregion

    # region location
    first_dic = f"/code/AC2D{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir: {config['prepare']['project']} ---")
    # endregion

    # region model
    _trans = PARTIAL(Wrapper, [fft_forward, fft_forward])
    _itrans = PARTIAL(Wrapper, [fft_inverse, fft_inverse])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOL2D_FFT, T)
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    out_channels = config['model']['output_channel']
    dim = config['model']['dim']
    tril = config['model']['triL']
    input_channel = initial_step + 2 + 1
    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    # endregion

    # region load model
    if args.pretrain is not None:
        model_name = args.pretrain[:-8]
        model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['model'])
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint.get('test_loss_list', [])
        grad_array = checkpoint['grad']
        lr_list = checkpoint.get('lr_list', [])
        epoch = checkpoint.get('epoch', 0)
        print(f"{datetime.now()} --- model loaded, {epoch} epochs trained")
    else:
        checkpoint = torch.load('checkpoint-best.pth.tar')
        grad_array = checkpoint['grad']
        model.load_state_dict(checkpoint['model'])
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        epoch = checkpoint['epoch']
        model_name = 'checkpoint-best'
        print(f"{datetime.now()} --- best model loaded, {epoch} epochs trained")
    # endregion

    # region plot loss
    plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array,
                               f'{first_dic}/loss_carve_for_{model_name}')
    # endregion

    # region evaluate
    dtype = torch.float32
    init_t = int(data_config['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    pre_mode = config['train'].get('pre_mode', 'delta')

    test_loaders = {
        f'test_{origin_nx}': (test_loader_1x, 1),
        f'test_{int((origin_nx - 1) / 2) + 1}': (test_loader_1_2x, 2),
        f'test_{int((origin_nx - 1) / 4) + 1}': (test_loader_1_4x, 4),
    }

    results = []
    errors_for_talk_all = {}
    visualize_results_all = []

    for name, (test_loader, sub_x) in test_loaders.items():
        errors_for_talk = []
        model.eval()

        with torch.no_grad():
            test_iter = iter(test_loader)
            for b in tqdm(range(len(test_loader))):
                xx, yy, grid, bc_params = next(test_iter)  # ← AC
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

                pred = ac_autoregressive_predict(
                    model, xx, grid, bc_params, init_t, t_train, device, pre_mode, dtype)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]

                assert pred.shape == yy.shape
                _batch = yy.size(0)
                _yy = yy[..., init_t + 1:t_train, :]
                _pred = pred[..., init_t + 1:t_train, :]
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                # ← AC residual
                residual_p = compute_ac2d_residual_fd(pred, bc_params, epsilon=epsilon, T=T_total)
                loss_f_p = torch.abs(residual_p).mean()
                residual_y = compute_ac2d_residual_fd(yy, bc_params, epsilon=epsilon, T=T_total)
                loss_f_y = torch.abs(residual_y).mean()

                errors_for_talk.append([l2_error, loss_f_p.item(), loss_f_y.item(), int(b)])

            error_records_sorted = sorted(errors_for_talk, key=lambda x: x[0])
            n_samples = len(error_records_sorted)
            selected_indices = set(
                [r[-1] for r in error_records_sorted[:3]] +
                [r[-1] for r in error_records_sorted[-3:]] +
                [r[-1] for r in error_records_sorted[n_samples // 2 - 1: n_samples // 2 + 2]]
            )

            test_iter = iter(test_loader)
            visualize_results = []
            for b in tqdm(range(len(test_loader))):
                xx, yy, grid, bc_params = next(test_iter)  # ← AC
                if b not in selected_indices:
                    continue
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

                pred = ac_autoregressive_predict(
                    model, xx, grid, bc_params, init_t, t_train, device, pre_mode, dtype)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]

                assert pred.shape == yy.shape
                f_p = compute_ac2d_residual_fd(pred, bc_params, epsilon=epsilon, T=T_total)
                f_y = compute_ac2d_residual_fd(yy, bc_params, epsilon=epsilon, T=T_total)

                l2_error = error_records_sorted[[r[-1] for r in error_records_sorted].index(b)][0]
                if b in [r[-1] for r in error_records_sorted[:3]]:
                    category = 'best'
                elif b in [r[-1] for r in error_records_sorted[-3:]]:
                    category = 'worst'
                else:
                    category = 'mid'

                visualize_results.append({
                    'Dataset': name,
                    'index': b,
                    'category': category,
                    'l2_error': l2_error,
                    'pred': pred.squeeze().cpu().numpy(),
                    'yy': yy.squeeze().cpu().numpy(),
                    'pred_du': f_p.squeeze().cpu().numpy(),
                    'yy_du': f_y.squeeze().cpu().numpy(),
                })
            visualize_results_all.append(visualize_results)

        errors_for_talk = np.array(errors_for_talk)
        results.append({
            'Dataset': name,
            'Mean Relative L2 Error': np.mean(errors_for_talk[:, 0]),
            'Std Relative L2 Error': np.std(errors_for_talk[:, 0]),
            'Max Relative L2 Error': np.max(errors_for_talk[:, 0]),
            'Min Relative L2 Error': np.min(errors_for_talk[:, 0]),
            'Mean PDE Error': np.mean(errors_for_talk[:, 1]),
            'Std PDE Error': np.std(errors_for_talk[:, 1]),
            'YY Mean PDE Error': np.mean(errors_for_talk[:, 2]),
        })
        errors_for_talk_all[name] = errors_for_talk

    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)
    with open('visualize_results.pkl', 'wb') as f:
        pickle.dump(visualize_results_all, f)
    with open('error_for_talk_all.pkl', 'wb') as f:
        pickle.dump(errors_for_talk_all, f)
    print("Results saved.")
    # endregion

    # region visualize
    for visualize_results in visualize_results_all:
        case = visualize_results[0]['Dataset']
        plot_burgers2d_comparison(visualize_results, save_dir=f'./figures_{case}')
    print('可视化完成~')
    # endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='Allen-Cahn 2D Training')
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
            with open(config_file, encoding="gb18030") as stream:
                config = yaml.load(stream, yaml.FullLoader)

    if args.mode == 'train':
        run(config, args)
        test(config, args)
    else:
        test(config, args)
