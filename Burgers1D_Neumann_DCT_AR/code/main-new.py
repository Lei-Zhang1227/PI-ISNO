"""
DCT-I for 1D Burgers Equation with Neumann BCs
u_t + u·u_x = ν·u_xx

Autoregressive training with:
  - DCT-I spectral basis (hard Neumann BC enforcement)
  - Multi-resolution dataloader support (Fixed/Mixed/Progressive)
  - Curriculum learning for temporal dimension
  - Adaptive gradient clipping
  - LR warmup (constant-value warmup before main training)

History:
  24/10/22 - Initial version
  25/01/14 - Width consistency fix; fixed time-step size
  25/06/10 - Ongoing refinement
  25/07/18 - LR warmup + cosine decay
  25/07/29 - Checkpoint logic, adaptive grad clip, curriculum scheduler
  25/08/15 - Multi-resolution dataloader
  25/08/21 - Iterator bug fix
  25/09/xx - Refactored to match AC2D code style; added warmup phase
"""
import os
import sys
import re
import math
import random
import pandas as pd
from argparse import ArgumentParser
import yaml
import shutil
from tqdm import tqdm, trange
import pickle
from functools import partial as PARTIAL
from datetime import datetime
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))

from model import SOL1dII
from Burger.datasets import *
from Burger.loss import *
from Burger.utils import *


# ================================================================
# 1D Burgers 自回归预测
# ================================================================

def burgers1d_autoregressive_predict(model, xx, grid, init_t, t_end,
                                     device, pre_mode='direct', dtype=torch.float32):
    """
    1D Burgers autoregressive prediction.

    模型输入: [u_window, grid] → 预测 u^{t+1} (direct) 或 δ (delta)
    Shape:
        xx:   [batch, nx, initial_step, 1]
        grid: [batch, nx, 1]

    Returns:
        pred: [batch, nx, t_end, 1]
    """
    batch, nx = xx.shape[0], xx.shape[1]

    inp_shape = list(xx.shape[:-2]) + [-1]       # [b, nx, -1]
    outp_shape = list(xx.shape[:-2]) + [1, -1]   # [b, nx, 1, -1]

    pred = torch.empty(batch, nx, t_end, 1, device=device, dtype=dtype)
    pred[..., :init_t, :] = xx[..., :init_t, :]

    u_window = xx.clone()

    for t in range(init_t, t_end):
        inp = u_window.reshape(inp_shape)
        out = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)

        if pre_mode == 'delta':
            last_step = u_window[..., -1:, :]
            out = last_step + out

        pred[..., t:t + 1, :] = out
        u_window = torch.cat((u_window[..., 1:, :], out), dim=-2)

    return pred


# ================================================================
# Training
# ================================================================

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
    data_config = config['data']
    batch_size = config['train']['batchsize']
    v = 0.01

    if data_config['loader'] == 'FixedRes':
        train_data = h5DatasetFor1DBurgersII(data_config['datapath'],
                                             sub_x=data_config['sub_x'],
                                             sub_t=data_config['sub_t'],
                                             initial_step=data_config['initial_step'])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        test_data = h5DatasetFor1DBurgersII(data_config['datapath'],
                                            sub_x=data_config['sub_x'],
                                            sub_t=data_config['sub_t'],
                                            initial_step=data_config['initial_step'], if_test=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                                 num_workers=2, pin_memory=True)
    else:
        file_path = r'/code/Burger/resolution_dfy.pkl'
        with open(file_path, 'rb') as f:
            resolution_df = pickle.load(f)
        train_data = h5DatasetFor1DBurgers_muti(
            filepath=data_config['datapath'],
            initial_step=data_config['initial_step'],
            sub_t=1, sub_x=1,
            resolution_df=resolution_df,
            label_num=500
        )
        test_data = h5DatasetFor1DBurgersII(data_config['datapath'],
                                            sub_x=data_config['sub_x'],
                                            sub_t=1,
                                            initial_step=data_config['initial_step'], if_test=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                                 num_workers=2, pin_memory=True)
        resolution_groups = {}
        for idx, name in enumerate(train_data.data_list):
            res = resolution_df.loc[resolution_df["sample_id"] == name, "recommended_res"].values[0]
            if res not in resolution_groups:
                resolution_groups[res] = []
            resolution_groups[res].append(idx)
        group_weights = {"129": 1, "257": 1, "517": 3}

        if data_config['loader'] == 'MixedRes':
            batch_sampler = ResolutionWeightedBatchSampler(
                resolution_groups=resolution_groups,
                batch_size=batch_size,
                group_weights=group_weights,
                shuffle=True
            )
            train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=2)

        elif data_config['loader'] == 'ProgRes':
            batch_sampler = ResolutionWeightedBatchSamplerII(
                resolution_groups=resolution_groups,
                batch_size=batch_size,
                group_weights=group_weights,
                shuffle=True
            )
            phases = config['data']["phases"]
            phase_manager = ResolutionPhaseManager(phases)
            train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=8)

            # 预计算总步数 (用于 cosine scheduler)
            sum_steps = 0
            for epoch in range(config['train']['epochs']):
                for phase in phases:
                    if epoch == phase["start_epoch"]:
                        batch_sampler.set_active_resolutions(phase["resolutions"])
                        break
                sum_steps += len(train_loader)

        print(f'{datetime.now()} --- set dataloader (mode={data_config["loader"]}) ---')

    train_size = train_data.data_list.shape[0]
    test_size = test_data.data_list.shape[0]
    ntrain, ntest = train_size, test_size
    print(f'Train/Test size: {train_size}/{test_size}')
    # endregion

    # region location
    first_dic = f"/code/Burger{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)

    # 拷贝配置文件 (处理同路径冲突)
    src = args.config_path
    base = os.path.basename(src)
    name_base, ext = os.path.splitext(base)
    dst = os.path.join(first_dic, base)
    if args.pretrain is not None and os.path.abspath(src) == os.path.abspath(dst):
        i = 1
        while True:
            new_name = f"{name_base}-retrain-{i}{ext}"
            new_dst = os.path.join(first_dic, new_name)
            if not os.path.exists(new_dst):
                dst = new_dst
                break
            i += 1
    shutil.copy(src, dst)
    print(f"{datetime.now()} --- set save dir: {first_dic} ---")
    # endregion

    # region model
    _trans = PARTIAL(Wrapper, [dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [idctI_SPFNO])
    T = Transform(_trans, _itrans)
    Model = PARTIAL(SOL1dII, T)

    # input = u_window(initial_step) + grid(1), 不含 time embedding
    input_channel = config['model']['input_channel'] * config['data']['initial_step'] + 1
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params:,}")
    print(f"  input_channel={input_channel} (u:{config['data']['initial_step']} + grid:1)")
    for layer_name, layer in model.named_children():
        num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"  {layer_name}: {num_params:,} parameters")
    # endregion

    # region optimizer
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                 lr=config['train']['base_lr'])
    base_lr = config['train']['base_lr']
    scheduler_name = config['train']['scheduler']

    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config['train']['milestones'], gamma=config['train']['gamma'])
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config['train']['gamma'],
            threshold=1e-2, patience=config['train']['patience'], verbose=True)
    elif scheduler_name == 'cosine_schedule_with_warmup':
        epoch_set = config['train']['epochs']
        bfe = math.ceil(train_size / batch_size)
        step = bfe * epoch_set if data_config['loader'] != 'ProgRes' else sum_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=step * config['train']['cosine_schedul'],
            num_training_steps=step
        )
        print(f'Cosine+warmup: warm={int(config["train"]["cosine_schedul"] * epoch_set)} epochs, steps={step}')
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config['train']['patience'], gamma=config['train']['gamma'])

    # 梯度裁剪
    clipper = RobustAdaptiveGradientClipper()

    # 课程学习
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

    # Warmup 配置 (与 AC2D 完全一致)
    warmup_epochs = config['train']['warmup_epochs']
    warmup_lr = config['train']['warmup_lr']

    print(f'{datetime.now()} --- set optimizer, lr={base_lr}, scheduler={scheduler_name}, '
          f'curriculum={use_curriculum}, warmup_epochs={warmup_epochs} ---')
    # endregion

    # region load model
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['model'])
        if config['train']['retrain_load_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print('Loaded optimizer and scheduler states')
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
        f.write(f"\n├─ Problem: 1D Burgers equation with Neumann BCs\n")
        f.write(f"│   ├── Viscosity: ν={v:.4f}\n")
        f.write(f"│   ├── Data Path: {data_config['datapath']}\n")
        f.write(f"│   └── Sub-sampling: spatial={data_config['sub_x']}, temporal={data_config['sub_t']}\n")
        f.write(f"\n├─ Model: SOL1dII + DCT-I (SPFNO)\n")
        f.write(f"│   ├── Input Channels: {input_channel} (u:{config['data']['initial_step']} + grid:1)\n")
        f.write(f"│   ├── Modes: {config['model']['modes']}, Width: {config['model']['width']}\n")
        f.write(f"│   ├── Bandwidth: {config['model']['bandwidth']}\n")
        f.write(f"│   └── Total Parameters: {total_params:,}\n")
        f.write(f"\n├─ Data: train={ntrain}, test={ntest}, batch={batch_size}\n")
        f.write(f"│   ├── Loader: {data_config['loader']}\n")
        f.write(f"│   └── t_train={t_train}, init_t={init_t}\n")
        f.write(f"\n├─ Optimizer: Adam, lr={base_lr}, scheduler={scheduler_name}\n")
        f.write(f"│   └── Warmup: {warmup_epochs} epochs\n")
        if args.pretrain is not None:
            f.write(f"│   └── Pretrained from: {args.pretrain}, loss={best_error:.4e}\n")
        f.write(f"└─ Device: {device}, Seed: {config['prepare']['seed']}\n")
        f.write(f"\n{'=' * 60}\n")
    # endregion

    # region train
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    print(f't_train={t_train}')

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
    print(f'rx={rx}, rt={rt}')

    x_length = config['data']['x_length']
    time_length = config['data']['t_length']
    loss_mode = config['train']['loss_mode']
    residual_mode = config['train'].get('residual_mode', 'Spectral')
    pre_mode = config['train'].get('pre_mode', 'direct')
    print(f'pre_mode: {pre_mode}, loss_mode: {loss_mode}')
    desc = DescStr()
    time_0 = time.time()
    time_old = time.time()
    dtype = torch.float32

    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n├─ Training Start! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")

    # ===== 初始 eval =====
    model.eval()
    val_l2_full = 0
    with torch.no_grad():
        for xx, yy, grid, _ in test_loader:
            xx = xx.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            grid = grid.to(device, non_blocking=True)

            pred = burgers1d_autoregressive_predict(
                model, xx, grid, init_t, t_train, device, pre_mode, dtype)

            _batch = yy.size(0)
            _pred = pred[..., init_t:t_train, :]
            _yy = yy[..., init_t:t_train, :]
            val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
        test_l2 = val_l2_full / ntest
        print(f'epoch:{pre_epoch}, test_l2: {test_l2}')
    model.train()

    First_data = True

    for e in ebar:
        epoch_loss = 0
        epoch_total_loss = 0

        current_t_train = curriculum.update(e - pre_epoch) if use_curriculum else t_train

        # ProgRes 分辨率切换
        if data_config['loader'] == 'ProgRes':
            current_resolutions = phase_manager.get_resolutions(e)
            if not hasattr(batch_sampler, 'last_resolutions'):
                print(f"\n[Epoch {e}] Initial resolutions: {current_resolutions}")
                batch_sampler.last_resolutions = current_resolutions.copy()
            elif sorted(batch_sampler.last_resolutions) != sorted(current_resolutions):
                print(f"\n[Epoch {e}] Resolution changed: "
                      f"{batch_sampler.last_resolutions} → {current_resolutions}")
                batch_sampler.last_resolutions = current_resolutions.copy()
            batch_sampler.set_active_resolutions(current_resolutions)

        train_iter = iter(train_loader)
        for b in trange(len(train_loader), file=desc, desc="batch"):
            # 数据加载 (兼容 FixedRes 和 MultiRes)
            if data_config['loader'] != 'FixedRes':
                xx, yy, grid, resolution, label = next(train_iter)
            else:
                xx, yy, grid, _ = next(train_iter)
                resolution = [None]
                label = None

            xx = xx.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            grid = grid.to(device, non_blocking=True)

            optimizer.zero_grad()

            init_x = yy[..., 0:init_t, :].squeeze(-1)
            yy = yy[..., :current_t_train, :]

            # 自回归预测
            pred = burgers1d_autoregressive_predict(
                model, xx, grid, init_t, current_t_train, device, pre_mode, dtype)
            pred[..., 0:init_t, :] = yy[..., 0:init_t, :]

            assert pred.shape == yy.shape, f"Shape mismatch: {pred.shape} != {yy.shape}"

            _batch = yy.size(0)

            # ===== Loss 计算 =====
            loss_f = torch.tensor(0.0, device=device)
            loss_b = torch.tensor(0.0, device=device)
            loss_data = torch.tensor(0.0, device=device)

            if warmup_epochs and e < warmup_epochs:
                # ====== Warmup 阶段: 恒等映射预训练 (与 AC2D 一致) ======
                last_input = yy[..., init_t - 1:init_t, :]  # [b, nx, 1, 1]
                target = last_input.expand(-1, -1, current_t_train - init_t, -1)
                pred_part = pred[..., init_t:current_t_train, :]
                loss_pretrain = ((pred_part - target) ** 2).mean()
                total_loss = loss_pretrain
                if b == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr[e]
                    print(f"[Warmup {e}/{warmup_epochs}] Pretrain loss: {loss_pretrain.item():.4e}")

            elif loss_mode == 'data':
                # 纯数据驱动
                out_data = pred[:, ::rx, ::rt, :]
                y_data = yy[:, ::rx, ::rt, :]
                if First_data:
                    print(f'[data mode] out_data.shape={out_data.shape}, y_data.shape={y_data.shape}')
                    First_data = False
                loss_data = F.mse_loss(out_data, y_data).to(dtype)
                total_loss = loss_data

            else:
                # 物理驱动 (+ 可选数据监控)
                out_data = pred[:, ::rx, ::rt, :]
                y_data = yy[:, ::rx, ::rt, :]
                if First_data:
                    print(f'[physics mode] out_data.shape={out_data.shape}, y_data.shape={y_data.shape}')
                    First_data = False
                loss_data = F.mse_loss(out_data, y_data).to(dtype)  # 仅用于监控, 不参与反传

                loss_fn = PINO_loss_1D if residual_mode == 'Spectral' else PINO_loss_1DII
                loss_init, loss_f, loss_b = loss_fn(pred, init_x.permute(0, 2, 1), init_t, x_length, time_length)
                total_loss = loss_f.to(dtype)

            assert not torch.isnan(total_loss).any(), "NaN in loss"

            total_loss.backward()
            grad_norm = clipper.step(model)
            optimizer.step()

            if scheduler_name == 'cosine_schedule_with_warmup' and not (warmup_epochs and e < warmup_epochs):
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            loss_list.append([loss_data.item(), loss_f.item(), total_loss.item(), e])
            grad.append([e, total_loss.item(), grad_norm])

            epoch_total_loss += total_loss.item()
            epoch_loss += loss_data.item()

            new_desc = (
                f"Epoch {e + 1}: {desc.read(b)}, "
                f"t_train: {current_t_train}, "
                f"Loss_total: {total_loss.item():.4e}, "
                f"Loss_data: {loss_data.item():.4e}, "
                f"Loss_phy: {loss_f.item():.4e}, "
                f"lr: {current_lr:.2e}, "
                f"Grad: {grad_norm:.2f}")
            ebar.set_description(new_desc)

        # Epoch 结束

        avg_epoch_loss = epoch_total_loss / len(train_loader)

        if not (warmup_epochs and e < warmup_epochs):
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(avg_epoch_loss)
            elif scheduler_name == 'cosine_schedule_with_warmup':
                pass  # 每个 batch 已更新
            else:
                scheduler.step()

        lr_list.append([current_lr, current_t_train, e])

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
                for xx, yy, grid, _ in test_loader:
                    xx = xx.to(device, non_blocking=True)
                    yy = yy.to(device, non_blocking=True)
                    grid = grid.to(device, non_blocking=True)

                    pred = burgers1d_autoregressive_predict(
                        model, xx, grid, init_t, t_train, device, pre_mode, dtype)

                    _batch = yy.size(0)
                    _pred = pred[..., init_t:t_train, :]
                    _yy = yy[..., init_t:t_train, :]
                    val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                test_l2 = val_l2_full / ntest
                test_loss_list.append([test_l2, e])
                print(f'epoch:{e}, test_l2: {test_l2}')

                if e % config['train']['check_epochs'] == 0:
                    save_checkpoint(model, e, optimizer, scheduler, loss_list, test_loss_list,
                                    lr_list, model_save_record, grad,
                                    filename=f'{first_dic}/checkpoint-{e}')
                    if e % 100 == 0:
                        elapsed = time.time() - time_0
                        elapsed_100 = time.time() - time_old
                        with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
                            f.write(f"├── Test l2 at epoch {e}: {test_l2:.4e}\n")
                            f.write(f"│   ├── Total time: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m\n")
                            f.write(f"│   ├── Per 100 epoch: {int(elapsed_100//3600)}h {int((elapsed_100%3600)//60)}m\n")
                            f.write(f"│   └── Best: epoch {model_save_record[-1][0]}, "
                                    f"loss={model_save_record[-1][1]:.4e}\n")
                        time_old = time.time()
            model.train()

    print(f'{datetime.now()} --- training succeed ---')
    elapsed = time.time() - time_0
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        f.write(f"├── Training complete | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"│   └── Total time: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m\n")
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

    test_data_1x = h5DatasetFor1DBurgersII(data_config['datapath'], sub_x=1,
                                           sub_t=data_config['sub_t'],
                                           initial_step=data_config['initial_step'], if_test=True)
    test_loader_1x = DataLoader(test_data_1x, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    test_data_1_2x = h5DatasetFor1DBurgersII(data_config['datapath'], sub_x=2,
                                             sub_t=data_config['sub_t'],
                                             initial_step=data_config['initial_step'], if_test=True)
    test_loader_1_2x = DataLoader(test_data_1_2x, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    test_data_1_4x = h5DatasetFor1DBurgersII(data_config['datapath'], sub_x=4,
                                             sub_t=data_config['sub_t'],
                                             initial_step=data_config['initial_step'], if_test=True)
    test_loader_1_4x = DataLoader(test_data_1_4x, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    test_data_1_8x = h5DatasetFor1DBurgersII(data_config['datapath'], sub_x=8,
                                             sub_t=data_config['sub_t'],
                                             initial_step=data_config['initial_step'], if_test=True)
    test_loader_1_8x = DataLoader(test_data_1_8x, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    test_data_1_16x = h5DatasetFor1DBurgersII(data_config['datapath'], sub_x=16,
                                              sub_t=data_config['sub_t'],
                                              initial_step=data_config['initial_step'], if_test=True)
    test_loader_1_16x = DataLoader(test_data_1_16x, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    ntest = test_data_1_16x.data_list.shape[0]
    # endregion

    # region location
    first_dic = f"/code/Burger/{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- Save dir: {first_dic}")
    # endregion

    # region model
    _trans = PARTIAL(Wrapper, [dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [idctI_SPFNO])
    T = Transform(_trans, _itrans)
    Model = PARTIAL(SOL1dII, T)

    # 与 run 保持一致: input = u_window + grid, 不含 time
    input_channel = config['model']['input_channel'] * config['data']['initial_step'] + 1
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- Model params: {total_params:,}")
    # endregion

    # region load model
    if args.pretrain is not None:
        model_name = os.path.basename(args.pretrain).replace('.pth.tar', '')
        model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['model'])
        loss_list = checkpoint.get('loss_list', [])
        test_loss_list = checkpoint.get('test_loss_list', [])
        grad_array = checkpoint.get('grad', [])
        lr_list = checkpoint.get('lr_list', [])
        epoch = checkpoint.get('epoch', 0)
        print(f"{datetime.now()} --- Loaded: {args.pretrain}, epoch={epoch}")
    else:
        checkpoint = torch.load('checkpoint-best.pth.tar')
        model.load_state_dict(checkpoint['model'])
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        grad_array = checkpoint['grad']
        epoch = checkpoint['epoch']
        model_name = 'checkpoint-best'
        print(f"Loaded checkpoint-best, epoch={epoch}")
    # endregion

    # region plot loss curve
    plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array,
                               f'{first_dic}/loss_curve_for_{model_name}')
    print(f"Loss curve saved.")
    # endregion

    # region evaluate
    dtype = torch.float32
    pre_mode = config['train'].get('pre_mode', 'direct')
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
    visualize_results_all = []

    for name, test_loader in test_loaders.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        errors_for_talk = []
        model.eval()

        with torch.no_grad():
            # ===== 第一遍: 计算所有样本误差 =====
            test_iter = iter(test_loader)
            for b in tqdm(range(len(test_loader)), desc="Computing errors"):
                xx, yy, grid, _ = next(test_iter)
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)

                pred = burgers1d_autoregressive_predict(
                    model, xx, grid, init_t, t_train, device, pre_mode, dtype)
                pred[..., 0:init_t, :] = yy[..., 0:init_t, :]

                assert pred.shape == yy.shape

                _yy = yy[..., init_t + 1:t_train, :]
                _pred = pred[..., init_t + 1:t_train, :]
                _batch = yy.size(0)
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                _, Du_pred = burgers_residual(pred.permute(0, 2, 1, 3).squeeze(-1))
                _, Du_yy = burgers_residual(yy.permute(0, 2, 1, 3).squeeze(-1))
                mean_residual_pred = torch.mean(torch.abs(Du_pred)).item()
                mean_residual_yy = torch.mean(torch.abs(Du_yy)).item()

                errors_for_talk.append([l2_error, mean_residual_pred, mean_residual_yy, int(b)])

            # ===== 选择可视化样本: best/mid/worst =====
            error_records_sorted = sorted(errors_for_talk, key=lambda x: x[0])
            n_samples = len(error_records_sorted)
            selected_indices = set(
                [r[-1] for r in error_records_sorted[:3]] +
                [r[-1] for r in error_records_sorted[-3:]] +
                [r[-1] for r in error_records_sorted[n_samples // 2 - 1: n_samples // 2 + 2]]
            )
            print(f"Selected indices for visualization: {sorted(selected_indices)}")

            # ===== 第二遍: 保存选中样本 =====
            test_iter = iter(test_loader)
            visualize_results = []
            for b in tqdm(range(len(test_loader)), desc="Collecting vis data"):
                xx, yy, grid, _ = next(test_iter)
                if b not in selected_indices:
                    continue

                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)

                pred = burgers1d_autoregressive_predict(
                    model, xx, grid, init_t, t_train, device, pre_mode, dtype)
                pred[..., 0:init_t, :] = yy[..., 0:init_t, :]

                _, Du_pred = burgers_residual(pred.permute(0, 2, 1, 3).squeeze(-1))
                _, Du_yy = burgers_residual(yy.permute(0, 2, 1, 3).squeeze(-1))

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
                    'Du_pred': torch.abs(Du_pred).squeeze().mT.cpu().numpy(),
                    'Du_yy': torch.abs(Du_yy).squeeze().mT.cpu().numpy(),
                    'Du_pred_mean': torch.mean(torch.abs(Du_pred)).item(),
                    'Du_yy_mean': torch.mean(torch.abs(Du_yy)).item(),
                })

            visualize_results_all.append(visualize_results)

        # 计算统计量
        errors_for_talk = np.array(errors_for_talk)
        results.append({
            'Dataset': name,
            'Mean Relative L2 Error': np.mean(errors_for_talk[:, 0]),
            'Std Relative L2 Error': np.std(errors_for_talk[:, 0]),
            'Max Relative L2 Error': np.max(errors_for_talk[:, 0]),
            'Min Relative L2 Error': np.min(errors_for_talk[:, 0]),
            'Mean PDE Error': np.mean(errors_for_talk[:, 1]),
            'Std PDE Error': np.std(errors_for_talk[:, 1]),
            'Max PDE Error': np.max(errors_for_talk[:, 1]),
            'Min PDE Error': np.min(errors_for_talk[:, 1]),
            'YY Mean PDE Error': np.mean(errors_for_talk[:, 2]),
            'YY Std PDE Error': np.std(errors_for_talk[:, 2]),
        })
        errors_for_talk_all[name] = errors_for_talk

        print(f"\n{name} Results:")
        print(f"  L2 Error: {np.mean(errors_for_talk[:, 0]):.4e} ± {np.std(errors_for_talk[:, 0]):.4e}")
        print(f"  PDE Error (pred): {np.mean(errors_for_talk[:, 1]):.4e}")
        print(f"  PDE Error (truth): {np.mean(errors_for_talk[:, 2]):.4e}")

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)

    with open('visualize_results.pkl', 'wb') as f:
        pickle.dump(visualize_results_all, f)

    with open('error_for_talk_all.pkl', 'wb') as f:
        pickle.dump(errors_for_talk_all, f)

    print(f"\nResults saved to test_results.csv")
    # endregion

    # region visualize
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    for vis_results in visualize_results_all:
        if len(vis_results) == 0:
            continue
        case = vis_results[0]['Dataset']
        save_dir = f'./figures_{case}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            for demo in vis_results:
                plt_name = f"{demo['category']}_{demo['index']}"
                plot_solution_with_Du(
                    demo['yy'], demo['pred'], plt_name,
                    demo['Du_pred'], demo['Du_pred_mean'],
                    demo['Du_yy'], demo['Du_yy_mean'],
                    demo['l2_error'],
                    save_svg=False, Aspect_Ratio=1 / 1.4)
            print(f"Visualizations saved to: {save_dir}")
        except Exception as exc:
            print(f"Visualization failed for {case}: {exc}")

    # region error talk
    traing_nx = data_config['sub_x']
    training_data = f'test_{int((origin_nx - 1) / traing_nx) + 1}'
    if training_data in errors_for_talk_all:
        print(f'\nError analysis on: {training_data}')
        talk_data = errors_for_talk_all[training_data]
        sample_ids = talk_data[:, 3]

        mean_loss = np.mean(talk_data[:, 0])
        std_loss = np.std(talk_data[:, 0])
        threshold = mean_loss + 1 * std_loss
        high_loss_samples = talk_data[talk_data[:, 0] > threshold]

        corr_pde = np.corrcoef(talk_data[:, 0], talk_data[:, 1])[0, 1]
        corr_ref = np.corrcoef(talk_data[:, 0], talk_data[:, 2])[0, 1]

        print(f"  High loss samples: {len(high_loss_samples)}")
        print(f"  Corr(data_loss, pde_residual): {corr_pde:.3f}")
        print(f"  Corr(data_loss, ref_residual): {corr_ref:.3f}")

        with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n├── Error Analysis | {timestamp}\n")
            f.write(f"│   ├── Dataset: {training_data}\n")
            f.write(f"│   ├── High loss samples: {len(high_loss_samples)}\n")
            f.write(f"│   ├── Corr(loss, pde_res): {corr_pde:.3f}\n")
            f.write(f"│   └── Corr(loss, ref_res): {corr_ref:.3f}\n")

        # 误差分布图
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(sample_ids, talk_data[:, 0], color='#0B5873', label='data_loss', alpha=0.7, s=10)
        ax1.set_xlabel('Sample ID')
        ax1.set_ylabel('data_loss', color='#0B5873')
        ax1.tick_params(axis='y', labelcolor='#0B5873')
        ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

        ax2 = ax1.twinx()
        ax2.scatter(sample_ids, talk_data[:, 1], color='#8A0011', label='pde_residual', marker='x', s=10)
        ax2.scatter(sample_ids, talk_data[:, 2], color='#107A38', label='ref_residual', marker='^', s=10)
        ax2.set_ylabel('Residuals', color='#8A0011')
        ax2.tick_params(axis='y', labelcolor='#8A0011')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title('Data Loss vs. PDE/Ref Residuals')
        plt.savefig('error_talk.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    # endregion

    print(f"\n{'='*60}")
    print("Testing completed!")
    print(f"{'='*60}")
    # endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='1D Burgers DCT-I Training')
    parser.add_argument('--config_path', type=str, default='./information.yaml',
                        help='Path to the configuration file')
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
        test(config, args)
    elif args.mode == 'test':
        test(config, args)