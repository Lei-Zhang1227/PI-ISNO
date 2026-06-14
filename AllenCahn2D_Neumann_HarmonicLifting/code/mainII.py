"""
DCT-I for 2D Porous Medium Equation (PME)
u_t = K · Δ(u³/3) with non-homogeneous Neumann BC
"""
import os
import sys
import pandas as pd
from argparse import ArgumentParser
import yaml
import shutil
from dataloader import *
from loss import *  # ← PME: 使用 PME loss
from utils import *
from datetime import datetime
from model import *
import h5py
import torch
from tqdm import tqdm
import pickle
from functools import partial as PARTIAL
import numpy as np
import time
import torch._dynamo

torch._dynamo.config.suppress_errors = True


# ================================================================
# PME 前/后处理
# ================================================================

def compute_p_b(Q_x, Nx, Ny, device):
    """计算 lifting p_b = Q(x)/4 * (y+1)²，每个 batch 只算一次"""
    batch = Q_x.shape[0]
    y_grid = torch.linspace(-1, 1, Ny, device=device, dtype=Q_x.dtype)
    y_plus_1_sq = (y_grid + 1) ** 2
    p_b = Q_x.view(batch, Nx, 1) / 4.0 * y_plus_1_sq.view(1, 1, Ny)
    return p_b  # [batch, Nx, Ny]


def u_to_ph(u, p_b):
    """u → p_h = u³/3 - p_b"""
    p = u ** 3 / 3.0
    return p - p_b


def ph_to_u(p_h, p_b):
    """p_h → u = (3(p_h + p_b))^{1/3}"""
    p = p_h + p_b
    return (3 * p).clamp(min=1e-12) ** (1.0 / 3.0)


def pme_autoregressive_predict(model, xx_u, yy_shape, grid, Q_x,
                               init_t, t_end, device, dtype=torch.float32,
                               verbose=False):
    batch, Nx, Ny = xx_u.shape[0], xx_u.shape[1], xx_u.shape[2]
    p_b = compute_p_b(Q_x, Nx, Ny, device)
    Q_channel = Q_x.view(batch, Nx, 1).expand(-1, -1, Ny)
    u_window = xx_u.squeeze(-1)
    p_h_window = u_window ** 3 / 3.0 - p_b.unsqueeze(-1)
    pred_ph = torch.empty(batch, Nx, Ny, t_end, device=device, dtype=dtype)
    for i in range(init_t):
        pred_ph[..., i] = p_h_window[..., i]
    for t in range(init_t, t_end):
        inp = torch.cat([p_h_window, Q_channel.unsqueeze(-1), grid], dim=-1)
        delta_ph = model(inp)
        p_h_current = p_h_window[..., -1]
        p_h_new = p_h_current + delta_ph.squeeze(-1)
        pred_ph[..., t] = p_h_new
        p_h_window = torch.cat([p_h_window[..., 1:], p_h_new.unsqueeze(-1)], dim=-1)
    pred_u = ph_to_u(pred_ph, p_b.unsqueeze(-1))
    pred = pred_u.unsqueeze(-1)
    return pred, pred_ph, p_b


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

    # ← PME: 读取全局物理参数
    with h5py.File(filepath, 'r') as f:
        K_global = float(f.attrs['K_global'])
    T_total = 1
    print(f"PME params: K_global={K_global}, T={T_total}")

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
        f'{datetime.now()} --- set dataset，batch size: {batch_size}, '
        f'Train loader lens：{train_size}, Test loader lens：{test_size}')
    # endregion

    # region location
    first_dic = f"/code/PME{config['prepare']['project']}"  # ← PME
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
    print(f"{datetime.now()} --- set save dir :{first_dic} ---")
    # endregion

    # region model
    _trans = PARTIAL(Wrapper, [dctI, dctI])
    _itrans = PARTIAL(Wrapper, [idctI, idctI])
    T = Transform(_trans, _itrans)
    Model = PARTIAL(SOL2D, T)
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    out_channels = config['model']['output_channel']
    dim = config['model']['dim']
    tril = config['model']['triL']

    input_channel = initial_step + 3  # ← PME: m(p_h steps) + 1(Q channel) + 2(grid)

    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    print(f"  input_channel={input_channel} (p_h:{initial_step} + Q:1 + grid:2)")
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
        print(f'模型【{args.pretrain}】已加载,当前训练loss为：{best_error}')
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
        f.write(f"\n├─ Problem: 2D PME  u_t = K·Δ(u³/3)\n")  # ← PME
        f.write(f"│   ├── K_global: {K_global}\n")
        f.write(f"│   ├── T: {T_total}\n")
        f.write(f"│   ├── Data Path: {filepath}\n")
        f.write(f"│   └── Sub-sampling: spatial={sub_x}, temporal={sub_t}\n")
        f.write(f"\n├─ Model: SOL2D + DCT-I\n")
        f.write(f"│   ├── Input Channels: {input_channel} (p_h:{initial_step} + Q:1 + grid:2)\n")
        f.write(f"│   ├── Modes: {modes}, Width: {width}\n")
        f.write(f"│   └── Total Parameters: {total_params:,}\n")
        f.write(f"\n├─ Data: train={ntrain}, test={ntest}, batch={batch_size}\n")
        f.write(f"│   └── t_train={t_train}, init_t={init_t}\n")
        f.write(f"\n├─ Optimizer: Adam, lr={base_lr}, scheduler={scheduler_name}\n")
        f.write(f"└─ Device: {device}, Seed: {config['prepare']['seed']}\n")
        f.write(f"\n{'=' * 60}\n")
    # endregion

    # region train
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
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
        for xx, yy, grid, Q_x in test_loader:
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            Q_x = Q_x.to(device, dtype=dtype, non_blocking=True)
            pred, _, _ = pme_autoregressive_predict(
                model, xx, yy.shape, grid, Q_x, init_t, t_train, device, dtype, verbose=True)
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
        # 每个 epoch 的累计计时
        current_t_train = curriculum.update(e - pre_epoch, current_loss=prev_loss) if use_curriculum else t_train
        train_iter = iter(train_loader)

        for b in trange(len(train_loader), file=desc, desc="batch"):
            xx, yy, grid, Q_x = next(train_iter)
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            Q_x = Q_x.to(device, dtype=dtype, non_blocking=True)
            # p_b = compute_p_b(Q_x, Nx, Ny, device)
            optimizer.zero_grad()
            if not torch.isfinite(xx).all():
                print(f"[Error] NaN in input xx at batch {b}")
                continue

            yy = yy[..., :current_t_train, :]

            # ← PME: 自回归预测
            pred, pred_ph, p_b = pme_autoregressive_predict(
                model, xx, yy.shape, grid, Q_x, init_t, current_t_train, device, dtype)
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
                epoch_physics_loss += 0.0
            else:
                if config['train']['loss_mode'] != 'data':
                    residual = compute_pme_residual_dct(pred_ph, Q_x, K_global=K_global, T=T_total)

                    if residual_mode == 'mae':
                        weight = 1.0 / (pred[..., 2:-2, :].detach() ** 2 + 1e-4)
                        weight = weight / weight.mean()
                        loss_f = (residual.abs() * weight).mean()
                    else:
                        weight = 1.0 / (pred[..., 2:-2, :].detach() ** 2 + 1e-4)
                        weight = weight / weight.mean()
                        loss_f = (residual ** 2 * weight).mean()
                    epoch_physics_loss += loss_f.item()

                if config['train']['loss_mode'] == 'both':
                    total_loss = loss_f * f_weight + loss_data * data_weight
                elif config['train']['loss_mode'] == 'data':
                    # 把 yy 转到 p_h 空间
                    yy_p = yy.squeeze(-1) ** 3 / 3.0  # [batch, Nx, Ny, Nt]
                    yy_ph = yy_p - p_b.unsqueeze(-1)  # [batch, Nx, Ny, Nt]

                    # p_h 空间的数据 loss
                    _batch = yy.size(0)
                    out_ph = pred_ph[:, ::rx, ::rx, ::rt].reshape(_batch, -1)
                    y_ph = yy_ph[:, ::rx, ::rx, ::rt].reshape(_batch, -1)
                    loss_data_ph = myloss(out_ph, y_ph)
                    total_loss = loss_data_ph
                else:
                    total_loss = loss_f
            if not torch.isfinite(total_loss).all():
                print(f"[Warning] Invalid loss at epoch {e}, batch {b}, skipping...")
                break

            total_loss.backward()
            clip_info = clipper.step(model)
            optimizer.step()

            if not use_lr_schedule_in_curriculum and scheduler_name == 'cosine_schedule_with_warmup':
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            loss_list.append([loss_data.item(), loss_f.item(), total_loss.item(), e])
            epoch_grad_norms.append(clip_info['grad_norm_after'])
            epoch_loss += total_loss.item()

        # Epoch 结束打印计时
        new_desc = (
            f"Epoch {e + 1} | "
            f"Loss_total: {total_loss.item():.4e}, Test L2: {test_l2:.4e}, "
            f"Loss_data: {loss_data.item():.4e}, Loss_phy: {loss_f.item():.4e}, "
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
                for xx, yy, grid, Q_x in test_loader:  # ← PME
                    xx = xx.to(device, dtype=dtype, non_blocking=True)
                    yy = yy.to(device, dtype=dtype, non_blocking=True)
                    grid = grid.to(device, dtype=dtype, non_blocking=True)
                    Q_x = Q_x.to(device, dtype=dtype, non_blocking=True)  # ← PME

                    pred, _, _ = pme_autoregressive_predict(  # ← PME
                        model, xx, yy.shape, grid, Q_x, init_t, t_train, device, dtype)
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
                    if e % 100 == 0:
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

    # ← PME: 读取物理参数
    with h5py.File(filepath, 'r') as f:
        K_global = float(f.attrs['K_global'])
    T_total = 1

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
    first_dic = f"/code/PME{config['prepare']['project']}"  # ← PME
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion

    # region model
    _trans = PARTIAL(Wrapper, [dctI, dctI])
    _itrans = PARTIAL(Wrapper, [idctI, idctI])
    T = Transform(_trans, _itrans)
    Model = PARTIAL(SOL2D, T)
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    out_channels = config['model']['output_channel']
    dim = config['model']['dim']
    tril = config['model']['triL']
    input_channel = initial_step + 3  # ← PME
    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
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

    test_loaders = {
        f'test_{origin_nx}': test_loader_1x,
        f'test_{int((origin_nx - 1) / 2) + 1}': test_loader_1_2x,
        f'test_{int((origin_nx - 1) / 4) + 1}': test_loader_1_4x,
    }

    results = []
    errors_for_talk_all = {}
    visualize_results_all = []

    for name, test_loader in test_loaders.items():
        errors_for_talk = []
        model.eval()

        with torch.no_grad():
            test_iter = iter(test_loader)
            for b in tqdm(range(len(test_loader))):
                xx, yy, grid, Q_x = next(test_iter)  # ← PME
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                Q_x = Q_x.to(device, dtype=dtype, non_blocking=True)  # ← PME

                pred, _, _ = pme_autoregressive_predict(  # ← PME
                    model, xx, yy.shape, grid, Q_x, init_t, t_train, device, dtype)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]

                assert pred.shape == yy.shape
                _batch = yy.size(0)
                _yy = yy[..., init_t + 1:t_train, :]
                _pred = pred[..., init_t + 1:t_train, :]
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                # ← PME: 使用 PME 残差
                residual_p = compute_pme_residual_dct(pred, Q_x, K_global=K_global, T=T_total)
                loss_f_p = torch.abs(residual_p).mean()
                residual_y = compute_pme_residual_dct(yy, Q_x, K_global=K_global, T=T_total)
                loss_f_y = torch.abs(residual_y).mean()

                errors_for_talk.append([l2_error, loss_f_p.item(), loss_f_y.item(), int(b)])

            # 选择 best/mid/worst 样本做可视化
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
                xx, yy, grid, Q_x = next(test_iter)  # ← PME
                if b not in selected_indices:
                    continue
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                Q_x = Q_x.to(device, dtype=dtype, non_blocking=True)  # ← PME

                pred, _, _ = pme_autoregressive_predict(  # ← PME
                    model, xx, yy.shape, grid, Q_x, init_t, t_train, device, dtype)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]

                assert pred.shape == yy.shape
                f_p = compute_pme_residual_dct(pred, Q_x, K_global=K_global, T=T_total)
                f_y = compute_pme_residual_dct(yy, Q_x, K_global=K_global, T=T_total)

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
                    'Q_x': Q_x.squeeze().cpu().numpy(),  # ← PME: 保存 Q(x) 用于可视化
                })
            visualize_results_all.append(visualize_results)

        # 统计
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
    parser = ArgumentParser(description='PME 2D Training')
    parser.add_argument('--config_path', type=str, default='./yaml/informationyaml',
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
