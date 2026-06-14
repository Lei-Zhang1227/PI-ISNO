"""
DCT-I for 2D Allen-Cahn Equation (Direct prediction in u_h space)
∂u/∂t = ε∇²u + u - u³  with non-homogeneous Neumann BC

Key design:
  - Lifting: u_b = αx² + βy² + γx + δy (harmonic, Δu_b=0)
  - u_h = u - u_b satisfies homogeneous Neumann → DCT-I safe
  - Network predicts u_h directly (not delta)
  - Recovery: u = u_h + u_b (LINEAR, no amplification!)
  - u_b as extra input channel (reaction term couples u_b into dynamics)
"""
import os
import sys
import pandas as pd
from argparse import ArgumentParser
import yaml
import shutil
from dataloader import *
from loss import *
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
# AC 前/后处理
# ================================================================

def compute_u_b(bc_params, Nx, Ny, device=None):
    """Construct per-sample harmonic lifting from BC parameters.
    Args:
        bc_params: [batch, 4] tensor of (a, b, c, d) per sample
        Nx: int, spatial resolution in x direction
        Ny: int, spatial resolution in y direction
        device: torch device (e.g., 'cuda:0', 'cpu')
    Returns:
        u_b: [batch, 1, Nx, Ny] lifting field
    """
    if device is None:
        device = bc_params.device
    bc_params = bc_params.to(device=device, dtype=torch.float32)

    x = torch.linspace(-1, 1, Nx, device=device, dtype=torch.float32)
    y = torch.linspace(-1, 1, Ny, device=device, dtype=torch.float32)

    a = bc_params[:, 0]
    b = bc_params[:, 1]
    c = bc_params[:, 2]
    d = bc_params[:, 3]
    alpha = (b - a) / 4.0
    beta = (d - c) / 4.0
    gamma = (a + b) / 2.0
    delta = (c + d) / 2.0

    X2 = (x ** 2).view(1, 1, -1, 1)
    Y2 = (y ** 2).view(1, 1, 1, -1)
    Xv = x.view(1, 1, -1, 1)
    Yv = y.view(1, 1, 1, -1)
    alpha = alpha.view(-1, 1, 1, 1)
    beta = beta.view(-1, 1, 1, 1)
    gamma = gamma.view(-1, 1, 1, 1)
    delta = delta.view(-1, 1, 1, 1)

    u_b = alpha * X2 + beta * Y2 + gamma * Xv + delta * Yv
    return u_b.squeeze(1) 

def u_to_uh(u, u_b):
    """u → u_h = u - u_b"""
    return u - u_b


def uh_to_u(u_h, u_b):
    """u_h → u = u_h + u_b  (纯线性，无放大！)"""
    return u_h + u_b


def ac_autoregressive_predict(model, xx_u, yy_shape, grid, bc_params,
                               init_t, t_end, device, dtype=torch.float32,
                               verbose=False):
    """
    Allen-Cahn 自回归预测（u_h 空间操作，u 空间输出）
    Direct mode: 模型直接预测 u_h_new

    Args:
        model:      网络
        xx_u:       [batch, Nx, Ny, m, 1] 初始 m 步的 u
        yy_shape:   输出 pred 的目标 shape
        grid:       [batch, Nx, Ny, 2]
        bc_params:  [batch, 4] → (a, b, c, d)
        init_t:     初始步数 m
        t_end:      预测到的时间步
    Returns:
        pred:       [batch, Nx, Ny, t_end, 1] 预测的 u
        pred_uh:    [batch, Nx, Ny, t_end] 预测的 u_h
    """
    batch, Nx, Ny = xx_u.shape[0], xx_u.shape[1], xx_u.shape[2]

    # 预计算 lifting 和 u_b channel
    u_b = compute_u_b(bc_params, Nx, Ny, device)  # [batch, Nx, Ny]
    u_b_channel = u_b  # 直接用 u_b 场作为输入通道 [batch, Nx, Ny]

    # 初始 u → u_h 窗口
    u_window = xx_u.squeeze(-1)  # [batch, Nx, Ny, m]
    u_h_window = u_window - u_b.unsqueeze(-1)  # [batch, Nx, Ny, m]

    # 输出容器
    pred_uh = torch.empty(batch, Nx, Ny, t_end, device=device, dtype=dtype)
    for i in range(init_t):
        pred_uh[..., i] = u_h_window[..., i]

    for t in range(init_t, t_end):
        # 构建输入: [u_h_steps(m), u_b_channel(1), grid(2)]
        inp = torch.cat([u_h_window, u_b_channel.unsqueeze(-1), grid], dim=-1)

        # 网络直接预测 u_h_new（Direct mode）
        u_h_new = model(inp).squeeze(-1)  # [batch, Nx, Ny]

        pred_uh[..., t] = u_h_new

        # 滑动 u_h 窗口
        u_h_window = torch.cat([u_h_window[..., 1:], u_h_new.unsqueeze(-1)], dim=-1)

    # 恢复 u（纯线性加法，无放大！）
    pred_u = pred_uh + u_b.unsqueeze(-1)  # [batch, Nx, Ny, t_end]
    pred = pred_u.unsqueeze(-1)  # [batch, Nx, Ny, t_end, 1]

    return pred, pred_uh


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

    # AC 物理参数
    with h5py.File(filepath, 'r') as f:
        epsilon = float(f.attrs.get('epsilon', 0.05))
    T_total = float(config['data'].get('T', 1.0))
    print(f"AC params: epsilon={epsilon}, T={T_total}")

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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False)
    train_size, test_size = len(train_data), len(test_data)
    ntrain, ntest = train_size, test_size
    print(
        f'{datetime.now()} --- set dataset, batch size: {batch_size}, '
        f'Train: {train_size}, Test: {test_size}')
    # endregion

    # region location
    first_dic = f"/code/AC2D{config['prepare']['project']}"
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
                new_name = "{}-retrain--{}{}".format(name, i, ext)
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

    input_channel = initial_step + 3  # m(u_h steps) + 1(u_b channel) + 2(grid)

    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Params: {total_params}")
    print(f"  input_channel={input_channel} (u_h:{initial_step} + u_b:1 + grid:2)")
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
                'cosine_schedule_with_warmup, warm epoch is {}, total steps is {}'.format(
                    int(cosine_schedul * epoch_set), step))
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['patience'],
                                                        gamma=config['train']['gamma'])

    clipper = RobustAdaptiveGradientClipperV2(
        initial_max_norm=500.0, window_size=60, trim_k=5, multiplier=5)
    print('{} --- set optimizer, lr={}, scheduler:{}, curriculum:{}---'.format(
        datetime.now(), base_lr, scheduler_name, use_curriculum))
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
        print('模型已加载, 当前训练loss为：{}'.format(best_error))
    else:
        loss_list = []
        test_loss_list = []
        lr_list = []
        grad = []
        best_error = 100.0
    # endregion

    # region information
    with open('{}/Experiment_record.txt'.format(first_dic), 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("\n{}\n".format('=' * 60))
        f.write("[Experiment Log | {}]\n".format(timestamp))
        f.write("{}\n".format('=' * 60))
        f.write("\n├─ Problem: 2D Allen-Cahn  ∂u/∂t = ε∇²u + u - u³\n")
        f.write("│   ├── epsilon: {}\n".format(epsilon))
        f.write("│   ├── T: {}\n".format(T_total))
        f.write("│   ├── Data Path: {}\n".format(filepath))
        f.write("│   └── Sub-sampling: spatial={}, temporal={}\n".format(sub_x, sub_t))
        f.write("\n├─ Model: SOL2D + DCT-I (Direct u_h prediction)\n")
        f.write("│   ├── Input: {} (u_h:{} + u_b:1 + grid:2)\n".format(input_channel, initial_step))
        f.write("│   ├── Modes: {}, Width: {}\n".format(modes, width))
        f.write("│   └── Params: {:,}\n".format(total_params))
        f.write("\n├─ Data: train={}, test={}, batch={}\n".format(ntrain, ntest, batch_size))
        f.write("│   └── t_train={}, init_t={}\n".format(t_train, init_t))
        f.write("\n├─ Optimizer: Adam, lr={}, scheduler={}\n".format(base_lr, scheduler_name))
        f.write("└─ Device: {}, Seed: {}\n".format(device, config['prepare']['seed']))
        f.write("\n{}\n".format('=' * 60))
    # endregion

    # region train
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    init_t = data_config['initial_step']
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    print('t_train is {}'.format(t_train))
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
    print('rx:{}, rt:{}'.format(rx, rt))
    desc = DescStr()
    time_0 = time.time()
    time_old = time.time()
    dtype = torch.float32
    residual_mode = config['train'].get('residual_mode', 'mse')

    # ===== 初始 eval =====
    model.eval()
    val_l2_full = 0
    with torch.no_grad():
        for xx, yy, grid, bc_params in test_loader:  # ← AC: bc_params 代替 Q_x
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

            pred, pred_uh = ac_autoregressive_predict(
                model, xx, yy.shape, grid, bc_params, init_t, t_train, device, dtype)

            _batch = yy.size(0)
            _pred = pred[..., init_t:, :]
            _yy = yy[..., init_t:, :]
            val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
        test_l2 = val_l2_full * batch_size / ntest
        print('epoch:{}, test_l2: {}'.format(pre_epoch, test_l2))
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
            bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

            optimizer.zero_grad()

            if not torch.isfinite(xx).all():
                print("[Error] NaN in input xx at batch {}".format(b))
                continue

            yy = yy[..., :current_t_train, :]

            # AC: 自回归预测 (u_h 空间, direct mode)
            pred, pred_uh = ac_autoregressive_predict(
                model, xx, yy.shape, grid, bc_params, init_t, current_t_train, device, dtype)
            pred[..., 0:init_t, :] = yy[..., 0:init_t, :]

            assert pred.shape == yy.shape, "Shape mismatch: {} != {}".format(pred.shape, yy.shape)

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
                # Warmup 在 u_h 空间（线性恢复，无需担心放大）
                u_b = compute_u_b(bc_params, xx.shape[1], xx.shape[2], device)
                yy_uh = yy.squeeze(-1) - u_b.unsqueeze(-1)
                last_uh = pred_uh[..., init_t - 1:init_t]
                target_uh = last_uh.expand(-1, -1, -1, current_t_train - init_t)
                pred_uh_part = pred_uh[..., init_t:current_t_train]
                loss_pretrain = ((pred_uh_part - target_uh) ** 2).mean()
                total_loss = loss_pretrain
                epoch_physics_loss += 0.0
                if b == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr[e]
                    print("[Warmup {}/{}] Pretrain loss: {:.4e}".format(e, warmup_epochs, loss_pretrain.item()))
            else:
                if config['train']['loss_mode'] != 'data':
                    # AC: PDE 残差 u_t - ε∇²u - u + u³ = 0
                    residual = compute_ac2d_residual_batch(pred, bc_params, epsilon=epsilon, T=T_total)

                    if residual_mode == 'mae':
                        loss_f = residual.abs().mean()
                    else:
                        loss_f = (residual ** 2).mean()
                    epoch_physics_loss += loss_f.item()

                if config['train']['loss_mode'] == 'both':
                    total_loss = loss_f * f_weight + loss_data * data_weight
                elif config['train']['loss_mode'] == 'data':
                    total_loss = loss_data
                else:
                    total_loss = loss_f

            if not torch.isfinite(total_loss).all():
                print("[Warning] Invalid loss at epoch {}, batch {}, skipping...".format(e, b))
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

        new_desc = (
            "Epoch {} | "
            "Loss_total: {:.4e}, Test L2: {:.4e}, "
            "Loss_data: {:.4e}, Loss_phy: {:.4e}, "
            "lr: {:.2e}, Grad: {:.2f}").format(
            e + 1, total_loss.item(), test_l2,
            loss_data.item(), loss_f.item(),
            current_lr, clip_info['grad_norm_after'])
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
                            filename='{}/checkpoint-best'.format(first_dic))
        save_checkpoint(model, e, optimizer, scheduler, loss_list,
                        test_loss_list, lr_list, model_save_record, grad,
                        filename='{}/checkpoint_newst'.format(first_dic))

        if e % config['train']['verbose_interval'] == 0:
            model.eval()
            val_l2_full = 0
            with torch.no_grad():
                for xx, yy, grid, bc_params in test_loader:
                    xx = xx.to(device, dtype=dtype, non_blocking=True)
                    yy = yy.to(device, dtype=dtype, non_blocking=True)
                    grid = grid.to(device, dtype=dtype, non_blocking=True)
                    bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

                    pred, pred_uh = ac_autoregressive_predict(
                        model, xx, yy.shape, grid, bc_params, init_t, t_train, device, dtype)

                    _batch = yy.size(0)
                    _pred = pred[..., init_t:, :]
                    _yy = yy[..., init_t:, :]
                    val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                test_l2 = val_l2_full * batch_size / ntest
                print('epoch:{}, test_l2: {}'.format(e, test_l2))
                test_loss_list.append([test_l2, e])
                if e % config['train']['check_epochs'] == 0:
                    save_checkpoint(model, e, optimizer, scheduler, loss_list, test_loss_list,
                                    lr_list, model_save_record, grad,
                                    filename='{}/checkpoint-{}'.format(first_dic, e))
                    if e % 100 == 0:
                        time_elapsed = time.time() - time_0
                        hours = int(time_elapsed // 3600)
                        minutes = int((time_elapsed % 3600) // 60)
                        time_elapsed_100 = time.time() - time_old
                        hours_100 = int(time_elapsed_100 // 3600)
                        minutes_100 = int((time_elapsed_100 % 3600) // 60)
                        with open('{}/Experiment_record.txt'.format(first_dic), 'a', encoding='utf-8') as f:
                            f.write("├── Test l2 error in epoch {}: {:.4e}\n".format(e, test_l2))
                            f.write("│   └── Time: {}h {}m\n".format(hours, minutes))
                            f.write("│   └── Per 100: {}h {}m\n".format(hours_100, minutes_100))
                            f.write("│   └── Best: epoch {}, loss {:.4e}\n".format(
                                model_save_record[-1][0], model_save_record[-1][1]))
                        time_old = time.time()
            model.train()

    print('{} --- training succeed ---'.format(datetime.now()))
    time_elapsed = time.time() - time_0
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    with open('{}/Experiment_record.txt'.format(first_dic), 'a', encoding='utf-8') as f:
        f.write("├── training succeed | [{}]\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("│   └── Time: {}h {}m\n".format(hours, minutes))
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
    filepath = '/data/zhanglei/BurgersEquationII/ac2d_extend_200.h5'#data_config['datapath']
    sub_t = data_config['sub_t']

    with h5py.File(filepath, 'r') as f:
        epsilon = float(f.attrs.get('epsilon', 0.05))
    T_total = float(config['data'].get('T', 1.0))

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
    print("{} --- set save dir: {} ---".format(datetime.now(), config['prepare']['project']))
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
    input_channel = initial_step + 3
    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} --- set model. Params: {}".format(datetime.now(), total_params))
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
        print("{} --- model loaded, {} epochs trained".format(datetime.now(), epoch))
    else:
        checkpoint = torch.load('checkpoint-best.pth.tar')
        grad_array = checkpoint['grad']
        model.load_state_dict(checkpoint['model'])
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        epoch = checkpoint['epoch']
        model_name = 'checkpoint-best'
        print("{} --- best model loaded, {} epochs trained".format(datetime.now(), epoch))
    # endregion

    # region plot loss
    plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array,
                               '{}/loss_carve_for_{}'.format(first_dic, model_name))
    # endregion

    # region evaluate
    dtype = torch.float32
    init_t = int(data_config['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=True)
    loss_fn = myloss

    test_loaders = {
        'test_{}'.format(origin_nx): test_loader_1x,
        'test_{}'.format(int((origin_nx - 1) / 2) + 1): test_loader_1_2x,
        'test_{}'.format(int((origin_nx - 1) / 4) + 1): test_loader_1_4x,
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
                xx, yy, grid, bc_params = next(test_iter)  # ← AC
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

                pred, pred_uh = ac_autoregressive_predict(
                    model, xx, yy.shape, grid, bc_params, init_t, t_train, device, dtype)

                assert pred.shape == yy.shape
                _batch = yy.size(0)
                _yy = yy[..., init_t + 1:t_train, :]
                _pred = pred[..., init_t + 1:t_train, :]
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                # AC PDE 残差
                residual_p = compute_ac2d_residual_batch(pred, bc_params, epsilon=epsilon, T=T_total)
                loss_f_p = torch.abs(residual_p).mean()
                residual_y = compute_ac2d_residual_batch(yy, bc_params, epsilon=epsilon, T=T_total)
                loss_f_y = torch.abs(residual_y).mean()

                errors_for_talk.append([l2_error, loss_f_p.item(), loss_f_y.item(), int(b)])

            # 选择 best/mid/worst
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
                xx, yy, grid, bc_params = next(test_iter)
                if b not in selected_indices:
                    continue
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

                pred, pred_uh = ac_autoregressive_predict(
                    model, xx, yy.shape, grid, bc_params, init_t, t_train, device, dtype)

                assert pred.shape == yy.shape
                f_p = compute_ac2d_residual_batch(pred, bc_params, epsilon=epsilon, T=T_total)
                f_y = compute_ac2d_residual_batch(yy, bc_params, epsilon=epsilon, T=T_total)

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
                    'bc_params': bc_params.squeeze().cpu().numpy(),
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
        plot_burgers2d_comparison(visualize_results, save_dir='./figures_{}'.format(case))
    print('可视化完成~')
    # endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='AC 2D Training (Direct u_h)')
    parser.add_argument('--config_path', type=str, default='./yaml/ac_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on wandb')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
    parser.add_argument('--load_lr', action='store_true', help='load lr from pretrain')
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