"""
DCT-I for 2D Heat equation with harmonic lifting (u_h + u_b decomposition)
Delta prediction mode: model predicts Δu_h instead of u_h directly
"""
import os
from pathlib import Path
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

CASE_ROOT = Path(__file__).resolve().parents[1]

def resolve_case_path(path_like):
    path = Path(path_like)
    return str(path if path.is_absolute() else CASE_ROOT / path)



def compute_ub_fixed_bc(Nx, device, dtype=torch.float32, a=0.5, b=-0.5):
    """
    计算固定边界条件下的 u_b
    
    对于 BC: (a, b, c, d) = (0.5, -0.5, 0, 0)
    解析解: u_b(x, y) = -0.5*x^2 + 0.5*y^2
    
    Args:
        Nx: 网格点数
        device: torch.device
        dtype: 数据类型
        a, b: 左右边界的 Neumann BC 值
    
    Returns:
        u_b: [1, Nx, Nx, 1, 1]
    """
    x_coord = torch.linspace(-1, 1, Nx, device=device, dtype=dtype)
    y_coord = torch.linspace(-1, 1, Nx, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x_coord, y_coord, indexing='ij')
    
    # 解析解: u_b(x,y) = -0.5*x^2 + 0.5*y^2
    # 这满足 ∂u_b/∂x|_{x=-1} = x = -(-1) = 1 ... 等等
    u_b = (-0.5 * X ** 2 + 0.5 * Y ** 2).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    return u_b  # [1, Nx, Nx, 1, 1]


def autoregressive_rollout_delta(model, xx_h, grid, u_b, init_t, t_train, dtype=torch.float32):
    """
    自回归预测 u_h (delta 模式)
    
    模型预测: Δu_h(t) = u_h(t+1) - u_h(t)
    
    Args:
        model: 神经网络模型
        xx_h: [batch, Nx, Ny, init_step, 1] - u_h 的初始条件
        grid: [batch, Nx, Ny, 2] - 空间网格
        u_b: [1, Nx, Ny, 1, 1] - 边界部分 (时间不变)
        init_t: 初始步数
        t_train: 总时间步数
        dtype: 数据类型
    
    Returns:
        pred_h: [batch, Nx, Ny, t_train, 1] - 预测的 u_h
        pred: [batch, Nx, Ny, t_train, 1] - 还原的 u = u_h + u_b
    """
    device = xx_h.device
    batch_size = xx_h.size(0)
    
    # 初始化预测数组
    pred_h = torch.empty(
        (batch_size, xx_h.size(1), xx_h.size(2), t_train, 1),
        device=device,
        dtype=dtype
    )
    
    # 复制初始条件
    pred_h[..., :init_t, :] = xx_h[..., :init_t, :]
    
    # 自回归预测
    xx_h_current = xx_h.clone()  # [batch, Nx, Ny, init_step, 1]
    
    for t in range(init_t, t_train):
        # 输入: 最近的 init_step 个 u_h + grid
        inp = xx_h_current.squeeze(-1)  # [batch, Nx, Ny, init_step]
        inp_with_grid = torch.cat([inp, grid], dim=-1)  # [batch, Nx, Ny, init_step+2]
        
        # 模型预测 Δu_h
        delta_uh = model(inp_with_grid).unsqueeze(-1)  # [batch, Nx, Ny, 1, 1]
        
        # 更新: u_h(t+1) = u_h(t) + Δu_h
        last_uh = xx_h_current[..., -1:, :]  # [batch, Nx, Ny, 1, 1]
        next_uh = last_uh + delta_uh
        
        # 保存预测
        pred_h[..., t:t+1, :] = next_uh
        
        # 滑动窗口更新输入
        xx_h_current = torch.cat([xx_h_current[..., 1:, :], next_uh], dim=-2)
    
    # 还原完整解: u = u_h + u_b
    pred = pred_h + u_b  # broadcast u_b to all time steps
    
    return pred_h, pred


def autoregressive_rollout_direct(model, xx_h, grid, u_b, init_t, t_train, dtype=torch.float32):
    """
    自回归预测 u_h (直接预测模式)
    
    模型直接预测: u_h(t+1)
    
    Args:
        (同上)
    
    Returns:
        pred_h: [batch, Nx, Ny, t_train, 1]
        pred: [batch, Nx, Ny, t_train, 1]
    """
    device = xx_h.device
    batch_size = xx_h.size(0)
    
    pred_h = torch.empty(
        (batch_size, xx_h.size(1), xx_h.size(2), t_train, 1),
        device=device,
        dtype=dtype
    )
    
    pred_h[..., :init_t, :] = xx_h[..., :init_t, :]
    xx_h_current = xx_h.clone()
    
    for t in range(init_t, t_train):
        inp = xx_h_current.squeeze(-1)
        inp_with_grid = torch.cat([inp, grid], dim=-1)
        
        # 直接预测 u_h(t+1)
        next_uh = model(inp_with_grid).unsqueeze(-1)
        
        pred_h[..., t:t+1, :] = next_uh
        xx_h_current = torch.cat([xx_h_current[..., 1:, :], next_uh], dim=-2)
    
    pred = pred_h + u_b
    
    return pred_h, pred


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
    filepath = resolve_case_path(data_config['datapath'])
    sub_x = data_config['sub_x']
    sub_t = data_config['sub_t']
    Nx = int((data_config['nx'] - 1) / sub_x) + 1
    
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                              num_workers=2, shuffle=False)
    train_size, test_size = len(train_data), len(test_data)
    ntrain, ntest = train_size, test_size
    
    print(f'{datetime.now()} --- Dataset loaded: batch_size={batch_size}, '
          f'train={train_size}, test={test_size}, Nx={Nx}')
    # endregion
    
    # region location
    first_dic = resolve_case_path(config['prepare']['project'])
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
    print(f"{datetime.now()} --- Save dir: {first_dic}")
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
    input_channel = initial_step + 2
    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- Model initialized. Params: {total_params:,}")
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

    if not use_lr_schedule_in_curriculum:
        scheduler_name = config['train']['scheduler']
        if config['train']['scheduler'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=config['train']['milestones'],
                                                             gamma=config['train']['gamma'])
        elif config['train']['scheduler'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=config['train']['gamma'],
                                                                   threshold=1e-2, 
                                                                   patience=config['train']['patience'],
                                                                   verbose=True)
        elif config['train']['scheduler'] == 'cosine_schedule_with_warmup':
            epoch_set = config['train']['epochs']
            bfe = math.ceil(train_size / batch_size)
            step = bfe * epoch_set
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=step * config['train']['cosine_schedul'], 
                num_training_steps=step
            )
            cosine_schedul = config['train']['cosine_schedul']
            print(f'Cosine schedule: warmup={int(cosine_schedul * epoch_set)} epochs, '
                  f'total_steps={step}')
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=config['train']['patience'],
                                                        gamma=config['train']['gamma'])

    clipper = RobustAdaptiveGradientClipperV2(
        initial_max_norm=500.0,
        window_size=60,
        trim_k=5,
        multiplier=5)
    
    print(f'{datetime.now()} --- Optimizer: lr={base_lr}, scheduler={scheduler_name}, '
          f'curriculum={use_curriculum}')
    
    warmup_epochs = config['train']['warmup_epochs']
    warmup_lr = config['train']['warmup_lr']
    # endregion
    
    # region load model
    if args.pretrain is not None:
        checkpoint = torch.load(resolve_case_path(args.pretrain))
        model.load_state_dict(checkpoint['model'])
        if config['train']['retrain_load_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print('Loaded optimizer and scheduler')
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        grad = checkpoint['grad']
        epoch = checkpoint['epoch']
        best_error = loss_list[-1][0]
        print(f'Loaded model from {args.pretrain}, best_loss={best_error:.4e}')
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
        f.write(f"\n├─ Problem: 2D Heat equation with harmonic lifting\n")
        f.write(f"│   ├── Data: {filepath}\n")
        f.write(f"│   ├── Sub-sampling: x={sub_x}, t={sub_t}\n")
        f.write(f"│   ├── Prediction mode: Delta (Δu_h)\n")
        f.write(f"│   └── BC: (a,b,c,d) = (0.5, -0.5, 0, 0)\n")
        f.write(f"\n├─ Model: SOL2D + DCT-I\n")
        f.write(f"│   ├── Params: {total_params:,}\n")
        f.write(f"│   ├── Modes: {modes}, Width: {width}, Bandwidth: {bandwidth}\n")
        f.write(f"│   └── Input: {input_channel} channels (init_step={initial_step} + grid)\n")
        f.write(f"\n├─ Training\n")
        f.write(f"│   ├── Samples: train={ntrain}, test={ntest}\n")
        f.write(f"│   ├── Batch: {batch_size}, Steps: {t_train}\n")
        f.write(f"│   └── Device: {device}\n")
        f.write(f"\n{'=' * 60}\n")
    # endregion
    
    # region train
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    init_t = data_config['initial_step']
    
    # 预计算 u_b (固定BC，只需计算一次)
    dtype = torch.float32
    u_b = compute_ub_fixed_bc(Nx, device, dtype=dtype)
    print(f'u_b computed: shape={u_b.shape}')
    
    nx = int((data_config['nx'] - 1) / data_config['sub_x']) + 1
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
    pre_mode = config['train'].get('pre_mode', 'delta')  # 'delta' or 'direct'
    desc = DescStr()
    time_0 = time.time()
    time_old = time.time()
    
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n├─ Training Start! | {timestamp}]\n")
        f.write(f"│   └── Prediction mode: {pre_mode}\n")
    
    # 选择预测函数
    if pre_mode == 'delta':
        rollout_fn = autoregressive_rollout_delta
        print("Using DELTA prediction mode: model predicts Δu_h")
    else:
        rollout_fn = autoregressive_rollout_direct
        print("Using DIRECT prediction mode: model predicts u_h(t+1)")
    
    # Initial evaluation
    model.eval()
    val_l2_full = 0
    with torch.no_grad():
        for xx, yy, grid in test_loader:
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            
            # 分离 u_h = u - u_b
            xx_h = xx - u_b[..., :init_t, :]
            
            # 自回归预测
            _, pred = rollout_fn(model, xx_h, grid, u_b, init_t, t_train, dtype)
            
            _batch = yy.size(0)
            _pred = pred[..., init_t:, :]
            _yy = yy[..., init_t:, :]
            val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
        
        test_l2 = val_l2_full * batch_size / ntest
        print(f'Initial test_l2: {test_l2:.4e}')
    
    model.train()
    First_data = True
    prev_loss = 10000
    residual_mode = config['train'].get('residual_mode', 'mse')

    for e in ebar:
        epoch_loss = 0
        epoch_physics_loss = 0
        epoch_grad_norms = []

        current_t_train = curriculum.update(e - pre_epoch, current_loss=prev_loss) if use_curriculum else t_train
        train_iter = iter(train_loader)

        for b in trange(len(train_loader), file=desc, desc="batch"):
            xx, yy, grid = next(train_iter)
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            _batch = yy.size(0)

            optimizer.zero_grad()

            if not torch.isfinite(xx).all():
                print(f"[Error] NaN in input at batch {b}")
                continue

            # 分离 u_h = u - u_b
            xx_h = xx - u_b[..., :init_t, :]
            
            # 截断到当前课程长度
            yy_truncated = yy[..., :current_t_train, :]
            
            # 自回归预测 (仅到 current_t_train)
            _, pred = rollout_fn(model, xx_h, grid, u_b, init_t, current_t_train, dtype)
            
            assert pred.shape == yy_truncated.shape, \
                f"Shape mismatch: {pred.shape} != {yy_truncated.shape}"
            
            # 数据损失
            out_data = pred[:, ::rx, ::rx, ::rt, :].reshape(_batch, -1)
            y_data = yy_truncated[:, ::rx, ::rx, ::rt, :].reshape(_batch, -1)
            loss_data = myloss(out_data, y_data)
            
            if First_data:
                print(f'Output shape: {pred[:, ::rx, ::rx, ::rt, :].shape}')
                First_data = False

            # 物理损失
            loss_f = torch.tensor(0.0, device=device)
            
            if warmup_epochs and e < warmup_epochs:
                # Warmup: 预训练到常数初始状态
                last_input = yy_truncated[..., init_t - 1:init_t, :]
                target = last_input.expand(-1, -1, -1, current_t_train - init_t, -1)
                pred_part = pred[..., init_t:current_t_train, :]
                loss_pretrain = ((pred_part - target) ** 2).mean()
                total_loss = loss_pretrain
                epoch_physics_loss += 0.0
                
                if b == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr[e]
                    print(f"[Warmup {e}/{warmup_epochs}] loss={loss_pretrain.item():.4e}")
            else:
                if config['train']['loss_mode'] != 'data':
                    residual = compute_heat2d_residual_batch(pred, kappa=0.02, T=1)
                    
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
                print(f"[Warning] Invalid loss at epoch {e}, batch {b}")
                break

            # 反向传播
            total_loss.backward()
            clip_info = clipper.step(model)
            optimizer.step()

            if not use_lr_schedule_in_curriculum and scheduler_name == 'cosine_schedule_with_warmup':
                scheduler.step()

            # 记录
            current_lr = optimizer.param_groups[0]['lr']
            loss_list.append([loss_data.item(), loss_f.item(), total_loss.item(), e])
            epoch_grad_norms.append(clip_info['grad_norm_after'])
            epoch_loss += total_loss.item()

        new_desc = (
            f"Epoch {e+1} | Loss={total_loss.item():.4e}, "
            f"Data={loss_data.item():.4e}, Phy={loss_f.item():.4e}, "
            f"Test={test_l2:.4e}, lr={current_lr:.2e}")
        ebar.set_description(new_desc)
        
        # Epoch 结束
        prev_loss = epoch_physics_loss / len(train_loader)
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_grad_norm = np.mean(epoch_grad_norms)
        grad.append([e, avg_epoch_loss, avg_grad_norm])
        lr_list.append([current_lr, current_t_train, e])
        
        # 更新学习率
        if use_lr_schedule_in_curriculum:
            curriculum.step_lr_scheduler(avg_epoch_loss)
        elif scheduler is not None:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(avg_epoch_loss)
            elif scheduler_name == 'cosine_schedule_with_warmup':
                pass
            else:
                scheduler.step()
        
        # 保存最佳模型
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
                for xx, yy, grid in test_loader:
                    xx = xx.to(device, dtype=dtype, non_blocking=True)
                    yy = yy.to(device, dtype=dtype, non_blocking=True)
                    grid = grid.to(device, dtype=dtype, non_blocking=True)
                    
                    xx_h = xx - u_b[..., :init_t, :]
                    _, pred = rollout_fn(model, xx_h, grid, u_b, init_t, t_train, dtype)
                    
                    _batch = yy.size(0)
                    _pred = pred[..., init_t:, :]
                    _yy = yy[..., init_t:, :]
                    val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                
                test_l2 = val_l2_full * batch_size / ntest
                print(f'Epoch {e}: test_l2={test_l2:.4e}')
                test_loss_list.append([test_l2, e])
                
                if e % config['train']['check_epochs'] == 0:
                    save_checkpoint(model, e, optimizer, scheduler, loss_list, test_loss_list,
                                    lr_list, model_save_record, grad, 
                                    filename=f'{first_dic}/checkpoint-{e}')
                    
                    if e % 100 == 0:
                        time_elapsed = time.time() - time_0
                        hours = int(time_elapsed // 3600)
                        minutes = int((time_elapsed % 3600) // 60)
                        time_elapsed_100 = time.time() - time_old
                        hours_100 = int(time_elapsed_100 // 3600)
                        minutes_100 = int((time_elapsed_100 % 3600) // 60)
                        
                        with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
                            f.write(f"├── Epoch {e}: test_l2={test_l2:.4e}\n")
                            f.write(f"│   ├── Time: {hours}h {minutes}m total\n")
                            f.write(f"│   ├── Time/100ep: {hours_100}h {minutes_100}m\n")
                            f.write(f"│   └── Best: epoch={model_save_record[-1][0]}, "
                                   f"loss={model_save_record[-1][1]:.4e}\n")
                        time_old = time.time()
            model.train()
    
    print(f'{datetime.now()} --- Training completed ---')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_elapsed = time.time() - time_0
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n├─ Training completed | {timestamp}\n")
        f.write(f"│   └── Total time: {hours}h {minutes}m\n")
        f.write("-" * 60 + "\n")
    # endregion


def test(config, args):
    """
    测试函数 - 在多个分辨率上评估模型
    """
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
    filepath = resolve_case_path(data_config['datapath'])
    sub_t = data_config['sub_t']
    
    # 创建三个不同分辨率的测试集
    test_data_1x = FNODatasetMult(file_path=filepath,
                                  initial_step=initial_step,
                                  sub_x=1,
                                  sub_t=sub_t,
                                  if_test=True,
                                 
                                  )
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)
 
    test_data_1_2x = FNODatasetMult(file_path=filepath,
                                    initial_step=initial_step,
                                    sub_x=2,
                                    sub_t=sub_t,
                                    if_test=True,
                                    
                                    )
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)
 
    test_data_1_4x = FNODatasetMult(file_path=filepath,
                                    initial_step=initial_step,
                                    sub_x=4,
                                    sub_t=sub_t,
                                    if_test=True,
                                   
                                    )
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)
 
    test_size = len(test_data_1x)
    ntest = test_size
    # endregion
    
    # region location
    first_dic = resolve_case_path(config['prepare']['project'])
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- Save dir: {first_dic}")
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
    input_channel = initial_step + 2
    
    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- Model params: {total_params:,}")
    # endregion
    
    # region load model
    if args.pretrain is not None:
        model_name = os.path.basename(args.pretrain).replace('.pth.tar', '')
        model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
        checkpoint = torch.load(resolve_case_path(args.pretrain))
        model.load_state_dict(checkpoint['model'])
        
        loss_list = checkpoint.get('loss_list', [])
        test_loss_list = checkpoint.get('test_loss_list', [])
        grad_array = checkpoint.get('grad', [])
        lr_list = checkpoint.get('lr_list', [])
        epoch = checkpoint.get('epoch', 0)
        
        print(f"{datetime.now()} --- Loaded model: {args.pretrain}, epoch={epoch}")
    else:
        print(f"{datetime.now()} --- Loading best checkpoint...")
        checkpoint = torch.load(os.path.join(first_dic, 'checkpoint-best.pth.tar'))
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
    from utils import plot_loss_with_analysis_II
    plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array,
                               f'{first_dic}/loss_curve_for_{model_name}')
    print(f"Loss curve saved to: loss_curve_for_{model_name}.png")
    # endregion
    
    # region evaluate
    dtype = torch.float32
    pre_mode = config['train'].get('pre_mode', 'delta')
    init_t = int(data_config['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    
   
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    
    # 选择预测模式
    if pre_mode == 'delta':
        rollout_fn = autoregressive_rollout_delta
        print(f"Using DELTA prediction mode")
    else:
        rollout_fn = autoregressive_rollout_direct
        print(f"Using DIRECT prediction mode")
    
    # 测试数据字典: {名称: (loader, sub_x)}
    test_loaders = {
        f'test_{origin_nx}': (test_loader_1x, 1),
        f'test_{int((origin_nx - 1) / 2) + 1}': (test_loader_1_2x, 2),
        f'test_{int((origin_nx - 1) / 4) + 1}': (test_loader_1_4x, 4),
    }
    
    results = []
    errors_for_talk_all = {}
    visualize_results_all = []
    
    for name, (test_loader, sub_x) in test_loaders.items():
        print(f"\n{'='*70}")
        print(f"Evaluating on: {name} (sub_x={sub_x})")
        print(f"{'='*70}")
        
        errors_for_talk = []
        Nx = int((origin_nx - 1) / sub_x) + 1
        
        # 计算对应分辨率的 u_b
        u_b = compute_ub_fixed_bc(Nx, device, dtype=dtype)
        print(f"u_b computed for Nx={Nx}, shape={u_b.shape}")
        
        model.eval()
        
        with torch.no_grad():
            test_iter = iter(test_loader)
            
            # 第一遍: 计算所有样本的误差
            for b in tqdm(range(len(test_loader)), desc="Computing errors"):
                xx, yy, grid = next(test_iter)
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                
                # 分离 u_h
                xx_h = xx - u_b[..., :init_t, :]
                
                # 自回归预测
                pred_h, pred = rollout_fn(model, xx_h, grid, u_b, init_t, t_train, dtype)
                
                # 计算 L2 误差
                _yy = yy[..., init_t + 1:t_train, :]
                _pred = pred[..., init_t + 1:t_train, :]
                _batch = yy.size(0)
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                
                # 计算物理残差
                residual_pred = compute_heat2d_residual_batch(pred, kappa=0.02, T=1)
                loss_f_pred = torch.abs(residual_pred).mean().item()
                
                residual_yy = compute_heat2d_residual_batch(yy, kappa=0.02, T=1)
                loss_f_yy = torch.abs(residual_yy).mean().item()
                
                errors_for_talk.append([l2_error, loss_f_pred, loss_f_yy, int(b)])
            
            # 排序找出最好/最差/中等样本
            error_records_sorted = sorted(errors_for_talk, key=lambda x: x[0])
            n_samples = len(error_records_sorted)
            
            selected_indices = set(
                [r[-1] for r in error_records_sorted[:3]] +  # best 3
                [r[-1] for r in error_records_sorted[-3:]] +  # worst 3
                [r[-1] for r in error_records_sorted[n_samples // 2 - 1: n_samples // 2 + 2]]  # mid 3
            )
            print(f"\nSelected sample indices for visualization: {sorted(selected_indices)}")
            
            # 第二遍: 保存选中样本的详细结果
            test_iter = iter(test_loader)
            visualize_results = []
            
            for b in tqdm(range(len(test_loader)), desc="Collecting visualization data"):
                xx, yy, grid = next(test_iter)
                
                if b not in selected_indices:
                    continue
                
                print(f"Processing sample {b} for dataset {name}")
                
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                
                # 分离 u_h
                xx_h = xx - u_b[..., :init_t, :]
                
                # 自回归预测
                pred_h, pred = rollout_fn(model, xx_h, grid, u_b, init_t, t_train, dtype)
                
                # 计算残差
                residual_pred = compute_heat2d_residual_batch(pred, kappa=0.02, T=1)
                residual_yy = compute_heat2d_residual_batch(yy, kappa=0.02, T=1)
                
                # 找到该样本的 L2 误差
                l2_error = error_records_sorted[[r[-1] for r in error_records_sorted].index(b)][0]
                
                # 分类
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
                    'pred_h': pred_h.squeeze().cpu().numpy(),
                    'u_b': u_b.squeeze().cpu().numpy(),
                    'pred_residual': residual_pred.squeeze().cpu().numpy(),
                    'yy_residual': residual_yy.squeeze().cpu().numpy(),
                })
            
            visualize_results_all.append(visualize_results)
        
        # 计算统计量
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
        
        print(f"\n{name} Results:")
        print(f"  L2 Error: {mean_l2_error:.4e} ± {std_l2_error:.4e}")
        print(f"  PDE Error (pred): {mean_PDE_error:.4e} ± {std_PDE_error:.4e}")
        print(f"  PDE Error (truth): {mean_PDE_error_yy:.4e} ± {std_PDE_error_yy:.4e}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)
    print(f"\n✅ Test results saved to: test_results.csv")
    
    with open('visualize_results.pkl', 'wb') as f:
        pickle.dump(visualize_results_all, f)
    print(f"✅ Visualization data saved to: visualize_results.pkl")
    
    with open('error_for_talk_all.pkl', 'wb') as f:
        pickle.dump(errors_for_talk_all, f)
    print(f"✅ Error records saved to: error_for_talk_all.pkl")
    # endregion
    
    # region visualize
    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}")
    
   
    
    for visualize_results in visualize_results_all:
        if len(visualize_results) == 0:
            continue
        case = visualize_results[0]['Dataset']
        save_dir = f'./figures_{case}'
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        try:
            plot_heat2d_comparison(visualize_results, save_dir=save_dir)
            print(f"✅ Visualizations saved to: {save_dir}")
        except Exception as e:
            print(f"⚠️  Visualization failed for {case}: {e}")
    
    print(f"\n{'='*70}")
    print("Testing completed!")
    print(f"{'='*70}")
    # endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='ISNO with harmonic lifting')
    parser.add_argument('--config_path', type=str, default='./yaml/information.yaml')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--load_lr', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.FullLoader)
    
    if args.mode == 'train':
        run(config, args)
        # test(config, args)
    else:
        test(config, args)