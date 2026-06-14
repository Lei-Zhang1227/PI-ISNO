"""
heat equation with Chebshev transform in CGL points
增量预测
"""
import os, sys
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))
from argparse import ArgumentParser
import yaml
import tqdm
import shutil
from tqdm import tqdm
from model import *
from datasets import *
from loss import *
from utils import *
from datetime import datetime
import pandas as pd
import pickle
import math
import torch
import numpy as np
from utilsII import *
from NOs_dict.models import CosNO_II as Model
import os
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import gc
import torch

gc.collect()
torch.cuda.empty_cache()


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
    initial_step = config['train']['initial_step']
    file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5'
    train_data = FNODatasetMult(file_path=file_path,
                                initial_step=10,
                                sub_x=1,
                                sub_t=1,
                                if_test=False,
                                )
    test_data = FNODatasetMult(file_path=file_path,
                               initial_step=10,
                               sub_x=1,
                               sub_t=1,
                               if_test=True,
                               )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=5, num_workers=4, shuffle=False)
    train_size, test_size = len(train_data), len(test_data)
    ntrain, ntest = train_size, test_size
    print('size-of-train/val:', train_size, test_size)
    # endregion
    # region location
    ################################################################
    first_dic = f"/code/ex3{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    if args.pretrain is not None:
        shutil.copy(args.config_path, f"/code/ex3{config['prepare']['project']}")
    else:
        shutil.copy(args.config_path, f"/code/ex3{config['prepare']['project']}")
    # os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :/code/ex3{config['prepare']['project']} ---")
    # endregion
    # region model
    ################################################################
    # 这里有所不同的是，不再将这个一维含时的问题视为二维问题
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    model = Model(10 * 2 + 3, modes, width, bandwidth, out_channels=2, dim=2, triL=0).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Total trainable parameters: {total_params}")

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
    # ============ 定义课程学习调度器 ============
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                 lr=config['train']['base_lr'])
    now_lr = config['train']['base_lr']
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    init_t = data_config['initial_step']

    # ============ 定义课程学习调度器 ============
    use_curriculum = config['train']['curriculum']
    use_lr_schedule_in_curriculum = False  # 标记是否使用curriculum内部的lr调度

    if use_curriculum:
        curriculum = CausalCurriculumScheduler(
            max_t_train=t_train,
            min_steps=init_t + config['train']['curriculum_para'][0],
            warmup_epochs=config['train']['curriculum_para'][1],
            rollback_prob=config['train']['curriculum_para'][2],
            # 自适应门控
            adaptive_gate=config['train'].get('adaptive_gate', False),
            loss_plateau_patience=config['train'].get('loss_plateau_patience', 12),
            loss_plateau_threshold=config['train'].get('loss_plateau_threshold', 0.001),
            force_expand_patience=config['train'].get('force_expand_patience', 30),
            force_patience_early_ratio=config['train'].get('force_patience_early_ratio', 1.5),
            force_patience_late_ratio=config['train'].get('force_patience_late_ratio', 0.5),
            # 因果权重
            use_causal_weights=config['train'].get('use_causal_weights', False),
            epsilon_start=config['train'].get('epsilon_start', 1.0),
            epsilon_end=config['train'].get('epsilon_end', 0.1),
            # 学习率调度
            use_lr_schedule=config['train'].get('use_lr_schedule', False),
            lr_boost=config['train'].get('lr_boost', 5.0),
            lr_warmup_epochs=config['train'].get('lr_warmup_epochs', 5),
            lr_scheduler_patience=config['train'].get('lr_scheduler_patience', 5),
            lr_scheduler_factor=config['train'].get('gamma', 0.5),
            lr_min_ratio=config['train'].get('lr_min_ratio', 0.1),
            # 日志
            log_file=f'{first_dic}/Experiment_record.txt',
        )
        # 如果启用curriculum内部的lr调度
        print(
            f"DEBUG: config use_lr_schedule= {config['train'].get('use_lr_schedule', False)}")

        if config['train'].get('use_lr_schedule', False):
            print("DEBUG: Calling init_lr_scheduler")
            curriculum.init_lr_scheduler(optimizer)
            use_lr_schedule_in_curriculum = True
            scheduler_name = 'P in C'
            scheduler = curriculum.lr_scheduler
        else:
            print("DEBUG: NOT calling init_lr_scheduler")
            use_lr_schedule_in_curriculum = False

    if not use_lr_schedule_in_curriculum:
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
    clipper = RobustGradientClipper()
    print(
        f'{datetime.now()} --- set optimizer,lr is {now_lr}, scheduler:{scheduler_name},  curriculum: {use_curriculum}---')
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
            print(f'加载优化器调度器')
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
        f.write(f"│   ├── input_channels: {22}\n")
        f.write(f"│   ├── modes: {modes}\n")
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
        f.write(f"├── Data: {data_config['datapath']}\n")
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
    print(f'rx:{rx},rt:{rt}')
    desc = DescStr()
    time_0 = time.time()
    time_old = time.time()
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n├─ Training Start! | {timestamp}] \n")
        f.close()
    model.eval()
    val_l2_full = 0
    dtype = torch.float32
    with torch.no_grad():
        for xx, yy, grid in test_loader:
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            print(f'xx.shape{xx.shape},yy.shape：{yy.shape},grid.shape：{grid.shape}')
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
            outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]

            pred = torch.empty(yy.shape, device=xx.device)
            gridt = torch.tensor(np.linspace(0, 1, t_train), dtype=dtype, device=xx.device).reshape(
                t_train, 1)
            pred[..., :initial_step, :] = yy[..., :initial_step, :]
            for t in range(init_t, t_train):
                current_time = gridt[t:t + 1, :]
                inp = xx.reshape(inp_shape)

                current_time = current_time.view(1, 1, 1, 1).expand(xx.size(0), xx.size(1), xx.size(2),
                                                                    1)  # 扩展为[batch, nx,ny, 1]
                delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                last_step = xx[..., -1:, :]
                out = last_step + delta
                pred[..., t:t + 1, :] = out
                print(f'inp.shape{inp.shape},out.shape：{out.shape}')
                xx = torch.cat((xx[..., 1:, :], out), dim=-2)
            assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
            _batch = yy.size(0)
            _pred = pred[..., init_t:, :]
            _yy = yy[..., init_t:, :]
            val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
        test_l2 = val_l2_full / ntest
        print(f'epoch:{pre_epoch}, test_l2: {test_l2}')
    model.train()
    First_data = True
    prev_loss = 10000
    delta_history = []
    delta_history_size = 20
    delta_min_history = 10
    delta_spike_threshold = 3.0
    # spike_log_buffer = []  # 缓存日志，批量写入
    # monitor = QuickMonitor(
    #     enable_timing=False,
    #     enable_health=False,
    #     loss_spike_threshold=3.0,
    #     grad_threshold=50.0,
    # )
    residual_calculator = ReactionDiffusionResidual(
        dx=0.015625, dt=5 / 100, du=0.001, dv=0.005, k=0.005,
        dtype=torch.float32
    ).to(device)
    for e in ebar:
        epoch_loss = 0
        epoch_physics_loss = 0
        epoch_grad_norms = []

        current_t_train = curriculum.update(e - pre_epoch, current_loss=prev_loss) if use_curriculum else t_train
        gridt = torch.tensor(np.linspace(0, 1, t_train), dtype=dtype, device=device).reshape(t_train, 1)
        train_iter = iter(train_loader)

        for b in trange(len(train_loader), file=desc, desc="batch"):
            xx, yy, grid = next(train_iter)
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            # monitor.tick('data_load')
            optimizer.zero_grad()

            if not torch.isfinite(xx).all():
                print(f"[Error] NaN in input xx at batch {b}")
                continue

            # 构造输入输出形状
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)
            outp_shape = inp_shape[:-1] + [1, -1]

            # 自回归训练
            yy = yy[..., :current_t_train, :]
            pred = torch.empty(yy.shape, device=xx.device, dtype=dtype)
            pred[..., 0:init_t, :] = yy[..., 0:init_t, :]

            # 预计算 avg_delta（每个 batch 只算一次）
            avg_delta = np.mean(delta_history[-delta_min_history + 5:-5]) if len(
                delta_history) >= delta_min_history else None
            clamp_threshold = avg_delta * delta_spike_threshold if avg_delta else None

            for t in range(init_t, current_t_train):
                current_time = gridt[t:t + 1, :]
                inp = xx.reshape(inp_shape)
                current_time = current_time.view(1, 1, 1, 1).expand(xx.size(0), xx.size(1), xx.size(2), 1)

                delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)

                # # ===== 只在第一个时间步检测 =====
                # if t == init_t and clamp_threshold is not None:
                #     delta_max = delta.abs().max().item()
                #     if delta_max > clamp_threshold:
                #         spike_log_buffer.append(f"│  [Delta Spike] Epoch {e}, Batch {b}, Max={delta_max:.2e}\n")
                #         # 对整个 batch 的所有时间步都用 clamp
                #         clamp_value = avg_delta
                #
                # # 如果检测到异常，所有时间步都 clamp
                # if 'clamp_value' in locals() and clamp_value is not None:
                #     delta = torch.clamp(delta, -clamp_value, clamp_value)

                last_step = xx[..., -1:, :]
                out = last_step + delta

                pred[..., t:t + 1, :] = out
                xx = torch.cat((xx[..., 1:, :], out), dim=-2)
            # monitor.tick('forward')
            # monitor.check_delta(delta, e, b)
            # 重置 clamp_value
            clamp_value = None

            # 每个 batch 只记录一次 delta（用第一个时间步）
            # with torch.no_grad():
            #     first_delta = pred[..., init_t, :] - yy[..., init_t - 1, :]
            #     delta_history.append(first_delta.abs().max().item())
            # if len(delta_history) > delta_history_size:
            #     delta_history.pop(0)

            # 下采样评估
            assert pred.shape == yy.shape
            out_data = pred[:, ::rx, ::rx, ::rt, :].reshape(_batch, -1)
            y_data = yy[:, ::rx, ::rx, ::rt, :].reshape(_batch, -1)
            loss_data = myloss(out_data, y_data)

            if First_data:
                print('out_data.shape, y_data.shape:', pred[:, ::rx, ::rx, ::rt, :].shape,
                      yy[:, ::rx, ::rx, ::rt, :].shape)
                First_data = False

            # ===== 统一计算 loss =====
            loss_f = torch.tensor(0.0, device=device)
            warmup_epochs = 10

            if e < warmup_epochs:
                last_input = yy[..., init_t - 1:init_t, :]
                target = last_input.expand(-1, -1, -1, current_t_train - init_t, -1)
                pred_part = pred[..., init_t:current_t_train, :]
                loss_pretrain = ((pred_part - target) ** 2).mean()
                total_loss = loss_pretrain
                epoch_physics_loss += 0.0

                if b == 0:
                    print(f"[Warmup {e}/{warmup_epochs}] Pretrain loss: {loss_pretrain.item():.4e}")
            else:
                if config['train']['loss_mode'] != 'data':
                    f_u, f_v = residual_calculator(pred)
                    f_u, f_v = f_u[..., 3:], f_v[..., 3:]
                    loss_u = (f_u ** 2).mean()
                    loss_v = (f_v ** 2).mean()
                    loss_f = loss_u + loss_v
                    epoch_physics_loss += loss_f.item()

                if config['train']['loss_mode'] == 'both':
                    total_loss = loss_f * f_weight + loss_data * data_weight
                elif config['train']['loss_mode'] == 'data':
                    total_loss = loss_data
                else:
                    total_loss = loss_f
            # monitor.tick('loss')
            # monitor.check_loss(total_loss.item(), e, b)
            # 检查 NaN/Inf
            if not torch.isfinite(total_loss).all():
                print(f"[Warning] Invalid loss at epoch {e}, batch {b}, skipping...")
                break

            # 反向传播
            total_loss.backward()
            if e > warmup_epochs:
                grad_norm, skipped = clipper.step(model, epoch=e, batch=b)
            else:
                skipped = False
                grad_norm = 0.0
            if skipped:
                optimizer.zero_grad()  # 确保清干净
                continue  # 跳过参数更新
            optimizer.step()
            # monitor.tick('backward')
            # monitor.check_grad(grad_norm, e, b)
            # monitor.batch_done()
            # Scheduler（batch级别）
            if not use_lr_schedule_in_curriculum and scheduler_name == 'cosine_schedule_with_warmup':
                scheduler.step()

            # 记录
            current_lr = optimizer.param_groups[0]['lr']
            loss_list.append([loss_data.item(), loss_f.item(), total_loss.item(), e])
            epoch_grad_norms.append(grad_norm)
            epoch_loss += total_loss.item()

            # 更新进度条
            state = curriculum.get_state() if use_curriculum else {'t_train': t_train}
            new_desc = (
                f"Epoch {e + 1}: {desc.read(b)},"
                f"t_train: {state['t_train']}, ε={state.get('epsilon', 0):.3f}, loss={prev_loss:.2e}, "
                f"Loss_total: {total_loss.item():.4e},Test L2: {test_l2:.4e},  "
                f"Loss_data: {loss_data.item():.4e}, Loss_phy: {loss_f.item():.4e}, "
                f"lr: {current_lr:.2e}, Grad Norm: {grad_norm:.2f}")
            ebar.set_description(new_desc)
            # Epoch结束
        # monitor.epoch_done(e)
        clipper.print_stats()
        # if spike_log_buffer:
        #     with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        #         f.writelines(spike_log_buffer)
        #     spike_log_buffer.clear()
        prev_loss = epoch_physics_loss / len(train_loader)
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_grad_norm = np.mean(epoch_grad_norms)
        grad.append([e, avg_epoch_loss, avg_grad_norm])
        lr_list.append([current_lr, current_t_train, e])
        # ============ Epoch 结束后更新 lr ============
        if use_lr_schedule_in_curriculum:
            curriculum.step_lr_scheduler(avg_epoch_loss)
        elif scheduler is not None:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(avg_epoch_loss)
            elif scheduler_name == 'cosine_schedule_with_warmup':
                pass  # cosine scheduler 在每个 batch 更新
            else:
                scheduler.step()
        # signs = monitor.get_early_warning_signs()
        # if signs:
        #     print(f"⚠️ 早期预警: {signs}")
        # 保存最佳模型
        if best_error > avg_epoch_loss:
            best_error = avg_epoch_loss
            model_save_record.append([e, avg_epoch_loss])
            save_checkpoint(model, e, optimizer, scheduler, loss_list,
                            test_loss_list, lr_list, model_save_record, grad,
                            filename=f'{first_dic}/checkpoint-best')
        # 保存当前模型
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
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
                    outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]

                    pred = torch.empty(yy.shape, device=xx.device)
                    gridt = torch.tensor(np.linspace(0, 1, t_train), dtype=dtype, device=xx.device).reshape(
                        t_train, 1)
                    pred[..., :initial_step, :] = yy[..., :initial_step, :]
                    for t in range(init_t, t_train):
                        current_time = gridt[t:t + 1, :]
                        inp = xx.reshape(inp_shape)
                        current_time = current_time.view(1, 1, 1, 1).expand(xx.size(0), xx.size(1), xx.size(2),
                                                                            1)  # 扩展为[batch, nx, 1]
                        delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                        last_step = xx[..., -1:, :]
                        out = last_step + delta
                        pred[..., t:t + 1, :] = out
                        xx = torch.cat((xx[..., 1:, :], out), dim=-2)
                    assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
                    _batch = yy.size(0)
                    _pred = pred[..., init_t:, :]
                    _yy = yy[..., init_t:, :]
                    val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                test_l2 = val_l2_full / ntest
                print(f'epoch:{pre_epoch}, test_l2: {test_l2}')
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
    file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_NA_NA.h5'
    test_data_1x = FNODatasetMult(file_path=file_path,
                                  initial_step=10,
                                  sub_x=1,
                                  sub_t=1,
                                  if_test=True,
                                  )
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)

    test_data_1_2x = FNODatasetMult(file_path=file_path,
                                    initial_step=10,
                                    sub_x=2,
                                    sub_t=1,
                                    if_test=True,
                                    )
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_4x = FNODatasetMult(file_path=file_path,
                                    initial_step=10,
                                    sub_x=4,
                                    sub_t=1,
                                    if_test=True,
                                    )
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_8x = FNODatasetMult(file_path=file_path,
                                    initial_step=10,
                                    sub_x=8,
                                    sub_t=1,
                                    if_test=True,
                                    )
    test_loader_1_8x = torch.utils.data.DataLoader(test_data_1_8x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_16x = FNODatasetMult(file_path=file_path,
                                     initial_step=10,
                                     sub_x=16,
                                     sub_t=1,
                                     if_test=True,
                                     )
    test_loader_1_16x = torch.utils.data.DataLoader(test_data_1_16x, batch_size=1, shuffle=False,
                                                    num_workers=0, pin_memory=True)

    test_size = len(test_data_1x)
    ntest = test_size
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    first_dic = f"/code/ex3{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    dtype = torch.float32
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    model = Model(10 * 2 + 3, modes, width, bandwidth, out_channels=2, dim=2, triL=0).to(device)
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
    plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array,
                               f'{first_dic}/loss_carve_for_{model_name}')
    # region evaluate
    residual_calculator = ReactionDiffusionResidual(
        dx=0.015625, dt=5 / 100, du=0.001, dv=0.005, k=0.005,
        dtype=torch.float32
    ).to(device)
    init_t = int(data_config['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    test_loaders = {
        f'test_{origin_nx}': test_loader_1x,
#         f'test_{int((origin_nx - 1) / 2) + 1}': test_loader_1_2x,
#         f'test_{int((origin_nx - 1) / 4) + 1}': test_loader_1_4x,
#         f'test_{int((origin_nx - 1) / 8) + 1}': test_loader_1_8x,
        #         f'test_{int((origin_nx - 1) / 16) + 1}': test_loader_1_16x,
    }
    results = []
    errors_for_talk_all = {}
    visualize_results_all = []
    i = 0
    for name, test_loader in test_loaders.items():
        errors_for_talk = []
        model.eval()  # 将模型设置为评估模式
        first = True
        with torch.no_grad():
            test_iter = iter(test_loader)
            for b in tqdm(range(len(test_loader))):
                xx, yy, grid = next(test_iter)
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
                outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]

                pred = torch.empty(yy.shape, device=xx.device)
                gridt = torch.tensor(np.linspace(0, 1, t_train), dtype=dtype, device=xx.device).reshape(
                    t_train, 1)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]
                for t in range(init_t, t_train):
                    current_time = gridt[t:t + 1, :]
                    inp = xx.reshape(inp_shape)
                    current_time = current_time.view(1, 1, 1, 1).expand(xx.size(0), xx.size(1), xx.size(2),
                                                                        1)  # 扩展为[batch, nx, 1]
                    delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                    last_step = xx[..., -1:, :]
                    out = last_step + delta
                    pred[..., t:t + 1, :] = out
                    xx = torch.cat((xx[..., 1:, :], out), dim=-2)
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"  # pred.shape： [1,nx,101,1]

                _yy = yy[..., init_t + 1:t_train, :]  # if t_train is not -1
                _pred = pred[..., init_t + 1:t_train, :]
                _batch = yy.size(0)
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                f_u_p, f_v_p = residual_calculator(pred)
                f_u_p, f_v_p = f_u_p[..., 8:], f_v_p[..., 8:]
                loss_f_p = (torch.abs(f_u_p).mean() + torch.abs(f_v_p).mean()) / 2

                f_u_y, f_v_y = residual_calculator(yy)
                f_u_y, f_v_y = f_u_y[..., 8:], f_v_y[..., 8:]
                loss_f_y = (torch.abs(f_u_y).mean() + torch.abs(f_v_y).mean()) / 2

                mean_residual_pred = loss_f_p.item()
                mean_residual_yy = loss_f_y.item()
                errors_for_talk.append([l2_error, mean_residual_pred, mean_residual_yy, int(b)])
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
                xx, yy, grid = next(test_iter)
                if b not in selected_indices:
                    continue
                print(f'b is {b},dataset is {name}')
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)

                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
                outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]

                pred = torch.empty(yy.shape, device=xx.device)
                gridt = torch.tensor(np.linspace(0, 1, t_train), dtype=dtype, device=xx.device).reshape(
                    t_train, 1)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]
                for t in range(init_t, t_train):
                    current_time = gridt[t:t + 1, :]
                    inp = xx.reshape(inp_shape)
                    current_time = current_time.view(1, 1, 1, 1).expand(xx.size(0), xx.size(1), xx.size(2),
                                                                        1)  # 扩展为[batch, nx, 1]
                    delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                    last_step = xx[..., -1:, :]
                    out = last_step + delta
                    pred[..., t:t + 1, :] = out
                    xx = torch.cat((xx[..., 1:, :], out), dim=-2)
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"  # pred.shape： [1,nx,101,1]

                _yy = yy[..., init_t + 1:t_train, :]  # if t_train is not -1
                _pred = pred[..., init_t + 1:t_train, :]
                _batch = yy.size(0)
                f_u_p, f_v_p = residual_calculator(pred)

                f_u_y, f_v_y = residual_calculator(yy)

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
                    'pred_du': [f_u_p.squeeze().cpu().numpy(), f_v_p.squeeze().cpu().numpy()],
                    'yy_du': [f_u_y.squeeze().cpu().numpy(), f_v_y.squeeze().cpu().numpy()]
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
        plot_visualization_results_2d(visualize_results, save_dir=f'./figures_{case}')
    print('可视化完成~')
    # endregion


def test_extend(config, args):
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
    file_path = '/data/zhanglei/BurgersEquationII/2D_diff-react_full_0_15.h5'
    test_data_unseen = FNODatasetMult_B(file_path=file_path,
                                        initial_step=10,
                                        sub_x=1,
                                        sub_t=1,
                                        mode='unseen',
                                        )
    test_loader_unseen = torch.utils.data.DataLoader(test_data_unseen, batch_size=1, shuffle=False,
                                                     num_workers=0, pin_memory=True)

    test_data_seen = FNODatasetMult_B(file_path=file_path,
                                      initial_step=10,
                                      sub_x=1,
                                      sub_t=1,
                                      mode='seen',
                                      )
    test_loader_seen = torch.utils.data.DataLoader(test_data_seen, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_size = len(test_data_unseen)
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    first_dic = f"/code/ex3{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    dtype = torch.float32
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    model = Model(10 * 2 + 3, modes, width, bandwidth, out_channels=2, dim=2, triL=0).to(device)
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
    init_t = int(data_config['initial_step'])
    t_train = 151
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    test_loaders = {
        f'test_unseen': test_loader_unseen,
        f'test_seen': test_loader_seen,
    }
    errors_for_talk_all = {}
    for name, test_loader in test_loaders.items():
        model.eval()
        errors_for_talk = []
        errors_segments = []
        with torch.no_grad():
            for b, (xx, yy, grid) in enumerate(tqdm(test_loader, desc=name)):
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)

                inp_shape = list(xx.shape)[:-2] + [-1]
                outp_shape = list(xx.shape)[:-2] + [1, -1]

                # 预分配
                pred = torch.empty(yy.shape, device=xx.device)
                # print(f"yy.shape: {yy.shape}, pred.shape: {pred.shape}")

                xx_input = xx.clone()
                gridt = torch.linspace(0, 3, 301, dtype=dtype, device=device)

                # 自回归预测
                for t in range(init_t, 101):
                    current_time = gridt[t:t + 1]
                    # print(current_time)
                    inp = xx_input.reshape(inp_shape)
                    current_time = current_time.view(1, 1, 1, 1).expand(xx.size(0), xx.size(1), xx.size(2),
                                                                        1)  # 扩展为[batch, nx, 1]
                    delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                    last_step = xx[..., -1:, :]
                    out = last_step + delta
                    pred[..., t:t + 1, :] = out
                    xx = torch.cat((xx[..., 1:, :], out), dim=-2)

                # print(pred.shape)

                # segments = {
                #     '0-5': (init_t + 1, 101),
                #     '5-7.5': (101, 151),
                #     # '7.5-10': (151, 201),
                #     # '10-12.5': (201, 251),
                #     # '12.5-15': (251, 301),
                # }
                #
                # sample_errors = []
                # for seg_name, (s, e) in segments.items():
                _yy = yy[..., 11:101, :]  # if t_train is not -1
                _pred = pred[..., 11:101, :]
                _batch = yy.size(0)
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                # sample_errors.append(l2_error)
                errors_for_talk.append(l2_error)
        errors_for_talk  = np.array(errors_for_talk)
        print('errors_for_talk.shape:', errors_for_talk.shape)
        errors_for_talk = np.array(errors_for_talk)
        mean_l2_error = np.mean(errors_for_talk)
        std_l2_error = np.std(errors_for_talk)
        max_l2_error = np.max(errors_for_talk)
        min_l2_error = np.min(errors_for_talk)

        # e_mean_l2_error = np.mean(errors_for_talk])
        # e_std_l2_error = np.std(errors_for_talk[:, 1])
        # e_max_l2_error = np.max(errors_for_talk[:, 1])
        # e_min_l2_error = np.min(errors_for_talk[:, 1])

        print(f'0-5:{mean_l2_error}')
    # endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='./information.yaml',
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
    if args.mode == 'test':
        test(config, args)
    if args.mode == 'test_extend':
        test_extend(config, args)
