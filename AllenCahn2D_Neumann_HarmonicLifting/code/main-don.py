"""
TI(L)-DeepONet for 2D Allen-Cahn Equation
==========================================
对标原版 2D TI-DeepONet (Nayak & Goswami, arXiv:2505.17341)

改动要点（相对原版 main-don.py）:
  1. Branch: CNN + Pooling + Flatten + MLP（对标原版，替代 GAP）
  2. RK4Net: CNN + Pooling + Flatten + MLP（对标原版，替代 GAP）
  3. Trunk:  最后一层加 Tanh（对标原版）
  4. 双 optimizer 交替更新 DeepONet 和 RK4（对标原版）
  5. 保留滑窗输入 u_window 和边界条件 u_b
  6. 所有模型超参数从 yaml 读取
  7. 支持 data/pde/both 三种 loss mode + warmup 预训练

Branch 设计:
  输入通道拼接: [u_window(init_step) | u_interp(1) | u_b(1)]
      ↓ CNN layers with Pooling
      ↓ Flatten
      ↓ MLP layers
  输出: [B, latent_dim]

Trunk 设计:
  (x, y) → MLP (all layers with Tanh) → [N_pts, latent_dim]

RK4Net 设计:
  u_curr [B, Nx, Ny]
      ↓ CNN layers with Pooling
      ↓ Flatten
      ↓ MLP layers
  输出: [B, 4] (softmax)
"""

import os
import re
import yaml
import shutil
import pickle
import time
import math
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 复用主工程模块 ──────────────────────────────────────────────
from dataloader import FNODatasetMult
from loss import build_lifting_batch, compute_ac2d_residual_batch,compute_neumann_bc_loss
from utils import save_checkpoint, plot_loss_with_analysis_II, \
    plot_burgers2d_comparison, RobustAdaptiveGradientClipperV2, \
    DescStr, LpLoss


# ================================================================
# 辅助: 计算 CNN + Pooling 之后的空间尺寸
# ================================================================

def _calc_spatial_size_after_cnn(nx, ny, cnn_cfg):
    """
    给定初始空间尺寸 (nx, ny) 和 CNN 配置列表,
    逐层计算经过 Conv + Pool 之后的空间尺寸。
    返回 (h_out, w_out)
    """
    h, w = nx, ny
    for layer_cfg in cnn_cfg:
        pool_type = layer_cfg.get('pool', 'none')
        if pool_type != 'none':
            pool_size = layer_cfg.get('pool_size', 2)
            h = math.ceil(h / pool_size)
            w = math.ceil(w / pool_size)
    return h, w


# ================================================================
# 1. 模型定义
# ================================================================

class BranchNet(nn.Module):
    """
    Branch Net: CNN + Pooling + Flatten + MLP (对标原版 2D)

    输入通道: u_window(init_step) + u_interp(1) + u_b(1)
    输出: [B, latent_dim]
    """

    def __init__(self, in_channels: int, latent_dim: int,
                 cnn_cfg: list, mlp_hidden: list,
                 nx: int, ny: int,
                 mlp_activation: str = 'gelu'):
        super().__init__()

        # ── 构建 CNN 层 ──
        cnn_layers = nn.ModuleList()
        ch_in = in_channels
        for layer_cfg in cnn_cfg:
            ch_out = layer_cfg['out_channels']
            ks = layer_cfg.get('kernel_size', 3)
            act_name = layer_cfg.get('activation', 'relu')
            pool_type = layer_cfg.get('pool', 'none')
            pool_size = layer_cfg.get('pool_size', 2)

            block = nn.ModuleList()
            block.append(nn.Conv2d(ch_in, ch_out, kernel_size=ks, padding='same'))
            if act_name == 'relu':
                block.append(nn.ReLU())
            elif act_name == 'gelu':
                block.append(nn.GELU())
            elif act_name == 'tanh':
                block.append(nn.Tanh())
            if pool_type == 'max':
                block.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size,
                                          ceil_mode=True))
            elif pool_type == 'avg':
                block.append(nn.AvgPool2d(kernel_size=pool_size, stride=pool_size,
                                          ceil_mode=True))
            cnn_layers.append(block)
            ch_in = ch_out

        self.cnn_layers = cnn_layers

        # ── 计算 Flatten 维度 ──
        h_out, w_out = _calc_spatial_size_after_cnn(nx, ny, cnn_cfg)
        flatten_dim = ch_in * h_out * w_out

        # ── 构建 MLP 层 ──
        _act_map = {'gelu': nn.GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}
        act_cls = _act_map.get(mlp_activation, nn.GELU)
        mlp_layers = []
        in_d = flatten_dim
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(in_d, h), act_cls()]
            in_d = h
        mlp_layers.append(nn.Linear(in_d, latent_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        for block in self.cnn_layers:
            for layer in block:
                x = layer(x)
        x = x.flatten(start_dim=1)
        return self.mlp(x)


class TrunkMLP(nn.Module):
    """
    Trunk Net: (x, y) → MLP → latent_dim
    对标原版: 所有层都加 Tanh (包括最后一层)
    """

    def __init__(self, hidden_dims: list, latent_dim: int):
        super().__init__()
        layers = []
        in_d = 2
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        layers += [nn.Linear(in_d, latent_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


class LearnableRK4Net(nn.Module):
    """
    根据当前 u 场生成自适应 RK4 系数 α1~α4 (softmax)。
    对标原版 2D: CNN + Pooling + Flatten + MLP
    """

    def __init__(self, cnn_cfg: list, mlp_hidden: list,
                 nx: int, ny: int):
        super().__init__()

        cnn_layers = nn.ModuleList()
        ch_in = 1
        for layer_cfg in cnn_cfg:
            ch_out = layer_cfg['out_channels']
            ks = layer_cfg.get('kernel_size', 3)
            act_name = layer_cfg.get('activation', 'relu')
            pool_type = layer_cfg.get('pool', 'none')
            pool_size = layer_cfg.get('pool_size', 2)

            block = nn.ModuleList()
            block.append(nn.Conv2d(ch_in, ch_out, kernel_size=ks, padding='same'))
            if act_name == 'relu':
                block.append(nn.ReLU())
            elif act_name == 'gelu':
                block.append(nn.GELU())
            elif act_name == 'tanh':
                block.append(nn.Tanh())
            if pool_type == 'max':
                block.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size,
                                          ceil_mode=True))
            elif pool_type == 'avg':
                block.append(nn.AvgPool2d(kernel_size=pool_size, stride=pool_size,
                                          ceil_mode=True))
            cnn_layers.append(block)
            ch_in = ch_out

        self.cnn_layers = cnn_layers

        h_out, w_out = _calc_spatial_size_after_cnn(nx, ny, cnn_cfg)
        flatten_dim = ch_in * h_out * w_out

        mlp_layers = []
        in_d = flatten_dim
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        mlp_layers.append(nn.Linear(in_d, 4))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, u_curr):
        x = u_curr.unsqueeze(1)
        for block in self.cnn_layers:
            for layer in block:
                x = layer(x)
        x = x.flatten(start_dim=1)
        return F.softmax(self.mlp(x), dim=-1)


class TIDeepONet2D(nn.Module):
    """
    TI(L)-DeepONet for 2D Allen-Cahn.
    本类只包含 branch + trunk + bias (DeepONet 部分)。
    rk4 作为独立网络，在训练时用独立 optimizer 更新。
    """

    def __init__(self, init_step: int, latent_dim: int,
                 branch_cnn_cfg: list, branch_mlp_hidden: list,
                 trunk_hidden: list,
                 nx: int, ny: int,
                 branch_mlp_activation: str = 'gelu'):
        super().__init__()
        in_channels = init_step + 2
        self.branch = BranchNet(in_channels, latent_dim,
                                branch_cnn_cfg, branch_mlp_hidden,
                                nx, ny,
                                mlp_activation=branch_mlp_activation)
        self.trunk = TrunkMLP(trunk_hidden, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def derivative_field(self, u_interp, u_window, u_b, trunk_out):
        B, Nx, Ny = u_interp.shape
        branch_in = torch.cat([
            u_window,
            u_interp.unsqueeze(1),
            u_b.unsqueeze(1),
        ], dim=1)
        b_out = self.branch(branch_in)
        k_flat = torch.mm(b_out, trunk_out.T) + self.bias
        return k_flat.reshape(B, Nx, Ny)


def rk4_step(deeponet, rk4_net, u_curr, u_window, u_b, grid_flat, dt):
    trunk_out = deeponet.trunk(grid_flat)
    alpha = rk4_net(u_curr)
    k1 = deeponet.derivative_field(u_curr, u_window, u_b, trunk_out)
    k2 = deeponet.derivative_field(u_curr + 0.5 * dt * k1, u_window, u_b, trunk_out)
    k3 = deeponet.derivative_field(u_curr + 0.5 * dt * k2, u_window, u_b, trunk_out)
    k4 = deeponet.derivative_field(u_curr + dt * k3, u_window, u_b, trunk_out)
    a = alpha.reshape(-1, 4, 1, 1)
    slopes = torch.stack([k1, k2, k3, k4], dim=1)
    u_next = u_curr + dt * (a * slopes).sum(dim=1)
    return u_next, alpha


# ================================================================
# 2. 自回归预测
# ================================================================

def ti_autoregressive_predict(deeponet, rk4_net, xx, grid, bc_params,
                              init_t, t_end, device, dt,
                              dtype=torch.float32):
    B, Nx, Ny = xx.shape[0], xx.shape[1], xx.shape[2]
    grid_flat = grid[0].reshape(-1, 2).to(device)
    x_coord = grid[0, :, 0, 0]
    y_coord = grid[0, 0, :, 1]
    u_b_raw = build_lifting_batch(bc_params, x_coord, y_coord)
    u_b = u_b_raw.squeeze(1).to(device, dtype=dtype)

    pred = torch.zeros(B, Nx, Ny, t_end, 1, device=device, dtype=dtype)
    for i in range(init_t):
        pred[..., i, :] = xx[..., i, :]

    u_window = xx[..., 0].permute(0, 3, 1, 2).contiguous()
    u_curr = xx[..., -1, 0]

    for t in range(init_t, t_end):
        u_next, _ = rk4_step(deeponet, rk4_net, u_curr, u_window, u_b, grid_flat, dt)
        pred[..., t, 0] = u_next
        u_window = torch.cat(
            [u_window[:, 1:, :, :],
             u_next.unsqueeze(1)], dim=1)
        u_curr = u_next

    return pred


# ================================================================
# 3. 训练
# ================================================================

def run(config, args):
    # ── prepare ──────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # ── dataloader ───────────────────────────────────────────────
    data_config = config['data']
    batch_size = config['train']['batchsize']
    initial_step = config['train']['initial_step']
    filepath = data_config['datapath']
    sub_x = data_config['sub_x']
    sub_t = data_config['sub_t']
    epsilon = config['data'].get('epsilon', 0.05)
    T_total = config['data'].get('T', 1.0)

    train_data = FNODatasetMult(filepath, initial_step=initial_step, sub_x=sub_x, sub_t=sub_t)
    test_data = FNODatasetMult(filepath, initial_step=initial_step, sub_x=sub_x, sub_t=sub_t, if_test=True)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=3, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=3, shuffle=False)

    ntrain, ntest = len(train_data), len(test_data)
    t_train = (data_config['nt'] - 1) // sub_t + 1
    init_t = initial_step
    dt = T_total / (t_train - 1)

    print(f"[TI-DeepONet] epsilon={epsilon}, T={T_total}, "
          f"t_train={t_train}, dt={dt:.6f}")
    print(f"Train={ntrain}, Test={ntest}, batch={batch_size}")

    # ── save dir ─────────────────────────────────────────────────
    first_dic = f"/code/AC2D{config['prepare']['project']}"
    os.makedirs(first_dic, exist_ok=True)
    shutil.copy(args.config_path, first_dic)

    # ── 推断空间尺寸 ──────────────────────────────────────────────
    _xx, _, _, _ = next(iter(train_loader))
    nx, ny = _xx.shape[1], _xx.shape[2]
    in_channels = initial_step + 2
    print(f"Spatial: Nx={nx}, Ny={ny}")
    print(f"Branch CNN in_channels = {initial_step}(hist) + 1(interp) + 1(u_b) = {in_channels}")

    # ── model config ─────────────────────────────────────────────
    ti_cfg = config.get('ti_deeponet', {})
    latent_dim = ti_cfg.get('latent_dim', 100)

    branch_cfg = ti_cfg.get('branch', {})
    branch_cnn_cfg = branch_cfg.get('cnn', [
        {'out_channels': 64, 'kernel_size': 3, 'activation': 'relu', 'pool': 'max', 'pool_size': 2},
        {'out_channels': 64, 'kernel_size': 3, 'activation': 'relu', 'pool': 'max', 'pool_size': 2},
        {'out_channels': 64, 'kernel_size': 2, 'activation': 'relu', 'pool': 'avg', 'pool_size': 2},
    ])
    branch_mlp_hidden = branch_cfg.get('mlp_hidden', [256, 128])
    branch_mlp_activation = branch_cfg.get('mlp_activation', 'gelu')

    trunk_hidden = ti_cfg.get('trunk_hidden', [128, 128, 128, 128, 128, 128])

    rk4_cfg = ti_cfg.get('rk4', {})
    rk4_cnn_cfg = rk4_cfg.get('cnn', [
        {'out_channels': 32, 'kernel_size': 3, 'activation': 'relu', 'pool': 'max', 'pool_size': 2},
        {'out_channels': 32, 'kernel_size': 3, 'activation': 'relu', 'pool': 'max', 'pool_size': 2},
        {'out_channels': 32, 'kernel_size': 2, 'activation': 'relu', 'pool': 'avg', 'pool_size': 2},
    ])
    rk4_mlp_hidden = rk4_cfg.get('mlp_hidden', [32, 32])

    # ── 构建模型 ─────────────────────────────────────────────────
    deeponet = TIDeepONet2D(
        init_step=initial_step,
        latent_dim=latent_dim,
        branch_cnn_cfg=branch_cnn_cfg,
        branch_mlp_hidden=branch_mlp_hidden,
        trunk_hidden=trunk_hidden,
        nx=nx, ny=ny,
        branch_mlp_activation=branch_mlp_activation,
    ).to(device)

    rk4_net = LearnableRK4Net(
        cnn_cfg=rk4_cnn_cfg,
        mlp_hidden=rk4_mlp_hidden,
        nx=nx, ny=ny,
    ).to(device)

    don_params = sum(p.numel() for p in deeponet.parameters() if p.requires_grad)
    rk4_params = sum(p.numel() for p in rk4_net.parameters() if p.requires_grad)
    total_params = don_params + rk4_params
    print(f"DeepONet params: {don_params:,}")
    print(f"RK4Net params:   {rk4_params:,}")
    print(f"Total params:    {total_params:,}")

    # ── 计算并打印 Flatten 维度 ──────────────────────────────────
    branch_h, branch_w = _calc_spatial_size_after_cnn(nx, ny, branch_cnn_cfg)
    branch_flatten = branch_cnn_cfg[-1]['out_channels'] * branch_h * branch_w
    rk4_h, rk4_w = _calc_spatial_size_after_cnn(nx, ny, rk4_cnn_cfg)
    rk4_flatten = rk4_cnn_cfg[-1]['out_channels'] * rk4_h * rk4_w
    print(f"Branch: CNN out {branch_h}×{branch_w} → Flatten {branch_flatten}")
    print(f"RK4:    CNN out {rk4_h}×{rk4_w} → Flatten {rk4_flatten}")

    # ── 双 optimizer (对标原版) ──────────────────────────────────
    train_cfg = config['train']
    don_lr = ti_cfg.get('deeponet_lr', train_cfg['base_lr'])
    rk4_lr = ti_cfg.get('rk4_lr', don_lr * 2)

    optimizer_don = torch.optim.Adam(deeponet.parameters(), betas=(0.9, 0.999), lr=don_lr)
    optimizer_rk4 = torch.optim.Adam(rk4_net.parameters(), betas=(0.9, 0.999), lr=rk4_lr)

    sched_name = train_cfg['scheduler']

    def _make_scheduler(opt, name):
        if name == 'MultiStepLR':
            return torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=train_cfg['milestones'], gamma=train_cfg['gamma'])
        elif name == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=train_cfg['gamma'],
                threshold=1e-2, patience=train_cfg['patience'])
        else:
            return torch.optim.lr_scheduler.StepLR(
                opt, step_size=train_cfg['patience'], gamma=train_cfg['gamma'])

    scheduler_don = _make_scheduler(optimizer_don, sched_name)
    scheduler_rk4 = _make_scheduler(optimizer_rk4, sched_name)

    clipper = RobustAdaptiveGradientClipperV2(
        initial_max_norm=500.0, window_size=60, trim_k=5, multiplier=5)

    # ── load pretrain ────────────────────────────────────────────
    if args.pretrain is not None:
        ckpt = torch.load(args.pretrain, map_location=device)
        deeponet.load_state_dict(ckpt['deeponet'])
        rk4_net.load_state_dict(ckpt['rk4_net'])
        optimizer_don.load_state_dict(ckpt['optimizer_don'])
        optimizer_rk4.load_state_dict(ckpt['optimizer_rk4'])
        scheduler_don.load_state_dict(ckpt['scheduler_don'])
        scheduler_rk4.load_state_dict(ckpt['scheduler_rk4'])
        loss_list = ckpt['loss_list']
        test_loss_list = ckpt.get('test_loss_list', [])
        lr_list = ckpt.get('lr_list', [])
        grad = ckpt['grad']
        start_epoch = ckpt['epoch'] + 1
        best_error = min(x[0] for x in loss_list) if loss_list else 100.0
        print(f"Resume from epoch {start_epoch}")
    else:
        loss_list = []
        test_loss_list = []
        lr_list = []
        grad = []
        start_epoch = 0
        best_error = 100.0

    # ── experiment record ────────────────────────────────────────
    with open(f'{first_dic}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'=' * 60}\n[TI(L)-DeepONet | {ts}]\n{'=' * 60}\n")
        f.write(f"├─ Allen-Cahn: epsilon={epsilon}, T={T_total}, dt={dt:.6f}\n")
        f.write(f"├─ Grid: Nx={nx}, Ny={ny}, init_step={initial_step}\n")
        f.write(f"├─ Branch CNN: {branch_cnn_cfg}\n")
        f.write(f"├─ Branch MLP: {branch_mlp_hidden} → latent={latent_dim}\n")
        f.write(f"├─ Branch Flatten: {branch_h}×{branch_w} = {branch_flatten}\n")
        f.write(f"├─ Trunk MLP: {trunk_hidden} → latent={latent_dim}\n")
        f.write(f"├─ RK4 CNN: {rk4_cnn_cfg}\n")
        f.write(f"├─ RK4 MLP: {rk4_mlp_hidden} → 4\n")
        f.write(f"├─ Params: DeepONet={don_params:,} + RK4={rk4_params:,} = {total_params:,}\n")
        f.write(f"├─ DeepONet lr={don_lr}, RK4 lr={rk4_lr}\n")
        f.write(f"└─ scheduler={sched_name}\n{'=' * 60}\n")

    # ── save checkpoint helper ───────────────────────────────────
    def _save_ckpt(filename, epoch):
        torch.save({
            'deeponet': deeponet.state_dict(),
            'rk4_net': rk4_net.state_dict(),
            'optimizer_don': optimizer_don.state_dict(),
            'optimizer_rk4': optimizer_rk4.state_dict(),
            'scheduler_don': scheduler_don.state_dict(),
            'scheduler_rk4': scheduler_rk4.state_dict(),
            'epoch': epoch,
            'loss_list': loss_list,
            'test_loss_list': test_loss_list,
            'lr_list': lr_list,
            'model_save_record': model_save_record,
            'grad': grad,
        }, f'{filename}.pth.tar')

    # ── eval helper ──────────────────────────────────────────────
    myloss = LpLoss(size_average=True)
    dtype = torch.float32

    def eval_model():
        deeponet.eval()
        rk4_net.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for xx, yy, grid, bc_params in test_loader:
                xx = xx.to(device, dtype=dtype)
                yy = yy.to(device, dtype=dtype)
                grid = grid.to(device, dtype=dtype)
                bc_params = bc_params.to(device, dtype=dtype)
                pred = ti_autoregressive_predict(
                    deeponet, rk4_net, xx, grid, bc_params,
                    init_t, t_train, device, dt, dtype)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]
                _b = yy.size(0)
                val_l2 += myloss(
                    pred[..., init_t:, :].reshape(_b, -1),
                    yy[..., init_t:, :].reshape(_b, -1)).item()
        deeponet.train()
        rk4_net.train()
        return val_l2 * batch_size / ntest

    test_l2 = eval_model()
    print(f"Initial test L2: {test_l2:.4e}")

    # ── loss mode & physics 参数 ─────────────────────────────────
    loss_mode = train_cfg.get('loss_mode', 'data')
    residual_mode = train_cfg.get('residual_mode', 'mae')
    f_weight = train_cfg.get('f_loss', 1.0)
    data_weight = train_cfg.get('ic_loss', 1.0)
    bc_weight = train_cfg.get('bc_loss', 1.0)
    warmup_epochs = train_cfg.get('warmup_epochs', 0)
    warmup_lr_list = train_cfg.get('warmup_lr', None)

    print(f"Loss mode: {loss_mode}, residual_mode: {residual_mode}")
    print(f"f_weight: {f_weight}, data_weight: {data_weight}")
    if warmup_epochs:
        print(f"Warmup: {warmup_epochs} epochs")

    # ── training loop ────────────────────────────────────────────
    model_save_record = [[0, 100.0]]
    desc = DescStr()
    time_0 = time_old = time.time()
    prev_loss = 10000
    ebar = trange(start_epoch, start_epoch + train_cfg['epochs'],
                  desc="Epoch")

    for e in ebar:
        epoch_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_grad_norms = []

        for xx, yy, grid, bc_params in tqdm(
                train_loader, file=desc, desc="batch", leave=False):
            xx = xx.to(device, dtype=dtype, non_blocking=True)
            yy = yy.to(device, dtype=dtype, non_blocking=True)
            grid = grid.to(device, dtype=dtype, non_blocking=True)
            bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

            if not torch.isfinite(xx).all():
                print(f"[Error] NaN in input xx, skipping batch")
                continue

            # ══════════════════════════════════════════════════════
            # Step 1: 更新 DeepONet
            # ══════════════════════════════════════════════════════
            for p in rk4_net.parameters():
                p.requires_grad_(False)
            optimizer_don.zero_grad()
            pred = ti_autoregressive_predict(
                deeponet, rk4_net, xx, grid, bc_params,
                init_t, t_train, device, dt, dtype)
            pred[..., :initial_step, :] = yy[..., :initial_step, :]

            _b = yy.size(0)
            loss_data = myloss(
                pred[..., init_t:, :].reshape(_b, -1),
                yy[..., init_t:, :].reshape(_b, -1))

            # ── 计算 total_loss ──
            loss_f = torch.tensor(0.0, device=device)
            loss_bc = torch.tensor(0.0, device=device)
            if warmup_epochs and e < warmup_epochs:
                # Warmup: 预训练，让预测保持在初始条件附近
                last_input = yy[..., init_t - 1:init_t, :]
                target = last_input.expand(-1, -1, -1, t_train - init_t, -1)
                pred_part = pred[..., init_t:t_train, :]
                loss_pretrain = ((pred_part - target) ** 2).mean()
                total_loss = loss_pretrain
                epoch_physics_loss += 0.0

                if warmup_lr_list and e < len(warmup_lr_list):
                    for param_group in optimizer_don.param_groups:
                        param_group['lr'] = warmup_lr_list[e]
                    for param_group in optimizer_rk4.param_groups:
                        param_group['lr'] = warmup_lr_list[e]
            else:
                if loss_mode != 'data':
                    # 物理残差 (Allen-Cahn PDE)
                    residual = compute_ac2d_residual_batch(
                        pred, bc_params, epsilon=epsilon, T=T_total)
                    loss_bc  = compute_neumann_bc_loss(pred, bc_params)
                    if residual_mode == 'mae':
                        loss_f = residual.abs().mean()
                    else:
                        loss_f = (residual ** 2).mean()
                    epoch_physics_loss += loss_f.item()

                if loss_mode == 'both':
                    total_loss = loss_f * f_weight + loss_data * data_weight
                elif loss_mode == 'data':
                    total_loss = loss_data
                else:  # 'pde'
                    total_loss = loss_f * f_weight + bc_weight * loss_bc

            if not torch.isfinite(total_loss):
                print(f"[Warning] Non-finite loss at epoch {e}, skip")
                continue

            total_loss.backward()
            clip_info = clipper.step(deeponet)
            optimizer_don.step()
            for p in rk4_net.parameters():
                p.requires_grad_(True)

            # ══════════════════════════════════════════════════════
            # Step 2: 更新 RK4Net (用已更新的 DeepONet)
            # ══════════════════════════════════════════════════════
            for p in deeponet.parameters():
                p.requires_grad_(False)
            optimizer_rk4.zero_grad()
            pred2 = ti_autoregressive_predict(
                deeponet, rk4_net, xx, grid, bc_params,
                init_t, t_train, device, dt, dtype)
            pred2[..., :initial_step, :] = yy[..., :initial_step, :]

            loss_data_rk4 = myloss(
                pred2[..., init_t:, :].reshape(_b, -1),
                yy[..., init_t:, :].reshape(_b, -1))

            # RK4 的 loss 也根据 loss_mode 计算
            loss_f_rk4 = torch.tensor(0.0, device=device)
            if warmup_epochs and e < warmup_epochs:
                last_input = yy[..., init_t - 1:init_t, :]
                target = last_input.expand(-1, -1, -1, t_train - init_t, -1)
                pred2_part = pred2[..., init_t:t_train, :]
                total_loss_rk4 = ((pred2_part - target) ** 2).mean()
            else:
                if loss_mode != 'data':
                    residual_rk4 = compute_ac2d_residual_batch(
                        pred2, bc_params, epsilon=epsilon, T=T_total)
                    loss_bc_rk4  = compute_neumann_bc_loss(pred2, bc_params)
                    if residual_mode == 'mae':
                        loss_f_rk4 = residual_rk4.abs().mean()
                    else:
                        loss_f_rk4 = (residual_rk4 ** 2).mean()

                if loss_mode == 'both':
                    total_loss_rk4 = loss_f_rk4 * f_weight + loss_data_rk4 * data_weight
                elif loss_mode == 'data':
                    total_loss_rk4 = loss_data_rk4
                else:
                     total_loss_rk4 = loss_f_rk4 * f_weight + bc_weight * loss_bc_rk4 

            if torch.isfinite(total_loss_rk4):
                total_loss_rk4.backward()
                torch.nn.utils.clip_grad_norm_(rk4_net.parameters(), max_norm=500.0)
                optimizer_rk4.step()
            for p in deeponet.parameters():
                p.requires_grad_(True)

            # ── 记录 ──
            current_lr_don = optimizer_don.param_groups[0]['lr']
            current_lr_rk4 = optimizer_rk4.param_groups[0]['lr']
            loss_list.append([loss_data.item(), loss_f.item(), loss_bc.item(), total_loss.item(), e])
            epoch_grad_norms.append(clip_info['grad_norm_after'])
            epoch_loss += total_loss.item()

        # ── epoch 结束 ──
        prev_loss = epoch_physics_loss / max(len(train_loader), 1)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        avg_grad = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
        grad.append([e, avg_loss, avg_grad])
        lr_list.append([current_lr_don, t_train, e])

        if sched_name == 'ReduceLROnPlateau':
            scheduler_don.step(avg_loss)
            scheduler_rk4.step(avg_loss)
        else:
            scheduler_don.step()
            scheduler_rk4.step()

        ebar.set_description(
            f"Epoch {e + 1} | Loss:{total_loss.item():.4e} TestL2:{test_l2:.4e} "
            f"Data:{loss_data.item():.4e}, Phy:{loss_f.item():.4e} "
            f"Loss_BC: {loss_bc.item():.4e},"
            f"lr_don:{current_lr_don:.2e} lr_rk4:{current_lr_rk4:.2e} "
            f"Grad:{avg_grad:.2f}")

        if best_error > avg_loss:
            best_error = avg_loss
            model_save_record.append([e, avg_loss])
            _save_ckpt(f'{first_dic}/checkpoint-best', e)
        _save_ckpt(f'{first_dic}/checkpoint_newst', e)

        if e % train_cfg['verbose_interval'] == 0:
            test_l2 = eval_model()
            print(f"epoch:{e}  test_l2:{test_l2:.4e}")
            test_loss_list.append([test_l2, e])

            if e % train_cfg.get('check_epochs', 50) == 0:
                _save_ckpt(f'{first_dic}/checkpoint-{e}', e)
                t_el = time.time() - time_0
                t_100 = time.time() - time_old
                with open(f'{first_dic}/Experiment_record.txt', 'a',
                          encoding='utf-8') as f:
                    f.write(f"├── epoch {e}: test_l2={test_l2:.4e}  "
                            f"elapsed={int(t_el // 3600)}h{int((t_el % 3600) // 60)}m  "
                            f"per_interval={int(t_100 // 3600)}h"
                            f"{int((t_100 % 3600) // 60)}m\n")
                time_old = time.time()

    print(f"{datetime.now()} --- training finished ---")


# ================================================================
# 4. 测试 & 可视化
# ================================================================

def test(config, args):
    # ── prepare ──────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    dtype = torch.float32

    data_config = config['data']
    origin_nx = data_config['nx']
    initial_step = config['train']['initial_step']
    filepath = data_config['datapath']
    sub_t = data_config['sub_t']
    T_total = config['data'].get('T', 5.0)

    test_loader_1x = torch.utils.data.DataLoader(
        FNODatasetMult(filepath, initial_step=initial_step, sub_x=1, sub_t=sub_t, if_test=True),
        batch_size=1, shuffle=False)
    test_loader_2x = torch.utils.data.DataLoader(
        FNODatasetMult(filepath, initial_step=initial_step, sub_x=2, sub_t=sub_t, if_test=True),
        batch_size=1, shuffle=False)
    test_loader_4x = torch.utils.data.DataLoader(
        FNODatasetMult(filepath, initial_step=initial_step, sub_x=4, sub_t=sub_t, if_test=True),
        batch_size=1, shuffle=False)

    t_train = (data_config['nt'] - 1) // sub_t + 1
    init_t = initial_step
    dt = T_total / (t_train - 1)

    first_dic = f"/code/AC2D{config['prepare']['project']}"
    os.makedirs(first_dic, exist_ok=True)
    os.chdir(first_dic)

    # ── model config ─────────────────────────────────────────────
    ti_cfg = config.get('ti_deeponet', {})
    latent_dim = ti_cfg.get('latent_dim', 100)

    branch_cfg = ti_cfg.get('branch', {})
    branch_cnn_cfg = branch_cfg.get('cnn', [
        {'out_channels': 64, 'kernel_size': 3, 'activation': 'relu', 'pool': 'max', 'pool_size': 2},
        {'out_channels': 64, 'kernel_size': 3, 'activation': 'relu', 'pool': 'max', 'pool_size': 2},
        {'out_channels': 64, 'kernel_size': 2, 'activation': 'relu', 'pool': 'avg', 'pool_size': 2},
    ])
    branch_mlp_hidden = branch_cfg.get('mlp_hidden', [256, 128])
    branch_mlp_activation = branch_cfg.get('mlp_activation', 'gelu')
    trunk_hidden = ti_cfg.get('trunk_hidden', [128, 128, 128, 128, 128, 128])

    rk4_cfg = ti_cfg.get('rk4', {})
    rk4_cnn_cfg = rk4_cfg.get('cnn', [
        {'out_channels': 32, 'kernel_size': 3, 'activation': 'relu', 'pool': 'max', 'pool_size': 2},
        {'out_channels': 32, 'kernel_size': 3, 'activation': 'relu', 'pool': 'max', 'pool_size': 2},
        {'out_channels': 32, 'kernel_size': 2, 'activation': 'relu', 'pool': 'avg', 'pool_size': 2},
    ])
    rk4_mlp_hidden = rk4_cfg.get('mlp_hidden', [32, 32])

    # ── 构建模型 (用训练分辨率) ───────────────────────────────────
    sub_x_train = data_config['sub_x']
    nx_train = (origin_nx - 1) // sub_x_train + 1
    ny_train = nx_train

    deeponet = TIDeepONet2D(
        init_step=initial_step,
        latent_dim=latent_dim,
        branch_cnn_cfg=branch_cnn_cfg,
        branch_mlp_hidden=branch_mlp_hidden,
        trunk_hidden=trunk_hidden,
        nx=nx_train, ny=ny_train,
        branch_mlp_activation=branch_mlp_activation,
    ).to(device)

    rk4_net = LearnableRK4Net(
        cnn_cfg=rk4_cnn_cfg,
        mlp_hidden=rk4_mlp_hidden,
        nx=nx_train, ny=ny_train,
    ).to(device)

    # ── load checkpoint ──────────────────────────────────────────
    ckpt_path = args.pretrain if args.pretrain else 'checkpoint-best.pth.tar'
    model_name = re.sub(r'[^a-zA-Z0-9_]', '_',
                        os.path.splitext(os.path.basename(ckpt_path))[0])
    ckpt = torch.load(ckpt_path, map_location=device)
    deeponet.load_state_dict(ckpt['deeponet'])
    rk4_net.load_state_dict(ckpt['rk4_net'])
    deeponet.eval()
    rk4_net.eval()
    print(f"Loaded [{ckpt_path}], epoch={ckpt.get('epoch', 0)}")

    plot_loss_with_analysis_II(
        ckpt['loss_list'], ckpt.get('lr_list', []),
        ckpt.get('test_loss_list', []), ckpt['grad'],
        f'{first_dic}/loss_carve_{model_name}')

    # ── evaluate ─────────────────────────────────────────────────
    myloss = LpLoss(size_average=True)

    test_loaders = {
        f'test_{origin_nx}': (test_loader_1x, 1),
        f'test_{(origin_nx - 1) // 2 + 1}': (test_loader_2x, 2),
        f'test_{(origin_nx - 1) // 4 + 1}': (test_loader_4x, 4),
    }

    results = []
    errors_for_talk_all = {}
    visualize_results_all = []

    for name, (loader, sub_x_ratio) in test_loaders.items():
        errors_for_talk = []
        visualize_results = []

        _xx_tmp, _, _, _ = next(iter(loader))
        cur_nx, cur_ny = _xx_tmp.shape[1], _xx_tmp.shape[2]

        if cur_nx != nx_train or cur_ny != ny_train:
            print(f"[{name}] Resolution {cur_nx}×{cur_ny} != train {nx_train}×{ny_train}")
            print(f"  → Flatten 架构不支持分辨率泛化，跳过")
            continue

        with torch.no_grad():
            for b, (xx, yy, grid, bc_params) in enumerate(
                    tqdm(loader, desc=f"{name} eval")):
                xx = xx.to(device, dtype=dtype)
                yy = yy.to(device, dtype=dtype)
                grid = grid.to(device, dtype=dtype)
                bc_params = bc_params.to(device, dtype=dtype)

                pred = ti_autoregressive_predict(
                    deeponet, rk4_net, xx, grid, bc_params,
                    init_t, t_train, device, dt, dtype)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]

                _b = yy.size(0)
                l2err = myloss(
                    pred[..., init_t + 1:t_train, :].reshape(_b, -1),
                    yy[..., init_t + 1:t_train, :].reshape(_b, -1)).item()
                errors_for_talk.append([l2err, b])

            sorted_errs = sorted(errors_for_talk, key=lambda x: x[0])
            n_s = len(sorted_errs)
            selected = set(
                [r[1] for r in sorted_errs[:3]] +
                [r[1] for r in sorted_errs[-3:]] +
                [r[1] for r in sorted_errs[n_s // 2 - 1: n_s // 2 + 2]]
            )

            for b, (xx, yy, grid, bc_params) in enumerate(
                    tqdm(loader, desc=f"{name} vis")):
                if b not in selected:
                    continue
                xx = xx.to(device, dtype=dtype)
                yy = yy.to(device, dtype=dtype)
                grid = grid.to(device, dtype=dtype)
                bc_params = bc_params.to(device, dtype=dtype)

                pred = ti_autoregressive_predict(
                    deeponet, rk4_net, xx, grid, bc_params,
                    init_t, t_train, device, dt, dtype)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]

                l2err = next(r[0] for r in sorted_errs if r[1] == b)
                if b in [r[1] for r in sorted_errs[:3]]:
                    cat = 'best'
                elif b in [r[1] for r in sorted_errs[-3:]]:
                    cat = 'worst'
                else:
                    cat = 'mid'

                visualize_results.append({
                    'Dataset': name,
                    'index': b,
                    'category': cat,
                    'l2_tidon': l2err,
                    'pred_tidon': pred.squeeze().cpu().numpy(),
                    'yy': yy.squeeze().cpu().numpy(),
                })

        errs = np.array([r[0] for r in errors_for_talk])
        results.append({
            'Dataset': name,
            'Mean Relative L2': errs.mean(),
            'Std Relative L2': errs.std(),
            'Max Relative L2': errs.max(),
            'Min Relative L2': errs.min(),
        })
        errors_for_talk_all[name] = np.array(errors_for_talk)
        visualize_results_all.append(visualize_results)
        print(f"[{name}] Mean L2: {errs.mean():.4e} ± {errs.std():.4e}")

    pd.DataFrame(results).to_csv('test_results_tidon.csv', index=False)
    with open('visualize_results_tidon.pkl', 'wb') as f:
        pickle.dump(visualize_results_all, f)
    with open('error_for_talk_tidon.pkl', 'wb') as f:
        pickle.dump(errors_for_talk_all, f)
    print("Results saved.")

    for vis in visualize_results_all:
        if vis:
            plot_burgers2d_comparison(
                vis, save_dir=f'./figures_tidon_{vis[0]["Dataset"]}')
    print("可视化完成~")


# ================================================================
# 5. 主入口
# ================================================================

if __name__ == '__main__':
    parser = ArgumentParser(description='TI(L)-DeepONet for 2D Allen-Cahn')
    parser.add_argument('--config_path', type=str,
                        default='./information_don.yaml')
    parser.add_argument('--mode', type=str, default='train',
                        help='train / test')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='path to pretrained checkpoint')
    args = parser.parse_args()

    with open(args.config_path, encoding='utf-8') as f:
        config = yaml.load(f, yaml.FullLoader)

    if args.mode == 'train':
        run(config, args)
        test(config, args)
    else:
        test(config, args)