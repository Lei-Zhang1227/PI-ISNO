"""
Test BC Generalization for FFT Model WITH BC Transformation
测试FFT + Delta模式 + BC转换的BC泛化能力

关键实验：验证BC转换是否需要显式Lifting作为前提
"""
import os
from pathlib import Path
import sys
import pandas as pd
from argparse import ArgumentParser
import yaml
import h5py
import torch
from tqdm import tqdm
import pickle
from functools import partial as PARTIAL
import numpy as np
import time
from datetime import datetime

# 导入FFT模型相关模块
from model import *

CASE_ROOT = Path(__file__).resolve().parents[1]

def resolve_case_path(path_like):
    path = Path(path_like)
    return str(path if path.is_absolute() else CASE_ROOT / path)

from loss import *
from dataloaderII import FNODatasetMult
from utils import *


def compute_lifting(bc_params, X, Y):
    """
    计算谐波lifting函数 u_b
    
    u_b = a*x^2 + b*y^2 + c*x + d*y
    
    Args:
        bc_params: [batch, 4] 或 [4]
        X, Y: [Nx, Ny] meshgrid
        
    Returns:
        u_b: [batch, Nx, Ny, 1, 1] 或 [1, Nx, Ny, 1, 1]
    """
    if bc_params.ndim == 1:
        bc_params = bc_params.unsqueeze(0)  # [1, 4]
    
    batch = bc_params.shape[0]
    a = bc_params[:, 0:1, None, None, None]  # [batch, 1, 1, 1, 1]
    b = bc_params[:, 1:2, None, None, None]
    c = bc_params[:, 2:3, None, None, None]
    d = bc_params[:, 3:4, None, None, None]
    
    X = X.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, Nx, Ny, 1, 1]
    Y = Y.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    u_b = a * X**2 + b * Y**2 + c * X + d * Y  # [batch, Nx, Ny, 1, 1]
    
    return u_b


def autoregressive_rollout_fft_with_bc_transform(
    model, xx, yy, grid, bc_params_test, bc_params_train, init_t, t_train, X, Y, pre_mode='delta'
):
    """
    FFT模型的自回归rollout WITH BC TRANSFORMATION
    
    关键：虽然FFT没有显式Lifting训练，但测试时尝试做BC转换
    
    流程：
    1. 提取 u_h = u_test - u_b_test
    2. 转换 u_pseudo = u_h + u_b_train (对齐到训练分布)
    3. FFT模型预测
    4. 转换回 pred_test = (pred_pseudo - u_b_train) + u_b_test
    
    Args:
        model: FFT神经算子模型
        xx: [batch, Nx, Ny, init_t, 1] 初始条件
        yy: [batch, Nx, Ny, T, 1] ground truth
        grid: [batch, Nx, Ny, 2] 空间网格
        bc_params_test: [batch, 4] 测试样本的BC参数
        bc_params_train: [4] 训练时的固定BC
        init_t: 初始步数
        t_train: 总时间步数
        X, Y: [Nx, Ny] 空间坐标meshgrid
        pre_mode: 'delta' 或 'direct'
    
    Returns:
        pred: [batch, Nx, Ny, t_train, 1] 预测的完整解u
    """
    batch, Nx, Ny = xx.shape[:3]
    device = xx.device
    dtype = xx.dtype
    
    # 1. 计算测试BC的u_b
    u_b_test = compute_lifting(bc_params_test, X, Y)  # [batch, Nx, Ny, 1, 1]
    
    # 2. 计算训练BC的u_b
    train_bc_tensor = torch.tensor(bc_params_train, device=device, dtype=dtype).unsqueeze(0)
    u_b_train = compute_lifting(train_bc_tensor, X, Y)  # [1, Nx, Ny, 1, 1]
    
    # 3. 提取u_h（虽然FFT训练时没这样做）
    xx_h = xx - u_b_test
    
    # 4. 转换到训练BC空间
    xx_pseudo = xx_h + u_b_train
    
    # 5. FFT模型在"伪训练空间"预测
    pred_pseudo = torch.empty((batch, Nx, Ny, t_train, 1), device=device, dtype=dtype)
    pred_pseudo[..., :init_t, :] = xx_pseudo
    
    for t in range(init_t, t_train):
        inp = xx_pseudo.squeeze(-1)  # [batch, Nx, Ny, init_t]
        out = model(torch.cat([inp, grid], dim=-1)).unsqueeze(-1)
        
        if pre_mode == 'delta':
            out = xx_pseudo[..., -1:, :] + out
        
        pred_pseudo[..., t:t+1, :] = out
        xx_pseudo = torch.cat((xx_pseudo[..., 1:, :], out), dim=-2)
    
    # 6. 转换回测试BC空间
    pred_h = pred_pseudo - u_b_train
    pred = pred_h + u_b_test
    
    return pred


def test_bc_generalization_fft_with_transform(config, args):
    """
    测试FFT模型 + BC转换的BC泛化能力
    """
    # ================================================================
    # 1. 准备工作
    # ================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    
    print(f"\n{'='*80}")
    print(f"BC Generalization Test - FFT Model WITH BC Transformation")
    print(f"{'='*80}\n")
    
    # ================================================================
    # 2. 加载数据集
    # ================================================================
    data_config = config['data']
    initial_step = config['train']['initial_step']
    filepath = resolve_case_path('data/heat2d_bc_gen_test.h5')
    sub_t = data_config['sub_t']
    full_step = data_config.get('full_step', 101)
    
    # 测试多个分辨率
    test_datasets = {}
    test_loaders = {}
    
    for sub_x in [1, 2, 4]:
        test_data = FNODatasetMult(
            file_path=filepath,
            initial_step=initial_step,
            full_step=full_step,
            sub_x=sub_x,
            sub_t=sub_t,
            if_test=True
        )
        test_datasets[f'{sub_x}x'] = test_data
        test_loaders[f'{sub_x}x'] = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=0
        )
    
    ntest = len(test_datasets['1x'])
    print(f"Total test samples: {ntest}")
    print(f"Time steps used: {full_step} (with sub_t={sub_t})\n")
    
    # ================================================================
    # 3. 创建保存目录
    # ================================================================
    first_dic = resolve_case_path(config['prepare']['project'])
    save_dir = os.path.join(first_dic, 'bc_generalization_test_fft_with_transform')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}\n")
    
    # ================================================================
    # 4. 加载FFT模型
    # ================================================================
    _trans = PARTIAL(Wrapper, [fft_forward, fft_forward])
    _itrans = PARTIAL(Wrapper, [fft_inverse, fft_inverse])
    T = Transform(_trans, _itrans)
    Model = PARTIAL(SOL2D_FFT, T)
    
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    out_channels = config['model']['output_channel']
    dim = config['model']['dim']
    tril = config['model']['triL']
    input_channel = initial_step + 2
    
    model = Model(input_channel, modes, width, bandwidth, 
                  out_channels=out_channels, dim=dim, triL=tril, 
                  double_weights=False, skip=True, flat=False).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SOL2D_FFT with FFT transform")
    print(f"Total trainable parameters: {total_params:,}\n")
    
    # 加载checkpoint
    if args.pretrain is not None:
        checkpoint_path = args.pretrain
    else:
        checkpoint_path = os.path.join(first_dic, 'checkpoint-best.pth.tar')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint.get('epoch', checkpoint.get('epochs', 0))
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Trained for {epoch} epochs\n")
    
    model.eval()
    
    # ================================================================
    # 5. 定义训练BC
    # ================================================================
    TRAIN_BC = args.train_bc if hasattr(args, 'train_bc') and args.train_bc else [-0.5, 0.5, 0.0, 0.0]
    
    print(f"Training BC (used for BC transformation):")
    print(f"  [a, b, c, d] = {TRAIN_BC}")
    print(f"  u_b = {TRAIN_BC[0]}*x^2 + {TRAIN_BC[1]}*y^2 + {TRAIN_BC[2]}*x + {TRAIN_BC[3]}*y")
    print(f"\n  ⚠️  KEY EXPERIMENT:")
    print(f"  Testing whether BC transformation works WITHOUT explicit lifting training.")
    print(f"  FFT model was trained on full u (not u_h), but we apply BC transform at test time.\n")
    
    # ================================================================
    # 6. 测试配置
    # ================================================================
    pre_mode = config['train'].get('pre_mode', 'delta')
    init_t = initial_step
    t_train = (full_step - 1) // sub_t + 1
    dtype = torch.float32
    
    myloss = LpLoss(size_average=True)
    
    print(f"Test configuration:")
    print(f"  Prediction mode: {pre_mode}")
    print(f"  Initial steps: {init_t}")
    print(f"  Total time steps: {t_train}")
    print(f"  Data type: {dtype}\n")
    
    # ================================================================
    # 7. 收集测试集中的BC分布统计
    # ================================================================
    print(f"Analyzing BC distribution in test set...")
    all_bc_params = []
    with h5py.File(filepath, 'r') as f:
        data_list = sorted(f.keys())
        data_list = [k for k in data_list if k not in ['grid', 'params']]
        test_keys = data_list[:100]
        
        for key in test_keys:
            if 'bc' in f[key]:
                a = f[f'{key}/bc/a'][()]
                b = f[f'{key}/bc/b'][()]
                c = f[f'{key}/bc/c'][()]
                d = f[f'{key}/bc/d'][()]
                all_bc_params.append([a, b, c, d])
    
    all_bc_params = np.array(all_bc_params)
    print(f"\nBC Statistics in test set ({len(all_bc_params)} samples):")
    print(f"  a: min={all_bc_params[:, 0].min():.3f}, max={all_bc_params[:, 0].max():.3f}, mean={all_bc_params[:, 0].mean():.3f}")
    print(f"  b: min={all_bc_params[:, 1].min():.3f}, max={all_bc_params[:, 1].max():.3f}, mean={all_bc_params[:, 1].mean():.3f}")
    print(f"  c: min={all_bc_params[:, 2].min():.3f}, max={all_bc_params[:, 2].max():.3f}, mean={all_bc_params[:, 2].mean():.3f}")
    print(f"  d: min={all_bc_params[:, 3].min():.3f}, max={all_bc_params[:, 3].max():.3f}, mean={all_bc_params[:, 3].mean():.3f}")
    print()
    
    # ================================================================
    # 8. 主测试循环
    # ================================================================
    all_results = {}
    
    for resolution, test_loader in test_loaders.items():
        print(f"\n{'='*80}")
        print(f"Testing Resolution: {resolution}")
        print(f"{'='*80}\n")
        
        errors_list = []
        
        # 获取当前分辨率的网格
        sub_x = int(resolution[0])
        Nx = int((data_config['nx'] - 1) / sub_x) + 1
        x_coord = torch.linspace(-1, 1, Nx, device=device, dtype=dtype)
        y_coord = torch.linspace(-1, 1, Nx, device=device, dtype=dtype)
        X, Y = torch.meshgrid(x_coord, y_coord, indexing='ij')
        
        with torch.no_grad():
            first_batch = True
            for xx, yy, grid, bc_params in tqdm(test_loader, desc=f"Testing {resolution}"):
                # 移动到设备
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)
                
                batch = xx.shape[0]
                
                # Debug: 打印形状（第一次）
                if first_batch:
                    print(f"\nDebug - Tensor shapes:")
                    print(f"  xx shape: {xx.shape}")
                    print(f"  yy shape: {yy.shape}")
                    print(f"  grid shape: {grid.shape}")
                    print(f"  bc_params shape: {bc_params.shape}")
                    print(f"  BC for first sample: [{bc_params[0, 0].item():.3f}, {bc_params[0, 1].item():.3f}, {bc_params[0, 2].item():.3f}, {bc_params[0, 3].item():.3f}]")
                    print()
                    first_batch = False
                
                # 关键：FFT模型 + BC转换
                pred = autoregressive_rollout_fft_with_bc_transform(
                    model, xx, yy, grid, bc_params, TRAIN_BC,
                    init_t, t_train, X, Y, pre_mode
                )
                
                # 计算L2误差
                _pred = pred[..., init_t:, :]
                _yy = yy[..., init_t:, :]
                l2_error = myloss(_pred.reshape(batch, -1), _yy.reshape(batch, -1)).item()
                
                # 计算PDE残差
                residual_pred = compute_heat2d_residual_fd(pred, kappa=0.02, T=1.0)
                residual_yy = compute_heat2d_residual_fd(yy, kappa=0.02, T=1.0)
                pde_error_pred = torch.abs(residual_pred).mean().item()
                pde_error_yy = torch.abs(residual_yy).mean().item()
                
                # 计算BC损失
                bc_loss_pred = compute_neumann_bc_loss(pred).item()
                bc_loss_yy = compute_neumann_bc_loss(yy).item()
                
                # 记录BC参数和误差
                errors_list.append({
                    'bc_a': bc_params[0, 0].item(),
                    'bc_b': bc_params[0, 1].item(),
                    'bc_c': bc_params[0, 2].item(),
                    'bc_d': bc_params[0, 3].item(),
                    'l2_error': l2_error,
                    'pde_error_pred': pde_error_pred,
                    'pde_error_yy': pde_error_yy,
                    'bc_loss_pred': bc_loss_pred,
                    'bc_loss_yy': bc_loss_yy,
                })
        
        # 汇总统计
        errors_array = np.array([
            [e['l2_error'], e['pde_error_pred'], e['pde_error_yy'], 
             e['bc_loss_pred'], e['bc_loss_yy']] 
            for e in errors_list
        ])
        bc_array = np.array([
            [e['bc_a'], e['bc_b'], e['bc_c'], e['bc_d']]
            for e in errors_list
        ])
        
        mean_l2 = np.mean(errors_array[:, 0])
        std_l2 = np.std(errors_array[:, 0])
        max_l2 = np.max(errors_array[:, 0])
        min_l2 = np.min(errors_array[:, 0])
        median_l2 = np.median(errors_array[:, 0])
        
        mean_pde_pred = np.mean(errors_array[:, 1])
        std_pde_pred = np.std(errors_array[:, 1])
        
        mean_pde_yy = np.mean(errors_array[:, 2])
        std_pde_yy = np.std(errors_array[:, 2])
        
        mean_bc_pred = np.mean(errors_array[:, 3])
        std_bc_pred = np.std(errors_array[:, 3])
        
        mean_bc_yy = np.mean(errors_array[:, 4])
        std_bc_yy = np.std(errors_array[:, 4])
        
        print(f"\nResults for {resolution}:")
        print(f"  L2 Error:         {mean_l2:.6e} ± {std_l2:.6e}")
        print(f"    Min:            {min_l2:.6e}")
        print(f"    Median:         {median_l2:.6e}")
        print(f"    Max:            {max_l2:.6e}")
        print(f"  PDE Error (pred): {mean_pde_pred:.6e} ± {std_pde_pred:.6e}")
        print(f"  PDE Error (true): {mean_pde_yy:.6e} ± {std_pde_yy:.6e}")
        print(f"  BC Loss (pred):   {mean_bc_pred:.6e} ± {std_bc_pred:.6e}")
        print(f"  BC Loss (true):   {mean_bc_yy:.6e} ± {std_bc_yy:.6e}")
        
        all_results[resolution] = {
            'mean_l2': mean_l2,
            'std_l2': std_l2,
            'max_l2': max_l2,
            'min_l2': min_l2,
            'median_l2': median_l2,
            'mean_pde_pred': mean_pde_pred,
            'std_pde_pred': std_pde_pred,
            'mean_pde_yy': mean_pde_yy,
            'std_pde_yy': std_pde_yy,
            'mean_bc_pred': mean_bc_pred,
            'std_bc_pred': std_bc_pred,
            'mean_bc_yy': mean_bc_yy,
            'std_bc_yy': std_bc_yy,
            'errors_list': errors_list,
            'bc_params': bc_array,
        }
    
    # ================================================================
    # 9. 保存结果
    # ================================================================
    print(f"\n{'='*80}")
    print("Saving Results...")
    print(f"{'='*80}\n")
    
    # 保存详细CSV
    csv_rows = []
    for resolution, results in all_results.items():
        for error_rec in results['errors_list']:
            csv_rows.append({
                'Resolution': resolution,
                'BC_a': error_rec['bc_a'],
                'BC_b': error_rec['bc_b'],
                'BC_c': error_rec['bc_c'],
                'BC_d': error_rec['bc_d'],
                'L2_Error': error_rec['l2_error'],
                'PDE_Error_Pred': error_rec['pde_error_pred'],
                'PDE_Error_True': error_rec['pde_error_yy'],
                'BC_Loss_Pred': error_rec['bc_loss_pred'],
                'BC_Loss_True': error_rec['bc_loss_yy'],
            })
    
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(save_dir, 'bc_generalization_detailed_fft_with_transform.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed CSV to: {csv_path}")
    
    # 保存摘要统计
    summary_rows = []
    for resolution, results in all_results.items():
        summary_rows.append({
            'Resolution': resolution,
            'Mean_L2': results['mean_l2'],
            'Std_L2': results['std_l2'],
            'Min_L2': results['min_l2'],
            'Median_L2': results['median_l2'],
            'Max_L2': results['max_l2'],
            'Mean_PDE_Pred': results['mean_pde_pred'],
            'Std_PDE_Pred': results['std_pde_pred'],
            'Mean_PDE_True': results['mean_pde_yy'],
            'Std_PDE_True': results['std_pde_yy'],
            'Mean_BC_Pred': results['mean_bc_pred'],
            'Std_BC_Pred': results['std_bc_pred'],
            'Mean_BC_True': results['mean_bc_yy'],
            'Std_BC_True': results['std_bc_yy'],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(save_dir, 'bc_generalization_summary_fft_with_transform.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV to: {summary_csv_path}")
    
    # 保存完整结果
    pkl_path = os.path.join(save_dir, 'bc_generalization_results_fft_with_transform.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved complete results to: {pkl_path}")
    
    # 保存文本报告
    report_path = os.path.join(save_dir, 'bc_generalization_report_fft_with_transform.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BC Generalization Test Report - FFT Model WITH BC Transformation\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Training Epochs: {epoch}\n")
        f.write(f"Number of test samples: {ntest}\n\n")
        
        f.write(f"Model: FFT (no explicit lifting during training)\n")
        f.write(f"Training BC (fixed): {TRAIN_BC}\n")
        f.write(f"  u_b = {TRAIN_BC[0]}*x^2 + {TRAIN_BC[1]}*y^2 + {TRAIN_BC[2]}*x + {TRAIN_BC[3]}*y\n\n")
        
        f.write(f"⚠️  KEY EXPERIMENT:\n")
        f.write(f"   Testing whether BC transformation works without explicit lifting.\n")
        f.write(f"   FFT model trained on full u (not u_h), but BC transform applied at test time.\n\n")
        
        f.write("Test BC Distribution:\n")
        f.write(f"  a: [{all_bc_params[:, 0].min():.3f}, {all_bc_params[:, 0].max():.3f}], mean={all_bc_params[:, 0].mean():.3f}\n")
        f.write(f"  b: [{all_bc_params[:, 1].min():.3f}, {all_bc_params[:, 1].max():.3f}], mean={all_bc_params[:, 1].mean():.3f}\n")
        f.write(f"  c: [{all_bc_params[:, 2].min():.3f}, {all_bc_params[:, 2].max():.3f}], mean={all_bc_params[:, 2].mean():.3f}\n")
        f.write(f"  d: [{all_bc_params[:, 3].min():.3f}, {all_bc_params[:, 3].max():.3f}], mean={all_bc_params[:, 3].mean():.3f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("Results Summary\n")
        f.write("="*80 + "\n\n")
        
        for resolution, results in all_results.items():
            f.write(f"Resolution {resolution}:\n")
            f.write(f"  L2 Error:         {results['mean_l2']:.6e} ± {results['std_l2']:.6e}\n")
            f.write(f"    Min:            {results['min_l2']:.6e}\n")
            f.write(f"    Median:         {results['median_l2']:.6e}\n")
            f.write(f"    Max:            {results['max_l2']:.6e}\n")
            f.write(f"  PDE Error (pred): {results['mean_pde_pred']:.6e} ± {results['std_pde_pred']:.6e}\n")
            f.write(f"  PDE Error (true): {results['mean_pde_yy']:.6e} ± {results['std_pde_yy']:.6e}\n")
            f.write(f"  BC Loss (pred):   {results['mean_bc_pred']:.6e} ± {results['std_bc_pred']:.6e}\n")
            f.write(f"  BC Loss (true):   {results['mean_bc_yy']:.6e} ± {results['std_bc_yy']:.6e}\n")
            f.write("\n")
    
    print(f"Saved report to: {report_path}")
    
    # ================================================================
    # 10. BC参数与误差的相关性分析
    # ================================================================
    print(f"\n{'='*80}")
    print("BC-Error Correlation Analysis")
    print(f"{'='*80}\n")
    
    try:
        from scipy.stats import pearsonr, spearmanr
        
        # 使用1x分辨率的结果
        res_1x = all_results['1x']
        bc_params_1x = res_1x['bc_params']
        l2_errors_1x = np.array([e['l2_error'] for e in res_1x['errors_list']])
        
        # 计算相关系数
        corr_results = {}
        for i, param_name in enumerate(['a', 'b', 'c', 'd']):
            pearson_corr, pearson_p = pearsonr(bc_params_1x[:, i], l2_errors_1x)
            spearman_corr, spearman_p = spearmanr(bc_params_1x[:, i], l2_errors_1x)
            
            corr_results[param_name] = {
                'pearson': pearson_corr,
                'pearson_p': pearson_p,
                'spearman': spearman_corr,
                'spearman_p': spearman_p,
            }
            
            print(f"BC parameter '{param_name}' vs L2 Error:")
            print(f"  Pearson correlation:  {pearson_corr:.4f} (p={pearson_p:.4e})")
            print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
            print()
        
        # 保存相关性分析
        corr_csv_path = os.path.join(save_dir, 'bc_error_correlation_fft_with_transform.csv')
        corr_df = pd.DataFrame([
            {
                'BC_Param': param,
                'Pearson_Corr': results['pearson'],
                'Pearson_P': results['pearson_p'],
                'Spearman_Corr': results['spearman'],
                'Spearman_P': results['spearman_p'],
            }
            for param, results in corr_results.items()
        ])
        corr_df.to_csv(corr_csv_path, index=False)
        print(f"Saved correlation analysis to: {corr_csv_path}")
        
    except ImportError:
        print("Warning: scipy not installed, skipping correlation analysis")
        print("Install with: pip install scipy")
    
    # ================================================================
    # 11. 最终摘要
    # ================================================================
    print(f"\n{'='*80}")
    print("Final Summary")
    print(f"{'='*80}\n")
    
    print("Mean L2 Errors across resolutions:")
    for resolution in ['1x', '2x', '4x']:
        if resolution in all_results:
            print(f"  {resolution}: {all_results[resolution]['mean_l2']:.6e} ± {all_results[resolution]['std_l2']:.6e}")
    
    print(f"\n{'='*80}")
    print("Test Completed Successfully!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = ArgumentParser(description='BC Generalization Test for FFT Model WITH BC Transformation')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Path to checkpoint')
    parser.add_argument('--train_bc', type=float, nargs=4, default=None,
                        help='Training BC parameters [a, b, c, d]')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.FullLoader)
    
    # 运行测试
    test_bc_generalization_fft_with_transform(config, args)