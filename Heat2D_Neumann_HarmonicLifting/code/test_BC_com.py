"""
固定BC验证测试 - 与随机BC测试完全相同的代码
只修改：
1. dataloader返回固定BC参数
2. 数据文件路径改为固定BC数据集
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

# 导入实际的模块
from model import *

CASE_ROOT = Path(__file__).resolve().parents[1]

def resolve_case_path(path_like):
    path = Path(path_like)
    return str(path if path.is_absolute() else CASE_ROOT / path)

from lossII import *
from dataloader import FNODatasetMult  # 使用实际的dataloader
from utils import *


# ============ 唯一修改：自定义dataloader ============
class FNODatasetMultFixedBC(FNODatasetMult):
    """
    固定BC数据集的dataloader
    唯一修改：返回固定的BC参数
    """
    def __getitem__(self, idx):
        # 调用父类获取数据
        xx, yy, grid = super().__getitem__(idx)
        
        # 返回固定的BC参数（与训练BC相同）
        bc_params = torch.tensor([-0.5, 0.5, 0.0, 0.0], dtype=torch.float)
        
        return xx, yy, grid, bc_params
# ===================================================


# ============ 以下代码与Document 6完全相同 ============
def compute_lifting_batch(bc_params, grid):
    """
    计算batch的lifting函数: u_b = a*x^2 + b*y^2 + c*x + d*y
    
    Args:
        bc_params: [batch, 4] tensor (a, b, c, d)
        grid: [batch, Nx, Ny, 2] or [Nx, Ny, 2]
    Returns:
        u_b: [batch, Nx, Ny, 1]
    """
    batch = bc_params.shape[0]
    device = bc_params.device
    dtype = bc_params.dtype
    
    a = bc_params[:, 0].view(batch, 1, 1, 1)
    b = bc_params[:, 1].view(batch, 1, 1, 1)
    c = bc_params[:, 2].view(batch, 1, 1, 1)
    d = bc_params[:, 3].view(batch, 1, 1, 1)
    
    if grid.dim() == 3:  # [Nx, Ny, 2]
        grid = grid.unsqueeze(0).expand(batch, -1, -1, -1)
    
    xx = grid[..., 0:1]  # [batch, Nx, Ny, 1]
    yy = grid[..., 1:2]
    
    u_b = a * xx**2 + b * yy**2 + c * xx + d * yy
    return u_b


def compute_lifting_single(bc_params, grid):
    """
    计算单个lifting函数
    
    Args:
        bc_params: [4] tensor or list
        grid: [Nx, Ny, 2]
    Returns:
        u_b: [Nx, Ny, 1]
    """
    if isinstance(bc_params, (list, tuple)):
        bc_params = torch.tensor(bc_params, device=grid.device, dtype=grid.dtype)
    
    a, b, c, d = bc_params[0], bc_params[1], bc_params[2], bc_params[3]
    xx = grid[..., 0:1]
    yy = grid[..., 1:2]
    return a * xx**2 + b * yy**2 + c * xx + d * yy


def autoregressive_rollout_with_bc_transform(
    model, xx, yy, grid, target_bc_params, train_bc_params,
    init_t, t_train, pre_mode='delta', current_t_train=None
):
    """
    使用BC转换的自回归rollout
    
    Args:
        model: 神经算子模型（训练在固定BC上）
        xx: [batch, Nx, Ny, init_t, 1] 初始条件（目标BC空间）
        yy: [batch, Nx, Ny, T, 1] ground truth（目标BC空间）
        grid: [batch, Nx, Ny, 2] 或 [Nx, Ny, 2] 空间网格
        target_bc_params: [batch, 4] 目标BC参数
        train_bc_params: [4] 训练时的固定BC参数
        init_t: 初始步数
        t_train: 总时间步数
        pre_mode: 'delta' 或 'direct'
        current_t_train: 当前预测的时间步数
    
    Returns:
        pred: [batch, Nx, Ny, current_t_train, 1] 目标BC空间的预测
    """
    if current_t_train is None:
        current_t_train = t_train
    
    batch, Nx, Ny = xx.shape[:3]
    device = xx.device
    dtype = xx.dtype
    
    # 确保grid维度与xx匹配
    if grid.dim() == 3:  # [Nx, Ny, 2]
        if grid.shape[0] != Nx or grid.shape[1] != Ny:
            raise ValueError(
                f"Grid spatial dimensions {grid.shape[:2]} don't match xx dimensions {(Nx, Ny)}. "
            )
        grid_for_lifting = grid.unsqueeze(0).expand(batch, -1, -1, -1)
    else:  # [batch, Nx, Ny, 2]
        if grid.shape[1] != Nx or grid.shape[2] != Ny:
            raise ValueError(
                f"Grid spatial dimensions {grid.shape[1:3]} don't match xx dimensions {(Nx, Ny)}. "
            )
        grid_for_lifting = grid
    
    # 计算liftings
    u_b_target = compute_lifting_batch(target_bc_params, grid_for_lifting)  # [batch, Nx, Ny, 1]
    u_b_train = compute_lifting_single(train_bc_params, grid if grid.dim() == 3 else grid[0])  # [Nx, Ny, 1]
    u_b_train = u_b_train.unsqueeze(0).expand(batch, -1, -1, -1)  # [batch, Nx, Ny, 1]
    
    # 1. 从目标BC空间提取 u_h
    u_h_init = xx - u_b_target.unsqueeze(-2)  # [batch, Nx, Ny, init_t, 1]
    
    # 2. 转换到训练BC空间
    xx_train_space = u_h_init + u_b_train.unsqueeze(-2)  # [batch, Nx, Ny, init_t, 1]
    
    # 3. 在训练BC空间进行rollout
    pred_train_space = torch.empty(
        (batch, Nx, Ny, current_t_train, 1), 
        device=device, dtype=dtype
    )
    pred_train_space[..., :init_t, :] = xx_train_space
    
    for t in range(init_t, current_t_train):
        inp = xx_train_space.squeeze(-1)  # [batch, Nx, Ny, init_t]
        out = model(torch.cat([inp, grid], dim=-1)).unsqueeze(-1)  # [batch, Nx, Ny, 1, 1]
        
        if pre_mode == 'delta':
            out = xx_train_space[..., -1:, :] + out
        
        pred_train_space[..., t:t+1, :] = out
        xx_train_space = torch.cat((xx_train_space[..., 1:, :], out), dim=-2)
    
    # 4. 转换回目标BC空间
    u_h_pred = pred_train_space - u_b_train.unsqueeze(-2)  # [batch, Nx, Ny, T, 1]
    pred_target_space = u_h_pred + u_b_target.unsqueeze(-2)  # [batch, Nx, Ny, T, 1]
    
    return pred_target_space


def test_bc_generalization(config, args):
    """
    测试BC泛化能力的主函数
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
    print(f"固定BC验证测试 - 使用与随机BC测试完全相同的代码")
    print(f"{'='*80}\n")
    
    # ================================================================
    # 2. 加载数据集
    # ================================================================
    data_config = config['data']
    initial_step = config['train']['initial_step']
    
    # ============ 修改：使用固定BC数据集 ============
    filepath = resolve_case_path('data/heat2d_neumann_1100.h5')
    print(f"Data file: {filepath}")
    # ===============================================
    
    sub_t = data_config['sub_t']
    full_step = data_config.get('full_step', 101)
    
    # 测试多个分辨率
    test_datasets = {}
    test_loaders = {}
    
    for sub_x in [1, 2, 4]:
        # ============ 修改：使用自定义dataloader ============
        test_data = FNODatasetMultFixedBC(
            file_path=filepath,
            initial_step=initial_step,
            full_step=full_step,
            sub_x=sub_x,
            sub_t=sub_t,
            if_test=True  # 使用前100个样本作为测试集
        )
        # =================================================
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
    save_dir = os.path.join(first_dic, 'fixed_bc_verification')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}\n")
    
    # ================================================================
    # 4. 加载模型
    # ================================================================
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
    
    model = Model(input_channel, modes, width, bandwidth, 
                  out_channels=out_channels, dim=dim, triL=tril, 
                  double_weights=False, skip=True, flat=False).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SOL2D with DCT-I transform")
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
    
    print(f"Training BC (fixed during training):")
    print(f"  [a, b, c, d] = {TRAIN_BC}")
    print(f"  u_b = {TRAIN_BC[0]}*x^2 + {TRAIN_BC[1]}*y^2 + {TRAIN_BC[2]}*x + {TRAIN_BC[3]}*y")
    print(f"  Note: target_BC = train_BC (should cancel out in BC transform)\n")
    
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
    # 7. BC分布统计（固定BC，所以全是相同值）
    # ================================================================
    print(f"BC in test set (all samples have same BC):")
    print(f"  [a, b, c, d] = {TRAIN_BC}\n")
    
    # ================================================================
    # 8. 主测试循环
    # ================================================================
    all_results = {}
    
    for resolution, test_loader in test_loaders.items():
        print(f"\n{'='*80}")
        print(f"Testing Resolution: {resolution}")
        print(f"{'='*80}\n")
        
        errors_list = []
        
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
                    print()
                    first_batch = False
                
                # 构造训练BC参数
                train_bc_params = torch.tensor(TRAIN_BC, device=device, dtype=dtype)
                
                # 使用BC转换的rollout预测
                pred = autoregressive_rollout_with_bc_transform(
                    model, xx, yy, grid, 
                    bc_params,  # 目标BC（固定BC）
                    train_bc_params,  # 训练BC（相同，理论上抵消）
                    init_t, t_train, pre_mode, 
                    current_t_train=None
                )
                
                # yy是目标BC下的ground truth
                # 计算L2误差（与Document 6完全相同）
                _pred = pred[...,init_t+1:t_train, :]
                _yy = yy[..., init_t+1:t_train, :]
                l2_error = myloss(_pred.reshape(batch, -1), _yy.reshape(batch, -1)).item()
                
                # 计算PDE残差
                residual_pred = compute_heat2d_residual_batch(pred, bc_params, kappa=0.02, T=1.0)
                residual_yy = compute_heat2d_residual_batch(yy, bc_params, kappa=0.02, T=1.0)
                pde_error_pred = torch.abs(residual_pred).mean().item()
                pde_error_yy = torch.abs(residual_yy).mean().item()
                
                # 记录BC参数和误差
                errors_list.append({
                    'bc_a': bc_params[0, 0].item(),
                    'bc_b': bc_params[0, 1].item(),
                    'bc_c': bc_params[0, 2].item(),
                    'bc_d': bc_params[0, 3].item(),
                    'l2_error': l2_error,
                    'pde_error_pred': pde_error_pred,
                    'pde_error_yy': pde_error_yy,
                })
        
        # 汇总统计
        errors_array = np.array([
            [e['l2_error'], e['pde_error_pred'], e['pde_error_yy']] 
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
        
        print(f"\nResults for {resolution}:")
        print(f"  L2 Error:         {mean_l2:.6e} ± {std_l2:.6e}")
        print(f"    Min:            {min_l2:.6e}")
        print(f"    Median:         {median_l2:.6e}")
        print(f"    Max:            {max_l2:.6e}")
        print(f"  PDE Error (pred): {mean_pde_pred:.6e} ± {std_pde_pred:.6e}")
        print(f"  PDE Error (true): {mean_pde_yy:.6e} ± {std_pde_yy:.6e}")
        
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
            })
    
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(save_dir, 'fixed_bc_verification_detailed.csv')
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
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(save_dir, 'fixed_bc_verification_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV to: {summary_csv_path}")
    
    # 保存完整结果
    pkl_path = os.path.join(save_dir, 'fixed_bc_verification_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved complete results to: {pkl_path}")
    
    # 保存文本报告
    report_path = os.path.join(save_dir, 'fixed_bc_verification_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("固定BC验证测试报告 - 与随机BC测试完全相同的代码\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Training Epochs: {epoch}\n")
        f.write(f"Number of test samples: {ntest}\n")
        f.write(f"Data file: {filepath}\n\n")
        
        f.write(f"Training BC (fixed): {TRAIN_BC}\n")
        f.write(f"  u_b = {TRAIN_BC[0]}*x^2 + {TRAIN_BC[1]}*y^2 + {TRAIN_BC[2]}*x + {TRAIN_BC[3]}*y\n\n")
        
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
            f.write("\n")
    
    print(f"Saved report to: {report_path}")
    
    # ================================================================
    # 10. 最终摘要
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
    parser = ArgumentParser(description='Fixed BC Verification Test')
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
    test_bc_generalization(config, args)