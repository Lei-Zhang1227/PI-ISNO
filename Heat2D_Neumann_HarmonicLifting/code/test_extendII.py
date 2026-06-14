"""
Test Extend - 适配 u_h + Delta 预测模式
评估模型在长时间外推上的性能
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



# ============================================================================
# 辅助函数
# ============================================================================

def compute_ub_exact(Nx, device, dtype=torch.float32):
    """计算 u_b 精确解: u_b(x,y) = -0.5*x² + 0.5*y²"""
    x_coord = torch.linspace(-1, 1, Nx, device=device, dtype=dtype)
    y_coord = torch.linspace(-1, 1, Nx, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x_coord, y_coord, indexing='ij')
    
    u_b = -0.5 * X**2 + 0.5 * Y**2
    u_b = u_b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    return u_b


def autoregressive_rollout(model, xx_h, grid, u_b, init_t, t_train, pre_mode='delta', dtype=torch.float32):
    """
    统一的自回归预测函数
    
    Args:
        model: 神经网络模型
        xx_h: [batch, Nx, Ny, init_step, 1] - u_h 的初始条件
        grid: [batch, Nx, Ny, 2] - 空间网格
        u_b: [1, Nx, Ny, 1, 1] - 边界部分
        init_t: 初始步数
        t_train: 总时间步数
        pre_mode: 'delta' 或 'direct'
        dtype: 数据类型
    
    Returns:
        pred_h: [batch, Nx, Ny, t_train, 1] - 预测的 u_h
        pred: [batch, Nx, Ny, t_train, 1] - 还原的 u = u_h + u_b
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
        
        out_h = model(inp_with_grid).unsqueeze(-1)
        
        if pre_mode == 'delta':
            # Delta 模式: 预测 Δu_h
            last_uh = xx_h_current[..., -1:, :]
            next_uh = last_uh + out_h
        else:
            # Direct 模式: 直接预测 u_h(t+1)
            next_uh = out_h
        
        pred_h[..., t:t+1, :] = next_uh
        xx_h_current = torch.cat([xx_h_current[..., 1:, :], next_uh], dim=-2)
    
    # 还原完整解: u = u_h + u_b
    pred = pred_h + u_b
    
    return pred_h, pred


# ============================================================================
# 主测试函数
# ============================================================================

def testII_with_lifting(config, args):
    """
    长时间外推测试 - 支持 u_h + Delta 预测模式
    """
    # region prepare
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.float32
    # endregion
    
    # region dataloader
    data_config = config['data']
    batch_size = config['train']['batchsize']
    initial_step = config['train']['initial_step']
    filepath = resolve_case_path(data_config['datapath'])
    sub_x = data_config['sub_x']
    sub_t = data_config['sub_t']
    
    train_data = FNODatasetMult(file_path=filepath,
                                initial_step=initial_step,
                                sub_x=sub_x,
                                sub_t=sub_t,
                                full_step=501,
                              
                                )
    test_data = FNODatasetMult(file_path=filepath,
                               initial_step=initial_step,
                               sub_x=sub_x,
                               sub_t=sub_t,
                               full_step=501,
                               if_test=True,
                            
                               )
    
    test_loader_seen = torch.utils.data.DataLoader(train_data, batch_size=1, 
                                                    num_workers=2, shuffle=True)
    test_loader_unseen = torch.utils.data.DataLoader(test_data, batch_size=1, 
                                                      num_workers=2, shuffle=False)
    train_size, test_size = len(train_data), len(test_data)
    
    print(f'{datetime.now()} --- Dataset: batch_size={batch_size}, '
          f'train={train_size}, test={test_size}')
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
    
    # region location
    first_dic = resolve_case_path(config['prepare']['project'])
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    
    if args.pretrain is not None:
        src = args.config_path
        base = os.path.basename(src)
        name, ext = os.path.splitext(base)
        dst = os.path.join(first_dic, base)
        
        if os.path.abspath(src) == os.path.abspath(dst):
            i = 1
            while True:
                new_name = f"{name}-test-extend--{i}{ext}"
                new_dst = os.path.join(first_dic, new_name)
                if not os.path.exists(new_dst):
                    dst = new_dst
                    break
                i += 1
        shutil.copy(src, dst)
    else:
        shutil.copy(args.config_path, first_dic)
    
    print(f"{datetime.now()} --- Working dir: {first_dic}")
    # endregion
    
    # region load model
    if args.pretrain is not None:
        checkpoint = torch.load(resolve_case_path(args.pretrain))
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', checkpoint.get('epochs', 0))
        print(f"Loaded: {args.pretrain}, epoch={epoch}")
    else:
        checkpoint = torch.load(os.path.join(first_dic, 'checkpoint-best.pth.tar'))
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint-best, epoch={epoch}")
    # endregion
    
    # region evaluate
    init_t = 3
    t_train = 251
    
    # 计算 u_b (只需一次)
    Nx = int((data_config['nx'] - 1) / sub_x) + 1
    u_b = compute_ub_exact(Nx, device, dtype)
    print(f"u_b computed: Nx={Nx}, shape={u_b.shape}")
    
    myloss = LpLoss(size_average=True)
    
    # 获取预测模式
    pre_mode = config['train'].get('pre_mode', 'delta')
    print(f"Prediction mode: {pre_mode}")
    
    test_loaders = {
        'unseen_extend': test_loader_unseen,
        'seen_extend': test_loader_seen,
    }
    
    selected_samples_all = {}
    timestep_errors_all = {}
    timestep_errors_per_sample_all = {}
    
    for name, test_loader in test_loaders.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"{'='*70}")
        
        errors_for_talk = []
        sample_details = []
        timestep_errors_list = []
        
        model.eval()
        with torch.no_grad():
            test_iter = iter(test_loader)
            
            for b in tqdm(range(len(test_loader)), desc=name):
                xx, yy, grid = next(test_iter)
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                
                # 分离 u_h = u - u_b
                xx_h = xx - u_b[..., :initial_step, :]
                
                # 自回归预测
                pred_h, pred = autoregressive_rollout(
                    model, xx_h, grid, u_b, init_t, t_train, pre_mode, dtype
                )
                
                assert pred.shape == yy.shape, f"Shape mismatch: {pred.shape} != {yy.shape}"
                
                # 定义时间段
                segments = {
                    '0.1-1': (init_t + 1, 51),
                    '1-1.5': (51, 76),
                    '1.5-2': (76, 101),
                    '2-2.5': (101, 126),
                    '2.5-3': (126, 151),
                    '3-3.5': (151, 176),
                    '3.5-4': (176, 201),
                    '4-4.5': (201, 226),
                    '4.5-5': (226, 251),
                }
                
                _batch = yy.size(0)
                sample_errors = []
                
                # 计算各时间段的 L2 误差
                for seg_name, (s, e) in segments.items():
                    _yy = yy[..., s:e, :]
                    _pred = pred[..., s:e, :]
                    err = myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                    sample_errors.append(err)
                
                errors_for_talk.append(sample_errors)
                
                # 计算每个时间步的 L2 误差
                sample_timestep_errors = []
                for t in range(init_t, t_train):
                    _yy_t = yy[..., t:t+1, :]
                    _pred_t = pred[..., t:t+1, :]
                    err_t = myloss(_pred_t.reshape(_batch, -1), _yy_t.reshape(_batch, -1)).item()
                    sample_timestep_errors.append(err_t)
                
                timestep_errors_list.append(sample_timestep_errors)
                
                # 计算整体 L2 误差
                l2_total = myloss(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                
                sample_details.append({
                    'index': b,
                    'l2_total': l2_total,
                    'segment_errors': sample_errors,
                    'timestep_errors': sample_timestep_errors,
                    'yy': yy.squeeze().cpu().numpy(),
                    'pred': pred.squeeze().cpu().numpy(),
                    'pred_h': pred_h.squeeze().cpu().numpy(),
                    'u_b': u_b.squeeze().cpu().numpy(),
                })
        
        # 转换为 numpy 数组
        errors_for_talk = np.array(errors_for_talk)  # [n_samples, 9]
        timestep_errors_array = np.array(timestep_errors_list)  # [n_samples, t_train - init_t]
        
        # 保存分段误差
        seg_names = ['0.1-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', 
                     '3-3.5', '3.5-4', '4-4.5', '4.5-5']
        
        results = []
        for i, seg_name in enumerate(seg_names):
            results.append({
                'Segment': seg_name,
                'Mean_L2': errors_for_talk[:, i].mean(),
                'Std_L2': errors_for_talk[:, i].std(),
                'Max_L2': errors_for_talk[:, i].max(),
                'Min_L2': errors_for_talk[:, i].min(),
            })
        
        df = pd.DataFrame(results)
        df.to_csv(f'{name}_segment_errors.csv', index=False)
        print(f"\n{name} - Segment errors:")
        print(df.to_string(index=False))
        
        # 保存时间步误差
        timestep_mean = timestep_errors_array.mean(axis=0)
        timestep_std = timestep_errors_array.std(axis=0)
        timestep_max = timestep_errors_array.max(axis=0)
        timestep_min = timestep_errors_array.min(axis=0)
        
        timestep_results = []
        for t_idx, t in enumerate(range(init_t, t_train)):
            timestep_results.append({
                'Timestep': t,
                'Mean_L2': timestep_mean[t_idx],
                'Std_L2': timestep_std[t_idx],
                'Max_L2': timestep_max[t_idx],
                'Min_L2': timestep_min[t_idx],
            })
        
        df_timestep = pd.DataFrame(timestep_results)
        df_timestep.to_csv(f'{name}_timestep_errors.csv', index=False)
        print(f"\n{name} - Timestep errors saved")
        print(f"  Timesteps: {len(timestep_results)}")
        print(f"  Mean L2 range: [{timestep_mean.min():.6e}, {timestep_mean.max():.6e}]")
        
        # 保存统计数据
        timestep_errors_all[name] = {
            'timesteps': list(range(init_t, t_train)),
            'mean': timestep_mean,
            'std': timestep_std,
            'max': timestep_max,
            'min': timestep_min,
            'all_samples': timestep_errors_array,
        }
        
        timestep_errors_per_sample_all[name] = {
            'timesteps': list(range(init_t, t_train)),
            'samples': [
                {
                    'index': d['index'],
                    'timestep_errors': d['timestep_errors'],
                }
                for d in sample_details
            ],
        }
        
        # 样本选择逻辑 (仅对 unseen_extend)
        if name == 'unseen_extend':
            train_seg_start, train_seg_end = 5, 51
            
            # 计算训练阶段误差
            for sample in sample_details:
                yy = sample['yy']
                pred = sample['pred']
                
                yy_train = yy[..., train_seg_start:train_seg_end]
                pred_train = pred[..., train_seg_start:train_seg_end]
                
                l2_diff = np.sqrt(np.sum((pred_train - yy_train) ** 2))
                l2_target = np.sqrt(np.sum(yy_train ** 2))
                sample['l2_train'] = l2_diff / l2_target
            
            # 排序
            sorted_by_train = sorted(sample_details, key=lambda x: x['l2_train'])
            sorted_by_total = sorted(sample_details, key=lambda x: x['l2_total'])
            
            # 选择样本
            selected = []
            selected_indices = set()
            
            # 按训练阶段误差选择 8 个
            percentiles_train = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
            n_total = len(sorted_by_train)
            
            for p in percentiles_train:
                idx = min(int(n_total * p), n_total - 1)
                sample = sorted_by_train[idx].copy()
                sample['selection_type'] = 'train_best'
                sample['percentile'] = f'{int(p * 100)}%'
                sample['rank_train'] = idx
                
                if sample['index'] not in selected_indices:
                    selected.append(sample)
                    selected_indices.add(sample['index'])
            
            # 按整体误差选择 2 个
            count_total = 0
            for idx, sample in enumerate(sorted_by_total):
                if sample['index'] not in selected_indices:
                    sample_copy = sample.copy()
                    sample_copy['selection_type'] = 'total_best'
                    sample_copy['percentile'] = f'total_{idx}'
                    sample_copy['rank_total'] = idx
                    selected.append(sample_copy)
                    selected_indices.add(sample['index'])
                    count_total += 1
                    if count_total >= 2:
                        break
            
            selected_samples_all[name] = selected
            
            # 打印信息
            print(f"\n{name} - Selected samples ({len(selected)}):")
            print(f"  By train phase error (0.1-1.0):")
            for s in selected:
                if s['selection_type'] == 'train_best':
                    print(f"    {s['percentile']:>5} (rank {s['rank_train']:>3}): "
                          f"idx={s['index']}, l2_train={s['l2_train']:.6e}, "
                          f"l2_total={s['l2_total']:.6e}")
            
            print(f"  By total error:")
            for s in selected:
                if s['selection_type'] == 'total_best':
                    print(f"    {s['percentile']:>12} (rank {s['rank_total']:>3}): "
                          f"idx={s['index']}, l2_train={s['l2_train']:.6e}, "
                          f"l2_total={s['l2_total']:.6e}")
    
    # 保存结果
    with open('selected_samples_for_visualization.pkl', 'wb') as f:
        pickle.dump(selected_samples_all, f)
    
    with open('timestep_errors.pkl', 'wb') as f:
        pickle.dump(timestep_errors_all, f)
    
    with open('timestep_errors_per_sample.pkl', 'wb') as f:
        pickle.dump(timestep_errors_per_sample_all, f)
    
    print("\n" + "="*70)
    print("Files saved:")
    print("  ✅ selected_samples_for_visualization.pkl")
    print("  ✅ timestep_errors.pkl")
    print("  ✅ timestep_errors_per_sample.pkl")
    print("  ✅ {name}_segment_errors.csv")
    print("  ✅ {name}_timestep_errors.csv")
    print("="*70)
    # endregion


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = ArgumentParser(description='Test extend with u_h lifting')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='test_extend')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Pretrained model path')
    parser.add_argument('--load_lr', action='store_true')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.FullLoader)
    
    # 运行测试
    testII_with_lifting(config, args)