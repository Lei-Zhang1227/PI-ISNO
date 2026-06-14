"""
检查训练集（固定BC）和测试集（随机BC）的u_h是否有重复

目的：验证两个数据集是否共享相同的u_h，只是BC不同
"""
import h5py
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
from pathlib import Path
import torch
from tqdm import tqdm


def compute_uh_from_data(u, bc_params, only_initial=True):
    """
    从完整解u中提取u_h
    
    Args:
        u: [Nt, Nx, Ny, 1] 完整解
        bc_params: [a, b, c, d] BC参数
        only_initial: 是否只返回初始条件
    
    Returns:
        u_h: [Nx, Ny, 1] 或 [Nt, Nx, Ny, 1]
    """
    Nt, Nx, Ny = u.shape[:3]
    
    # 创建网格
    X, Y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny), indexing='ij')
    
    # 计算lifting
    a, b, c, d = bc_params
    u_b = a * X**2 + b * Y**2 + c * X + d * Y
    u_b = u_b[..., None]  # [Nx, Ny, 1]
    
    if only_initial:
        # 只提取初始条件
        u_0 = u[0, :, :, :]  # [Nx, Ny, 1]
        u_h_0 = u_0 - u_b    # [Nx, Ny, 1]
        return u_h_0
    else:
        # 提取整个时间序列
        u_b = u_b[None, :, :, :]  # [1, Nx, Ny, 1]
        u_h = u - u_b  # [Nt, Nx, Ny, 1]
        return u_h


def compute_uh_hash(u_h_0):
    """
    计算u_h初始条件的哈希值（用于快速比较）
    
    Args:
        u_h_0: [Nx, Ny, 1] 初始条件
    
    Returns:
        hash_values: tuple of floats
    """
    u = u_h_0[:, :, 0]  # [Nx, Ny]
    
    # 计算多个统计量作为"指纹"
    hash_values = [
        u.mean(),
        u.std(),
        u.min(),
        u.max(),
        np.percentile(u, 25),
        np.percentile(u, 50),
        np.percentile(u, 75),
        # 添加梯度信息
        np.abs(np.gradient(u, axis=0)).mean(),
        np.abs(np.gradient(u, axis=1)).mean(),
    ]
    
    return tuple(hash_values)


def check_uh_overlap_two_files(fixed_bc_filepath, random_bc_filepath, 
                                fixed_bc_indices, random_bc_indices, 
                                tolerance=1e-6):
    """
    检查两个不同文件中的u_h是否有重叠
    
    Args:
        fixed_bc_filepath: 固定BC数据集文件路径
        random_bc_filepath: 随机BC数据集文件路径
        fixed_bc_indices: 固定BC测试集样本索引
        random_bc_indices: 随机BC测试集样本索引
        tolerance: 判定相等的容差
    
    Returns:
        overlap_pairs: 重叠的样本对列表
        fixed_bc_stats: 固定BC数据集u_h统计
        random_bc_stats: 随机BC数据集u_h统计
    """
    print("="*80)
    print("检查两个数据集的u_h是否有重叠")
    print("="*80)
    print(f"\n固定BC数据集: {fixed_bc_filepath}")
    print(f"  样本索引: {fixed_bc_indices[0]}-{fixed_bc_indices[-1]} (共{len(fixed_bc_indices)}个)")
    print(f"\n随机BC数据集: {random_bc_filepath}")
    print(f"  样本索引: {random_bc_indices[0]}-{random_bc_indices[-1]} (共{len(random_bc_indices)}个)")
    print(f"\n相等容差: {tolerance}\n")
    
    # ================================================================
    # 1. 加载固定BC数据集的u_h初始条件
    # ================================================================
    print("加载固定BC数据集的初始条件（固定BC: -0.5, 0.5, 0, 0）...")
    fixed_uh_list = []
    fixed_uh_hashes = []
    fixed_bc = [-0.5, 0.5, 0.0, 0.0]
    
    with h5py.File(fixed_bc_filepath, 'r') as f:
        keys = sorted([k for k in f.keys() if k not in ['grid', 'params']])
        
        for idx in tqdm(fixed_bc_indices):
            key = keys[idx]
            u = np.array(f[f'{key}/data'])  # [Nt, Nx, Ny, 1]
            
            # 只提取初始条件的u_h
            u_h_0 = compute_uh_from_data(u, fixed_bc, only_initial=True)
            
            # 计算哈希
            u_h_hash = compute_uh_hash(u_h_0)
            
            fixed_uh_list.append(u_h_0)
            fixed_uh_hashes.append(u_h_hash)
    
    print(f"固定BC数据集加载完成，共{len(fixed_uh_list)}个样本\n")
    
    # ================================================================
    # 2. 加载随机BC数据集的u_h初始条件
    # ================================================================
    print("加载随机BC数据集的初始条件...")
    random_uh_list = []
    random_uh_hashes = []
    random_bc_list = []
    
    with h5py.File(random_bc_filepath, 'r') as f:
        keys = sorted([k for k in f.keys() if k not in ['grid', 'params']])
        
        for idx in tqdm(random_bc_indices):
            key = keys[idx]
            u = np.array(f[f'{key}/data'])  # [Nt, Nx, Ny, 1]
            
            # 读取BC参数
            if 'bc' in f[key]:
                a = f[f'{key}/bc/a'][()]
                b = f[f'{key}/bc/b'][()]
                c = f[f'{key}/bc/c'][()]
                d = f[f'{key}/bc/d'][()]
                bc = [a, b, c, d]
            else:
                # 如果没有BC参数，假设也是固定BC
                bc = [-0.5, 0.5, 0.0, 0.0]
                print(f"  Warning: 样本 {key} 没有BC参数，使用默认固定BC")
            
            # 只提取初始条件的u_h
            u_h_0 = compute_uh_from_data(u, bc, only_initial=True)
            
            # 计算哈希
            u_h_hash = compute_uh_hash(u_h_0)
            
            random_uh_list.append(u_h_0)
            random_uh_hashes.append(u_h_hash)
            random_bc_list.append(bc)
    
    print(f"随机BC数据集加载完成，共{len(random_uh_list)}个样本\n")
    
    # ================================================================
    # 3. 快速哈希比较
    # ================================================================
    print("="*80)
    print("步骤1: 快速哈希比较")
    print("="*80)
    print("使用统计指纹进行初步筛选...\n")
    
    potential_matches = []
    
    for i, random_hash in enumerate(tqdm(random_uh_hashes, desc="比较随机BC数据")):
        for j, fixed_hash in enumerate(fixed_uh_hashes):
            # 计算哈希值的差异
            hash_diff = np.abs(np.array(random_hash) - np.array(fixed_hash))
            max_diff = np.max(hash_diff)
            
            if max_diff < tolerance * 100:  # 哈希容差放宽100倍
                potential_matches.append((i, j, max_diff))
    
    print(f"找到 {len(potential_matches)} 对潜在匹配\n")
    
    # ================================================================
    # 4. 精确比较（逐元素）
    # ================================================================
    print("="*80)
    print("步骤2: 精确比较（逐元素）")
    print("="*80)
    
    overlap_pairs = []
    
    if len(potential_matches) > 0:
        print(f"对 {len(potential_matches)} 对潜在匹配进行精确验证...\n")
        
        for random_idx, fixed_idx, hash_diff in tqdm(potential_matches):
            random_uh = random_uh_list[random_idx]
            fixed_uh = fixed_uh_list[fixed_idx]
            
            # 计算逐元素差异
            diff = np.abs(random_uh - fixed_uh)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            if max_diff < tolerance:
                overlap_pairs.append({
                    'random_idx': random_bc_indices[random_idx],
                    'fixed_idx': fixed_bc_indices[fixed_idx],
                    'random_bc': random_bc_list[random_idx],
                    'fixed_bc': fixed_bc,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'hash_diff': hash_diff,
                })
                
                print(f"✓ 找到匹配!")
                print(f"  随机BC样本 {random_bc_indices[random_idx]} (BC={random_bc_list[random_idx]})")
                print(f"  固定BC样本 {fixed_bc_indices[fixed_idx]} (BC={fixed_bc})")
                print(f"  最大差异: {max_diff:.2e}")
                print(f"  平均差异: {mean_diff:.2e}\n")
    else:
        print("没有找到潜在匹配\n")
    
    # ================================================================
    # 5. 统计分析（初始条件）
    # ================================================================
    print("="*80)
    print("统计分析（初始条件 u_h(t=0)）")
    print("="*80)
    
    # 固定BC数据集u_h初始条件统计
    fixed_stats = {
        'mean': np.mean([uh[:, :, 0].mean() for uh in fixed_uh_list]),
        'std': np.mean([uh[:, :, 0].std() for uh in fixed_uh_list]),
        'min': np.mean([uh[:, :, 0].min() for uh in fixed_uh_list]),
        'max': np.mean([uh[:, :, 0].max() for uh in fixed_uh_list]),
    }
    
    # 随机BC数据集u_h初始条件统计
    random_stats = {
        'mean': np.mean([uh[:, :, 0].mean() for uh in random_uh_list]),
        'std': np.mean([uh[:, :, 0].std() for uh in random_uh_list]),
        'min': np.mean([uh[:, :, 0].min() for uh in random_uh_list]),
        'max': np.mean([uh[:, :, 0].max() for uh in random_uh_list]),
    }
    
    print("\n固定BC数据集 u_h(t=0) 统计:")
    print(f"  平均值: {fixed_stats['mean']:.6f}")
    print(f"  标准差: {fixed_stats['std']:.6f}")
    print(f"  最小值: {fixed_stats['min']:.6f}")
    print(f"  最大值: {fixed_stats['max']:.6f}")
    
    print("\n随机BC数据集 u_h(t=0) 统计:")
    print(f"  平均值: {random_stats['mean']:.6f}")
    print(f"  标准差: {random_stats['std']:.6f}")
    print(f"  最小值: {random_stats['min']:.6f}")
    print(f"  最大值: {random_stats['max']:.6f}")
    
    # ================================================================
    # 6. 初始条件复杂度比较
    # ================================================================
    print("\n" + "="*80)
    print("初始条件复杂度比较")
    print("="*80)
    
    fixed_ic_complexity = []
    for u_h_0 in fixed_uh_list:
        u = u_h_0[:, :, 0]  # [Nx, Ny]
        complexity = {
            'std': u.std(),
            'grad_x': np.abs(np.gradient(u, axis=0)).mean(),
            'grad_y': np.abs(np.gradient(u, axis=1)).mean(),
        }
        fixed_ic_complexity.append(complexity)
    
    random_ic_complexity = []
    for u_h_0 in random_uh_list:
        u = u_h_0[:, :, 0]  # [Nx, Ny]
        complexity = {
            'std': u.std(),
            'grad_x': np.abs(np.gradient(u, axis=0)).mean(),
            'grad_y': np.abs(np.gradient(u, axis=1)).mean(),
        }
        random_ic_complexity.append(complexity)
    
    fixed_ic_std = np.mean([c['std'] for c in fixed_ic_complexity])
    fixed_ic_grad = np.mean([c['grad_x'] + c['grad_y'] for c in fixed_ic_complexity])
    
    random_ic_std = np.mean([c['std'] for c in random_ic_complexity])
    random_ic_grad = np.mean([c['grad_x'] + c['grad_y'] for c in random_ic_complexity])
    
    print(f"\n固定BC数据集初始条件 u_h(t=0):")
    print(f"  平均标准差: {fixed_ic_std:.6f}")
    print(f"  平均梯度:   {fixed_ic_grad:.6f}")
    
    print(f"\n随机BC数据集初始条件 u_h(t=0):")
    print(f"  平均标准差: {random_ic_std:.6f}")
    print(f"  平均梯度:   {random_ic_grad:.6f}")
    
    print(f"\n复杂度比值（固定BC/随机BC）:")
    print(f"  标准差比: {fixed_ic_std / random_ic_std:.2f}")
    print(f"  梯度比:   {fixed_ic_grad / random_ic_grad:.2f}")
    
    # ================================================================
    # 7. 最终结论
    # ================================================================
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    
    if len(overlap_pairs) > 0:
        print(f"\n✓ 找到 {len(overlap_pairs)} 对重叠的u_h!")
        print(f"  这意味着这些样本共享相同的u_h，只是BC不同。")
        print(f"  重叠比例: 固定BC {len(overlap_pairs)/len(fixed_uh_list)*100:.1f}%, 随机BC {len(overlap_pairs)/len(random_uh_list)*100:.1f}%")
    else:
        print(f"\n✗ 没有找到重叠的u_h")
        print(f"  两个数据集的u_h完全不同。")
    
    if abs(fixed_ic_std / random_ic_std - 1.0) > 0.2:
        print(f"\n⚠️  警告: 初始条件复杂度差异较大!")
        if fixed_ic_std > random_ic_std:
            print(f"  固定BC数据集的u_h(t=0)比随机BC复杂 {fixed_ic_std/random_ic_std:.1f}倍")
            print(f"  这可能解释为什么固定BC测试误差更大")
        else:
            print(f"  随机BC数据集的u_h(t=0)比固定BC复杂 {random_ic_std/fixed_ic_std:.1f}倍")
    else:
        print(f"\n✓ 初始条件复杂度相似")
        print(f"  差异在20%以内，可以认为是相同难度")
    
    print("\n" + "="*80)
    
    return overlap_pairs, fixed_stats, random_stats


def check_uh_overlap(filepath, train_indices, test_indices, tolerance=1e-6):
    """
    检查两个数据集的u_h是否有重叠
    
    Args:
        filepath: HDF5文件路径
        train_indices: 训练集样本索引（例如100-200）
        test_indices: 测试集样本索引（例如0-100）
        tolerance: 判定相等的容差
    
    Returns:
        overlap_pairs: 重叠的样本对列表
        train_uh_stats: 训练集u_h统计
        test_uh_stats: 测试集u_h统计
    """
    print("="*80)
    print("检查训练集（固定BC）和测试集（随机BC）的u_h是否有重叠")
    print("="*80)
    print(f"\n文件路径: {filepath}")
    print(f"训练集索引: {train_indices[0]}-{train_indices[-1]} (共{len(train_indices)}个)")
    print(f"测试集索引: {test_indices[0]}-{test_indices[-1]} (共{len(test_indices)}个)")
    print(f"相等容差: {tolerance}\n")
    
    with h5py.File(filepath, 'r') as f:
        keys = sorted([k for k in f.keys() if k not in ['grid', 'params']])
        
        # ================================================================
        # 1. 加载训练集的u_h（固定BC）
        # ================================================================
        print("加载训练集数据（固定BC: -0.5, 0.5, 0, 0）...")
        train_uh_list = []
        train_uh_hashes = []
        train_bc = [-0.5, 0.5, 0.0, 0.0]
        
        for idx in tqdm(train_indices):
            key = keys[idx]
            u = np.array(f[f'{key}/data'])  # [Nt, Nx, Ny, 1]
            
            # 提取u_h
            u_h = compute_uh_from_data(u, train_bc)
            
            # 计算哈希（用于快速比较）
            u_h_hash = compute_uh_hash(u_h)
            
            train_uh_list.append(u_h)
            train_uh_hashes.append(u_h_hash)
        
        print(f"训练集加载完成，共{len(train_uh_list)}个样本\n")
        
        # ================================================================
        # 2. 加载测试集的u_h（随机BC）
        # ================================================================
        print("加载测试集数据（随机BC）...")
        test_uh_list = []
        test_uh_hashes = []
        test_bc_list = []
        
        for idx in tqdm(test_indices):
            key = keys[idx]
            u = np.array(f[f'{key}/data'])  # [Nt, Nx, Ny, 1]
            
            # 读取BC参数
            if 'bc' in f[key]:
                a = f[f'{key}/bc/a'][()]
                b = f[f'{key}/bc/b'][()]
                c = f[f'{key}/bc/c'][()]
                d = f[f'{key}/bc/d'][()]
                bc = [a, b, c, d]
            else:
                # 如果没有BC参数，假设也是固定BC
                bc = [-0.5, 0.5, 0.0, 0.0]
            
            # 提取u_h
            u_h = compute_uh_from_data(u, bc)
            
            # 计算哈希
            u_h_hash = compute_uh_hash(u_h)
            
            test_uh_list.append(u_h)
            test_uh_hashes.append(u_h_hash)
            test_bc_list.append(bc)
        
        print(f"测试集加载完成，共{len(test_uh_list)}个样本\n")
    
    # ================================================================
    # 3. 快速哈希比较
    # ================================================================
    print("="*80)
    print("步骤1: 快速哈希比较")
    print("="*80)
    print("使用统计指纹进行初步筛选...\n")
    
    potential_matches = []
    
    for i, test_hash in enumerate(tqdm(test_uh_hashes, desc="比较测试集")):
        for j, train_hash in enumerate(train_uh_hashes):
            # 计算哈希值的差异
            hash_diff = np.abs(np.array(test_hash) - np.array(train_hash))
            max_diff = np.max(hash_diff)
            
            if max_diff < tolerance * 100:  # 哈希容差放宽100倍
                potential_matches.append((i, j, max_diff))
    
    print(f"找到 {len(potential_matches)} 对潜在匹配\n")
    
    # ================================================================
    # 4. 精确比较（逐元素）
    # ================================================================
    print("="*80)
    print("步骤2: 精确比较（逐元素）")
    print("="*80)
    
    overlap_pairs = []
    
    if len(potential_matches) > 0:
        print(f"对 {len(potential_matches)} 对潜在匹配进行精确验证...\n")
        
        for test_idx, train_idx, hash_diff in tqdm(potential_matches):
            test_uh = test_uh_list[test_idx]
            train_uh = train_uh_list[train_idx]
            
            # 计算逐元素差异
            diff = np.abs(test_uh - train_uh)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            if max_diff < tolerance:
                overlap_pairs.append({
                    'test_idx': test_indices[test_idx],
                    'train_idx': train_indices[train_idx],
                    'test_bc': test_bc_list[test_idx],
                    'train_bc': train_bc,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'hash_diff': hash_diff,
                })
                
                print(f"✓ 找到匹配!")
                print(f"  测试样本 {test_indices[test_idx]} (BC={test_bc_list[test_idx]})")
                print(f"  训练样本 {train_indices[train_idx]} (BC={train_bc})")
                print(f"  最大差异: {max_diff:.2e}")
                print(f"  平均差异: {mean_diff:.2e}\n")
    else:
        print("没有找到潜在匹配\n")
    
    # ================================================================
    # 5. 统计分析
    # ================================================================
    print("="*80)
    print("统计分析")
    print("="*80)
    
    # 训练集u_h统计
    train_uh_array = np.array([u_h.flatten() for u_h in train_uh_list])
    train_stats = {
        'mean': np.mean([uh.mean() for uh in train_uh_list]),
        'std': np.mean([uh.std() for uh in train_uh_list]),
        'min': np.mean([uh.min() for uh in train_uh_list]),
        'max': np.mean([uh.max() for uh in train_uh_list]),
    }
    
    # 测试集u_h统计
    test_uh_array = np.array([u_h.flatten() for u_h in test_uh_list])
    test_stats = {
        'mean': np.mean([uh.mean() for uh in test_uh_list]),
        'std': np.mean([uh.std() for uh in test_uh_list]),
        'min': np.mean([uh.min() for uh in test_uh_list]),
        'max': np.mean([uh.max() for uh in test_uh_list]),
    }
    
    print("\n训练集 u_h 统计:")
    print(f"  平均值: {train_stats['mean']:.6f}")
    print(f"  标准差: {train_stats['std']:.6f}")
    print(f"  最小值: {train_stats['min']:.6f}")
    print(f"  最大值: {train_stats['max']:.6f}")
    
    print("\n测试集 u_h 统计:")
    print(f"  平均值: {test_stats['mean']:.6f}")
    print(f"  标准差: {test_stats['std']:.6f}")
    print(f"  最小值: {test_stats['min']:.6f}")
    print(f"  最大值: {test_stats['max']:.6f}")
    
    # ================================================================
    # 6. 初始条件复杂度比较
    # ================================================================
    print("\n" + "="*80)
    print("初始条件复杂度比较")
    print("="*80)
    
    train_ic_complexity = []
    for u_h in train_uh_list:
        u_h_0 = u_h[0, :, :, 0]  # 初始条件
        complexity = {
            'std': u_h_0.std(),
            'grad_x': np.abs(np.gradient(u_h_0, axis=0)).mean(),
            'grad_y': np.abs(np.gradient(u_h_0, axis=1)).mean(),
        }
        train_ic_complexity.append(complexity)
    
    test_ic_complexity = []
    for u_h in test_uh_list:
        u_h_0 = u_h[0, :, :, 0]
        complexity = {
            'std': u_h_0.std(),
            'grad_x': np.abs(np.gradient(u_h_0, axis=0)).mean(),
            'grad_y': np.abs(np.gradient(u_h_0, axis=1)).mean(),
        }
        test_ic_complexity.append(complexity)
    
    train_ic_std = np.mean([c['std'] for c in train_ic_complexity])
    train_ic_grad = np.mean([c['grad_x'] + c['grad_y'] for c in train_ic_complexity])
    
    test_ic_std = np.mean([c['std'] for c in test_ic_complexity])
    test_ic_grad = np.mean([c['grad_x'] + c['grad_y'] for c in test_ic_complexity])
    
    print(f"\n训练集初始条件 u_h(t=0):")
    print(f"  平均标准差: {train_ic_std:.6f}")
    print(f"  平均梯度:   {train_ic_grad:.6f}")
    
    print(f"\n测试集初始条件 u_h(t=0):")
    print(f"  平均标准差: {test_ic_std:.6f}")
    print(f"  平均梯度:   {test_ic_grad:.6f}")
    
    print(f"\n复杂度比值（训练/测试）:")
    print(f"  标准差比: {train_ic_std / test_ic_std:.2f}")
    print(f"  梯度比:   {train_ic_grad / test_ic_grad:.2f}")
    
    # ================================================================
    # 7. 最终结论
    # ================================================================
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    
    if len(overlap_pairs) > 0:
        print(f"\n✓ 找到 {len(overlap_pairs)} 对重叠的u_h!")
        print(f"  这意味着这些样本共享相同的u_h，只是BC不同。")
        print(f"  重叠比例: 训练集 {len(overlap_pairs)/len(train_uh_list)*100:.1f}%, 测试集 {len(overlap_pairs)/len(test_uh_list)*100:.1f}%")
    else:
        print(f"\n✗ 没有找到重叠的u_h")
        print(f"  训练集和测试集的u_h完全不同。")
    
    if abs(train_ic_std / test_ic_std - 1.0) > 0.2:
        print(f"\n⚠️  警告: 初始条件复杂度差异较大!")
        if train_ic_std > test_ic_std:
            print(f"  训练集的u_h(t=0)比测试集复杂 {train_ic_std/test_ic_std:.1f}倍")
            print(f"  这可能解释为什么训练集测试误差更大")
        else:
            print(f"  测试集的u_h(t=0)比训练集复杂 {test_ic_std/train_ic_std:.1f}倍")
    else:
        print(f"\n✓ 初始条件复杂度相似")
        print(f"  差异在20%以内，可以认为是相同难度")
    
    print("\n" + "="*80)
    
    return overlap_pairs, train_stats, test_stats


if __name__ == '__main__':
    # 配置
    # 固定BC数据集
    fixed_bc_filepath = str(CASE_ROOT / 'data' / 'heat2d_neumann_1100.h5')
    fixed_bc_indices = list(range(0, 100))  # 固定BC测试集：样本0-100
    
    # 随机BC数据集  
    random_bc_filepath = str(CASE_ROOT / 'data' / 'heat2d_bc_gen_test.h5')
    random_bc_indices = list(range(0, 100))  # 随机BC测试集：样本0-100
    
    # 运行检查（需要修改函数以支持两个文件）
    overlap_pairs, train_stats, test_stats = check_uh_overlap_two_files(
        fixed_bc_filepath,
        random_bc_filepath,
        fixed_bc_indices, 
        random_bc_indices,
        tolerance=1e-6
    )
    
    # 保存结果
    import pickle
    with open('uh_overlap_check_results.pkl', 'wb') as f:
        pickle.dump({
            'overlap_pairs': overlap_pairs,
            'fixed_bc_stats': train_stats,
            'random_bc_stats': test_stats,
        }, f)
    
    print(f"\n结果已保存到: uh_overlap_check_results.pkl")