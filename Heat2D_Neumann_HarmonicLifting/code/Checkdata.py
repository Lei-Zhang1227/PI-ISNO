"""
快速检查HDF5数据的维度
"""
import h5py
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
from pathlib import Path

filepath = str(CASE_ROOT / 'data' / 'heat2d_param_bc.h5')  # 修改为你的路径

with h5py.File(filepath, 'r') as f:
    # 检查第一个样本
    key = '0000'
    
    print("="*60)
    print(f"Checking sample: {key}")
    print("="*60)
    
    # 数据维度
    data = f[f'{key}/data']
    print(f"\nData shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    
    # Grid维度
    if f'{key}/grid/x' in f:
        grid_x = f[f'{key}/grid/x'][:]
        grid_y = f[f'{key}/grid/y'][:]
        grid_t = f[f'{key}/grid/t'][:]
        print(f"\nGrid shapes:")
        print(f"  grid_x: {grid_x.shape}, range: [{grid_x.min():.3f}, {grid_x.max():.3f}]")
        print(f"  grid_y: {grid_y.shape}, range: [{grid_y.min():.3f}, {grid_y.max():.3f}]")
        print(f"  grid_t: {grid_t.shape}, range: [{grid_t.min():.3f}, {grid_t.max():.3f}]")
    
    # BC参数
    a = f[f'{key}/bc/a'][()]
    b = f[f'{key}/bc/b'][()]
    c = f[f'{key}/bc/c'][()]
    d = f[f'{key}/bc/d'][()]
    print(f"\nBC params:")
    print(f"  a = {a:.6f}")
    print(f"  b = {b:.6f}")
    print(f"  c = {c:.6f}")
    print(f"  d = {d:.6f}")
    
    # 降采样测试
    print(f"\n{'='*60}")
    print("Testing downsampling:")
    print(f"{'='*60}")
    
    for sub_x in [1, 2, 4]:
        data_sub = data[::sub_x, ::sub_x, ::1, :]
        grid_x_sub = grid_x[::sub_x]
        grid_y_sub = grid_y[::sub_x]
        
        print(f"\nsub_x = {sub_x}:")
        print(f"  data_sub shape: {data_sub.shape}")
        print(f"  grid_x_sub shape: {grid_x_sub.shape}")
        print(f"  grid_y_sub shape: {grid_y_sub.shape}")
        
        # 检查grid是否匹配data
        if data_sub.shape[0] == grid_x_sub.shape[0] and data_sub.shape[1] == grid_y_sub.shape[0]:
            print(f"  ✓ Grid dimensions match data")
        else:
            print(f"  ✗ MISMATCH: data ({data_sub.shape[0]}, {data_sub.shape[1]}) vs grid ({grid_x_sub.shape[0]}, {grid_y_sub.shape[0]})")

print("\n" + "="*60)
print("Check complete!")
print("="*60)