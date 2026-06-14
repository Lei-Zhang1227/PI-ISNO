import h5py
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
filepath = r"../data/burgers2d_solved.h5"

with h5py.File(filepath, 'r') as f:
    sample_keys = [k for k in f.keys() if k.isdigit()]
    print(f"总共 {len(sample_keys)} 个样本")

    # 先看一下实际shape
    test_key = sample_keys[0]
    print(f"data shape: {f[test_key]['data'].shape}")
    print(f"t shape: {f[test_key]['grid']['t'].shape}")
    print(f"x shape: {f[test_key]['grid']['x'].shape}")
    print(f"y shape: {f[test_key]['grid']['y'].shape}")

    np.random.seed(42)
    selected = np.random.choice(sample_keys, size=min(3, len(sample_keys)), replace=False)

    samples = {}
    for key in selected:
        samples[key] = {
            'data': f[key]['data'][:],
            't': f[key]['grid']['t'][:],
            'x': f[key]['grid']['x'][:],
            'y': f[key]['grid']['y'][:]
        }

# 检查实际维度
for key, sample in samples.items():
    print(
        f"Sample {key}: data={sample['data'].shape}, t={len(sample['t'])}, x={len(sample['x'])}, y={len(sample['y'])}")
    break

# 可视化
time_indices = [0, 9, 39, 69, -1]
time_labels = ['t=0', 't=10', 't=40', 't=70', 't=end']

fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for row, (key, sample) in enumerate(samples.items()):
    x, y = sample['x'], sample['y']
    X, Y = np.meshgrid(x, y)
    data = sample['data']  # 先看看shape再决定怎么索引

    for col, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
        ax = axes[row, col]

        # 根据实际shape调整索引方式
        # 如果 data.shape = (101, 41, 41) -> u = data[tidx, :, :]
        # 如果 data.shape = (41, 41, 101) -> u = data[:, :, tidx]
        if data.shape[0] == len(sample['t']):
            u = data[tidx, :, :]
        else:
            u = data[:, :, tidx]

        # 确保 u 和 X, Y 维度匹配
        if u.shape != X.shape:
            u = u.T

        c = ax.pcolormesh(X, Y, u, cmap='RdBu_r', shading='auto')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax, fraction=0.046)

        if row == 0:
            ax.set_title(f'{tlabel} (t={sample["t"][tidx]:.3f})')
        if col == 0:
            ax.set_ylabel(f'Sample {key}')

plt.tight_layout()
plt.savefig('burgers2d_visualization.png', dpi=150)
plt.show()