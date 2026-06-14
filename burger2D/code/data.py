#
#
# # import h5py
# # import scipy.io as sio
# #
# # filepath = r'../data/burgers2d.mat'
# #
# # with h5py.File(filepath, 'r') as f:
# #     u = f['u_unif'][:]
# #     print(f"原始 shape: {u.shape}")
# #
# # # 转置（如果需要）
# # # if u.shape[0] == 11:
# # #     u = u.transpose(3, 1, 2, 0)
# # #     print(f"转置后 shape: {u.shape}")
# #
# # # 保存
# # sio.savemat(r'../data/burgers2d_new.mat', {'data': u})
# # print("已保存")
#
#
#
# import h5py
# import scipy.io as sio
#
# filepath = r'../data/burgers2d_solved.h5'  # 修改为你的路径
#
# # # 先尝试 scipy（普通mat格式）
# # try:
# #     data = sio.loadmat(filepath)
# #     print("格式: MATLAB v5/v7")
# #     print("Keys:", [k for k in data.keys() if not k.startswith('__')])
# #     for key in data.keys():
# #         if not key.startswith('__'):
# #             val = data[key]
# #             print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
# # except:
# #     # 如果失败，用 h5py（HDF5/v7.3格式）
# #     print("格式: MATLAB v7.3 (HDF5)")
# #     with h5py.File(filepath, 'r') as f:
# #         u = f['u_unif'][:]
# #         # print("Keys:", list(f.keys()))
# #         # for key in f.keys():
# #         #     val = f[key]
# #         #     print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
# # sio.savemat(r'../data/burgers2d_new.mat', {'u_unif': u})
#
#
#
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
#
# # filepath = r'burgers2d.mat'  # 修改为你的路径
#
# # 读取数据
# with h5py.File(filepath, 'r') as f:
#     u_cgl = f['data'][:]  # (1100, 201, 201, 11)
#
# # 注意：HDF5读取可能需要转置，检查一下维度顺序
# print(f"u_cgl shape: {u_cgl.shape}")
#
# # 如果维度是 (11, 201, 201, 1100)，需要转置
# if u_cgl.shape[0] == 11:
#     u_cgl = u_cgl.transpose(3, 1, 2, 0)  # -> (1100, 201, 201, 11)
#     print(f"转置后 shape: {u_cgl.shape}")
#
# # 随机选取3个样本
# np.random.seed(42)
# sample_indices = np.random.choice(u_cgl.shape[0], 3, replace=False)
# print(f"选取样本索引: {sample_indices}")
#
# # 时间步索引（注意：共11个时间步，索引0-10，所以用10代替11）
# time_steps = [0, 1, 2, 8, 10]
# time_labels = ['t=0', 't=3', 't=5', 't=8', 't=10']
#
# # 绘图：3行（样本）× 5列（时间步）
# fig, axes = plt.subplots(3, 5, figsize=(15, 9))
# sub = 5
# for i, sample_idx in enumerate(sample_indices):
#     for j, t_idx in enumerate(time_steps):
#         ax = axes[i, j]
#         im = ax.imshow(u_cgl[sample_idx, ::sub, ::sub, t_idx], cmap='jet', aspect='equal')
#         ax.set_title(f'Sample {sample_idx}, {time_labels[j]}')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.colorbar(im, ax=ax, fraction=0.046)
#
# plt.suptitle('2D Burgers Equation - u_cgl', fontsize=14)
# plt.tight_layout()
# plt.savefig('burgers2d_samples.png', dpi=150)
# plt.show()

import scipy.io as sio
import numpy as np

# 加载数据
data = sio.loadmat(r"../data/burgers2d_new.mat")

# 查看所有键
print("Keys:", data.keys())
print()

# 查看每个变量的形状和类型
for key in data.keys():
    if not key.startswith('__'):
        val = data[key]
        print(f"{key}:")
        print(f"  shape: {val.shape}")
        print(f"  dtype: {val.dtype}")
        if val.ndim <= 2 and val.size < 20:
            print(f"  value: {val}")
        elif val.ndim > 0:
            print(f"  min: {val.min()}, max: {val.max()}")
        print()