import scipy.io as sio
import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import dct, idct
import h5py
import time

# 加载数据
data = sio.loadmat(r"../data/burgers2d_new.mat")['data']
print(f"Data shape: {data.shape}")  # (1100, 201, 201, 11)

n_samples, Nx_data, Ny_data, Nt_data = data.shape

# 参数
nu = 0.1
T = 1.0
Nx, Ny = 128, 128
Lx, Ly = 2.0, 2.0
Nt_save = 101  # 保存的时间步数

# 网格 (半网格 cell-centered)
x = -1 + Lx * (np.arange(Nx) + 0.5) / Nx
y = -1 + Ly * (np.arange(Ny) + 0.5) / Ny
t_eval = np.linspace(0, T, Nt_save)

# 波数
kx = np.pi * np.arange(Nx) / Lx
ky = np.pi * np.arange(Ny) / Ly
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX ** 2 + KY ** 2


def dct2(f):
    return dct(dct(f, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(f):
    return idct(idct(f, axis=0, norm='ortho'), axis=1, norm='ortho')


def spectral_grad(f):
    Nx, Ny = f.shape

    # x方向偶延拓
    f_ext = np.vstack([f, np.flipud(f)])
    kx_ext = np.concatenate([np.arange(Nx), [0], np.arange(-Nx + 1, 0)]) * np.pi / Lx
    fx_ext = np.real(np.fft.ifft(1j * kx_ext[:, None] * np.fft.fft(f_ext, axis=0), axis=0))
    fx = fx_ext[:Nx, :]

    # y方向偶延拓
    f_ext = np.hstack([f, np.fliplr(f)])
    ky_ext = np.concatenate([np.arange(Ny), [0], np.arange(-Ny + 1, 0)]) * np.pi / Ly
    fy_ext = np.real(np.fft.ifft(1j * ky_ext[None, :] * np.fft.fft(f_ext, axis=1), axis=1))
    fy = fy_ext[:, :Ny]

    return fx, fy


def rhs(t, u_flat):
    u = u_flat.reshape(Nx, Ny)
    ux, uy = spectral_grad(u)
    u_hat = dct2(u)
    lap_u = idct2(-K2 * u_hat)
    du = -u * ux - u * uy + nu * lap_u
    return du.flatten()


def extract_data(sample_idx, tidx):
    ix = np.linspace(0, Nx_data - 1, Nx, dtype=int)
    iy = np.linspace(0, Ny_data - 1, Ny, dtype=int)
    return data[sample_idx][np.ix_(ix, iy, [tidx])][:, :, 0]


# 输出文件
output_file = r"../data/burgers2d_spectral.h5"

print(f"开始求解 {n_samples} 个样本...")
print(f"分辨率: {Nx}x{Ny}, 时间步: {Nt_save}")
print("=" * 60)

total_start = time.time()

with h5py.File(output_file, 'w') as f:
    for sample_idx in range(n_samples):
        sample_start = time.time()

        # 提取初始条件
        u0 = extract_data(sample_idx, 0)

        # 求解
        sol = solve_ivp(rhs, (0, T), u0.flatten(), method='RK23', t_eval=t_eval,
                        rtol=1e-5, atol=1e-7)

        if sol.success:
            # 整理数据 (Nt, Nx, Ny, 1) -> 只有u分量
            U = sol.y.T.reshape(Nt_save, Nx, Ny)
            # 扩展为 (Nt, Nx, Ny, 1) 格式，如果需要两个分量可以改成 (Nt, Nx, Ny, 2)
            U_out = U[:, :, :, np.newaxis].astype(np.float32)

            # 写入 HDF5
            key = f"{sample_idx:04d}"
            grp = f.create_group(key)
            grp.create_dataset('data', data=U_out, dtype='float32')

            grid_grp = grp.create_group('grid')
            grid_grp.create_dataset('t', data=t_eval.astype(np.float32), dtype='float32')
            grid_grp.create_dataset('x', data=x.astype(np.float32), dtype='float32')
            grid_grp.create_dataset('y', data=y.astype(np.float32), dtype='float32')

            sample_time = time.time() - sample_start
            elapsed = time.time() - total_start
            eta = elapsed / (sample_idx + 1) * (n_samples - sample_idx - 1)

            print(f"\rSample {sample_idx + 1:4d}/{n_samples} | "
                  f"本次: {sample_time:.2f}s | "
                  f"已用: {elapsed / 60:.1f}min | "
                  f"剩余: {eta / 60:.1f}min | "
                  f"|u|_max: {np.abs(U).max():.4f}", end='')
        else:
            print(f"\nSample {sample_idx} 求解失败: {sol.message}")

total_time = time.time() - total_start
print(f"\n\n完成! 总用时: {total_time / 60:.1f} 分钟")
print(f"输出文件: {output_file}")

# 验证输出文件
print("\n验证输出文件结构:")
with h5py.File(output_file, 'r') as f:
    print(f"样本数: {len(f.keys())}")
    key = '0000'
    print(f"Name: {key}/data")
    print(f"  Dataset shape: {f[key]['data'].shape}")
    print(f"  Dataset dtype: {f[key]['data'].dtype}")
    print(f"Name: {key}/grid/t")
    print(f"  Dataset shape: {f[key]['grid']['t'].shape}")
    print(f"Name: {key}/grid/x")
    print(f"  Dataset shape: {f[key]['grid']['x'].shape}")
    print(f"Name: {key}/grid/y")
    print(f"  Dataset shape: {f[key]['grid']['y'].shape}")