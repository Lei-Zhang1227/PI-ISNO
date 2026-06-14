"""
DCT-I 两种实现的完整对比测试:
  1. dctI_SPFNO / idctI_SPFNO  (fft 版本)
  2. dctI_batch / idctI_batch   (rfft 版本)

测试内容:
  - 数值一致性
  - 逆变换精度 (round-trip)
  - 2D round-trip
  - 与 scipy 对比
  - 速度对比
"""

import torch
import numpy as np
import time
from scipy.fft import dct as scipy_dct


# ================================================================
# 实现 1: SPFNO 版本 (fft)
# ================================================================
def dctI_SPFNO(u):
    if not torch.is_tensor(u):
        u = torch.Tensor(u)
    Nx = u.shape[-1]
    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx - 1]], dim=-1)
    a = torch.fft.fft(V, dim=-1)[..., :Nx].real
    return a


def idctI_SPFNO(a):
    if not torch.is_tensor(a):
        a = torch.Tensor(a)
    Nx = a.shape[-1]
    V = torch.cat([a, a.flip(dims=[-1])[..., 1:Nx - 1]], dim=-1)
    u = torch.fft.ifft(V, dim=-1)[..., :Nx].real
    return u


# ================================================================
# 实现 2: batch 版本 (rfft)
# ================================================================
def dctI_batch(u, axis=-1):
    u = torch.moveaxis(u, axis, -1)
    N1 = u.shape[-1]
    N = N1 - 1
    y = torch.cat([u, u[..., 1:N].flip(dims=[-1])], dim=-1)
    Y = torch.fft.rfft(y, dim=-1)
    result = Y.real
    return torch.moveaxis(result, -1, axis)


def idctI_batch(c, axis=-1):
    c = torch.moveaxis(c, axis, -1)
    N1 = c.shape[-1]
    N = N1 - 1
    y = torch.cat([c, c[..., 1:N].flip(dims=[-1])], dim=-1)
    Y = torch.fft.rfft(y, dim=-1)
    result = Y.real / (2 * N)
    return torch.moveaxis(result, -1, axis)


# ================================================================
# 测试
# ================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    sizes = [(4, 65, 65), (4, 129, 129), (8, 64, 64)]

    for shape in sizes:
        print(f"{'='*70}")
        print(f"测试形状: {shape}")
        print(f"{'='*70}")
        x = torch.randn(*shape, device=device)

        # -------- 1. 正变换一致性 --------
        c_spfno = dctI_SPFNO(x)
        c_batch = dctI_batch(x, axis=-1)
        diff = torch.max(torch.abs(c_spfno - c_batch)).item()
        print(f"\n[正变换一致性] max diff: {diff:.2e}")

        # -------- 2. Round-trip axis=-1 --------
        err_sp = torch.max(torch.abs(x - idctI_SPFNO(dctI_SPFNO(x)))).item()
        err_bt = torch.max(torch.abs(x - idctI_batch(dctI_batch(x, -1), -1))).item()
        print(f"\n[Round-trip axis=-1]")
        print(f"  SPFNO: {err_sp:.2e}")
        print(f"  batch: {err_bt:.2e}")

        # -------- 3. Round-trip axis=-2 --------
        err_bt2 = torch.max(torch.abs(x - idctI_batch(dctI_batch(x, -2), -2))).item()
        x_t = x.transpose(-1, -2)
        err_sp2 = torch.max(torch.abs(x - idctI_SPFNO(dctI_SPFNO(x_t)).transpose(-1, -2))).item()
        print(f"\n[Round-trip axis=-2]")
        print(f"  SPFNO (手动转置): {err_sp2:.2e}")
        print(f"  batch (原生支持): {err_bt2:.2e}")

        # -------- 4. 2D Round-trip --------
        c2d = dctI_batch(dctI_batch(x, -1), -2)
        err_2d_bt = torch.max(torch.abs(x - idctI_batch(idctI_batch(c2d, -2), -1))).item()

        c2d_sp = dctI_SPFNO(x)
        c2d_sp = dctI_SPFNO(c2d_sp.transpose(-1, -2)).transpose(-1, -2)
        x_back_sp = idctI_SPFNO(c2d_sp.transpose(-1, -2)).transpose(-1, -2)
        x_back_sp = idctI_SPFNO(x_back_sp)
        err_2d_sp = torch.max(torch.abs(x - x_back_sp)).item()

        print(f"\n[2D Round-trip]")
        print(f"  SPFNO: {err_2d_sp:.2e}")
        print(f"  batch: {err_2d_bt:.2e}")

        # -------- 5. 与 scipy 对比 --------
        x_np = x[0].cpu().numpy()
        c_scipy = scipy_dct(x_np, type=1, axis=-1)
        err_sc_sp = np.max(np.abs(c_scipy - c_spfno[0].cpu().numpy()))
        err_sc_bt = np.max(np.abs(c_scipy - c_batch[0].cpu().numpy()))
        print(f"\n[scipy DCT-I 对比]")
        print(f"  scipy vs SPFNO: {err_sc_sp:.2e}")
        print(f"  scipy vs batch: {err_sc_bt:.2e}")

        # -------- 6. 速度对比 --------
        n_runs = 200

        # 预热
        for _ in range(10):
            _ = dctI_SPFNO(x)
            _ = dctI_batch(x, -1)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        def bench(fn, n=n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1000

        t_sp_f = bench(lambda: dctI_SPFNO(x))
        t_bt_f = bench(lambda: dctI_batch(x, -1))
        t_sp_i = bench(lambda: idctI_SPFNO(c_spfno))
        t_bt_i = bench(lambda: idctI_batch(c_batch, -1))

        # 2D 拉普拉斯完整流程
        Nx, Ny = shape[-2], shape[-1]
        kx = torch.arange(Nx, device=device, dtype=x.dtype)
        ky = torch.arange(Ny, device=device, dtype=x.dtype)
        K2 = -(kx * torch.pi / 2.0).view(1, -1, 1) ** 2 \
             - (ky * torch.pi / 2.0).view(1, 1, -1) ** 2

        def lap_batch():
            h = dctI_batch(dctI_batch(x, -1), -2)
            return idctI_batch(idctI_batch(h * K2, -2), -1)

        def lap_spfno():
            h = dctI_SPFNO(x)
            h = dctI_SPFNO(h.transpose(-1, -2)).transpose(-1, -2)
            h = h * K2
            h = idctI_SPFNO(h.transpose(-1, -2)).transpose(-1, -2)
            return idctI_SPFNO(h)

        t_bt_2d = bench(lap_batch)
        t_sp_2d = bench(lap_spfno)

        print(f"\n[速度对比] ({n_runs} 次平均)")
        print(f"  {'操作':<20} {'SPFNO(fft)':<16} {'batch(rfft)':<16} {'加速比':<10}")
        print(f"  {'-'*62}")
        print(f"  {'DCT-I 正变换':<20} {t_sp_f:>8.4f} ms    {t_bt_f:>8.4f} ms    {t_sp_f/t_bt_f:>5.2f}x")
        print(f"  {'IDCT-I 逆变换':<20} {t_sp_i:>8.4f} ms    {t_bt_i:>8.4f} ms    {t_sp_i/t_bt_i:>5.2f}x")
        print(f"  {'2D 拉普拉斯':<20} {t_sp_2d:>8.4f} ms    {t_bt_2d:>8.4f} ms    {t_sp_2d/t_bt_2d:>5.2f}x")
        print()

    # -------- 7. 功能总结 --------
    print(f"{'='*70}")
    print("功能对比总结")
    print(f"{'='*70}")
    print(f"  {'特性':<30} {'SPFNO (fft)':<20} {'batch (rfft)':<20}")
    print(f"  {'-'*70}")
    print(f"  {'FFT 类型':<30} {'fft (复数)':<20} {'rfft (实数优化)':<20}")
    print(f"  {'任意 axis 支持':<30} {'否 (仅 axis=-1)':<20} {'是':<20}")
    print(f"  {'自动归一化':<30} {'是 (ifft 内置)':<20} {'手动 / (2N)':<20}")
    print(f"  {'适合训练循环':<30} {'需转置处理多轴':<20} {'直接使用':<20}")