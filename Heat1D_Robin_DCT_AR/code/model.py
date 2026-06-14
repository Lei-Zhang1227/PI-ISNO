'''
This project references the following open-source projects.
1. [SPFNO](https://github.com/liu-ziyuan-math/SPFNO) by Ziyuan Liu.
2. [spectral operator learning] (https://github.com/liu-ziyuan-math/spectral_operator_learning) by Ziyuan Liu.
3. [physics_informed FNO] (https://github.com/neuraloperator/physics_informed) by Zongyi Li.
4. [FNO] () by Zongyi Li.
5. [DCT] by
6. [FC-FNO] by Haydn Maust
'''
import torch.nn.functional as F
import torch.nn as nn
import functools
from transform import *

x2phi = functools.partial(Wrapper, [dct, cmp_robin])
phi2x = functools.partial(Wrapper, [icmp_robin, idct])
idctn = functools.partial(Wrapper, [idct])
dctn = functools.partial(Wrapper, [dct])
# 时间变换（无 BC）
t2phi = functools.partial(Wrapper, [dct])
phi2t = functools.partial(Wrapper, [idct])


class PseudoSpectra_heat(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bandwidth):
        super(PseudoSpectra_heat, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = modes
        self.bandwidth = bandwidth

        self.scale = (2 / (in_channels + out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(modes, in_channels, out_channels, bandwidth, dtype=torch.float64))

    def quasi_diag(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        # x : (batches, nx, features)
        batch_size, width, Nx = u.shape
        b = dctn(u, -1)
        out = torch.zeros(batch_size, self.out_channels, Nx, device=u.device, dtype=torch.float64)
        out[..., :self.degree] = self.quasi_diag(b[..., :self.degree + 2], self.weights)
        u = phi2x(out, -1)
        return u


class SOL_heat(nn.Module):
    def __init__(self, in_channel, modes, width, bandwidth):
        super(SOL_heat, self).__init__()
        self.modes = modes
        self.width = width
        self.conv0 = PseudoSpectra_heat(self.width, self.width, self.modes, bandwidth)
        self.conv1 = PseudoSpectra_heat(self.width, self.width, self.modes, bandwidth)
        self.conv2 = PseudoSpectra_heat(self.width, self.width, self.modes, bandwidth)
        self.conv3 = PseudoSpectra_heat(self.width, self.width, self.modes, bandwidth)
        self.conv4 = PseudoSpectra_heat(self.width, self.width, self.modes, bandwidth)
        self.convl = PseudoSpectra_heat(in_channel, self.width - in_channel, self.modes, bandwidth)
        self.w0 = nn.Conv1d(self.width, self.width, 1).double()  # better
        self.w1 = nn.Conv1d(self.width, self.width, 1).double()
        self.w2 = nn.Conv1d(self.width, self.width, 1).double()
        self.w3 = nn.Conv1d(self.width, self.width, 1).double()
        self.fc1 = nn.Linear(self.width, 128).double()
        self.fc2 = nn.Linear(128, 1).double()

    def acti(self, x):
        return F.gelu(x)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([x, self.acti(self.convl(x))], dim=1)
        x = x + self.acti(self.w0(x) + self.conv0(x))
        x = x + self.acti(self.w1(x) + self.conv1(x))
        x = x + self.acti(self.w2(x) + self.conv2(x))
        x = x + self.acti(self.w3(x) + self.conv3(x))
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = phi2x(x2phi(x, -2), -2)
        return x


class PseudoSpectra1d(nn.Module):
    '''
    (T, width, width, modes, bandwidth, triL)
    T=DCT
    in_channels=2
    modes=20
    width=50
    bandwidth=4
    '''

    def __init__(self, T, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super().__init__()

        self.T = T
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-1, 0)

        scale = 1 / (in_channels * out_channels)  # 1/50*50
        self.weights = nn.Parameter(scale * torch.rand(modes, in_channels, out_channels, bandwidth))

    def quasi_diag_mul(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        #  # u: (1100, 2, 4097)
        batch_size, _, Nx = u.shape
        b = self.T(u, self.X_dims)
        out = torch.zeros((batch_size, self.out_channels, Nx), device=u.device, dtype=u.dtype)  # 第一次是50-2=48
        b = F.pad(b, (self.triL, 0, 0, 0, 0, 0))  # 本来是个填充流程，但是由于 self.triL=0， 所以实质上没有填充的；
        out[..., :self.modes] = self.quasi_diag_mul(b[..., :self.modes + self.bandwidth - 1], self.weights)
        u = self.T.inv(out, self.X_dims)
        return u


class PseudoSpectra_heat_32(nn.Module):
    def __init__(self, in_channels, out_channels, modes, bandwidth):
        super(PseudoSpectra_heat_32, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = modes
        self.bandwidth = bandwidth

        self.scale = (2 / (in_channels + out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(modes, in_channels, out_channels, bandwidth, dtype=torch.float32))

    def quasi_diag(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        batch_size, width, Nx = u.shape
        b = dctn(u, -1)
        out = torch.zeros(batch_size, self.out_channels, Nx, device=u.device, dtype=torch.float)
        # 改这一行：degree + 2 → degree + bandwidth - 1
        out[..., :self.degree] = self.quasi_diag(b[..., :self.degree + self.bandwidth - 1], self.weights)
        u = phi2x(out, -1)
        return u


class SOL_heat_32(nn.Module):
    def __init__(self, in_channel, modes, width, bandwidth, out_channel=1):
        super(SOL_heat_32, self).__init__()
        self.modes = modes
        self.width = width
        self.conv0 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.conv1 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.conv2 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.conv3 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.conv4 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.convl = PseudoSpectra_heat_32(in_channel, self.width - in_channel, self.modes, bandwidth)
        self.w0 = nn.Conv1d(self.width, self.width, 1)  # better
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channel)

    def acti(self, x):
        return F.gelu(x)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([x, self.acti(self.convl(x))], dim=1)
        x = x + self.acti(self.w0(x) + self.conv0(x))
        x = x + self.acti(self.w1(x) + self.conv1(x))
        x = x + self.acti(self.w2(x) + self.conv2(x))
        x = x + self.acti(self.w3(x) + self.conv3(x))
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = phi2x(x2phi(x, -2), -2)
        return x


class SOL_heat_all(nn.Module):
    def __init__(self, in_channel, out_channel, modes, width, bandwidth):
        super(SOL_heat_32, self).__init__()
        self.modes = modes
        self.width = width
        self.conv0 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.conv1 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.conv2 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.conv3 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.conv4 = PseudoSpectra_heat_32(self.width, self.width, self.modes, bandwidth)
        self.convl = PseudoSpectra_heat_32(in_channel, self.width - in_channel, self.modes, bandwidth)
        self.w0 = nn.Conv1d(self.width, self.width, 1)  # better
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channel)

    def acti(self, x):
        return F.gelu(x)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([x, self.acti(self.convl(x))], dim=1)
        x = x + self.acti(self.w0(x) + self.conv0(x))
        x = x + self.acti(self.w1(x) + self.conv1(x))
        x = x + self.acti(self.w2(x) + self.conv2(x))
        x = x + self.acti(self.w3(x) + self.conv3(x))
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = phi2x(x2phi(x, -2), -2)
        return x


class PseudoSpectra_heat_2D(nn.Module):
    """
    时空谱卷积层
    空间：带 Robin BC 的切比雪夫变换
    时间：普通切比雪夫变换（CGL 节点）
    """

    def __init__(self, in_channels, out_channels, modes_x, modes_t, bandwidth_x=3, bandwidth_t=3):
        super(PseudoSpectra_heat_2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_t = modes_t
        self.bandwidth_x = bandwidth_x
        self.bandwidth_t = bandwidth_t

        self.scale = (2 / (in_channels + out_channels))

        # 权重：[modes_x, modes_t, in_channels, out_channels, bandwidth_x, bandwidth_t]
        self.weights = nn.Parameter(
            self.scale * torch.rand(modes_x, modes_t, in_channels, out_channels,
                                    bandwidth_x, bandwidth_t, dtype=torch.float32))

    def quasi_diag(self, x, weights):
        """
        x: [batch, in_channels, modes_x + bw_x - 1, modes_t + bw_t - 1]
        weights: [modes_x, modes_t, in_channels, out_channels, bw_x, bw_t]
        """
        # 在空间和时间维度展开
        x_unfold = x.unfold(-2, self.bandwidth_x, 1).unfold(-2, self.bandwidth_t, 1)
        # x_unfold: [batch, in_ch, modes_x, modes_t, bw_x, bw_t]
        return torch.einsum("bixyuv,xyiouv->boxy", x_unfold, weights)

    def forward(self, u):
        """
        u: [batch, in_channels, nx, nt]
        return: [batch, out_channels, nx, nt]
        """
        batch_size, width, Nx, Nt = u.shape

        # 空间变换（带 Robin BC）
        b = dctn(u, -2)  # 先对空间做 DCT
        b = t2phi(b, -1)  # 再对时间做 DCT

        # 输出初始化
        out = torch.zeros(batch_size, self.out_channels, Nx, Nt, device=u.device, dtype=u.dtype)

        # 截取需要的模式进行谱乘法
        x_end = self.modes_x + self.bandwidth_x - 1
        t_end = self.modes_t + self.bandwidth_t - 1
        out[..., :self.modes_x, :self.modes_t] = self.quasi_diag(
            b[..., :x_end, :t_end], self.weights
        )

        # 逆变换
        out = phi2t(out, -1)  # 时间逆变换
        out = phi2x(out, -2)  # 空间逆变换（自动满足 Robin BC）

        return out


class SOL_heat_2D(nn.Module):
    def __init__(self, in_channel, modes_x, modes_t, width, bandwidth_x=3, bandwidth_t=3):
        super(SOL_heat_2D, self).__init__()

        self.modes_x = modes_x
        self.modes_t = modes_t
        self.width = width
        self.in_channel = in_channel

        self.conv0 = PseudoSpectra_heat_2D(width, width, modes_x, modes_t, bandwidth_x, bandwidth_t)
        self.conv1 = PseudoSpectra_heat_2D(width, width, modes_x, modes_t, bandwidth_x, bandwidth_t)
        self.conv2 = PseudoSpectra_heat_2D(width, width, modes_x, modes_t, bandwidth_x, bandwidth_t)
        self.conv3 = PseudoSpectra_heat_2D(width, width, modes_x, modes_t, bandwidth_x, bandwidth_t)
        self.convl = PseudoSpectra_heat_2D(in_channel, width - in_channel, modes_x, modes_t, bandwidth_x, bandwidth_t)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def acti(self, x):
        return F.gelu(x)

    def forward(self, x):
        """
        x: [batch, nx, nt, in_channel]
        return: [batch, nx, nt, 1]
        """
        # permute: [batch, nx, nt, ch] -> [batch, ch, nx, nt]
        x = x.permute(0, 3, 1, 2)

        x = torch.cat([x, self.acti(self.convl(x))], dim=1)
        x = x + self.acti(self.w0(x) + self.conv0(x))
        x = x + self.acti(self.w1(x) + self.conv1(x))
        x = x + self.acti(self.w2(x) + self.conv2(x))
        x = x + self.acti(self.w3(x) + self.conv3(x))

        # permute: [batch, ch, nx, nt] -> [batch, nx, nt, ch]
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)

        # BC 投影，对 nx 维度 (dim=-3)
        x = phi2x(x2phi(x, -3), -3)

        return x


def cheb_derivative(u, device='cpu'):
    """计算切比雪夫导数"""
    Nx = u.shape[0]
    x = torch.cos(torch.pi * torch.arange(Nx, dtype=torch.float64, device=device) / (Nx - 1))
    c = torch.ones(Nx, dtype=torch.float64, device=device)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** torch.arange(Nx, dtype=torch.float64, device=device))
    X = x.unsqueeze(1).expand(Nx, Nx)
    dX = X - X.T
    D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + torch.eye(Nx, dtype=torch.float64, device=device))
    D = D - torch.diag(D.sum(dim=1))
    return torch.matmul(D, u)


if __name__ == '__main__':
    device = 'cuda'
    # 定义模型
    # degree = 40
    # width = 50
    # in_channel = 2
    # model = SOL_heat(in_channel, degree, width).to(device)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"--- set model. Total trainable parameters: {total_params}")
    #
    #
    # def count_parameters(layer):
    #     return sum(p.numel() for p in layer.parameters() if p.requires_grad)
    #
    #
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params}")
    # print("\nTrainable parameters for each layer/module:")
    # for name, layer in model.named_children():
    #     num_params = count_parameters(layer)
    #     print(f"{name}: {num_params} parameters")
    # dummy_input = torch.randn(20, 129, 2).to(device).double()
    # output = model(dummy_input)
    # print(f"\n最终输出: {output.shape}")
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 参数
    batch = 4
    nx = 129
    nt = 129
    in_channel = 3  # u0 + x + t
    modes_x = 40
    modes_t = 32
    width = 32

    print("=" * 60)
    print("模型配置")
    print("=" * 60)
    print(f"  batch: {batch}")
    print(f"  nx: {nx}, nt: {nt}")
    print(f"  modes_x: {modes_x}, modes_t: {modes_t}")
    print(f"  width: {width}")
    print(f"  in_channel: {in_channel}")

    # 创建模型
    model = SOL_heat_2D(
        in_channel=in_channel,
        modes_x=modes_x,
        modes_t=modes_t,
        width=width,
        bandwidth_x=3,
        bandwidth_t=3
    ).to(device)

    # 计算参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {params:,}")

    # 构造输入: [batch, nx, nt, 3]
    print("\n" + "=" * 60)
    print("构造输入")
    print("=" * 60)

    # u0: 初始条件，扩展到所有时间步
    u0 = torch.randn(batch, nx, 1, device=device).expand(-1, -1, nt)

    # x 坐标（CGL 节点）
    k_x = torch.arange(nx, device=device, dtype=torch.float32)
    x_grid = torch.cos(torch.pi * k_x / (nx - 1))
    x_grid = x_grid.view(1, -1, 1).expand(batch, -1, nt)

    # t 坐标（CGL 节点）
    k_t = torch.arange(nt, device=device, dtype=torch.float32)
    t_grid = (1 - torch.cos(torch.pi * k_t / (nt - 1))) / 2
    t_grid = t_grid.view(1, 1, -1).expand(batch, nx, -1)

    # 拼接: [batch, nx, nt, 3]
    x_input = torch.stack([u0, x_grid, t_grid], dim=-1)
    print(f"  输入形状: {x_input.shape}")
    print(
        f"  输入范围: u0=[{u0.min():.2f}, {u0.max():.2f}], x=[{x_grid.min():.2f}, {x_grid.max():.2f}], t=[{t_grid.min():.2f}, {t_grid.max():.2f}]")

    # 前向传播
    print("\n" + "=" * 60)
    print("前向传播")
    print("=" * 60)

    with torch.no_grad():
        out = model(x_input)

    print(f"  输出形状: {out.shape}")
    print(f"  输出范围: [{out.min():.4f}, {out.max():.4f}]")
    print(f"  输出均值: {out.mean():.4f}, 标准差: {out.std():.4f}")

    # 验证 Robin BC
    print("\n" + "=" * 60)
    print("Robin BC 验证")
    print("=" * 60)

    max_robin_right = 0
    max_robin_left = 0

    for b_idx in range(batch):
        for t_idx in [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]:
            u_test = out[b_idx, :, t_idx, 0].cpu().double()
            u_x = cheb_derivative(u_test, device='cpu')
            robin_right = abs(u_test[0] + u_x[0]).item()
            robin_left = abs(u_test[-1] - u_x[-1]).item()
            max_robin_right = max(max_robin_right, robin_right)
            max_robin_left = max(max_robin_left, robin_left)

    print(f"  右边界 |u + u'| 最大误差 @ x=1:  {max_robin_right:.2e}")
    print(f"  左边界 |u - u'| 最大误差 @ x=-1: {max_robin_left:.2e}")

    if max_robin_right < 1e-10 and max_robin_left < 1e-10:
        print("  ✅ Robin BC 满足！")
    else:
        print("  ⚠️ Robin BC 误差较大，需检查")

    # 测试反向传播
    print("\n" + "=" * 60)
    print("反向传播测试")
    print("=" * 60)

    x_input.requires_grad_(False)
    out = model(x_input)
    loss = out.mean()
    loss.backward()

    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    print(f"  Loss: {loss.item():.6f}")
    print(f"  梯度范数: {grad_norm:.6f}")
    print("  ✅ 反向传播正常！")

    # 内存占用
    print("\n" + "=" * 60)
    print("内存占用")
    print("=" * 60)

    if device == 'cuda':
        print(f"  GPU 内存分配: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
        print(f"  GPU 内存缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

    # 在测试部分添加这个验证

    print("\n" + "=" * 60)
    print("变换精度验证（排除网络影响）")
    print("=" * 60)

    # 直接测试 phi2x(x2phi(x)) 的精度
    test_signal = torch.randn(4, 129, 129, 1, device=device, dtype=torch.float32)
    test_proj = phi2x(x2phi(test_signal, -3), -3)

    # 检查投影后的 BC
    for b_idx in range(1):
        for t_idx in [0, 64, 128]:
            u_test = test_proj[b_idx, :, t_idx, 0].cpu().double()
            u_x = cheb_derivative(u_test, device='cpu')
            robin_right = abs(u_test[0] + u_x[0]).item()
            robin_left = abs(u_test[-1] - u_x[-1]).item()
            print(f"  t={t_idx}: 右|u+u'|={robin_right:.2e}, 左|u-u'|={robin_left:.2e}")

    # float64 测试
    print("\n使用 float64:")
    test_signal_64 = test_signal.double()
    test_proj_64 = phi2x(x2phi(test_signal_64, -3), -3)

    for t_idx in [0, 64, 128]:
        u_test = test_proj_64[0, :, t_idx, 0].cpu()
        u_x = cheb_derivative(u_test, device='cpu')
        robin_right = abs(u_test[0] + u_x[0]).item()
        robin_left = abs(u_test[-1] - u_x[-1]).item()
        print(f"  t={t_idx}: 右|u+u'|={robin_right:.2e}, 左|u-u'|={robin_left:.2e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
