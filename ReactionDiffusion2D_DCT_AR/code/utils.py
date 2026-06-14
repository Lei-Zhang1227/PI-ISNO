import scipy.io
import numpy as np
import h5py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.animation import FuncAnimation


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self.h5 = False
        self._load_file()

    def _load_file(self):

        if self.file_path[-3:] == '.h5':
            self.data = h5py.File(self.file_path, 'r')
            self.h5 = True

        else:
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except:
                self.data = h5py.File(self.file_path, 'r')
                self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if self.h5:
            x = x[()]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


def FC1d(u, order=5):
    if not 1 <= order <= 5:
        raise ValueError(f"`order` must be between 1 and 5; got {order}")

    AQ1 = MatReader(f"FC_data/AlQl_d{order}_C_25.mat", to_cuda=True).read_field('AlQl').double()
    AQ2 = MatReader(f"FC_data/ArQr_d{order}_C_25.mat", to_cuda=True).read_field('ArQr').double()

    u1 = torch.einsum("xy,bcy->bcx", AQ1, u[..., :order])
    u2 = torch.einsum("xy,bcy->bcx", AQ2, u[..., -order:])
    return torch.cat([u, u1 + u2], dim=-1)


def save_and_record(num_epochs, optimizer, scheduler, model, filename='EndModel', ):
    state = {
        'epoch': num_epochs,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, filename + '.pth.tar')
    torch.save(model, filename + '.pkl')


def save_checkpoint(model, LogIter, optimizer, scheduler, loss_list, test_loss_list, lr_list, model_save_record,
                    filename='checkpoint'):
    state = {
        'epoch': LogIter,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_list': loss_list,
        'test_loss_list': test_loss_list,
        'lr_list': lr_list,
        'model_save_record': model_save_record,
    }
    torch.save(state, filename + '.pth.tar')
    torch.save(model, filename + '.pkl')


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional


def plot_vector_field_comparison(
        model_field: np.ndarray,
        ref_field: np.ndarray,
        output_gif_path: str = "field_comparison.gif",
        time_steps: Optional[list] = None,
        plot_type: str = "streamplot",
        downsample_stride: int = 4,
        frame_duration: int = 200,
        loop_gif: bool = True,
        dpi: int = 100,
) -> None:
    """
    绘制模型场与参考场的动态对比图，并保存为GIF。

    参数:
        model_field: 模型输出场，形状 [H, W, T, 2] (u和v分量)。
        ref_field: 参考场，形状 [H, W, T, 2]。
        output_gif_path: 输出GIF路径（默认当前目录）。
        time_steps: 指定要绘制的时间步列表（默认全绘制）。
        plot_type: 绘图类型，"streamplot" 或 "quiver"。
        downsample_stride: 降采样步长（提升性能）。
        frame_duration: 每帧显示时间（毫秒）。
        loop_gif: 是否循环播放GIF。
        dpi: 图像分辨率（默认100）。
    """
    # 检查输入数据
    assert model_field.shape == ref_field.shape, "模型场和参考场形状必须一致"
    assert plot_type in ["streamplot", "quiver"], "plot_type 必须是 'streamplot' 或 'quiver'"

    H, W, T, _ = model_field.shape
    time_steps = range(T) if time_steps is None else time_steps

    # 生成网格（降采样后）
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x_sub = x[::downsample_stride, ::downsample_stride]
    y_sub = y[::downsample_stride, ::downsample_stride]

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    fig.suptitle("Model vs Reference Field (Dynamic Comparison)")

    frames = []
    for t in time_steps:
        ax1.clear()
        ax2.clear()

        # 提取当前时间步的场（降采样）
        u_model = model_field[::downsample_stride, ::downsample_stride, t, 0]
        v_model = model_field[::downsample_stride, ::downsample_stride, t, 1]
        u_ref = ref_field[::downsample_stride, ::downsample_stride, t, 0]
        v_ref = ref_field[::downsample_stride, ::downsample_stride, t, 1]

        # 绘制模型场（左图）
        ax1.set_title(f"Model Field (t={t})")
        if plot_type == "streamplot":
            ax1.streamplot(x_sub, y_sub, u_model, v_model, color='blue', density=1.5, linewidth=1)
        else:
            ax1.quiver(x_sub, y_sub, u_model, v_model, scale=50, color='blue', width=0.002)

        # 绘制参考场（右图）
        ax2.set_title(f"Reference Field (t={t})")
        if plot_type == "streamplot":
            ax2.streamplot(x_sub, y_sub, u_ref, v_ref, color='red', density=1.5, linewidth=1)
        else:
            ax2.quiver(x_sub, y_sub, u_ref, v_ref, scale=50, color='red', width=0.002)

        # 统一坐标轴
        for ax in [ax1, ax2]:
            ax.set_xlim(0, W - 1)
            ax.set_ylim(0, H - 1)

        # 转换为PIL图像并保存帧
        fig.canvas.draw()
        frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(frame)
        print(f"Generated frame for t={t}")

    plt.close()

    # 保存GIF
    loop = 0 if loop_gif else 1
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=loop,
        optimize=True
    )
    print(f"GIF saved to: {output_gif_path}")


#
#
# def save_checkpoint(path, name, model, optimizer=None):
#     ckpt_dir = 'checkpoints/%s/' % path
#     if not os.path.exists(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     try:
#         model_state_dict = model.module.state_dict()
#     except AttributeError:
#         model_state_dict = model.state_dict()
#
#     if optimizer is not None:
#         optim_dict = optimizer.state_dict()
#     else:
#         optim_dict = 0.0
#
#     torch.save({
#         'model': model_state_dict,
#         'optim': optim_dict
#     }, ckpt_dir + name)
#     print('Checkpoint is saved at %s' % ckpt_dir + name)


def spectral_operator():
    pass
