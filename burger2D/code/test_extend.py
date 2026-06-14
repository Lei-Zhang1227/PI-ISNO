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

CASE_ROOT = Path(__file__).resolve().parents[1]

def resolve_case_path(path_like):
    path = Path(path_like)
    return str(path if path.is_absolute() else CASE_ROOT / path)

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


def testII(config, args):
    """
    一个全面的评测函数，可以再三个不同分辨率上计算模型的l2误差
    :param config:
    :param args:
    :return:
    """
    # region prepare
    ################################################################
    # prepare
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # endregion
    # region dataloader
    data_config = config['data']
    batch_size = config['train']['batchsize']
    initial_step = config['train']['initial_step']
    sub_x = data_config['sub_x']
    sub_t = data_config['sub_t']
    filepath = resolve_case_path(data_config.get('extend_datapath', data_config['datapath']))
    train_data = FNODatasetMult(file_path=filepath,
                                initial_step=initial_step,
                                sub_x=2,
                                sub_t=2,
                                )
    test_data = FNODatasetMult(file_path=filepath,
                               initial_step=initial_step,
                               sub_x=2,
                               sub_t=2,
                               if_test=True,
                               )

    test_loader_seen = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=2, shuffle=True)
    test_loader_unseen = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=2, shuffle=False)
    train_size, test_size = len(train_data), len(test_data)
    ntrain, ntest = train_size, test_size
    print(
        f'{datetime.now()} --- set dataset，batch size: {batch_size}, '
        f'Train loader lens：{train_size}, Test loader lens：{test_size}')
    ################################################################
    # location
    ################################################################
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    # region model
    _trans = PARTIAL(Wrapper, [dctII, dctII])
    _itrans = PARTIAL(Wrapper, [idctII, idctII])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOL2D, T)
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    out_channels = config['model']['output_channel']
    dim = config['model']['dim']
    tril = config['model']['triL']
    input_channel = initial_step + 3
    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    # endregion

    first_dic = resolve_case_path(config['prepare']['project'])
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    ################################################################
    # load model
    ################################################################
    if args.pretrain is not None:
        checkpoint = torch.load(resolve_case_path(args.pretrain))
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        epoch = checkpoint.get('epoch', checkpoint.get('epochs', None))
        if epoch is None:
            print("[Warning] Checkpoint does not contain 'epoch' or 'epochs'.")
            epoch = 0
        print(f"{datetime.now()} --- model【{args.pretrain}】has been loaded, {epoch} epochs has trained")
    else:
        print(os.getcwd())
        checkpoint = torch.load(os.path.join(first_dic, 'checkpoint-best.pth.tar'))
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        # 其他状态，如损失列表、学习率列表等
        epoch = checkpoint['epoch']
        print(f"{datetime.now()} --- current best model has been loaded, {epoch} epochs has trained")
    # endregion

    # region evaluate
    init_t = 5
    t_train = 251
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    test_loaders = {
        f'unseen_extend': test_loader_unseen,
        f'seen_extend': test_loader_seen,
    }

    # 保存所有数据集的选中样本
    selected_samples_all = {}
    # 保存每个时间步的平均L2损失
    timestep_errors_all = {}
    # 保存每个样本在每个时间步的L2误差
    timestep_errors_per_sample_all = {}

    pre_mode = config['train'].get('pre_mode', 'diret')
    dtype=torch.float32
    for name, test_loader in test_loaders.items():
        errors_for_talk = []
        sample_details = []
        timestep_errors_list = []

        model.eval()
        with torch.no_grad():
            test_iter = iter(test_loader)
            for b in tqdm(range(len(test_loader)), desc=name):
                '''
                    xx: torch.Size([1, 128, 128, 5, 1])
                    yy: torch.Size([1, 128, 128, 51, 1])
                    grid: torch.Size([1, 128, 128, 2])
                '''
                xx, yy, grid = next(test_iter)
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)

                pred = torch.empty(yy.shape, device=xx.device)
                gridt = torch.tensor(np.linspace(0, 5, t_train), dtype=dtype, device=xx.device).reshape(
                    t_train, 1)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]
                for t in range(init_t, t_train):
                    current_time = gridt[t:t + 1, :]
                    inp = xx.squeeze(-1)  # torch.Size([b, nx, ny, step])
                    current_time = current_time.view(1, 1, 1, 1).expand(xx.size(0), xx.size(1), xx.size(2),
                                                                        1)  # 扩展为[batch, nx,ny, 1]
                    out = model(torch.cat([inp, grid, current_time], dim=-1)).unsqueeze(-1)
                    if pre_mode == 'delta':
                        last_step = xx[..., -1:, :]
                        out = last_step + out
                    pred[..., t:t + 1, :] = out
                    xx = torch.cat((xx[..., 1:, :], out), dim=-2)
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"

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
                for seg_name, (s, e) in segments.items():
                    _yy = yy[..., s:e, :]
                    _pred = pred[..., s:e, :]
                    err = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                    sample_errors.append(err)

                errors_for_talk.append(sample_errors)

                # 计算每个时间步的L2误差
                sample_timestep_errors = []
                for t in range(init_t, t_train):
                    _yy_t = yy[..., t:t + 1, :]
                    _pred_t = pred[..., t:t + 1, :]
                    err_t = loss_fn(_pred_t.reshape(_batch, -1), _yy_t.reshape(_batch, -1)).item()
                    sample_timestep_errors.append(err_t)
                timestep_errors_list.append(sample_timestep_errors)

                # 计算整体L2误差并保存样本详情
                l2_total = loss_fn(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                sample_details.append({
                    'index': b,
                    'l2_total': l2_total,
                    'segment_errors': sample_errors,
                    'timestep_errors': sample_timestep_errors,
                    'yy': yy.squeeze().cpu().numpy(),
                    'pred': pred.squeeze().cpu().numpy(),
                })

        errors_for_talk = np.array(errors_for_talk)  # [n_samples, 9]
        timestep_errors_array = np.array(timestep_errors_list)  # [n_samples, t_train - init_t]

        seg_names = ['0.1-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5']

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

        # 计算并保存每个时间步的平均L2损失
        timestep_mean_errors = timestep_errors_array.mean(axis=0)
        timestep_std_errors = timestep_errors_array.std(axis=0)
        timestep_max_errors = timestep_errors_array.max(axis=0)
        timestep_min_errors = timestep_errors_array.min(axis=0)

        timestep_results = []
        for t_idx, t in enumerate(range(init_t, t_train)):
            timestep_results.append({
                'Timestep': t,
                'Mean_L2': timestep_mean_errors[t_idx],
                'Std_L2': timestep_std_errors[t_idx],
                'Max_L2': timestep_max_errors[t_idx],
                'Min_L2': timestep_min_errors[t_idx],
            })

        df_timestep = pd.DataFrame(timestep_results)
        df_timestep.to_csv(f'{name}_timestep_errors.csv', index=False)
        print(f"\n{name} - Timestep errors saved to {name}_timestep_errors.csv")
        print(f"  Total timesteps: {len(timestep_results)}")
        print(f"  Mean L2 range: [{timestep_mean_errors.min():.6e}, {timestep_mean_errors.max():.6e}]")

        # 保存时间步误差统计数据
        timestep_errors_all[name] = {
            'timesteps': list(range(init_t, t_train)),
            'mean': timestep_mean_errors,
            'std': timestep_std_errors,
            'max': timestep_max_errors,
            'min': timestep_min_errors,
            'all_samples': timestep_errors_array,
        }

        # 保存每个样本在每个时间步的L2误差
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

        # ========== 修改后的样本选择逻辑 ==========
        # 只处理 unseen_extend
        if name == 'unseen_extend':
            # 计算每个样本在训练阶段 (0.1-1.0) 的误差
            # 时间步: t=0.1 -> idx=5, t=1.0 -> idx=50
            train_seg_start, train_seg_end = 5, 51

            for sample in sample_details:
                yy = sample['yy']
                pred = sample['pred']

                # 计算训练阶段的 L2 误差
                yy_train = yy[..., train_seg_start:train_seg_end]
                pred_train = pred[..., train_seg_start:train_seg_end]

                l2_diff = np.sqrt(np.sum((pred_train - yy_train) ** 2))
                l2_target = np.sqrt(np.sum(yy_train ** 2))
                sample['l2_train'] = l2_diff / l2_target

            # 按训练阶段误差排序
            sorted_by_train = sorted(sample_details, key=lambda x: x['l2_train'])

            # 按整体误差排序
            sorted_by_total = sorted(sample_details, key=lambda x: x['l2_total'])

            # 选择样本: 8个按训练阶段误差 + 2个按整体误差
            selected = []
            selected_indices = set()

            # 先选8个训练阶段最好的
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

            # 再选2个整体最好的（如果还没被选中）
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

            # 打印选中样本信息
            print(f"\n{name} - Selected samples for visualization ({len(selected)} samples):")
            print(f"  按训练阶段误差 (0.1-1.0) 选择:")
            for s in selected:
                if s['selection_type'] == 'train_best':
                    print(f"    {s['percentile']:>5} (rank {s['rank_train']:>3}): index={s['index']}, "
                          f"l2_train={s['l2_train']:.6e}, l2_total={s['l2_total']:.6e}")
            print(f"  按整体误差选择:")
            for s in selected:
                if s['selection_type'] == 'total_best':
                    print(f"    {s['percentile']:>12} (rank {s['rank_total']:>3}): index={s['index']}, "
                          f"l2_train={s['l2_train']:.6e}, l2_total={s['l2_total']:.6e}")
    # 保存选中的样本用于可视化
    with open('selected_samples_for_visualization.pkl', 'wb') as f:
        pickle.dump(selected_samples_all, f)

    # 保存时间步误差统计数据
    with open('timestep_errors.pkl', 'wb') as f:
        pickle.dump(timestep_errors_all, f)

    # 保存每个样本在每个时间步的L2误差
    with open('timestep_errors_per_sample.pkl', 'wb') as f:
        pickle.dump(timestep_errors_per_sample_all, f)

    print("\n" + "=" * 60)
    print("文件已保存:")
    print("  - selected_samples_for_visualization.pkl (选中的样本)")
    print("  - timestep_errors.pkl (时间步误差统计)")
    print("  - timestep_errors_per_sample.pkl (每个样本每个时间步的L2)")
    print("  - {name}_segment_errors.csv (分段误差)")
    print("  - {name}_timestep_errors.csv (时间步误差)")
    print("=" * 60)
    # endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='test_extend', help='train or test')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='pretrain model path')
    parser.add_argument('--load_lr', action='store_true', help='pretrain model path')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        try:
            with open(config_file, encoding="utf-8") as stream:
                config = yaml.load(stream, yaml.FullLoader)
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试GB18030（兼容GBK）
            with open(config_file, encoding="gb18030") as stream:
                config = yaml.load(stream, yaml.FullLoader)
    testII(config, args)
