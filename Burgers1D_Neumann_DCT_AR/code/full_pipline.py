"""
Burger equation 1D for fixed t_dim
"""
import os, sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))

from argparse import ArgumentParser
import yaml
from functools import partial as PARTIAL
import tqdm
import shutil
from tqdm import tqdm
from Burger.model import SOL1dII
from Burger.datasets import *
from Burger.loss import *
from Burger.utils import *
from datetime import datetime
import pandas as pd
import pickle
import torch
import numpy as np


def run(config_a, config_b, args):
    # region prepare
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(config_a['prepare']['seed'])
    torch.manual_seed(config_a['prepare']['seed'])
    np.random.seed(config_a['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config_a['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # endregion
    # region location
    ################################################################
    if args.new_path is not None:
        output_dir = args.new_path
    else:
        output_dir = f"/code/Burger{config_a['prepare']['project']}_TwoStage"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # 复制配置文件到输出目录
    shutil.copy(args.config_a, os.path.join(output_dir, 'MODEL_A.yaml'))
    shutil.copy(args.config_b, os.path.join(output_dir, 'MODEL_B.yaml'))

    print(f"{datetime.now()} --- Output directory: {output_dir} ---")
    print(f"  - MODEL_A.yaml copied from: {args.config_a}")
    print(f"  - MODEL_B.yaml copied from: {args.config_b}")
    # endregion
    # region model A
    ################################################################
    # 这里有所不同的是，不再将这个一维含时的问题视为二维问题
    # 这里有所不同的是，不再将这个一维含时的问题视为二维问题
    _trans = PARTIAL(Wrapper, [dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [idctI_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOL1dII, T)
    input_channel = config_a['model']['input_channel'] * config_a['data']['initial_step'] + 1
    model_A = Model(input_channel, config_a['model']['modes'], config_a['model']['width'],
                    config_a['model']['bandwidth'], out_channels=config_a['model']['output_channel'],
                    dim=config_a['model']['dim'], triL=config_a['model']['triL']).to(device).to(torch.float32)
    total_params = sum(p.numel() for p in model_A.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")

    def count_parameters(layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)

    dtype_a = torch.float32
    total_params = sum(p.numel() for p in model_A.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("\nTrainable parameters for each layer/module:")
    for name, layer in model_A.named_children():
        num_params = count_parameters(layer)
        print(f"{name}: {num_params} parameters")

    # 加载Model A权重
    model_a_dir = f"/code/Burger{config_a['prepare']['project']}"
    checkpoint_a_path = os.path.join(model_a_dir, 'checkpoint-best.pth.tar')
    checkpoint_a = torch.load(checkpoint_a_path, map_location=device)
    model_A.load_state_dict(checkpoint_a['model'])
    model_A.eval()
    epoch_a = checkpoint_a['epoch']
    print(f"Model A loaded from: {checkpoint_a_path}")
    print(f"Model A trained for {epoch_a} epochs")
    # endregion
    # region model B
    _trans = PARTIAL(Wrapper, [dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [idctI_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOL1dII, T)
    input_channel = config_b['model']['input_channel'] * config_b['data']['initial_step'] + 2
    model_B = Model(input_channel, config_b['model']['modes'], config_b['model']['width'],
                    config_b['model']['bandwidth'], out_channels=config_b['model']['output_channel'],
                    dim=config_b['model']['dim'], triL=config_b['model']['triL']).to(device)  # .to(torch.float32)
    total_params = sum(p.numel() for p in model_B.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")

    def count_parameters(layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model_B.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("\nTrainable parameters for each layer/module:")
    for name, layer in model_B.named_children():
        num_params = count_parameters(layer)
        print(f"{name}: {num_params} parameters")
    # 加载Model B权重
    model_b_dir = f"/code/Burger{config_b['prepare']['project']}"
    checkpoint_b_path = os.path.join(model_b_dir, 'checkpoint-best.pth.tar')
    checkpoint_b = torch.load(checkpoint_b_path, map_location=device)
    model_B.load_state_dict(checkpoint_b['model'])
    model_B.eval()
    epoch_b = checkpoint_b['epoch']
    print(f"Model B loaded from: {checkpoint_b_path}")
    print(f"Model B trained for {epoch_b} epochs")
    # endregion
    # region information_write
    with open(f'{output_dir}/Experiment_record.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'=' * 60}\n")
        f.write(f"[Two-Stage Model Evaluation | {timestamp}]\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"├── Case: 1D Heat equation with Robin BCs\n")
        f.write(f"├── Method: Two-Stage Prediction\n")
        f.write(f"│   ├── Stage A: u0 -> first {config_a['model']['output_channel']} time steps\n")
        f.write(f"│   └── Stage B: Autoregressive prediction for remaining steps\n")

        # Model A 配置
        f.write(f"\n├─ Model A Configuration\n")
        f.write(f"├── Checkpoint: {checkpoint_a_path}\n")
        f.write(f"├── Trained Epochs: {epoch_a}\n")
        f.write(f"├── Architecture Parameters:\n")
        f.write(
            f"│   ├── input_channels: {config_a['model']['input_channel'] * config_a['data']['initial_step'] + 1}\n")
        f.write(f"│   ├── modes: {config_a['model']['modes']}\n")
        f.write(f"│   ├── width: {config_a['model']['width']}\n")
        f.write(f"│   ├── bandwidth: {config_a['model']['bandwidth']}\n")
        f.write(f"│   ├── dim: {config_a['model']['dim']}\n")
        f.write(f"│   ├── triL: {config_a['model']['triL']}\n")
        f.write(f"│   └── output_channel: {config_a['model']['output_channel']}\n")
        total_params_a = sum(p.numel() for p in model_A.parameters() if p.requires_grad)
        f.write(f"├── Total Parameters: {total_params_a:,}\n")
        f.write(f"├── Layer Details:\n")
        for name, layer in model_A.named_children():
            num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            f.write(f"│   ├── {name}: {num_params:,} parameters\n")

        # Model B 配置
        f.write(f"\n├─ Model B Configuration\n")
        f.write(f"├── Checkpoint: {checkpoint_b_path}\n")
        f.write(f"├── Trained Epochs: {epoch_b}\n")
        f.write(f"├── Architecture Parameters:\n")
        f.write(
            f"│   ├── input_channels: {config_b['model']['input_channel'] * config_b['data']['initial_step'] + 2}\n")
        f.write(f"│   ├── modes: {config_b['model']['modes']}\n")
        f.write(f"│   ├── width: {config_b['model']['width']}\n")
        f.write(f"│   ├── bandwidth: {config_b['model']['bandwidth']}\n")
        f.write(f"│   ├── dim: {config_b['model']['dim']}\n")
        f.write(f"│   ├── triL: {config_b['model']['triL']}\n")
        f.write(f"│   └── output_channel: {config_b['model']['output_channel']}\n")
        total_params_b = sum(p.numel() for p in model_B.parameters() if p.requires_grad)
        f.write(f"├── Total Parameters: {total_params_b:,}\n")
        f.write(f"├── Layer Details:\n")
        for name, layer in model_B.named_children():
            num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            f.write(f"│   ├── {name}: {num_params:,} parameters\n")

        # 设备信息
        f.write(f"\n├─ Runtime Configuration\n")
        f.write(f"├── Device: {device}\n")
        f.write(f"├── Output Directory: {output_dir}\n")
        f.write(f"└── Config Files:\n")
        f.write(f"    ├── MODEL_A.yaml: {args.config_a}\n")
        f.write(f"    └── MODEL_B.yaml: {args.config_b}\n")
        f.write(f"{'=' * 60}\n\n")
    # endregion
    # region dataloader

    data_config = config_a['data']
    test_data_1x = h5DatasetFor1DBurgers_TwoStage(data_config['datapath'],
                                                  sub_x=1,
                                                  sub_t=data_config['sub_t'], if_test=True)
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)

    test_data_1_2x = h5DatasetFor1DBurgers_TwoStage(data_config['datapath'],
                                                    sub_x=2,
                                                    sub_t=data_config['sub_t'], if_test=True)
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_4x = h5DatasetFor1DBurgers_TwoStage(data_config['datapath'],
                                                    sub_x=4,
                                                    sub_t=data_config['sub_t'], if_test=True)
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_8x = h5DatasetFor1DBurgers_TwoStage(data_config['datapath'],
                                                    sub_x=8,
                                                    sub_t=data_config['sub_t'], if_test=True)
    test_loader_1_8x = torch.utils.data.DataLoader(test_data_1_8x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_size = len(test_data_1x)
    ntest = test_size
    print('size-of-test:', test_size)
    # endregion
    # region evaluate
    ################################################################
    origin_nx = config_a['data']['nx']
    n_output_steps_a = config_a['data']['n_output_steps']
    initial_step_b = config_b['data']['initial_step']
    t_total = (data_config['nt'] - 1) // data_config['sub_t'] + 1

    assert initial_step_b == n_output_steps_a + 1, \
        f"Model B initial_step ({initial_step_b}) should equal Model A n_output_steps + 1 ({n_output_steps_a + 1})"

    myloss = LpLoss(size_average=True)
    loss_fn = myloss

    test_loaders = {
        f'test_{origin_nx}': test_loader_1x,
        f'test_{int((origin_nx - 1) / 2) + 1}': test_loader_1_2x,
        f'test_{int((origin_nx - 1) / 4) + 1}': test_loader_1_4x,
        f'test_{int((origin_nx - 1) / 8) + 1}': test_loader_1_8x,
    }

    results = []
    errors_for_talk_all = {}
    visualize_results_all = []

    for name, test_loader in test_loaders.items():
        errors_list = []
        pred_cache = {}  # 缓存预测结果，避免重复计算

        model_A.eval()
        model_B.eval()

        first = True
        with torch.no_grad():
            # ==================== 第一遍：计算所有样本误差并缓存预测 ====================
            for xx, yy, grid, sample_name in tqdm(test_loader, desc=f"Evaluating {name}"):
                if isinstance(sample_name, (list, tuple)):
                    sample_name = sample_name[0]

                batch_size, nx, _ = xx.shape

                xx = xx.to(device, dtype=dtype_a, non_blocking=True)
                yy = yy.to(device, dtype=dtype_a, non_blocking=True)
                grid = grid.to(device, dtype=dtype_a, non_blocking=True)

                if first:
                    print(f'xx.shape: {xx.shape}, yy.shape: {yy.shape}, grid.shape: {grid.shape}')
                    print(f'n_output_steps_a: {n_output_steps_a}, initial_step_b: {initial_step_b}, t_total: {t_total}')
                    first = False

                # ==================== Stage A ====================
                pred_A = model_A(xx)  # [batch, n_output_steps_a, nx]
                pred_A = pred_A.permute(0, 2, 1)  # [batch, nx, n_output_steps_a]
                # print(f'pred_A.shape:{pred_A.shape}')

                # ==================== 准备 Stage B 输入 ====================
                u0 = xx[..., 0:1]  # [batch, nx, 1]
                u0_expanded = u0.unsqueeze(-2)  # [batch, nx, 1, 1]
                # print(f'u0_expanded.shape:{u0_expanded.shape}')
                pred_A_expanded = pred_A.unsqueeze(-1)  # [batch, nx, n_output_steps_a, 1]
                # print(f'pred_A_expanded.shape:{pred_A_expanded.shape}')
                xx_B = torch.cat([u0_expanded, pred_A_expanded], dim=-2)  # [batch, nx, initial_step_b, 1]

                # ==================== Stage B: 自回归预测 ====================
                pred_full = torch.empty(batch_size, nx, t_total, 1, device=device, dtype=dtype_a)
                pred_full[..., :initial_step_b, :] = xx_B

                gridt = torch.linspace(0, 1, t_total, dtype=dtype_a, device=device).reshape(t_total, 1)

                for t in range(initial_step_b, t_total):
                    current_time = gridt[t:t + 1, :].view(1, 1, 1).expand(batch_size, nx, 1)
                    inp = xx_B.reshape(batch_size, nx, -1)
                    model_input = torch.cat([inp, grid, current_time], dim=-1)
                    # print(f'model_input.shape:{model_input.shape}')
                    out = model_B(model_input).permute(0, 2, 1).unsqueeze(-1)
                    # print(f'out.shape:{out.shape}')
                    pred_full[..., t:t + 1, :] = out
                    xx_B = torch.cat([xx_B[..., 1:, :], out], dim=-2)

                assert pred_full.shape == yy.shape, f"Shape mismatch: {pred_full.shape} != {yy.shape}"

                # 计算误差
                _yy = yy[..., 1:, :]
                _pred = pred_full[..., 1:, :]
                _batch = yy.size(0)
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                errors_list.append([l2_error, sample_name])

                # 缓存预测结果
                pred_cache[sample_name] = {
                    'pred': pred_full.squeeze().cpu().numpy(),
                    'yy': yy.squeeze().cpu().numpy(),
                }

            # ==================== 选择要可视化的样本 ====================
            error_records_sorted = sorted(errors_list, key=lambda x: x[0])
            n_samples = len(error_records_sorted)

            best_samples = [r[-1] for r in error_records_sorted[:3]]
            worst_samples = [r[-1] for r in error_records_sorted[-3:]]
            mid_samples = [r[-1] for r in error_records_sorted[n_samples // 2 - 1: n_samples // 2 + 2]]
            selected_indices = set(best_samples + worst_samples + mid_samples)

            print(f'Selected samples: {selected_indices}')

            # ==================== 构建可视化结果 ====================
            visualize_results = []
            error_dict = {r[-1]: r[0] for r in error_records_sorted}

            for sample_name in selected_indices:
                if sample_name in best_samples:
                    category = 'best'
                elif sample_name in worst_samples:
                    category = 'worst'
                else:
                    category = 'mid'

                visualize_results.append({
                    'Dataset': name,
                    'index': sample_name,
                    'category': category,
                    'l2_error': error_dict[sample_name],
                    'pred': pred_cache[sample_name]['pred'],
                    'yy': pred_cache[sample_name]['yy'],
                })

            visualize_results_all.append(visualize_results)

        # ==================== 统计结果 ====================
        errors_array = np.array([e[0] for e in errors_list], dtype=np.float64)
        mean_l2_error = np.mean(errors_array)
        std_l2_error = np.std(errors_array)
        max_l2_error = np.max(errors_array)
        min_l2_error = np.min(errors_array)

        results.append({
            'Dataset': name,
            'Mean Relative L2 Error': mean_l2_error,
            'Std Relative L2 Error': std_l2_error,
            'Max Relative L2 Error': max_l2_error,
            'Min Relative L2 Error': min_l2_error,
        })
        errors_for_talk_all[name] = errors_list

        print(f"\n{'=' * 60}")
        print(f"Results for {name}:")
        print(f"  Mean L2 Error: {mean_l2_error:.6e}")
        print(f"  Std  L2 Error: {std_l2_error:.6e}")
        print(f"  Max  L2 Error: {max_l2_error:.6e}")
        print(f"  Min  L2 Error: {min_l2_error:.6e}")
        print(f"{'=' * 60}")

    # ==================== 保存结果 ====================
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/test_results.csv', index=False)

    with open(f'{output_dir}/visualize_results.pkl', 'wb') as f:
        pickle.dump(visualize_results_all, f)

    with open(f'{output_dir}/error_for_talk_all.pkl', 'wb') as f:
        pickle.dump(errors_for_talk_all, f)

    print(f"结果已保存到 {output_dir}/")
    print(f"  - test_results.csv")
    print(f"  - visualize_results.pkl")
    print(f"  - error_for_talk_all.pkl")
    # endregion
    # region visualize
    ################################################################
    for visualize_results in visualize_results_all:
        case = visualize_results[0]['Dataset']
        plot_visualization_results(visualize_results, save_dir=f'{output_dir}/figures_{case}', plot_residual=False)
    print('可视化完成~')
    # endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='Two-Stage Model Evaluation')
    parser.add_argument('--config_a', type=str, required=True,
                        help='Path to Model A configuration file (.yaml)')
    parser.add_argument('--config_b', type=str, required=True,
                        help='Path to Model B configuration file (.yaml)')
    parser.add_argument('--new_path', type=str, default=None, help='full model path')
    args = parser.parse_args()

    # 加载Model A配置
    with open(args.config_a, 'r') as stream:
        try:
            with open(args.config_a, encoding="utf-8") as stream:
                config_a = yaml.load(stream, yaml.FullLoader)
        except UnicodeDecodeError:
            with open(args.config_a, encoding="gb18030") as stream:
                config_a = yaml.load(stream, yaml.FullLoader)

    # 加载Model B配置
    with open(args.config_b, 'r') as stream:
        try:
            with open(args.config_b, encoding="utf-8") as stream:
                config_b = yaml.load(stream, yaml.FullLoader)
        except UnicodeDecodeError:
            with open(args.config_b, encoding="gb18030") as stream:
                config_b = yaml.load(stream, yaml.FullLoader)

    run(config_a, config_b, args)
