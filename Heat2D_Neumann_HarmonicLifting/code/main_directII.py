"""
test() 函数片段 — direct 模式, per-sample BC lifting

替换原 test() 中的 evaluate region 即可。
主要改动:
  1. dataloader 解包加 bc_params
  2. u_b 从固定改为 build_lifting_batch(bc_params, ...)
  3. compute_heat2d_residual_batch 传入 bc_params
"""
import os
from pathlib import Path
import sys
import pandas as pd
from argparse import ArgumentParser
import yaml
import shutil
from dataloaderII import *
from lossII import *
from utils import *
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

CASE_ROOT = Path(__file__).resolve().parents[1]

def resolve_case_path(path_like):
    path = Path(path_like)
    return str(path if path.is_absolute() else CASE_ROOT / path)

# ============================================================
# 需要在文件头部导入:
# from loss import build_lifting_batch
# ============================================================

def test(config, args):
    # region prepare
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
    origin_nx = data_config['nx']
    initial_step = config['train']['initial_step']
    filepath = resolve_case_path(data_config['datapath'])
    sub_t = data_config['sub_t']
    test_data_1x = FNODatasetMult(file_path=filepath,
                                  initial_step=initial_step,
                                  sub_x=1,
                                  sub_t=sub_t,
                                  if_test=True,
                                  )
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)

    test_data_1_2x = FNODatasetMult(file_path=filepath,
                                    initial_step=initial_step,
                                    sub_x=2,
                                    sub_t=sub_t,
                                    if_test=True,
                                    )
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_4x = FNODatasetMult(file_path=filepath,
                                    initial_step=initial_step,
                                    sub_x=4,
                                    sub_t=sub_t,
                                    if_test=True,
                                    )
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_size = len(test_data_1x)
    ntest = test_size
    # endregion
    # region location
    first_dic = resolve_case_path(config['prepare']['project'])
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    _trans = PARTIAL(Wrapper, [dctI, dctI])
    _itrans = PARTIAL(Wrapper, [idctI, idctI])
    T = Transform(_trans, _itrans)
    Model = PARTIAL(SOL2D, T)
    modes = config['model']['modes']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    out_channels = config['model']['output_channel']
    dim = config['model']['dim']
    tril = config['model']['triL']
    input_channel = initial_step + 2
    model = Model(input_channel, modes, width, bandwidth, out_channels=out_channels,
                  dim=dim, triL=tril, double_weights=False,
                  skip=True, flat=False).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    # endregion
    # region load model
    if args.pretrain is not None:
        model_name = args.pretrain[:-8]
        model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
        checkpoint = torch.load(resolve_case_path(args.pretrain))
        model.load_state_dict(checkpoint['model'])
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint.get('test_loss_list')
        grad_array = checkpoint['grad']
        if test_loss_list is None:
            print("[Warning] Checkpoint does not contain 'test_loss_list'.")
            test_loss_list = []
        lr_list = checkpoint.get('lr_list')
        if lr_list is None:
            print("[Warning] Checkpoint does not contain 'lr_list'.")
            lr_list = []
        epoch = checkpoint.get('epoch', checkpoint.get('epochs', None))
        if epoch is None:
            print("[Warning] Checkpoint does not contain 'epoch' or 'epochs'.")
            epoch = 0
        print(f"{datetime.now()} --- model【{args.pretrain}】has been loaded, {epoch} epochs has trained")
    else:
        print(os.getcwd())
        checkpoint = torch.load(os.path.join(first_dic, 'checkpoint-best.pth.tar'))
        grad_array = checkpoint['grad']
        model.load_state_dict(checkpoint['model'])
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']
        print('len(loss_list0,len(test_loss_list),len(lr_list)):', len(loss_list), len(test_loss_list), len(lr_list))
        epoch = checkpoint['epoch']
        model_name = 'checkpoint-best'
        print(f"{datetime.now()} --- current best model has been loaded, {epoch} epochs has trained")
    # endregion
    # region plot loss carve
    plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array,
                               f'{first_dic}/loss_carve_for_{model_name}')
    # endregion
    # region evaluate
    dtype = torch.float32

    init_t = int(data_config['initial_step'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    test_loaders = {
        f'test_{origin_nx}': (test_loader_1x, 1),
        f'test_{int((origin_nx - 1) / 2) + 1}': (test_loader_1_2x, 2),
        f'test_{int((origin_nx - 1) / 4) + 1}': (test_loader_1_4x, 4),
    }
    results = []
    errors_for_talk_all = {}
    visualize_results_all = []
    i = 0
    for name, (test_loader, sub_x) in test_loaders.items():
        errors_for_talk = []
        Nx = int((origin_nx - 1) / sub_x) + 1
        model.eval()
        first = True
        with torch.no_grad():
            test_iter = iter(test_loader)
            for b in tqdm(range(len(test_loader))):
                xx, yy, grid, bc_params = next(test_iter)
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

                # ---- per-sample lifting ----
                x_coord = grid[0, :, 0, 0]  # [Nx]
                y_coord = grid[0, 0, :, 1]  # [Ny]
                u_b = build_lifting_batch(bc_params, x_coord, y_coord)  # [batch, 1, Nx, Ny]
                u_b = u_b.permute(0, 2, 3, 1).unsqueeze(-1)  # [batch, Nx, Ny, 1, 1]

                # ---- rollout in u_h space (direct mode) ----
                xx_h = xx - u_b
                pred_h = torch.empty(yy.shape, device=device, dtype=dtype)
                pred_h[..., :init_t, :] = yy[..., :init_t, :] - u_b
                for t in range(init_t, t_train):
                    inp = xx_h.squeeze(-1)
                    out = model(torch.cat([inp, grid], dim=-1)).unsqueeze(-1)
                    pred_h[..., t:t + 1, :] = out
                    xx_h = torch.cat((xx_h[..., 1:, :], out), dim=-2)
                pred = pred_h + u_b

                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"

                _yy = yy[..., init_t + 1:t_train, :]
                _pred = pred[..., init_t + 1:t_train, :]
                _batch = yy.size(0)
                l2_error = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

                residual = compute_heat2d_residual_batch(pred, bc_params, kappa=0.02, T=1)
                loss_f_p = torch.abs(residual).mean()

                residual = compute_heat2d_residual_batch(yy, bc_params, kappa=0.02, T=1)
                loss_f_y = torch.abs(residual).mean()

                mean_residual_pred = loss_f_p.item()
                mean_residual_yy = loss_f_y.item()
                errors_for_talk.append([l2_error, mean_residual_pred, mean_residual_yy, int(b)])

            error_records_sorted = sorted(errors_for_talk, key=lambda x: x[0])
            n_samples = len(error_records_sorted)

            selected_indices = set(
                [r[-1] for r in error_records_sorted[:3]] +
                [r[-1] for r in error_records_sorted[-3:]] +
                [r[-1] for r in error_records_sorted[n_samples // 2 - 1: n_samples // 2 + 2]]
            )
            print('selected_indices:', selected_indices)

            test_iter = iter(test_loader)
            visualize_results = []
            for b in tqdm(range(len(test_loader))):
                xx, yy, grid, bc_params = next(test_iter)
                if b not in selected_indices:
                    continue
                print(f'b is {b},dataset is {name}')
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                bc_params = bc_params.to(device, dtype=dtype, non_blocking=True)

                # ---- per-sample lifting ----
                x_coord = grid[0, :, 0, 0]
                y_coord = grid[0, 0, :, 1]
                u_b = build_lifting_batch(bc_params, x_coord, y_coord)
                u_b = u_b.permute(0, 2, 3, 1).unsqueeze(-1)

                # ---- rollout in u_h space (direct mode) ----
                xx_h = xx - u_b
                pred_h = torch.empty(yy.shape, device=device, dtype=dtype)
                pred_h[..., :init_t, :] = yy[..., :init_t, :] - u_b
                for t in range(init_t, t_train):
                    inp = xx_h.squeeze(-1)
                    out = model(torch.cat([inp, grid], dim=-1)).unsqueeze(-1)
                    pred_h[..., t:t + 1, :] = out
                    xx_h = torch.cat((xx_h[..., 1:, :], out), dim=-2)
                pred = pred_h + u_b

                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
                _yy = yy[..., init_t + 1:t_train, :]
                _pred = pred[..., init_t + 1:t_train, :]
                _batch = yy.size(0)
                f_p = compute_heat2d_residual_batch(pred, bc_params, kappa=0.02, T=1)
                f_y = compute_heat2d_residual_batch(yy, bc_params, kappa=0.02, T=1)
                l2_error = error_records_sorted[[r[-1] for r in error_records_sorted].index(b)][0]
                if b in [r[-1] for r in error_records_sorted[:3]]:
                    category = 'best'
                elif b in [r[-1] for r in error_records_sorted[-3:]]:
                    category = 'worst'
                else:
                    category = 'mid'
                print('pred.squeeze().shape:', pred.squeeze().shape)
                visualize_results.append({
                    'Dataset': name,
                    'index': b,
                    'category': category,
                    'l2_error': l2_error,
                    'pred': pred.squeeze().cpu().numpy(),
                    'yy': yy.squeeze().cpu().numpy(),
                    'pred_du': f_p.squeeze().cpu().numpy(),
                    'yy_du': f_y.squeeze().cpu().numpy(),
                })
                print('len(visualize_results):', len(visualize_results))
            visualize_results_all.append(visualize_results)

        i += 1
        errors_for_talk = np.array(errors_for_talk)
        mean_l2_error = np.mean(errors_for_talk[:, 0])
        std_l2_error = np.std(errors_for_talk[:, 0])
        max_l2_error = np.max(errors_for_talk[:, 0])
        min_l2_error = np.min(errors_for_talk[:, 0])

        mean_PDE_error = np.mean(errors_for_talk[:, 1])
        std_PDE_error = np.std(errors_for_talk[:, 1])
        max_PDE_error = np.max(errors_for_talk[:, 1])
        min_PDE_error = np.min(errors_for_talk[:, 1])

        mean_PDE_error_yy = np.mean(errors_for_talk[:, 2])
        std_PDE_error_yy = np.std(errors_for_talk[:, 2])
        max_PDE_error_yy = np.max(errors_for_talk[:, 2])
        min_PDE_error_yy = np.min(errors_for_talk[:, 2])
        results.append({
            'Dataset': name,
            'Mean Relative L2 Error': mean_l2_error,
            'Std Relative L2 Error': std_l2_error,
            'Max Relative L2 Error': max_l2_error,
            'Min Relative L2 Error': min_l2_error,
            'Mean PDE Error': mean_PDE_error,
            'Std PDE Error': std_PDE_error,
            'Max PDE Error': max_PDE_error,
            'Min PDE Error': min_PDE_error,
            'YY Mean PDE Error': mean_PDE_error_yy,
            'YY Std PDE Error': std_PDE_error_yy,
            'YY Max PDE Error': max_PDE_error_yy,
            'YY Min PDE Error': min_PDE_error_yy,
        })
        errors_for_talk_all[name] = errors_for_talk
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)
    print('len(visualize_results_all):', len(visualize_results_all))
    with open('visualize_results.pkl', 'wb') as f:
        pickle.dump(visualize_results, f)
    with open('error_for_talk_all.pkl', 'wb') as f:
        pickle.dump(errors_for_talk_all, f)
    print("测试结果已保存到 test_results.csv/error_for_talk_all.pkl,可视化数据已保存到visualize_results_results.pkl")
    # endregion
    # region visualize
    for visualize_results in visualize_results_all:
        case = visualize_results[0]['Dataset']
        plot_burgers2d_comparison(visualize_results, save_dir=f'./figures_{case}')
    print('可视化完成~')
    # endregion
if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='./yaml/information.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
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
    if args.mode == 'train':
        print('no training mode')
    else:
        test(config, args)