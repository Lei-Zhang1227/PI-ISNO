"""
heat equation with Chebshev transform in CGL points
增量预测
"""
import os, sys
from pathlib import Path
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))
from argparse import ArgumentParser
import yaml
import tqdm
import shutil
from tqdm import tqdm
from model import *

CASE_ROOT = Path(__file__).resolve().parents[1]

def resolve_case_path(path_like):
    path = Path(path_like)
    return str(path if path.is_absolute() else CASE_ROOT / path)

from dataset import *
from utils import *
from datetime import datetime
import pandas as pd
import pickle
import torch
import numpy as np


def test_extend(config, args):
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
    ################################################################
    # dataloader
    ################################################################
    origin_nx = 513
    initial_step = 5
    file_path = resolve_case_path(config.get('data', {}).get('extend_datapath', 'data/heat1D_robin_highprec_long.h5'))
    unseen_test_data_1x = h5DatasetFor1DHeat_extend(file_path,
                                                    sub_x=1,
                                                    sub_t=2,
                                                    initial_step=initial_step, unseen=True)
    unseen_test_loader_1x = torch.utils.data.DataLoader(unseen_test_data_1x, batch_size=1, shuffle=False,
                                                        num_workers=0, pin_memory=True)
    unseen_test_data_1_2x = h5DatasetFor1DHeat_extend(file_path,
                                                      sub_x=2,
                                                      sub_t=2,
                                                      initial_step=initial_step, unseen=True)
    unseen_test_loader_1_2x = torch.utils.data.DataLoader(unseen_test_data_1_2x, batch_size=1, shuffle=False,
                                                          num_workers=0, pin_memory=True)
    unseen_test_data_1_4x = h5DatasetFor1DHeat_extend(file_path,
                                                      sub_x=4,
                                                      sub_t=2,
                                                      initial_step=initial_step, unseen=True)
    unseen_test_loader_1_4x = torch.utils.data.DataLoader(unseen_test_data_1_4x, batch_size=1, shuffle=False,
                                                          num_workers=0, pin_memory=True)
    unseen_test_data_1_8x = h5DatasetFor1DHeat_extend(file_path,
                                                      sub_x=8,
                                                      sub_t=2,
                                                      initial_step=initial_step, unseen=True)
    unseen_test_loader_1_8x = torch.utils.data.DataLoader(unseen_test_data_1_8x, batch_size=1, shuffle=False,
                                                          num_workers=0, pin_memory=True)

    test_data_1x = h5DatasetFor1DHeat_extend(file_path,
                                             sub_x=1,
                                             sub_t=2,
                                             initial_step=initial_step, unseen=False)
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)

    test_data_1_2x = h5DatasetFor1DHeat_extend(file_path,
                                               sub_x=2,
                                               sub_t=2,
                                               initial_step=initial_step, unseen=False)
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_4x = h5DatasetFor1DHeat_extend(file_path,
                                               sub_x=4,
                                               sub_t=2,
                                               initial_step=initial_step, unseen=False)
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_8x = h5DatasetFor1DHeat_extend(file_path,
                                               sub_x=8,
                                               sub_t=2,
                                               initial_step=initial_step, unseen=False)
    test_loader_1_8x = torch.utils.data.DataLoader(test_data_1_8x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_size = test_data_1x.data_list.shape[0]
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    first_dic = resolve_case_path(config['prepare']['project'])
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    degree = config['model']['degree']
    width = config['model']['width']
    bandwidth = config['model']['bandwidth']
    in_channel = initial_step + 2
    FLOAT = 32
    FLOAT_CONFIG = {
        32: {'model_class': SOL_heat_32, 'dtype': torch.float32},
        64: {'model_class': SOL_heat, 'dtype': torch.float64},
    }
    model_config = FLOAT_CONFIG[FLOAT]
    model = model_config['model_class'](in_channel, degree, width, bandwidth).to(device)
    dtype = model_config['dtype']
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")

    def count_parameters(layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("\nTrainable parameters for each layer/module:")
    for name, layer in model.named_children():
        num_params = count_parameters(layer)
        print(f"{name}: {num_params} parameters")
    ################################################################
    # load model
    ################################################################
    if args.pretrain is not None:
        checkpoint = torch.load(resolve_case_path(args.pretrain))
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        # 其他状态，如损失列表、学习率列表等
        # 获取 epoch
        epoch = checkpoint.get('epoch', checkpoint.get('epochs', None))
        print(f"{datetime.now()} --- model【{args.pretrain}】has been loaded, {epoch} epochs has trained")
    else:
        print(os.getcwd())
        checkpoint = torch.load(os.path.join(first_dic, 'checkpoint-best.pth.tar'))
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        epoch = checkpoint['epoch']
        print(f"{datetime.now()} --- current best model has been loaded, {epoch} epochs has trained")
    # endregion
    # region evaluate
    init_t = 5
    t_train = 251
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    test_loaders_seen = {
        f'seen_{origin_nx}': test_loader_1x,
    }

    test_loaders_unseen = {
        f'unseen_{origin_nx}': unseen_test_loader_1x,
    }

    test_all = {
        'unseen': test_loaders_unseen,  # unseen 在前
        'seen': test_loaders_seen,
    }

    nx_list = [origin_nx, int((origin_nx - 1) / 2) + 1, int((origin_nx - 1) / 4) + 1, int((origin_nx - 1) / 8) + 1]
    seg_names = ['0.1-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5']
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

    all_results = []
    errors_for_talk_all = {}
    pre_mode = config['train'].get('pre_mode', 'delta')
    
    # 新增：保存每个数据集的详细信息
    sample_details_all = {}  # {dataset_name: [{sample_name, l2_total, yy, pred}, ...]}
    selected_samples_all = {}  # 保存随机选取的好样本

    for case, test_loaders_dict in test_all.items():
        for idx, (name, test_loader) in enumerate(test_loaders_dict.items()):
            nx = nx_list[idx]
            errors_for_talk = []
            sample_details = []  # 当前数据集的样本详情
            
            model.eval()
            with torch.no_grad():
                print(f'dataset: {name}, nx: {nx}')

                for b, (xx, yy, grid, sample_name) in enumerate(tqdm(test_loader, desc=name)):
                    xx = xx.to(device, dtype=dtype, non_blocking=True)
                    yy = yy.to(device, dtype=dtype, non_blocking=True)
                    grid = grid.to(device, dtype=dtype, non_blocking=True)

                    inp_shape = list(xx.shape)[:-2] + [-1]
                    outp_shape = list(xx.shape)[:-2] + [1, -1]

                    pred = torch.empty(yy.shape, device=xx.device, dtype=dtype)
                    pred[..., 0:init_t, :] = yy[..., 0:init_t, :]
                    gridt = torch.linspace(0, 5, t_train, dtype=dtype, device=xx.device)

                    for t in range(init_t, t_train):
                        current_time = gridt[t].view(1, 1, 1).expand(xx.size(0), xx.size(1), 1)
                        inp = xx.reshape(inp_shape)
                        out = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                        if pre_mode == 'delta':
                            out = xx[..., -1:, :] + out
                        pred[..., t:t + 1, :] = out
                        xx = torch.cat((xx[..., 1:, :], out), dim=-2)

                    # 分段计算误差
                    _batch = yy.size(0)
                    sample_errors = []
                    for seg_name, (s, e) in segments.items():
                        _yy = yy[..., s:e, :]
                        _pred = pred[..., s:e, :]
                        err = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                        sample_errors.append(err)
                    errors_for_talk.append(sample_errors)
                    
                    # 计算整体 L2 误差（用于排序选择好样本）
                    l2_total = loss_fn(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                    
                    # 处理 sample_name（可能是 tuple 或 list）
                    if isinstance(sample_name, (list, tuple)):
                        sample_name_str = sample_name[0]
                    else:
                        sample_name_str = sample_name
                    
                    # 保存样本详情
                    sample_details.append({
                        'sample_name': sample_name_str,
                        'index': b,
                        'l2_total': l2_total,
                        'segment_errors': sample_errors,
                        'yy': yy.squeeze().cpu().numpy(),
                        'pred': pred.squeeze().cpu().numpy(),
                    })

            errors_for_talk = np.array(errors_for_talk)
            errors_for_talk_all[name] = errors_for_talk
            sample_details_all[name] = sample_details

            # 汇总结果
            for i, seg_name in enumerate(seg_names):
                all_results.append({
                    'Case': case,
                    'Resolution': nx,
                    'Segment': seg_name,
                    'Mean_L2': errors_for_talk[:, i].mean(),
                    'Std_L2': errors_for_talk[:, i].std(),
                    'Max_L2': errors_for_talk[:, i].max(),
                    'Min_L2': errors_for_talk[:, i].min(),
                })
            
            # 按 L2 误差排序，选择结果好的样本（误差小的）
            sorted_samples = sorted(sample_details, key=lambda x: x['l2_total'])
            
            # 从前 30% 好的样本中随机选 2 个
            n_good = max(1, int(len(sorted_samples) * 0.3))
            good_samples = sorted_samples[:n_good]
            
            if len(good_samples) >= 2:
                selected_indices = np.random.choice(len(good_samples), size=2, replace=False)
                selected = [good_samples[i] for i in selected_indices]
            else:
                selected = good_samples[:2]
            
            selected_samples_all[name] = selected
            
            print(f"\n{name} - Selected good samples:")
            for s in selected:
                print(f"  sample_name: {s['sample_name']}, l2_total: {s['l2_total']:.6e}")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('extend_results.csv', index=False)

    with open('error_for_talk_all.pkl', 'wb') as f:
        pickle.dump(errors_for_talk_all, f)
    
    # 保存每个样本的 L2 误差详情（不包含 yy 和 pred，节省空间）
    sample_l2_summary = {}
    for name, details in sample_details_all.items():
        sample_l2_summary[name] = [
            {
                'sample_name': d['sample_name'],
                'index': d['index'],
                'l2_total': d['l2_total'],
                'segment_errors': d['segment_errors'],
            }
            for d in details
        ]
    
    with open('sample_l2_summary.pkl', 'wb') as f:
        pickle.dump(sample_l2_summary, f)
    
    # 保存选中的好样本（包含 yy 和 pred）
    with open('selected_good_samples.pkl', 'wb') as f:
        pickle.dump(selected_samples_all, f)
    
    # 打印选中样本的汇总信息
    print("\n" + "=" * 60)
    print("Selected Good Samples Summary:")
    print("=" * 60)
    for name, samples in selected_samples_all.items():
        print(f"\nDataset: {name}")
        for i, s in enumerate(samples):
            print(f"  [{i+1}] sample_name: {s['sample_name']}, l2_total: {s['l2_total']:.6e}")
            print(f"      yy shape: {s['yy'].shape}, pred shape: {s['pred'].shape}")

    print("\n测试结果已保存:")
    print("  - extend_results.csv")
    print("  - error_for_talk_all.pkl")
    print("  - sample_l2_summary.pkl (每个样本的L2误差)")
    print("  - selected_good_samples.pkl (选中的好样本，含yy和pred)")
    print(results_df.to_string(index=False))

    # endregion


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

    test_extend(config, args)