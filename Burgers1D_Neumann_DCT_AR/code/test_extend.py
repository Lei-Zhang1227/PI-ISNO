"""
heat equation with Chebshev transform in CGL points
增量预测
"""
import os, sys
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))
from argparse import ArgumentParser
import yaml
import tqdm
import shutil
from tqdm import tqdm
from functools import partial as PARTIAL
from Burger.model import SOL1dII
from Burger.datasets import *
from Burger.loss import *
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
    initial_step = 10
    file_path = '/data/zhanglei/BurgersEquationII/burgers_neumann_513x101_1s_old_format.h5'
    #'/data/zhanglei/BurgersEquationII/burgers_neumann_513x501_5s_old_format.h5'
    unseen_test_data_1x = h5DatasetFor1DBurgersII_extend(file_path,
                                                         sub_x=1,
                                                         sub_t=1,
                                                         initial_step=10, unseen=True)
    unseen_test_loader_1x = torch.utils.data.DataLoader(unseen_test_data_1x, batch_size=1, shuffle=False,
                                                        num_workers=0, pin_memory=True)
    unseen_test_data_1_2x = h5DatasetFor1DBurgersII_extend(file_path,
                                                           sub_x=2,
                                                           sub_t=1,
                                                           initial_step=10, unseen=True)
    unseen_test_loader_1_2x = torch.utils.data.DataLoader(unseen_test_data_1_2x, batch_size=1, shuffle=False,
                                                          num_workers=0, pin_memory=True)
    unseen_test_data_1_4x = h5DatasetFor1DBurgersII_extend(file_path,
                                                           sub_x=4,
                                                           sub_t=1,
                                                           initial_step=10, unseen=True)
    unseen_test_loader_1_4x = torch.utils.data.DataLoader(unseen_test_data_1_4x, batch_size=1, shuffle=False,
                                                          num_workers=0, pin_memory=True)
    unseen_test_data_1_8x = h5DatasetFor1DBurgersII_extend(file_path,
                                                           sub_x=8,
                                                           sub_t=1,
                                                           initial_step=10, unseen=True)
    unseen_test_loader_1_8x = torch.utils.data.DataLoader(unseen_test_data_1_8x, batch_size=1, shuffle=False,
                                                          num_workers=0, pin_memory=True)

    test_data_1x = h5DatasetFor1DBurgersII_extend(file_path,
                                                  sub_x=1,
                                                  sub_t=1,
                                                  initial_step=10, unseen=False)
    test_loader_1x = torch.utils.data.DataLoader(test_data_1x, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)

    test_data_1_2x = h5DatasetFor1DBurgersII_extend(file_path,
                                                    sub_x=2,
                                                    sub_t=1,
                                                    initial_step=10, unseen=False)
    test_loader_1_2x = torch.utils.data.DataLoader(test_data_1_2x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_4x = h5DatasetFor1DBurgersII_extend(file_path,
                                                    sub_x=4,
                                                    sub_t=1,
                                                    initial_step=10, unseen=False)
    test_loader_1_4x = torch.utils.data.DataLoader(test_data_1_4x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_data_1_8x = h5DatasetFor1DBurgersII_extend(file_path,
                                                    sub_x=8,
                                                    sub_t=1,
                                                    initial_step=10, unseen=False)
    test_loader_1_8x = torch.utils.data.DataLoader(test_data_1_8x, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    test_size = test_data_1x.data_list.shape[0]
    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    first_dic = f"/code/Burger{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    # endregion
    # region model
    _trans = PARTIAL(Wrapper, [dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [idctI_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOL1dII, T)
    input_channel = config['model']['input_channel'] * config['data']['initial_step'] + 2
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)  # .to(torch.float32)
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
        checkpoint = torch.load(args.pretrain)
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        # 其他状态，如损失列表、学习率列表等
        # 获取 epoch
        epoch = checkpoint.get('epoch', checkpoint.get('epochs', None))
        print(f"{datetime.now()} --- model【{args.pretrain}】has been loaded, {epoch} epochs has trained")
    else:
        print(os.getcwd())
        checkpoint = torch.load('checkpoint-best.pth.tar')
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        epoch = checkpoint['epoch']
        print(f"{datetime.now()} --- current best model has been loaded, {epoch} epochs has trained")
    # endregion
# region evaluate
    init_t = 10
    t_train = 101
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
        '0.1-1': (init_t + 1, 101),
#         '1-1.5': (101, 151),
#         '1.5-2': (151, 201),
#         '2-2.5': (201, 251),
#         '2.5-3': (251, 301),
#         '3-3.5': (301, 351),
#         '3.5-4': (351, 401),
#         '4-4.5': (401, 451),
#         '4.5-5': (451, 501),
    }

    all_results = []
    errors_for_talk_all = {}
    pre_mode = config['train'].get('pre_mode', 'delta')

    # 新增：保存每个数据集的详细信息
    sample_details_all = {}  # {dataset_name: [{sample_name, l2_total, yy, pred}, ...]}
    selected_samples_all = {}  # 保存随机选取的好样本
    timestep_errors_all = {}  # 新增：保存每个样本在每个时间步的L2误差

    for case, test_loaders_dict in test_all.items():
        for idx, (name, test_loader) in enumerate(test_loaders_dict.items()):
            nx = nx_list[idx]
            errors_for_talk = []
            sample_details = []  # 当前数据集的样本详情
            timestep_errors_list = []  # 新增：当前数据集每个样本的时间步误差

            model.eval()
            with torch.no_grad():
                print(f'dataset: {name}, nx: {nx}')
                for b, (xx, yy, grid, sample_name) in enumerate(tqdm(test_loader, desc=name)):
                    xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                               non_blocking=True)  # 确保数据在相同设备上
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
                    outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
                    pred = torch.empty(yy.shape, device=xx.device)
                    gridt = torch.tensor(np.linspace(0, 5, t_train), dtype=torch.float32, device=xx.device).reshape(
                        t_train, 1)
                    for t in range(init_t, t_train):
                    # print("t:", t)
                        current_time = gridt[t:t + 1, :]
                        inp = xx.reshape(inp_shape)
                        current_time = current_time.view(1, 1, 1).expand(xx.size(0), xx.size(1), 1)  # 扩展为[batch, nx, 1]
                        delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                        out = xx[..., -1:, :] + delta
                        pred[..., t:t + 1, :] = out
                        xx = torch.cat((xx[..., 1:, :], out), dim=-2)
                    assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"

                    # 分段计算误差
                    _batch = yy.size(0)
                    sample_errors = []
                    for seg_name, (s, e) in segments.items():
                        _yy = yy[..., s:e, :]
                        _pred = pred[..., s:e, :]
                        err = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                        sample_errors.append(err)
                    errors_for_talk.append(sample_errors)

                    # 新增：计算每个时间步的L2误差
                    sample_timestep_errors = []
                    for t in range(init_t, t_train):
                        _yy_t = yy[..., t:t+1, :]
                        _pred_t = pred[..., t:t+1, :]
                        err_t = loss_fn(_pred_t.reshape(_batch, -1), _yy_t.reshape(_batch, -1)).item()
                        sample_timestep_errors.append(err_t)

                    # 计算整体 L2 误差（用于排序选择好样本）
                    l2_total = loss_fn(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()

                    # 处理 sample_name（可能是 tuple 或 list）
                    if isinstance(sample_name, (list, tuple)):
                        sample_name_str = sample_name[0]
                    else:
                        sample_name_str = sample_name

                    # 新增：保存时间步误差（含sample_name和index）
                    timestep_errors_list.append({
                        'sample_name': sample_name_str,
                        'index': b,
                        'timestep_errors': sample_timestep_errors,
                    })

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
            timestep_errors_all[name] = {
                'timesteps': list(range(init_t, t_train)),
                'samples': timestep_errors_list,
            }

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

            # 选择10个样本，位次分别为 0%, 10%, 20%, ..., 90%
            percentiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            selected = []
            n_total = len(sorted_samples)

            for p in percentiles:
                idx = min(int(n_total * p), n_total - 1)
                selected.append(sorted_samples[idx])

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

    # 新增：保存每个样本在每个时间步的L2误差
    with open('timestep_errors_per_sample.pkl', 'wb') as f:
        pickle.dump(timestep_errors_all, f)

    # 打印选中样本的汇总信息
    print("\n" + "=" * 60)
    print("Selected Good Samples Summary:")
    print("=" * 60)
    for name, samples in selected_samples_all.items():
        print(f"\nDataset: {name}")
        for i, s in enumerate(samples):
            print(f"  [{i + 1}] sample_name: {s['sample_name']}, l2_total: {s['l2_total']:.6e}")
            print(f"      yy shape: {s['yy'].shape}, pred shape: {s['pred'].shape}")

    print("\n测试结果已保存:")
    print("  - extend_results.csv")
    print("  - error_for_talk_all.pkl")
    print("  - sample_l2_summary.pkl (每个样本的L2误差)")
    print("  - selected_good_samples.pkl (选中的好样本，含yy和pred)")
    print("  - timestep_errors_per_sample.pkl (每个样本在每个时间步的L2误差)")
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
