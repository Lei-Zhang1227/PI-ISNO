import os, sys
import pickle
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
sys.path.append(os.path.abspath('..'))
from argparse import ArgumentParser
import yaml
import tqdm
from tqdm import tqdm
from loss import *
import pandas as pd
from utilsII import *
from NOs_dict.models import CosNO_II as Model
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import gc
import torch

gc.collect()
torch.cuda.empty_cache()

CASE_ROOT = Path(__file__).resolve().parents[1]


def case_path(path_like):
    path = Path(path_like)
    return str(path if path.is_absolute() else CASE_ROOT / path)


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
    ################################################################
    # dataloader
    ################################################################
    data_config = config['data']
    origin_nx = data_config['nx']
    initial_step = data_config['initial_step']
    print(f'initial_step:{initial_step}')
    file_path = case_path(config['data'].get('extend_datapath', 'data/2D_diff-react_full_0_15.h5'))
    test_data_unseen = FNODatasetMult_B(file_path=file_path,
                                        initial_step=10,
                                        sub_x=1,
                                        sub_t=1,
                                        if_test=True,
                                        )
    test_loader_unseen = torch.utils.data.DataLoader(test_data_unseen, batch_size=1, shuffle=False,
                                                     num_workers=0, pin_memory=True)
    test_data_seen = FNODatasetMult_B(file_path=file_path,
                                      initial_step=10,
                                      sub_x=1,
                                      sub_t=1,
                                      if_test=False,
                                      )
    test_loader_seen = torch.utils.data.DataLoader(test_data_seen, batch_size=1, shuffle=False,
                                                   num_workers=0, pin_memory=True)

    # endregion
    # region location
    ################################################################
    # location
    ################################################################
    if args.pkl is not None:
        first_dic = case_path("result/data_case")
        if not os.path.exists(first_dic):
            os.makedirs(first_dic)
        os.chdir(first_dic)
        print(f"{datetime.now()} --- set save dir :{first_dic} ---")
        dtype = torch.float32
        model = torch.load(case_path(args.pkl))
    else:
        first_dic = case_path(config['prepare']['project'])
        if not os.path.exists(first_dic):
            os.makedirs(first_dic)
        os.chdir(first_dic)
        print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
        # endregion
        # region model
        dtype = torch.float32
        modes = config['model']['modes']
        width = config['model']['width']
        bandwidth = config['model']['bandwidth']
        model = Model(10 * 2 + 3, modes, width, bandwidth, out_channels=2, dim=2, triL=0).to(device)
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
            model_name = args.pretrain[:-8]
            model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
            checkpoint = torch.load(case_path(args.pretrain))
            # Load model and training state from checkpoint.
            model.load_state_dict(checkpoint['model'])  # 加载模型参数
            # Auxiliary state: loss history, learning-rate history, and gradients.
            loss_list = checkpoint['loss_list']  # checkpoint['loss_for_train']
            test_loss_list = checkpoint.get('test_loss_list')  # checkpoint.get('test_loss_list')
            grad_array = checkpoint['grad']
            if test_loss_list is None:
                print("[Warning] Checkpoint does not contain 'test_loss_list'.")
                test_loss_list = []
            lr_list = checkpoint.get('lr_list')
            if lr_list is None:
                print("[Warning] Checkpoint does not contain 'lr_list'.")
                lr_list = []

            # 获取 epoch
            epoch = checkpoint.get('epoch', checkpoint.get('epochs', None))
            if epoch is None:
                print("[Warning] Checkpoint does not contain 'epoch' or 'epochs'.")
                epoch = 0
            print(f"{datetime.now()} --- model【{args.pretrain}】has been loaded, {epoch} epochs has trained")
        else:
            print(os.getcwd())
            checkpoint = torch.load(os.path.join(first_dic, 'checkpoint-best.pth.tar'))
            grad_array = checkpoint['grad']
            # Load model and training state from checkpoint.
            model.load_state_dict(checkpoint['model'])  # 加载模型参数
            # Auxiliary state: loss history, learning-rate history, and gradients.
            loss_list = checkpoint['loss_list']
            test_loss_list = checkpoint['test_loss_list']
            lr_list = checkpoint['lr_list']
            print('len(loss_list0,len(test_loss_list),len(lr_list)):', len(loss_list), len(test_loss_list),
                  len(lr_list))
            # 获取 epoch
            epoch = checkpoint['epoch']
            model_name = 'checkpoint-best'
            print(f"{datetime.now()} --- current best model has been loaded, {epoch} epochs has trained")
        # endregion

    # region evaluate
    init_t = 10
    t_train = 301
    myloss = LpLoss(size_average=True)
    loss_fn = myloss
    test_loaders = {
        f'unseen_extend': test_loader_unseen,
        f'seen_extend': test_loader_seen,
    }

    # 新增：保存所有数据集的选中样本
    selected_samples_all = {}
    # 新增：保存每个时间步的平均L2损失
    timestep_errors_all = {}

    for name, test_loader in test_loaders.items():
        errors_for_talk = []
        sample_details = []  # 新增：保存每个样本的详细信息
        timestep_errors_list = []  # 新增：保存每个样本在每个时间步的L2误差

        model.eval()  # 将模型设置为评估模式
        first = True
        with torch.no_grad():
            test_iter = iter(test_loader)
            for b in tqdm(range(len(test_loader))):
                xx, yy, grid = next(test_iter)
                xx = xx.to(device, dtype=dtype, non_blocking=True)
                yy = yy.to(device, dtype=dtype, non_blocking=True)
                grid = grid.to(device, dtype=dtype, non_blocking=True)
                if first:
                    print(f'xx.shape:{xx.shape},yy.shape:{yy.shape},grid.shape:{grid.shape}')
                    first = False
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
                outp_shape = inp_shape[:-1] + [1, -1]  # Append [1, -1] to obtain [b, nx, 1, -1].

                pred = torch.empty(yy.shape, device=xx.device)
                gridt = torch.tensor(np.linspace(0, 3, t_train), dtype=dtype, device=xx.device).reshape(
                    t_train, 1)
                pred[..., :initial_step, :] = yy[..., :initial_step, :]
                for t in range(init_t, t_train):
                    current_time = gridt[t:t + 1, :]
                    inp = xx.reshape(inp_shape)
                    current_time = current_time.view(1, 1, 1, 1).expand(xx.size(0), xx.size(1), xx.size(2),
                                                                        1)  # 扩展为[batch, nx, 1]
                    delta = model(torch.cat([inp, grid, current_time], dim=-1)).reshape(outp_shape)
                    last_step = xx[..., -1:, :]
                    out = last_step + delta
                    pred[..., t:t + 1, :] = out
                    xx = torch.cat((xx[..., 1:, :], out), dim=-2)
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"

                segments = {
                    '0-5': (init_t + 1, 101),
                    '5-6': (101, 121),
                    '6-7': (121, 141),
                    '7-8': (141, 161),
                    '8-9': (161, 181),
                    '9-10': (181, 201),
                    '10-11': (201, 221),
                    '11-12': (221, 241),
                    '12-13': (241, 261),
                    '13-14': (261, 281),
                    '14-15': (281, 301),
                }

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
                    _yy_t = yy[..., t:t + 1, :]
                    _pred_t = pred[..., t:t + 1, :]
                    err_t = loss_fn(_pred_t.reshape(_batch, -1), _yy_t.reshape(_batch, -1)).item()
                    sample_timestep_errors.append(err_t)
                timestep_errors_list.append(sample_timestep_errors)

                # Save total L2 error and per-sample details.
                l2_total = loss_fn(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                sample_details.append({
                    'index': b,
                    'l2_total': l2_total,
                    'segment_errors': sample_errors,
                    'timestep_errors': sample_timestep_errors,  # 新增：保存时间步误差
                    'yy': yy.squeeze().cpu().numpy(),
                    'pred': pred.squeeze().cpu().numpy(),
                })

        errors_for_talk = np.array(errors_for_talk)  # [n_samples, 11]
        timestep_errors_array = np.array(timestep_errors_list)  # [n_samples, t_train - init_t]

        seg_names = ['0-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11', '11-12', '12-13', '13-14', '14-15']

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
        print(df.to_string(index=False))

        # 新增：计算并保存每个时间步的平均L2损失
        timestep_mean_errors = timestep_errors_array.mean(axis=0)  # [t_train - init_t]
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

        # Save timestep error data for later analysis.
        timestep_errors_all[name] = {
            'timesteps': list(range(init_t, t_train)),
            'mean': timestep_mean_errors,
            'std': timestep_std_errors,
            'max': timestep_max_errors,
            'min': timestep_min_errors,
            'all_samples': timestep_errors_array,  # 可选：保存所有样本的数据
        }

        # 新增：按 L2 误差排序，选择指定百分位的样本
        sorted_samples = sorted(sample_details, key=lambda x: x['l2_total'])

        # Select samples at 0%, 1%, 2%, 5%, 10%, and 30% ranks.
        percentiles = [0, 0.01, 0.02, 0.05, 0.10, 0.30]
        percentile_names = ['0%', '1%', '2%', '5%', '10%', '30%']
        selected = []
        n_total = len(sorted_samples)

        for p, p_name in zip(percentiles, percentile_names):
            idx = min(int(n_total * p), n_total - 1)
            sample = sorted_samples[idx].copy()
            sample['percentile'] = p_name
            sample['rank'] = idx
            selected.append(sample)

        selected_samples_all[name] = selected

        # 打印选中样本信息
        print(f"\n{name} - Selected samples for visualization:")
        for s in selected:
            print(f"  {s['percentile']:>4} (rank {s['rank']:>3}): index={s['index']}, l2_total={s['l2_total']:.6e}")
            print(f"       yy shape: {s['yy'].shape}, pred shape: {s['pred'].shape}")

    # 保存选中的样本用于可视化
    with open('selected_samples_for_visualization.pkl', 'wb') as f:
        pickle.dump(selected_samples_all, f)

    # 新增：保存时间步误差数据
    with open('timestep_errors.pkl', 'wb') as f:
        pickle.dump(timestep_errors_all, f)

    print("\n" + "=" * 60)
    print("Visualization data saved to selected_samples_for_visualization.pkl")
    print("Timestep error data saved to timestep_errors.pkl")
    print("=" * 60)
    # endregion


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='../yaml/information.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--pkl', default=None, help='Turn on the wandb')
    parser.add_argument('--mode', type=str, default='test_extend', help='train or test')
    parser.add_argument('--pretrain', type=str, default='result/exp/checkpoint-best.pth.tar',
                        help='pretrain model path')
    parser.add_argument('--load_lr', action='store_true', help='pretrain model path')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        try:
            with open(config_file, encoding="utf-8") as stream:
                config = yaml.load(stream, yaml.FullLoader)
        except UnicodeDecodeError:
            # Fallback for archived configs saved with GB18030/GBK-compatible encoding.
            with open(config_file, encoding="gb18030") as stream:
                config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'testII':
        testII(config, args)
