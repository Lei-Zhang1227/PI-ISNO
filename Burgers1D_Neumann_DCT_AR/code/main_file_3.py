"""
10/30:
对时间和空间维度都进行降维缩放；
11/5:
添加了test部分；
11/6 存档；
1. 统一分辨率的data loss和residual loss
"""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 添加上一级目录到 sys.path
sys.path.insert(0, parent_dir)
from argparse import ArgumentParser
import yaml
from functools import partial as PARTIAL
from model import SOLII
from datasets import BurgersLoader
from loss import *
import tqdm
from utils import *
import time
import shutil
from tqdm import tqdm
from datetime import datetime


def run(config):
    ################################################################
    # prepare
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{datetime.now()}--- set divice: {device} ---')
    torch.manual_seed(config['prepare']['seed'])
    np.random.seed(config['prepare']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['prepare']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f'{datetime.now()} --- set seed ---')
    ################################################################
    # dataloader
    ################################################################
    data_config = config['data']
    dataset = BurgersLoader(data_config['datapath'],
                            nx=data_config['nx'], nt=data_config['nt'],
                            sub_x=data_config['sub_x'], sub_t=data_config['sub_t'], new=False)
    train_loader = dataset.make_loader(n_sample=data_config['n_sample'],
                                       batch_size=config['train']['batchsize'],
                                       start=data_config['offset'])
    v = dataset.v
    print(v)
    batch_size = config['train']['batchsize']
    n_sample = data_config['n_sample']
    test_loader = dataset.make_loader(n_sample=data_config['total_num'] - data_config['n_sample'],
                                      batch_size=config['train']['batchsize'],
                                      start=data_config['n_sample'], train=False)
    print(f'{datetime.now()} --- set dataset，batch size: {batch_size}, loader lens：{len(train_loader)}, data size：{n_sample}')
    ################################################################
    # location
    ################################################################
    first_dic = f"./{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    shutil.copy('./information.yaml', f"./{config['prepare']['project']}")
    os.chdir(first_dic)
    print(f'{datetime.now()} --- set save dir ---')
    ################################################################
    # model
    ################################################################
    # 定义使用的离散变换
    _trans = PARTIAL(Wrapper, [fft_fun, dctII_SPFNO])
    _itrans = PARTIAL(Wrapper, [ifft_fun, idctII_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOLII, T)
    model = Model(config['model']['input_channel'], config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    ################################################################
    # 定义优化器
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                 lr=config['train']['base_lr'])
    now_lr = config['train']['base_lr']
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=config['train']['milestones'],
    #                                                  gamma=config['train']['scheduler_gamma'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=config['train']['scheduler_gamma'],
                                                           patience=config['train']['patience'],
                                                           threshold=1e-4,
                                                           threshold_mode='rel', cooldown=5,
                                                           min_lr=1e-8,
                                                           eps=1e-8, verbose=True)
    print(f'{datetime.now()} --- set optimizer, scheduler ---')
    ################################################################
    # loss
    ################################################################
    print(f'{datetime.now()} --- training start ---')
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    loss_list = []
    test_loss_list = []
    lr_list = []
    model_save_record = []
    test_Loss_data = 100.0
    pbar = range(config['train']['epochs'])
    if config['train']['use_tqdm'] == 0:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    for e in pbar:
        Loss_f = 0.0
        Loss_init = 0.0
        Loss_data = 0.0
        Loss_all = 0.0
        # time_5 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # 确保数据在相同设备上
            # time_0 = time.time()
            # print(f'循环回来耗时：{time_0 - time_5:.2f}')
            optimizer.zero_grad()
            out = model(x)
            out = out.reshape(y.shape)
            # time_1 = time.time()
            # print(f'模型推理耗时：{time_1 - time_0:.2f}')
            if config['train']['loss_mode'] == 'both':
                loss_data = F.mse_loss(out, y)
                # time_2 = time.time()
                loss_init, loss_f = PINO_loss_1D(out, x[:, 0, :, 0], v)
                # time_3 = time.time()
                # print(f'计算物理损失耗时：{time_3 - time_2:.2f}')
            elif config['train']['loss_mode'] == 'data':
                loss_init, loss_f = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                loss_data = F.mse_loss(out, y)
            else:
                loss_init, loss_f = PINO_loss_1D(out, x[:, 0, :, 0], v)
                loss_data = torch.tensor(0.0, device=device)
            total_loss = loss_init * ic_weight + loss_f * f_weight + loss_data * data_weight
            assert not torch.isnan(total_loss).any(), "NaN in loss"
            # time_4 = time.time()

            total_loss.backward()
            optimizer.step()
            # scaler.scale(total_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # time_5 = time.time()
            # print(f'反向传播耗时：{time_5 - time_4:.2f}')

            Loss_data += loss_data.item()
            Loss_init += loss_init.item()
            Loss_f += loss_f.item()
            Loss_all += total_loss.item()
            # time_6 = time.time()
            # print(f'累加loss耗时：{time_6 - time_5:.1f}')

        scheduler.step(total_loss)
        Loss_data /= len(train_loader)
        Loss_init /= len(train_loader)
        Loss_f /= len(train_loader)
        Loss_all /= len(train_loader)
        pbar.set_description(
            (f"epoch: {e + 1}, loss: {Loss_all:.5e}, loss_data: {Loss_data:.5e}, loss_PDE: {Loss_f:.5e},"
             f",loss_init: {Loss_init:.5e}, loss_test: {test_Loss_data:.5e}, now_lr: {now_lr:.2e}"))

        if e % config['train']['verbose_interval'] == 0:
            model.eval()
            now_lr = optimizer.state_dict()['param_groups'][0]['lr']  # 当前学习率查看
            loss_item = [e, Loss_all, Loss_init, Loss_f, Loss_data]
            loss_list.append(loss_item)
            lr_list.append(now_lr)
            # model test
            test_Loss_f = 0.0
            test_Loss_init = 0.0
            test_Loss_data = 0.0
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # 确保数据在相同设备上
                out = model(x)
                out = out.reshape(y.shape)
                test_loss_data = myloss(out, y)
                test_loss_init, test_loss_f = PINO_loss_1D(out, x[:, 0, :, 0], v)
                test_Loss_data += test_loss_data.item()
                test_Loss_init += test_loss_init.item()
                test_Loss_f += test_loss_f.item()
            test_Loss_data /= len(test_loader)
            test_Loss_init /= len(test_loader)
            test_Loss_f /= len(test_loader)
            test_loss_item = [e, test_Loss_init, test_Loss_f, test_Loss_data]
            test_loss_list.append(test_loss_item)
            save_checkpoint(model, e, optimizer, scheduler, loss_list,
                            lr_list, test_loss_list, filename=f'checkpoint')
            if e % config['train']['check_epochs'] == 0:
                save_checkpoint(model, e, optimizer, scheduler, loss_list,
                                lr_list, test_loss_list, filename=f'checkpoint-{e}')
            model.train()
    print(f'{datetime.now()} --- training succeed ---')


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='./information.yaml', help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(config)
