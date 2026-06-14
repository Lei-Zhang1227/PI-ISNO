"""
24/10/22：
模型：
输入某一时刻的值，得到下一时刻的值；
训练：
自回归训练
25/01/14
1.由于在模型中没有添加flat，所以在info中width要保持相同的数，不然会报错；
2.由于是向量自回归，时间步的大小最好固定；
3.和file3不同，这里模型输出值最后拼起来的大小是[b,nx,nt,1]，所以在计算f的时候，对数据进行转置处理；
"""
import os
import sys

sys.path.append(os.path.abspath('..'))
print(sys.path)
from argparse import ArgumentParser
import yaml
from functools import partial as PARTIAL
# from model import SOL1d
from loss import *
import tqdm
from utils import *
import time
import shutil
from tqdm import tqdm
import re
from tqdm import trange
from Burger.model import SOL1dII
from Burger.datasets import h5DatasetFor1DBurgersII
from Burger.loss import *
from Burger.utils import *
from datetime import datetime


class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        # 清理控制字符
        cleaned_instr = re.sub('\n|\x1b.*|\r', '', instr)
        # 将清理后的信息存储到 _desc
        self._desc += cleaned_instr

    def read(self, b):
        ret = self._desc
        self._desc = f'batch {b}:'
        return ret

    def flush(self):
        pass


def run(config):
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
    ################################################################
    # dataloader
    ################################################################
    ################################################################
    data_config = config['data']
    batch_size = config['train']['batchsize']
    v = 0.01
    train_data = h5DatasetFor1DBurgersII(data_config['datapath'],
                                         sub_x=data_config['sub_x'],
                                         sub_t=data_config['sub_t'],
                                         initial_step=data_config['initial_step'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=128, pin_memory=True)
    test_data = h5DatasetFor1DBurgersII(data_config['datapath'],
                                        sub_x=data_config['sub_x'],
                                        sub_t=data_config['sub_t'],
                                        initial_step=data_config['initial_step'], if_test=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=128, pin_memory=True)

    #     train_size, test_size = train_data.data_list.shape[0], test_data.data_list.shape[0]
    #     print('size-of-train/val:', train_size, test_size)
    print(
        f'{datetime.now()} --- set dataset，batch size: {batch_size}, Train loader lens：{len(train_loader)}, Test loader lens：{len(test_loader)}')
    ################################################################
    # location
    ################################################################
    first_dic = f"./{config['prepare']['project']}"
    if not os.path.exists(first_dic):
        os.makedirs(first_dic)
    if args.pretrain is not None:
        shutil.copy(f'./info_{args.pretrain}.yaml', f"./{config['prepare']['project']}")
    else:
        shutil.copy('./information.yaml', f"./{config['prepare']['project']}")
    os.chdir(first_dic)
    print(f"{datetime.now()} --- set save dir :{config['prepare']['project']} ---")
    ################################################################
    # model
    ################################################################
    # 这里有所不同的是，不再将这个一维含时的问题视为二维问题
    _trans = PARTIAL(Wrapper, [dctI_SPFNO])
    _itrans = PARTIAL(Wrapper, [idctI_SPFNO])
    T = Transform(_trans, _itrans)
    # 定义模型
    Model = PARTIAL(SOL1dII, T)
    input_channel = config['model']['input_channel']*config['data']['initial_step']+1
    model = Model(input_channel, config['model']['modes'], config['model']['width'],
                  config['model']['bandwidth'], out_channels=config['model']['output_channel'],
                  dim=config['model']['dim'], triL=config['model']['triL']).to(device)  # .to(torch.float32)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now()} --- set model. Total trainable parameters: {total_params}")
    ################################################################
    # 定义优化器
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                 lr=config['train']['base_lr'])
    now_lr = config['train']['base_lr']
    if config['train']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config['train']['milestones'],
                                                         gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['train']['gamma'],
                                                               threshold=1e-2, patience=config['train']['patience'],
                                                               verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['patience'],
                                                    gamma=config['train']['gamma'])

    print(f'{datetime.now()} --- set optimizer, scheduler ---')
    ################################################################
    # load model
    ################################################################
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        # 从 checkpoint 中提取模型、优化器和其他状态
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
        scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率调度器状态

        # 其他状态，如损失列表、学习率列表等
        loss_list = checkpoint['loss_list']
        test_loss_list = checkpoint['test_loss_list']
        lr_list = checkpoint['lr_list']

        # 获取 epoch
        epoch = checkpoint['epoch']
        print(f'模型【{args.pretrain}】已加载')
    else:
        loss_list = []
        test_loss_list = []
        lr_list = []
    ################################################################
    # loss
    ################################################################
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    init_t = int(config['train']['init_t'])
    t_train = (data_config['nt'] - 1) // data_config['sub_t'] + 1
    model_save_record = []
    best_error = 100.0
    myloss = LpLoss(size_average=True)
    ebar = trange(config['train']['epochs'], desc="Epoch")
    if args.pretrain is not None:
        ebar = trange(epoch, epoch + config['train']['epochs'], desc="Epoch")
    rx = int(config['data']['data_sub_x'] / config['data']['sub_x'])
    rt = int(config['data']['data_sub_t'] / config['data']['sub_t'])
    x_length, time_lentgh = config['data']['x_length'], config['data']['t_length']
    desc = DescStr()
    model.train()
    for e in ebar:
        # 内层进度条
        Loss_f = 0.0
        Loss_init = 0.0
        Loss_data = 0.0
        Loss_all = 0.0
        count=0
        for b in trange(len(train_loader), file=desc, desc="batch"):
            """
            x:[b, nx, self.initial_step，1]
            y:[b, nx, nt, 1]
            grid:[b, nx, 1]
            """
            loss = 0.0
            xx, yy, grid = next(iter(train_loader))
            xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                       non_blocking=True)  # 确保数据在相同设备上
            optimizer.zero_grad()
            init_x = xx.squeeze(-1)[:, :, 0:1]
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
            outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
            pred = yy[..., 0:init_t, :]
            for t in range(init_t, t_train):
                inp = xx.reshape(inp_shape)
                y = yy[..., t:t + 1, :]
                out = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
                _batch = out.size(0)
                loss += myloss(out[:,::rx,...].reshape(_batch, -1), y[:,::rx,...].reshape(_batch, -1))
                pred = torch.cat((pred, out), -2)
                xx = torch.cat((xx[..., 1:, :], out), dim=-2)
            assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
            if config['train']['loss_mode'] == 'both':
                out_data = pred[:, ::rt, ::rx,:]
                y_data = yy[:, ::rt, ::rx,:]
                loss_data = F.mse_loss(out_data, y_data).to(torch.float32)
#                 print(f'pred.shape:{pred.shape}, init_x.shape:{init_x.shape}')
                loss_init, loss_f, loss_b = PINO_loss_1DII(pred.permute(0,2,1,3).squeeze(), init_x.permute(0,2,1), v, x_length, time_lentgh)
                total_loss = (loss_init * ic_weight + loss_f * f_weight + loss * data_weight).to(torch.float32)
                total_loss.backward()
            elif config['train']['loss_mode'] == 'data':
                loss_init, loss_f = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                out_data = pred[:, ::rx, ::rt,:]
                y_data = yy[:, ::rx, ::rt,:]
                loss_data = F.mse_loss(out_data, y_data).to(torch.float32)
                total_loss = (loss_init * ic_weight + loss_f * f_weight + loss_data * data_weight).to(torch.float32)
                loss.backward()
            else:
                loss_init, loss_f, loss_b = PINO_loss_1DII(pred.permute(0,2,1,3).squeeze(), init_x.permute(0,2,1), v, x_length, time_lentgh)
                loss_data = torch.tensor(0.0, device=device)
                total_loss = (loss_init * ic_weight + loss_f * f_weight + loss_data * data_weight).to(torch.float32)
                total_loss.backward()
                assert not torch.isnan(total_loss).any(), "NaN in loss"
            optimizer.step()

            Loss_data = (Loss_data * count + loss_data) / (count + 1)
            Loss_init = (Loss_init * count + loss_init) / (count + 1)
            Loss_f = (Loss_f * count + loss_f) / (count + 1)
            Loss_all = (Loss_all * count + total_loss) / (count + 1)
            count += 1
            new_desc = f"epoch {e + 1}: {desc.read(b)},Loss: {loss.item():.4e}, loss_init: {Loss_init.item():.4e}, loss_PDE: {Loss_f.item():.4e},loss_Data: {Loss_data.item():.4e}"
            ebar.set_description(new_desc)
        if config['train']['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(Loss_all)
        else:
            scheduler.step()
        if best_error > Loss_data:
            best_error = Loss_data
            model_save_record.append([e, Loss_data])
        if e % config['train']['verbose_interval'] == 0:
            model.eval()
            now_lr = optimizer.state_dict()['param_groups'][0]['lr']  # 当前学习率查看
            loss_item = [e, Loss_all, Loss_init, Loss_f, Loss_data]
            loss_list.append(loss_item)
            lr_list.append(now_lr)
            test_Loss_f = 0.0
            test_Loss_init = 0.0
            test_Loss_data = 0.0
            test_Loss_b = 0.0
            count=0
            for xx, yy, grid in test_loader:
                """
                            x:[b, nx, self.initial_step，1]
                            y:[b, nx, nt, 1]
                            grid:[b, nx, 1]
                            """
                xx, yy, grid = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True), grid.to(device,
                                                                                                           non_blocking=True)  # 确保数据在相同设备上
                init_x = xx[:, 0, :, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)  # [b, nx, -1]，等于合并剩余的维度
                outp_shape = inp_shape[:-1] + [1, -1]  # 最后添加 [1, -1] 得到 [b, nx, 1, -1]
                pred = yy[..., 0:init_t, :]
                init_x = xx.squeeze(-1)[:, :, 0:1]
                for t in range(init_t, t_train):
                    inp = xx.reshape(inp_shape)
                    y = yy[..., t:t + 1, :]
                    out = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
                    pred = torch.cat((pred, out), -2)
                    xx = torch.cat((xx[..., 1:, :], out), dim=-2)
                assert pred.shape == yy.shape, f"Tensor shapes do not match: {pred.shape} != {yy.shape}"
                out_data = pred[:, ::rt, ::rx, :]
                y_data = yy[:, ::rt, ::rx, :]
                test_loss_data = myloss(out_data, y_data)
                test_loss_init, test_loss_f, test_loss_b = PINO_loss_1DII(pred.permute(0,2,1,3).squeeze(), init_x.permute(0,2,1), v, x_length, time_lentgh)
                test_Loss_data = (test_Loss_data * count + test_loss_data) / (count + 1)
                test_Loss_init = (test_Loss_init * count + test_loss_init) / (count + 1)
                test_Loss_b = (test_Loss_b * count + test_loss_b) / (count + 1)
                test_Loss_f = (test_Loss_f * count + test_loss_f) / (count + 1)
                count += 1
            test_loss_item = [e, test_Loss_init, test_Loss_f, test_Loss_data,test_Loss_b]
            test_loss_list.append(test_loss_item)

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
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
    parser.add_argument('--load_lr', action='store_true', help='pretrain model path')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(config)
