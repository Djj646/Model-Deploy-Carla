
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))
import random
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from learning.model_train_path import TrajFrom2Pic
from learning.dataset_gan_path import ImgRouteDataset
from utils import write_params

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

# 计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 设置交互命令
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="img2path-01", help='name of the train')
parser.add_argument('--dataset_path', type=str, default="./dataset/", help='path of the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--n_cpu', type=int, default=16, help='number of CPU threads to use during batches generating')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--train_time', type=int, default=1e3, help='total training time')
parser.add_argument('--gamma', type=float, default=0.1, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.001, help='xy and axy loss trade off')
parser.add_argument('--checkpoints_interval', type=int, default=2000, help='interval between model checkpoints')
parser.add_argument('--clip_value', type=float, default=1, help='Clip value for training to avoid gradient vanishing')
parser.add_argument('--adjust_dist', type=float, default=25., help='max distance') # 最大距离max_dist
parser.add_argument('--adjust_t', type=float, default=3., help='max time') # 最大时间max_t
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')

opt = parser.parse_args()

description = 'train'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

logger = SummaryWriter(log_dir=log_path)
write_params(log_path, parser, description)

""" 模型TrajFrom2Pic加载 """
model = TrajFrom2Pic().to(device)
# 如果别的模型，就把TrajFrom2Pic换成别的

""" 训练集加载 """
train_loader = DataLoader(ImgRouteDataset(data_index=[7], opt=opt), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

""" 验证集加载 """
test_loader = DataLoader(ImgRouteDataset(data_index=[1], opt=opt, evalmode=True), batch_size=1, shuffle=False, num_workers=1)
test_samples = iter(test_loader)  # 设置迭代器

""" 设置损失函数MSE和优化器Adam """
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

total_step = 0
print("Start Training ...")

for epoch in range(opt.epoch, opt.n_epochs):
    print(f"epoch: {epoch}")
    
    # enumerate返回index, item
    bar = enumerate(train_loader)
    
    length = len(train_loader)
    bar = tqdm(bar, total=length)
    
    for i, batch in bar:
        total_step += 1
        """ 数据移动到Cuda,['img']和['t']需要计算梯度 """
        batch['img'] = batch['img'].to(device)
        batch['routine'] = batch['routine'].to(device)
        batch['t'] = batch['t'].to(device)
        batch['v_0'] = batch['v_0'].to(device)
        batch['xy'] = batch['xy'].to(device)
        batch['vxy'] = batch['vxy'].to(device)
        batch['axy'] = batch['axy'].to(device)

        batch['img'].requires_grad = True
        batch['t'].requires_grad = True

        # 模型输出y_hat
        output = model(batch['img'], batch['routine'], batch['t'], batch['v_0'])

        # 输出控制信息
        # vx, vy和ax, ay
        vx = grad(output[:, 0].sum(), batch['t'], create_graph=True)[0]
        vy = grad(output[:, 1].sum(), batch['t'], create_graph=True)[0]
        # tensor拼接
        output_vxy = torch.cat([vx, vy], dim=1)
        output_vxy = (opt.adjust_dist / opt.adjust_t) * output_vxy
        # 反归一化，在dataset里面，xy除了opt.adjust_dist，t除了opt.adjust_t

        ax = grad(vx.sum(), batch['t'], create_graph=True)[0]
        ay = grad(vy.sum(), batch['t'], create_graph=True)[0]
        output_axy = torch.cat([ax, ay], dim=1)
        output_axy = (1. / opt.adjust_t) * output_axy
        # 反归一化，在dataset里面，t除了opt.adjust_t

        optimizer.zero_grad()

        # 损失计算
        loss_xy = criterion(output, batch['xy'])
        loss_vxy = criterion(output_vxy, batch['vxy'])
        loss_axy = criterion(output_axy, batch['axy'])
        loss = loss_xy + opt.gamma * loss_vxy + opt.gamma2 * loss_axy

        # 反向
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=opt.clip_value)
        # 防止梯度消失

        optimizer.step()

        logger.add_scalar('train/loss_xy', loss_xy.item(), total_step)
        logger.add_scalar('train/loss_vxy', loss_vxy.item(), total_step)
        logger.add_scalar('train/loss_axy', loss_axy.item(), total_step)
        logger.add_scalar('train/loss', loss.item(), total_step)

        # if total_step % opt.test_interval == 0:
        #     eval_error(total_step)  # 没搞懂这个函数想要干嘛

        if total_step % opt.checkpoints_interval == 0:
            torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth' % (opt.dataset_name, total_step))

