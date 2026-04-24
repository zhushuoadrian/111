# -*- coding: utf-8 -*-
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.LoL_DataLoader import TrainData_for_LOLv2Real, TestData_for_LOLv2Real
from numpy import *
from pytorch_msssim import ssim, SSIM
from models import *
import random
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# ============== 固定随机种子函数 ==============
def set_seed(seed=8001):
    """
    固定所有随机种子，确保实验可重复
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在最开始就固定种子
set_seed(8001)
# ============================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sfhformer_lol_m', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='lowlight', type=str, help='experiment setting')
args = parser.parse_args()

# 🔥 新增：Charbonnier Loss (比 L1 更容易收敛出细节)
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

# 🔥 升级训练函数：混合Loss + EMA + 梯度裁剪 + NaN检测
def train(train_loader, network, criterion_char, criterion_ssim, optimizer, ema_model):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()
    loop = tqdm(train_loader, leave=False)

    for batch in loop:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        pred_img = network(source_img)
        label_img = target_img

        # 🔥 混合 Loss 计算
        # 1. Charbonnier Loss (内容重建)
        loss_char = criterion_char(pred_img, label_img)

        # 2. SSIM Loss (结构相似度，直接提升 PSNR)
        loss_ssim = 1 - criterion_ssim(pred_img, label_img)

        # 3. FFT Loss (频域一致性)
        label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)
        pred_fft3 = torch.fft.fft2(pred_img, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)
        loss_fft = criterion_char(pred_fft3, label_fft3)

        # 🔥 权重配比（和第一个代码完全一致）
        loss = loss_char + 0.2 * loss_ssim + 0.05 * loss_fft

        # 损失NaN/Inf检测，防止训练崩溃
        if not torch.isfinite(loss):
            print("⚠️ Warning: Loss is NaN or Inf, skipping this batch!")
            optimizer.zero_grad()
            continue

        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，稳定训练
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1)
        optimizer.step()
        # EMA模型更新
        ema_model.update_parameters(network)

        loop.set_description(f"Loss: {loss.item():.4f}")

    return losses.avg


def valid(val_loader_full, network):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader_full:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            output = network(source_img).clamp_(0, 1)

        mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr_full = 10 * torch.log10(1 / mse_loss).mean()
        PSNR_full.update(psnr_full.item(), source_img.size(0))

        ssim_full = ssim(output, target_img, data_range=1, size_average=False).mean()
        SSIM_full.update(ssim_full.item(), source_img.size(0))

    return PSNR_full.avg, SSIM_full.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    print(setting_filename)
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    device_index = [0]
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=device_index).cuda()

    # 🔥 EMA 模型（指数移动平均，提升模型泛化能力）
    ema_model = AveragedModel(network, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    # 🔥 初始化高级损失函数
    criterion_char = CharbonnierLoss().cuda()
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3).cuda()

    # AdamW optimizer with betas=(0.9, 0.999) and initial lr=1e-3, as in your paper.
    if setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, betas=(0.9, 0.999))
    elif setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    # Cosine annealing learning rate schedule, from 1e-3 to 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=setting['epochs'],
        eta_min=1e-6
    )

    # ============== 固定 DataLoader 的随机种子 ==============
    def worker_init_fn(worker_id):
        worker_seed = 8001 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(8001)
    # ====================================================

    # ===================== 完全保留你的LOLv2数据集路径 ==================
    train_data_dir = '/root/lanyun-tmp/data/LOLv2/Real_captured/Train'
    test_data_dir = '/root/lanyun-tmp/data/LOLv2/Real_captured/Test'
    train_dataset = TrainData_for_LOLv2Real(256, train_data_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              sampler=RandomSampler(train_dataset, num_samples=setting['batch_size'] * 20,
                                                    replacement=False, generator=generator),
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True,
                              worker_init_fn=worker_init_fn)
    test_dataset = TestData_for_LOLv2Real(8, test_data_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=20,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    test_str = 'train_lolv2r'

    if not os.path.exists(os.path.join(save_dir, args.model + test_str + '.pth')):
        print('==> Start training, current model name: ' + args.model)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model, test_str))

        best_psnr = 0
        best_ssim = 0
        best_psnr_ema = 0
        best_ssim_ema = 0

        for epoch in tqdm(range(setting['epochs'] + 1)):
            # 调用升级后的训练函数
            train_loss = train(train_loader, network, criterion_char, criterion_ssim, optimizer, ema_model)
            writer.add_scalar('train_loss', train_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', current_lr, epoch)

            torch.save({'state_dict': network.state_dict()},
                       os.path.join(save_dir, args.model + test_str + '_newest' + '.pth'))

            if epoch % setting['eval_freq'] == 0:
                # 验证原始模型
                avg_psnr, avg_ssim = valid(test_loader, network)
                print(f"Epoch {epoch} | Normal: PSNR {avg_psnr:.4f} SSIM {avg_ssim:.4f}")
                # 验证EMA模型
                avg_psnr_ema, avg_ssim_ema = valid(test_loader, ema_model)
                print(f"Epoch {epoch} | EMA    : PSNR {avg_psnr_ema:.4f} SSIM {avg_ssim_ema:.4f}")

                # 写入TensorBoard
                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('valid_ssim', avg_ssim, epoch)
                writer.add_scalar('valid_psnr_ema', avg_psnr_ema, epoch)
                writer.add_scalar('valid_ssim_ema', avg_ssim_ema, epoch)

                # 保存最佳原始模型
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + test_str + '_best' + '.pth'))
                writer.add_scalar('best_psnr', best_psnr, epoch)

                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                writer.add_scalar('best_ssim', best_ssim, epoch)

                # 保存最佳EMA模型（效果通常更好）
                if avg_psnr_ema > best_psnr_ema:
                    best_psnr_ema = avg_psnr_ema

                    if isinstance(ema_model, AveragedModel):
                        real_model = ema_model.module
                        if isinstance(real_model, torch.nn.DataParallel):
                            save_content = real_model.module.state_dict()
                        else:
                            save_content = real_model.state_dict()
                    else:
                        save_content = ema_model.state_dict()

                    torch.save({'state_dict': save_content},
                               os.path.join(save_dir, args.model + test_str + '_best_ema' + '.pth'))

                writer.add_scalar('best_psnr_ema', best_psnr_ema, epoch)

                if avg_ssim_ema > best_ssim_ema:
                    best_ssim_ema = avg_ssim_ema
                writer.add_scalar('best_ssim_ema', best_ssim_ema, epoch)

            scheduler.step()

        writer.close()

    else:
        print('==> Existing trained model')
        exit(1)