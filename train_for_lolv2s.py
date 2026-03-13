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
from datasets.LoL_DataLoader import TrainData_for_LOLv2Synthetic, TestData_for_LOLv2Synthetic
from numpy import *
from pytorch_msssim import ssim
from models import *
import random
import numpy as np
# [New] 引入 EMA 模块
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
parser.add_argument('--model', default='sfhformer_lol_s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='lowlight', type=str, help='experiment setting')
args = parser.parse_args()

# [Modified] 增加了 ema_model 参数
def train(train_loader, network, criterion, optimizer, ema_model):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()
    
    # 使用 tqdm 显示进度，方便监控实时 Loss
    loop = tqdm(train_loader, leave=False)
    
    for batch in loop:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        pred_img = network(source_img)
        label_img = target_img
        
        # 1. 内容损失
        loss_content = criterion(pred_img, label_img)

        # 2. FFT 损失
        label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(pred_img, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

        loss_fft = criterion(pred_fft3, label_fft3)

        loss = loss_content + 0.1 * loss_fft

        # 🔥【关键修改 1】NaN 熔断机制 🔥
        # 如果 Loss 变成 NaN 或者 Inf，直接跳过这一步更新！防止模型崩坏。
        if not torch.isfinite(loss):
            print("⚠️ Warning: Loss is NaN or Inf, skipping this batch!")
            optimizer.zero_grad()
            continue

        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        
        # 🔥【关键修改 2】收紧梯度裁剪 (0.5 -> 0.1)
        # 对于曾出现崩塌的模型，0.1 更安全。
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1)
        
        optimizer.step()

        # [New] 更新 EMA 模型参数
        ema_model.update_parameters(network)
        
        # 更新进度条显示的 Loss
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

    # [New] 初始化 EMA 模型
    # decay=0.999 适合小数据集防止过拟合
    ema_model = AveragedModel(network, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, betas=(0.9, 0.999))
    elif setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

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

    # 请确保这里的路径是正确的
    train_data_dir = '/home/leng/data/almzqfl/data/LOLv2/Synthetic/Train'
    test_data_dir = '/home/leng/data/almzqfl/data/LOLv2/Synthetic/Test'
    
    train_dataset = TrainData_for_LOLv2Synthetic(128, train_data_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True,
                              worker_init_fn=worker_init_fn)
    test_dataset = TestData_for_LOLv2Synthetic(8, test_data_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=25,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    test_str = 'train_lolv2s'

    if not os.path.exists(os.path.join(save_dir, args.model + test_str + '.pth')):
        print('==> Start training, current model name: ' + args.model)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model, test_str))

        best_psnr = 0
        best_ssim = 0
        best_psnr_ema = 0
        best_ssim_ema = 0

        for epoch in tqdm(range(setting['epochs'] + 1)):
            train_loss = train(train_loader, network, criterion, optimizer, ema_model)
            writer.add_scalar('train_loss', train_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', current_lr, epoch)

            torch.save({'state_dict': network.state_dict()},
                       os.path.join(save_dir, args.model + test_str + '_newest' + '.pth'))

            if epoch % setting['eval_freq'] == 0:
                # 1. 验证普通模型
                avg_psnr, avg_ssim = valid(test_loader, network)
                print(f"Epoch {epoch} | Normal: PSNR {avg_psnr:.4f} SSIM {avg_ssim:.4f}")
                
                # 2. 验证 EMA 模型
                avg_psnr_ema, avg_ssim_ema = valid(test_loader, ema_model)
                print(f"Epoch {epoch} | EMA    : PSNR {avg_psnr_ema:.4f} SSIM {avg_ssim_ema:.4f}")

                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('valid_ssim', avg_ssim, epoch)
                writer.add_scalar('valid_psnr_ema', avg_psnr_ema, epoch)
                writer.add_scalar('valid_ssim_ema', avg_ssim_ema, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + test_str + '_best' + '.pth'))
                writer.add_scalar('best_psnr', best_psnr, epoch)

                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                writer.add_scalar('best_ssim', best_ssim, epoch)
                
                # ✅ 保存 EMA 模型 Best (Correct logic)
                if avg_psnr_ema > best_psnr_ema:
                    best_psnr_ema = avg_psnr_ema
                    
                    # 解包逻辑：AveragedModel -> DataParallel -> Module
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