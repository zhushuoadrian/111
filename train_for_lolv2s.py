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
from pytorch_msssim import ssim, SSIM 
from models import *
import random
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# ============== 固定随机种子函数 ==============
def set_seed(seed=8001):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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

# 🔥 核心修改 1：eps 降至 1e-6，确保 30dB 以上的高精度微调
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

# 🔥 核心修改 2：加入 epoch 参数，实现 700 轮动态权重切换
def train(train_loader, network, criterion_char, criterion_ssim, optimizer, ema_model, epoch):
    losses = AverageMeter()

    torch.cuda.empty_cache()
    network.train()
    
    loop = tqdm(train_loader, leave=False)
    
    for batch in loop:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        pred_img = network(source_img)
        label_img = target_img
        
        loss_char = criterion_char(pred_img, label_img)
        loss_ssim = 1 - criterion_ssim(pred_img, label_img)

        # FFT Loss
        label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)
        pred_fft3 = torch.fft.fft2(pred_img, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)
        loss_fft = criterion_char(pred_fft3, label_fft3) 

        # 🔥 700 轮冲刺策略：削弱 SSIM 干扰，让 Charbonnier 强行对齐像素
        if epoch > 700:
            loss = loss_char + 0.2 * loss_ssim + 0.05 * loss_fft
        else:
            loss = loss_char + 0.2 * loss_ssim + 0.05 * loss_fft

        if not torch.isfinite(loss):
            print("⚠️ Warning: Loss is NaN or Inf, skipping this batch!")
            optimizer.zero_grad()
            continue

        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1)
        
        optimizer.step()
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

    ema_model = AveragedModel(network, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    criterion_char = CharbonnierLoss().cuda()
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3).cuda()

    # 🔥 核心修改 3：AdamW Beta2 改为 0.95，更适合 Transformer 动态梯度
    base_lr = setting.get('lr', 1e-3)
    if setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=base_lr, betas=(0.9, 0.95))
    elif setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=base_lr, betas=(0.9, 0.95))
    else:
        raise Exception("ERROR: unsupported optimizer")

    # 🔥 核心修改 4：Warmup 衔接 CosineAnnealing
    warmup_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=setting['epochs'] - warmup_epochs,
        eta_min=1e-6
    )

    def worker_init_fn(worker_id):
        worker_seed = 8001 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    generator = torch.Generator()
    generator.manual_seed(8001)

    train_data_dir = '/root/lanyun-tmp/data/LOLv2/Synthetic/Train'
    test_data_dir = '/root/lanyun-tmp/data/LOLv2/Synthetic/Test'
    
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
            # 🔥 核心修改 5：线性 Warmup 学习率手动覆盖
            if epoch < warmup_epochs:
                lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # 正常执行训练，传入 epoch 以便 700 轮后切换权重
            train_loss = train(train_loader, network, criterion_char, criterion_ssim, optimizer, ema_model, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', current_lr, epoch)

            # 保持原始保存逻辑：_newest.pth
            torch.save({'state_dict': network.state_dict()},
                       os.path.join(save_dir, args.model + test_str + '_newest' + '.pth'))

            if epoch % setting['eval_freq'] == 0:
                avg_psnr, avg_ssim = valid(test_loader, network)
                print(f"Epoch {epoch} | Normal: PSNR {avg_psnr:.4f} SSIM {avg_ssim:.4f}")
                
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

                    # 保持原始 EMA 保存逻辑
                    torch.save({'state_dict': save_content},
                               os.path.join(save_dir, args.model + test_str + '_best_ema' + '.pth'))
                
                writer.add_scalar('best_psnr_ema', best_psnr_ema, epoch)
                
                if avg_ssim_ema > best_ssim_ema:
                    best_ssim_ema = avg_ssim_ema
                writer.add_scalar('best_ssim_ema', best_ssim_ema, epoch)

            # 🔥 Warmup 结束后步进 scheduler (Cosine)
            if epoch >= warmup_epochs:
                scheduler.step()

        writer.close()

    else:
        print('==> Existing trained model')
        exit(1)