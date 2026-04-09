import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.LoL_DataLoader import TrainData_for_LOLv2Synthetic, TestData_for_LOLv2Synthetic
from pytorch_msssim import ssim, SSIM
from models import *
import random
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


# ============== 固定随机种子 ==============
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
# =========================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sfhformer_lol_s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='lowlight', type=str, help='experiment setting')
args = parser.parse_args()


# ===================================================================================
# 🔥 Charbonnier Loss (比 L1 更平滑，比 L2 更抗过平滑)
# ===================================================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# ===================================================================================
# 🔥 TTA (Test Time Augmentation) — 8 种几何变换取平均
# ===================================================================================
def tta_predict(network, source_img):
    """
    对输入做 8 种翻转/旋转增强，分别推理后逆变换再平均，
    通常可稳定提升 +0.1～0.3 dB，无需额外训练。
    """
    preds = []
    for i in range(8):
        # 前向增强
        if i == 0:
            aug = source_img
        elif i == 1:
            aug = torch.flip(source_img, dims=[3])          # 水平翻转
        elif i == 2:
            aug = torch.flip(source_img, dims=[2])          # 垂直翻转
        elif i == 3:
            aug = torch.flip(source_img, dims=[2, 3])       # 180° 旋转
        elif i == 4:
            aug = torch.rot90(source_img, 1, dims=[2, 3])   # 90°
        elif i == 5:
            aug = torch.rot90(source_img, 1, dims=[2, 3])
            aug = torch.flip(aug, dims=[3])                  # 90° + 水平翻转
        elif i == 6:
            aug = torch.rot90(source_img, 3, dims=[2, 3])   # 270°
        else:
            aug = torch.rot90(source_img, 3, dims=[2, 3])
            aug = torch.flip(aug, dims=[3])                  # 270° + 水平翻转

        with torch.no_grad():
            pred = network(aug).clamp_(0, 1)

        # 逆变换
        if i == 0:
            pass
        elif i == 1:
            pred = torch.flip(pred, dims=[3])
        elif i == 2:
            pred = torch.flip(pred, dims=[2])
        elif i == 3:
            pred = torch.flip(pred, dims=[2, 3])
        elif i == 4:
            pred = torch.rot90(pred, 3, dims=[2, 3])
        elif i == 5:
            pred = torch.flip(pred, dims=[3])
            pred = torch.rot90(pred, 3, dims=[2, 3])
        elif i == 6:
            pred = torch.rot90(pred, 1, dims=[2, 3])
        else:
            pred = torch.flip(pred, dims=[3])
            pred = torch.rot90(pred, 1, dims=[2, 3])

        preds.append(pred)

    return torch.stack(preds, dim=0).mean(dim=0)


# ===================================================================================
# 训练 & 验证
# ===================================================================================
def train(train_loader, network, criterion_char, criterion_ssim, optimizer, ema_model):
    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.train()

    loop = tqdm(train_loader, leave=False)
    for batch in loop:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        pred_img = network(source_img)

        # 1. Charbonnier Loss (内容重建)
        loss_char = criterion_char(pred_img, target_img)

        # 2. SSIM Loss (结构相似度)
        loss_ssim = 1 - criterion_ssim(pred_img, target_img)

        # 3. FFT Loss (频域一致性)
        label_fft = torch.fft.fft2(target_img, dim=(-2, -1))
        label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
        pred_fft = torch.fft.fft2(pred_img, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        loss_fft = criterion_char(pred_fft, label_fft)

        # 混合 Loss: Charbonnier 主导 + SSIM 结构 + FFT 频域
        loss = loss_char + 0.2 * loss_ssim + 0.05 * loss_fft

        if not torch.isfinite(loss):
            print("⚠️ Warning: Loss is NaN or Inf, skipping batch!")
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


def valid(val_loader, network, use_tta=False):
    """
    验证函数。use_tta=True 时启用测试时增强 (推理阶段免费提分)。
    """
    PSNR_meter = AverageMeter()
    SSIM_meter = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        if use_tta:
            output = tta_predict(network, source_img)
        else:
            with torch.no_grad():
                output = network(source_img).clamp_(0, 1)

        mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR_meter.update(psnr.item(), source_img.size(0))

        ssim_val = ssim(output, target_img, data_range=1, size_average=False).mean()
        SSIM_meter.update(ssim_val.item(), source_img.size(0))

    return PSNR_meter.avg, SSIM_meter.avg


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

    def worker_init_fn(worker_id):
        np.random.seed(8001 + worker_id)
        random.seed(8001 + worker_id)

    generator = torch.Generator()
    generator.manual_seed(8001)

    train_data_dir = '/root/lanyun-tmp/data/LOLv2/Synthetic/Train'
    test_data_dir = '/root/lanyun-tmp/data/LOLv2/Synthetic/Test'

    train_dataset = TrainData_for_LOLv2Synthetic(128, train_data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=setting['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    test_dataset = TestData_for_LOLv2Synthetic(8, test_data_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=25,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

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
            train_loss = train(train_loader, network, criterion_char, criterion_ssim, optimizer, ema_model)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            torch.save({'state_dict': network.state_dict()},
                       os.path.join(save_dir, args.model + test_str + '_newest.pth'))

            if epoch % setting['eval_freq'] == 0:
                # 普通评估
                avg_psnr, avg_ssim = valid(test_loader, network, use_tta=False)
                # TTA 评估 (使用 EMA 模型 + TTA 双重加持)
                avg_psnr_ema, avg_ssim_ema = valid(test_loader, ema_model, use_tta=True)

                print(f"Epoch {epoch} | Normal : PSNR {avg_psnr:.4f}  SSIM {avg_ssim:.4f}")
                print(f"Epoch {epoch} | EMA+TTA: PSNR {avg_psnr_ema:.4f}  SSIM {avg_ssim_ema:.4f}")

                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('valid_ssim', avg_ssim, epoch)
                writer.add_scalar('valid_psnr_ema', avg_psnr_ema, epoch)
                writer.add_scalar('valid_ssim_ema', avg_ssim_ema, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + test_str + '_best.pth'))
                writer.add_scalar('best_psnr', best_psnr, epoch)

                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                writer.add_scalar('best_ssim', best_ssim, epoch)

                if avg_psnr_ema > best_psnr_ema:
                    best_psnr_ema = avg_psnr_ema
                    # 保存 EMA 模型权重
                    real_model = ema_model.module
                    if isinstance(real_model, nn.DataParallel):
                        save_content = real_model.module.state_dict()
                    else:
                        save_content = real_model.state_dict()
                    torch.save({'state_dict': save_content},
                               os.path.join(save_dir, args.model + test_str + '_best_ema.pth'))

                writer.add_scalar('best_psnr_ema', best_psnr_ema, epoch)

                if avg_ssim_ema > best_ssim_ema:
                    best_ssim_ema = avg_ssim_ema
                writer.add_scalar('best_ssim_ema', best_ssim_ema, epoch)

            scheduler.step()

        writer.close()

    else:
        print('==> Existing trained model')
        exit(1)
