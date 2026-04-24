# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import AverageMeter
from datasets.LoL_DataLoader import TestData_for_LOLv2Real
from pytorch_msssim import ssim
from models import *
import random
import lpips

# ============== 固定随机种子函数 ==============
def set_seed(seed=8001):
    """
    固定所有随机种子,确保实验可重复
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random seed set to: {seed}")

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
parser.add_argument('--save_img_dir', default='./result/', type=str, help='path to save test images')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_img_root = '/root/lanyun-tmp/newnet/result_lolv2r'
os.makedirs(save_img_root, exist_ok=True)


def valid(val_loader_full, network):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()
    LPIPS_full = AverageMeter()

    # 初始化 LPIPS 模型
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).cuda()

    network.eval()

    img_idx = 0
    for batch in val_loader_full:
        source_img = batch['source'].to(device)
        target_img = batch['target'].to(device)

        # =========================================================
        # ✅ 获取原本的图片名字
        # 假设 Dataset 返回的字典里有 'filename' 或 'name'。如果没有，则默认降级使用数字
        filename = batch.get('filename', batch.get('name', [f"{img_idx:04d}"] * source_img.size(0)))
        
        # 取出 batch 中第一张图的文件名，并去掉后缀
        if isinstance(filename, (list, tuple)):
            base_name = str(filename[0])
        else:
            base_name = str(filename)
        base_name = os.path.splitext(os.path.basename(base_name))[0]
        # =========================================================

        with torch.no_grad():
            output = network(source_img).clamp_(0, 1)

        # 计算PSNR
        mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr_full = 10 * torch.log10(1 / mse_loss).mean()
        PSNR_full.update(psnr_full.item(), source_img.size(0))

        # 计算SSIM
        ssim_full = ssim(output, target_img, data_range=1, size_average=False).mean()
        SSIM_full.update(ssim_full.item(), source_img.size(0))

        # 计算 LPIPS
        output_lpips = output * 2.0 - 1.0
        target_lpips = target_img * 2.0 - 1.0
        lpips_val = loss_fn_alex(output_lpips, target_lpips).mean()
        LPIPS_full.update(lpips_val.item(), source_img.size(0))

        # 提取当前单张图片指标用于命名
        cur_psnr = psnr_full.item()
        cur_ssim = ssim_full.item()
        cur_lpips = lpips_val.item()
        metrics_str = f"_PSNR{cur_psnr:.2f}_SSIM{cur_ssim:.4f}_LPIPS{cur_lpips:.4f}"

        # 保存图片
        source_np = (source_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        target_np = (target_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output_np = (output[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        source_bgr = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
        target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        # ✅ 保存图片时把原先的 f"{img_idx:04d}" 替换为 {base_name}
        cv2.imwrite(os.path.join(save_img_root, f"{base_name}_source.jpg"), source_bgr)
        cv2.imwrite(os.path.join(save_img_root, f"{base_name}_target.jpg"), target_bgr)
        
        # 带指标命名保存 output 和对比图
        cv2.imwrite(os.path.join(save_img_root, f"{base_name}{metrics_str}_output.jpg"), output_bgr)
        
        comparison = np.hstack([source_bgr, output_bgr, target_bgr])
        cv2.imwrite(os.path.join(save_img_root, f"{base_name}{metrics_str}_comparison.jpg"), comparison)
        
        img_idx += 1

    return PSNR_full.avg, SSIM_full.avg, LPIPS_full.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    print(setting_filename)
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=[0]).to(device)

    checkpoint_path = '/root/lanyun-tmp/newnet/saved_models/lowlight/sfhformer_lol_mtrain_lolv2r_best.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['state_dict'])

    def worker_init_fn(worker_id):
        worker_seed = 8001 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_data_dir = '/root/lanyun-tmp/data/LOLv2/Real_captured/Test'
    test_dataset = TestData_for_LOLv2Real(8, test_data_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    print(f"开始测试，结果将保存到: {save_img_root}")
    avg_psnr, avg_ssim, avg_lpips = valid(test_loader, network)
    print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    print(f"测试完成！图片已保存到: {save_img_root}")