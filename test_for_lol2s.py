# -*- coding: utf-8 -*-
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import AverageMeter
from datasets.LoL_DataLoader import TestData_for_LOLv2Synthetic
from numpy import *
from pytorch_msssim import ssim
from models import *
import random
import numpy as np
import cv2  
import lpips  

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

# ✅ 保持原样：创建保存目录
save_img_root = '/home/leng/data/qtacefz/the02/result'
os.makedirs(save_img_root, exist_ok=True)

def valid(val_loader_full, network):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()
    LPIPS_full = AverageMeter() 

    # ✅ 保持原样：初始化 LPIPS 模型
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).cuda()

    network.eval()

    img_idx = 0  
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

        # LPIPS 计算
        output_lpips = output * 2.0 - 1.0
        target_lpips = target_img * 2.0 - 1.0
        
        lpips_val = loss_fn_alex(output_lpips, target_lpips).mean()
        LPIPS_full.update(lpips_val.item(), source_img.size(0))

        # =========================================================
        # ✅ 修改重点：提取当前单张图片的指标数值用于命名
        cur_psnr = psnr_full.item()
        cur_ssim = ssim_full.item()
        cur_lpips = lpips_val.item()
        
        # 格式化字符串，例如: _PSNR25.55_SSIM0.9123_LPIPS0.1001
        metrics_str = f"_PSNR{cur_psnr:.2f}_SSIM{cur_ssim:.4f}_LPIPS{cur_lpips:.4f}"
        # =========================================================

        # ✅ 保持原样：转换图片格式
        source_np = (source_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        target_np = (target_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output_np = (output[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        source_bgr = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
        target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        # ✅ 修改：保存图片时加入 metrics_str 到文件名中
        # source 和 target 还是保持简洁命名（也可按需加上）
        cv2.imwrite(os.path.join(save_img_root, f"{img_idx:04d}_source.jpg"), source_bgr)
        cv2.imwrite(os.path.join(save_img_root, f"{img_idx:04d}_target.jpg"), target_bgr)
        
        # 给 output 和 comparison 加上指标
        cv2.imwrite(os.path.join(save_img_root, f"{img_idx:04d}{metrics_str}_output.jpg"), output_bgr)
        
        # 创建对比图
        comparison = np.hstack([source_bgr, output_bgr, target_bgr])
        cv2.imwrite(os.path.join(save_img_root, f"{img_idx:04d}{metrics_str}_comparison.jpg"), comparison)
        
        img_idx += 1

    return PSNR_full.avg, SSIM_full.avg, LPIPS_full.avg 


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
    network.load_state_dict(torch.load('/home/leng/data/qtacefz/the02/saved_models/lowlight/sfhformer_lol_strain_lolv2s_best.pth')['state_dict'])

    # ============== 固定 DataLoader 的随机种子 ==============
    def worker_init_fn(worker_id):
        worker_seed = 8001 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    # ====================================================

    test_data_dir = '/home/leng/data/qtacefz/data/LOLv2/Synthetic/Test'
    test_dataset = TestData_for_LOLv2Synthetic(8, test_data_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    print(f"开始测试，结果将保存到: {save_img_root}")
    avg_psnr, avg_ssim, avg_lpips = valid(test_loader, network)
    
    print("PSNR: {:.2f}, SSIM: {:.4f}, LPIPS: {:.4f}".format(avg_psnr, avg_ssim, avg_lpips))
    print(f"测试完成！图片已保存到: {save_img_root}")