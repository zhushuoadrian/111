import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image


# ===================================================================================
# 🔥 增强版数据增强工具函数
# ===================================================================================
def random_gamma(img, gamma_range=(0.7, 1.4)):
    """
    随机 Gamma 变换，模拟不同照度水平，扩充低光训练多样性。
    gamma < 1 → 图像变亮；gamma > 1 → 图像变暗。
    """
    gamma = random.uniform(*gamma_range)
    return img.pow(gamma)


def augment_pair(source, target, patch_size):
    """
    对 source/target 图像对做一致的随机增强:
      - 多尺度随机裁剪 (patch_size 或 patch_size*1.5, 再 resize 回去)
      - 随机水平/垂直翻转
      - 随机 90° 旋转
      - 对 source 单独做随机 Gamma 扰动 (target 不变，模拟更多照度)
    """
    # 多尺度裁剪: 以 1.0 或 1.5 倍 crop，再 resize 回 patch_size
    scale = random.choice([1.0, 1.5])
    crop_size = int(patch_size * scale)

    h, w = source.shape[1], source.shape[2]
    if h < crop_size or w < crop_size:
        crop_size = min(h, w, patch_size)

    i = random.randint(0, h - crop_size)
    j = random.randint(0, w - crop_size)
    source = source[:, i:i + crop_size, j:j + crop_size]
    target = target[:, i:i + crop_size, j:j + crop_size]

    if crop_size != patch_size:
        source = torch.nn.functional.interpolate(
            source.unsqueeze(0), size=(patch_size, patch_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        target = torch.nn.functional.interpolate(
            target.unsqueeze(0), size=(patch_size, patch_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

    # 随机水平翻转
    if random.random() > 0.5:
        source = torch.flip(source, dims=[2])
        target = torch.flip(target, dims=[2])

    # 随机垂直翻转
    if random.random() > 0.5:
        source = torch.flip(source, dims=[1])
        target = torch.flip(target, dims=[1])

    # 随机 90° 旋转
    k = random.randint(0, 3)
    if k > 0:
        source = torch.rot90(source, k, dims=[1, 2])
        target = torch.rot90(target, k, dims=[1, 2])

    # 🔥 对 source 单独做随机 Gamma 扰动 (target 保持不变)
    if random.random() > 0.3:
        source = random_gamma(source, gamma_range=(0.75, 1.35))
        source = source.clamp(0, 1)

    return source, target


# ===================================================================================
# LOLv2 Synthetic 数据集
# ===================================================================================
def load_image_pair(low_path, high_path):
    """加载图像对并转为 [0,1] float tensor (C,H,W)"""
    low = Image.open(low_path).convert('RGB')
    high = Image.open(high_path).convert('RGB')
    low = TF.to_tensor(low)    # [3, H, W], float32 in [0, 1]
    high = TF.to_tensor(high)
    return low, high


class TrainData_for_LOLv2Synthetic(Dataset):
    """
    LOLv2 Synthetic 训练集数据加载器。
    目录结构:
        train_data_dir/
            low/   *.png  (低光图)
            high/  *.png  (对应正常光图)
    """

    def __init__(self, patch_size, data_dir):
        super().__init__()
        self.patch_size = patch_size

        low_dir = os.path.join(data_dir, 'low')
        high_dir = os.path.join(data_dir, 'high')

        low_files = sorted([f for f in os.listdir(low_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.pairs = []
        for fname in low_files:
            low_path = os.path.join(low_dir, fname)
            high_path = os.path.join(high_dir, fname)
            if os.path.exists(high_path):
                self.pairs.append((low_path, high_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No image pairs found in {data_dir}. "
                "Expected sub-folders 'low/' and 'high/'."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        source, target = load_image_pair(low_path, high_path)
        source, target = augment_pair(source, target, self.patch_size)
        return {'source': source, 'target': target}


class TestData_for_LOLv2Synthetic(Dataset):
    """
    LOLv2 Synthetic 测试集数据加载器 (不做随机增强，仅裁剪对齐)。
    目录结构与训练集相同。
    """

    def __init__(self, patch_size, data_dir):
        super().__init__()
        self.patch_size = patch_size

        low_dir = os.path.join(data_dir, 'low')
        high_dir = os.path.join(data_dir, 'high')

        low_files = sorted([f for f in os.listdir(low_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.pairs = []
        for fname in low_files:
            low_path = os.path.join(low_dir, fname)
            high_path = os.path.join(high_dir, fname)
            if os.path.exists(high_path):
                self.pairs.append((low_path, high_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No image pairs found in {data_dir}. "
                "Expected sub-folders 'low/' and 'high/'."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        source, target = load_image_pair(low_path, high_path)
        # 测试集: 裁剪到 patch_size 的整数倍，不做随机增强
        h, w = source.shape[1], source.shape[2]
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size
        source = source[:, :new_h, :new_w]
        target = target[:, :new_h, :new_w]
        return {'source': source, 'target': target}
