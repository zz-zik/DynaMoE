# -*- coding: utf-8 -*-
"""
@Project : SegChange-R2
@FileName: transforms.py
@Time    : 2025/5/24 下午4:24
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 建筑物变化检测数据增强变换类
@Usage   :
数据增强策略建议：
对 T1 和 T2 图像进行同步变换（翻转、旋转、裁剪等）
使用 CutMix、Random Erasing 等方法模拟局部变化
添加噪声或颜色扰动来提升模型鲁棒性
"""
from typing import Tuple
import cv2
import numpy as np
import random
import torch
from torch import nn
from torchvision import transforms


class Transforms(nn.Module):
    """数据增强变换类"""

    def __init__(self, train=True, **kwargs):
        super().__init__()
        self.train = train
        self.a_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.b_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 获取数据增强的参数
        self.transforms = []

        transform_params = {
            'RandomFlip': kwargs.get('random_flip', 0.0),
            'RandomRotation': kwargs.get('random_rotation', 0.0),
            'RandomResize': kwargs.get('random_resize', (0.0, 0.0)),
            'ColorJitter': kwargs.get('color_jitter', (0.0, 0.0, 0.0, 0.0)),
            'GammaCorrection': kwargs.get('gamma_correction', (0.0, 0.0)),
            'GaussianBlur': kwargs.get('blur_sigma', (0.0, 0.0, 0.0)),
            'Clahe': kwargs.get('clahe', 0.0),
        }

        prob = kwargs.get('prob', 0.0)

        for name, param in transform_params.items():
            if isinstance(param, (list, tuple)) and param[0] > 0.0:
                if name == 'RandomResize' and len(param) == 2:
                    scale_range = (float(param[0]), float(param[1]))
                    self.transforms.append(RandomResize(scale_range=scale_range, prob=prob))
                elif name == 'ColorJitter' and len(param) == 4:
                    color_jitter = (float(param[0]), float(param[1]), float(param[2]), float(param[3]))
                    self.transforms.append(ColorJitter(color_jitter=color_jitter, prob=prob))
                elif name == 'GammaCorrection' and len(param) == 2:
                    gamma_range = (float(param[0]), float(param[1]))
                    self.transforms.append(GammaCorrection(gamma_range=gamma_range, prob=prob))
                elif name == 'GaussianBlur' and len(param) == 3:
                    kernel_size = int(param[0])
                    sigma_range = (float(param[1]), float(param[2]))
                    self.transforms.append(GaussianBlur(kernel_size=kernel_size, sigma=sigma_range, prob=prob))
            elif isinstance(param, (int, float)) and param > 0.0:
                if name == 'RandomFlip':
                    self.transforms.append(RandomFlip(prob=prob))
                elif name == 'RandomRotation':
                    self.transforms.append(RandomRotation(angle=param, prob=prob))
                elif name == 'Clahe':
                    self.transforms.append(CLAHE(prob=prob))

    def forward(self, a_img: np.ndarray, b_img: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a_img: [H, W, 3]
            b_img: [H, W, 3]
            target: [N, H, W]
        Returns:
            a_img: Tensor [3, H, W]
            b_img: Tensor [3, H, W]
            target: Tensor [N, H, W]
        """
        if self.train:
            for transform in self.transforms:
                a_img, b_img, target = transform(a_img, b_img, target)

        # 转换为 Tensor 并归一化
        a_img = self.a_transform(a_img)  # [3, H, W]
        b_img = self.b_transform(b_img)  # [3, H, W]
        target = torch.from_numpy(target).permute(2, 0, 1).long()  # [H, W, N] -> [N, H, W]

        return a_img, b_img, target


class RandomFlip(nn.Module):
    """随机水平或垂直翻转"""

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob


    def forward(self, a_img: np.ndarray, b_img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            if random.random() < 0.5:
                a_img = cv2.flip(a_img, 1)
                b_img = cv2.flip(b_img, 1)
                arget = cv2.flip(target, 1)
            else:
                a_img = cv2.flip(a_img, 0)
                b_img = cv2.flip(b_img, 0)
                target = np.flip(target, 0)
        return a_img, b_img, target


class RandomResize(nn.Module):
    """随机缩放"""

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5):
        super().__init__()
        self.scale_range = scale_range
        self.prob = prob

    def forward(self, a_img: np.ndarray, b_img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            scale = np.random.uniform(*self.scale_range)
            h, w = a_img.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)

            a_img = cv2.resize(a_img, (new_w, new_h))
            b_img = cv2.resize(b_img, (new_w, new_h))
            target = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Crop or Pad 回原始尺寸
            if scale > 1.0:
                top = (new_h - h) // 2
                left = (new_w - w) // 2
                a_img = a_img[top:top + h, left:left + w]
                b_img = b_img[top:top + h, left:left + w]
                target = target[top:top + h, left:left + w]
            else:
                pad_top = (h - new_h) // 2
                pad_left = (w - new_w) // 2
                pad_bottom = h - new_h - pad_top
                pad_right = w - new_w - pad_left

                a_img = cv2.copyMakeBorder(a_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                b_img = cv2.copyMakeBorder(b_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                target = cv2.copyMakeBorder(target, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        return a_img, b_img, target


class RandomRotation(nn.Module):
    """随机旋转"""

    def __init__(self, angle: float = 10.0, prob: float = 0.5):
        super().__init__()
        self.angle = angle
        self.prob = prob

    def forward(self, a_img: np.ndarray, b_img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            h, w = a_img.shape[:2]
            center = (w // 2, h // 2)
            m = cv2.getRotationMatrix2D(center, self.angle, 1)

            a_img = cv2.warpAffine(a_img, m, (w, h))
            b_img = cv2.warpAffine(b_img, m, (w, h))
            target = cv2.warpAffine(target, m, (w, h), flags=cv2.INTER_NEAREST)  # [H, W, N]

        return a_img, b_img, target


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """调整亮度"""
    return np.clip(img * factor, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """调整对比度"""
    mean = np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)


def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """调整饱和度"""
    # 转换到 HSV 空间
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
    # 调整饱和度通道
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    # 转换回 RGB
    img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
    # 返回三通道 RGB 图像
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_hue(img: np.ndarray, factor: float) -> np.ndarray:
    """调整色相"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
    hsv[..., 0] = (hsv[..., 0] + factor * 180) % 180
    return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)


class ColorJitter(nn.Module):
    """颜色扰动变换"""

    def __init__(self, color_jitter: Tuple[float, float, float, float], prob: float = 0.5):
        super().__init__()
        self.brightness = color_jitter[0]
        self.contrast = color_jitter[1]
        self.saturation = color_jitter[2]
        self.hue = color_jitter[3]
        self.prob = prob

    def forward(self, a_img: np.ndarray, b_img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            # 随机生成参数
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            s = 1.0 + random.uniform(-self.saturation, self.saturation)
            h = random.uniform(-self.hue, self.hue)

            # 应用到 a_img
            a_img = adjust_brightness(a_img, b)
            a_img = adjust_contrast(a_img, c)
            a_img = adjust_saturation(a_img, s)
            a_img = adjust_hue(a_img, h)

            # 应用到 b_img
            b_img = adjust_brightness(b_img, b)
            b_img = adjust_contrast(b_img, c)
            b_img = adjust_saturation(b_img, s)
            b_img = adjust_hue(b_img, h)

        return a_img, b_img, target


class GammaCorrection(nn.Module):
    """Gamma 校正变换"""

    def __init__(self, gamma_range: Tuple[float, float], prob: float = 0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.prob = prob

    def forward(self, a_img: np.ndarray, b_img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if torch.rand(1) < self.prob:
            gamma = float(torch.empty(1).uniform_(self.gamma_range[0], self.gamma_range[1]))

            # 归一化到[0,1]，应用gamma校正，再恢复
            a_img = np.clip(((a_img / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
            b_img = np.clip(((b_img / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)

            # 确保值在合理范围内
            a_img = np.clip(a_img, 0, 255).astype(np.uint8)
            b_img = np.clip(b_img, 0, 255).astype(np.uint8)

        return a_img, b_img, target


class GaussianBlur(nn.Module):
    """高斯模糊"""

    def __init__(self, kernel_size: int = 5, sigma: Tuple[float, float] = (0.1, 2.0), prob: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def forward(self, a_img: np.ndarray, b_img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if torch.rand(1) < self.prob:
            sigma = random.uniform(*self.sigma)
            a_img = cv2.GaussianBlur(a_img, (self.kernel_size, self.kernel_size), sigma)
            b_img = cv2.GaussianBlur(b_img, (self.kernel_size, self.kernel_size), sigma)

        return a_img, b_img, target


class CLAHE(nn.Module):
    """直方图均衡化变换"""

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def apply_clahe_to_image(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # ✅ 每次都新建
        if len(img.shape) == 3 and img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge((l_eq, a, b))
            img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        else:
            img_eq = clahe.apply(img)
        return img_eq

    def forward(self, a_img: np.ndarray, b_img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            a_img = self.apply_clahe_to_image(a_img)
            b_img = self.apply_clahe_to_image(b_img)

        return a_img, b_img, target

