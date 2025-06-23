# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: show_image.py
@Time    : 2025/5/21 上午10:43
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 测试数据和标签
@Usage   :
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataloader import loading_data, DeNormalize


def tensor_to_image(img_tensor):
    """将 PyTorch Tensor 转换为 NumPy 图像（HWC 格式）"""
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV


def tensor_to_masks(label_tensor, classes):
    """
    将多通道标签张量转换为类别名到 NumPy 掩码图像的映射字典
    label_tensor shape: [C, H, W] or [B, C, H, W]
    """
    if label_tensor.dim() == 4:
        label_tensor = label_tensor[0]  # 假设 batch_size=1，取第一个样本

    masks = {}
    for idx, cls in enumerate(classes):
        mask = label_tensor[idx].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        masks[cls] = mask
    return masks


if __name__ == '__main__':
    from utils import load_config

    cfg = load_config("../../configs/config.yaml")
    cfg.data.data_root = '../../dataset/change'
    classes = cfg.data.classes
    train_dataset, val_dataset = loading_data(cfg)

    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))

    img_a, img_b, labels = train_dataset[0]
    # 转换为图像
    denorm_a = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    denorm_b = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    rgb_img = tensor_to_image(denorm_a(img_a.clone()))
    tir_img = tensor_to_image(denorm_b(img_b.clone()))

    # 图像预处理
    # rgb_img = tensor_to_image(img_a.clone())
    # tir_img = tensor_to_image(img_b.clone())

    # 获取各个类别的掩码图像
    masks = tensor_to_masks(labels, classes)

    # 显示图像
    num_classes = len(classes)
    plt.figure(figsize=(6 * (num_classes + 2), 6))

    # RGB 图像
    plt.subplot(1, num_classes + 2, 1)
    plt.title("RGB Image")
    plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # TIR 图像
    plt.subplot(1, num_classes + 2, 2)
    plt.title("TIR Image")
    plt.imshow(cv2.cvtColor(tir_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 各类别 Mask 图像
    for i, (cls, mask) in enumerate(masks.items()):
        plt.subplot(1, num_classes + 2, i + 3)
        plt.title(f"Mask - {cls}")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("output.png")  # 保存图像到文件
    plt.close()
    print("图像已保存为 output.png")
