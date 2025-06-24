# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: tester.py
@Time    : 2025/6/24 上午9:42
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   : 
"""
import datetime
import os
import pprint
import time
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import MultiMaskChange, Transforms
from models import build_model
from utils import get_output_dir, setup_logging, collate_fn, get_environment_info
import cv2


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.test.device

        self.classes = cfg.data.classes
        self.threshold = cfg.test.threshold
        self.weights_dir = cfg.test.weights_dir

        self.output_dir = get_output_dir(cfg.test.save_dir, cfg.test.name)
        self.logger = setup_logging(cfg, self.output_dir)

        self._setup_model()
        self._setup_testloader()

    def _setup_model(self):
        """加载并初始化模型"""
        self.logger.info('------------------------ model params ------------------------')
        self.model = build_model(self.cfg, training=False)
        self.model.to(self.device)

        if self.weights_dir is not None:
            checkpont = torch.load(self.weights_dir, map_location=self.device)
            self.model.load_state_dict(checkpont['model'])
        self.model.eval()

    def _setup_testloader(self):
        """初始化测试数据集和数据加载器"""
        self.logger.info('------------------------ data params ------------------------')
        transforms_val = Transforms(train=False, **self.cfg.data.transforms.to_dict())
        test_dataset = MultiMaskChange(transform=transforms_val, train=False, test=True, **self.cfg.data.to_dict())
        self.dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    def _log_testing_info(self):
        """记录训练开始信息"""
        self.logger.info('Test Log %s' % time.strftime("%c"))
        env_info = get_environment_info()
        self.logger.info(env_info)
        self.logger.info('Running with config:')
        self.logger.info(pprint.pformat(self.cfg.__dict__))

    def run(self):
        # 记录训练开始信息
        self._log_testing_info()

        self.logger.info("------------------------ Start testing ------------------------")
        start_time = time.time()

        metrics = self.evaluate()

        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Test Metrics: {metric_str}")

        total_time = time.time() - start_time

        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('------------------------ Test completed ------------------------')
        self.logger.info(f'Total test time: {total_time_str}')
        self.logger.info(f'Results saved to: {self.output_dir}')

    def evaluate(self):
        class_preds = {cls: [] for cls in self.classes}
        class_labels = {cls: [] for cls in self.classes}

        with torch.no_grad(), tqdm(self.dataloader_test, desc='[Testing]') as pbar:
            for images_a, images_b, labels in pbar:
                images_a = images_a.to(self.device)
                images_b = images_b.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images_a, images_b)

                if isinstance(outputs, dict):
                    predictions = outputs['prediction']  # shape: [B, N, H, W]
                else:
                    predictions = outputs

                # 为每个类别处理预测和标签
                batch_preds = {}
                for idx, cls in enumerate(self.classes):
                    pred_channel = predictions[:, idx, :, :].squeeze(1)  # 正确提取第idx个类别的预测
                    label_channel = labels[:, idx, :, :].squeeze(1)  # 确保这是对应类别的真实标签

                    pred = (torch.sigmoid(pred_channel) > self.threshold).cpu().numpy()
                    true = label_channel.cpu().numpy()

                    class_preds[cls].extend(pred.flatten())
                    class_labels[cls].extend(true.flatten())

                    batch_preds[cls] = pred[0]  # 1个batch

                # 保存预测图像和对比图像
                if self.cfg.test.show:
                    pred_save_dir = os.path.join(self.output_dir, 'predictions')

                    # 为每个类别创建预测图像
                    for cls in self.classes:
                        cls_pred_dir = os.path.join(pred_save_dir, cls)
                        os.makedirs(cls_pred_dir, exist_ok=True)

                        # 保存预测图像
                        pred_img = (batch_preds[cls] * 255).astype(np.uint8)
                        pred_img = Image.fromarray(pred_img, mode='L')
                        save_path = os.path.join(cls_pred_dir, f'{pbar.n:04d}_pred.png')
                        pred_img.save(save_path)

                    # 保存带掩码的叠加图像
                    if self.cfg.test.show_overlay:
                        overlay_dir = os.path.join(self.output_dir, 'overlays')

                        for cls in self.classes:
                            cls_overlay_dir = os.path.join(overlay_dir, cls)
                            os.makedirs(cls_overlay_dir, exist_ok=True)

                            # 将掩码叠加到原始图像上
                            overlay_img = overlay_mask_on_image(
                                reverse_normalize(images_a.cpu(), mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])[0].numpy().transpose(1, 2, 0),
                                batch_preds[cls],
                                1  # 单类别掩码
                            )
                            overlay_save_path = os.path.join(cls_overlay_dir, f'{pbar.n:04d}_overlay.png')
                            overlay_img.save(overlay_save_path)

                    # 保存A, B, target, pred的对比图像
                    comparison_dir = os.path.join(self.output_dir, 'comparisons')

                    for cls in self.classes:
                        cls_comparison_dir = os.path.join(comparison_dir, cls)
                        os.makedirs(cls_comparison_dir, exist_ok=True)

                        # 还原 images_a 和 images_b 到原始图像
                        images_a_unnorm = reverse_normalize(images_a.cpu(), mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                        images_b_unnorm = reverse_normalize(images_b.cpu(), mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])

                        # 获取对应类别的真实标签
                        cls_idx = self.classes.index(cls)
                        target_mask = labels[0, cls_idx, :, :].cpu().numpy()

                        comparison_img = create_comparison_image(
                            images_a_unnorm[0],
                            images_b_unnorm[0],
                            target_mask,
                            batch_preds[cls],
                            1  # 单类别处理
                        )
                        comparison_save_path = os.path.join(cls_comparison_dir, f'{pbar.n:04d}_comparison.png')
                        comparison_img.save(comparison_save_path)

        metrics = {
            'precision': {},
            'recall': {},
            'f1': {},
            'iou': {},
            'oa': {}
        }

        for cls in self.classes:
            y_true = np.array(class_labels[cls])
            y_pred = np.array(class_preds[cls])

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            iou = jaccard_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)

            metrics['precision'][cls] = precision
            metrics['recall'][cls] = recall
            metrics['f1'][cls] = f1
            metrics['iou'][cls] = iou
            metrics['oa'][cls] = accuracy

        return metrics


def reverse_normalize(image_tensor, mean, std):
    """
    逆归一化操作，将归一化的图像张量还原为原始图像
    Args:
        image_tensor: 归一化的图像张量 (Tensor, B x C x H x W)
        mean: 均值 (list)
        std: 标准差 (list)
    Returns:
        还原后的图像张量 (Tensor, B x C x H x W)
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return image_tensor * std + mean

def overlay_mask_on_image(image, mask, num_classes, alpha=0.5):
    """
    将掩码叠加到原始图像上
    Args:
        image: 原始图像 (numpy array, H x W x C)
        mask: 掩码 (numpy array, H x W)
        num_classes: 类别数量
        alpha: 背景图像的透明度
    Returns:
        PIL Image
    """
    # 将图像从 numpy 格式转换为 PIL Image
    image = Image.fromarray((image * 255).astype(np.uint8))

    # 创建掩码图像
    if num_classes == 1:
        # 单类别：黑白掩码
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    else:
        # 多类别：彩色掩码
        cmap = plt.get_cmap('viridis', num_classes)
        mask_colored = (cmap(mask) * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_colored[:, :, :3])

    # 将掩码图像调整为与原始图像相同的尺寸
    mask_img = mask_img.resize(image.size)

    # 将掩码叠加到图像上
    overlay = Image.new('RGBA', image.size)
    overlay.paste(mask_img.convert('RGBA'), (0, 0), mask_img.convert('L'))
    overlayed_img = Image.alpha_composite(image.convert('RGBA'), overlay)

    return overlayed_img

def create_comparison_image(image_a, image_b, target_mask, pred_mask, num_classes):
    """
    创建对比图像，并在原始图像上绘制掩码边界（支持控制线条粗细）
    """
    # 将张量转换为 NumPy 数组并调整形状
    image_a = image_a.numpy().transpose(1, 2, 0)  # C x H x W -> H x W x C
    image_b = image_b.numpy().transpose(1, 2, 0)
    image_a = (image_a * 255).astype(np.uint8)
    image_b = (image_b * 255).astype(np.uint8)

    # 单类别处理
    if num_classes == 1:
        if target_mask.ndim == 3:
            target_mask = target_mask[0]
        if pred_mask.ndim == 3:
            pred_mask = pred_mask[0]

        target_mask_np = (target_mask * 255).astype(np.uint8)
        pred_mask_np = (pred_mask * 255).astype(np.uint8)

        contours_target, _ = cv2.findContours(target_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(pred_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 设置颜色和线宽
        color_target = (0, 0, 255)  # BGR 格式，红色
        color_pred = (255, 255, 255)  # 白色
        thickness = 2  # 控制线条粗细

        # 在图像 A 上同时绘制 target（红）和 pred（白）
        image_a_contour = image_a.copy()
        image_a_contour = cv2.drawContours(image_a_contour, contours_target, -1, color_target, thickness)
        image_a_contour = cv2.drawContours(image_a_contour, contours_pred, -1, color_pred, thickness)

        # 在图像 B 上也同时绘制 target（红）和 pred（白）
        image_b_contour = image_b.copy()
        image_b_contour = cv2.drawContours(image_b_contour, contours_target, -1, color_target, thickness)
        image_b_contour = cv2.drawContours(image_b_contour, contours_pred, -1, color_pred, thickness)

    else:
        # 多类别处理（仅绘制前景区域）
        target_mask_binary = (target_mask > 0).astype(np.uint8) * 255
        pred_mask_binary = (pred_mask > 0).astype(np.uint8) * 255

        contours_target, _ = cv2.findContours(target_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(pred_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_target = (0, 0, 255)
        color_pred = (0, 255, 0)
        thickness = 2

        image_a_contour = cv2.drawContours(image_a.copy(), contours_target, -1, color_target, thickness)
        image_b_contour = cv2.drawContours(image_b.copy(), contours_pred, -1, color_pred, thickness)

    # 转换回 PIL 图像用于拼接
    image_a_pil = Image.fromarray(image_a_contour)
    image_b_pil = Image.fromarray(image_b_contour)

    # 掩码图像（保持不变）
    if num_classes == 1:
        target_img = Image.fromarray((target_mask * 255).astype(np.uint8), mode='L')
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
    else:
        cmap = plt.get_cmap('viridis', num_classes)
        target_colored = (cmap(target_mask) * 255).astype(np.uint8)
        pred_colored = (cmap(pred_mask) * 255).astype(np.uint8)
        target_img = Image.fromarray(target_colored[:, :, :3])
        pred_img = Image.fromarray(pred_colored[:, :, :3])

    # 拼接图像
    comparison_img = Image.new('RGB', (image_a_pil.width * 4, image_a_pil.height))
    comparison_img.paste(image_a_pil, (0, 0))
    comparison_img.paste(image_b_pil, (image_a_pil.width, 0))
    comparison_img.paste(target_img.convert('RGB'), (image_a_pil.width * 2, 0))
    comparison_img.paste(pred_img.convert('RGB'), (image_a_pil.width * 3, 0))

    return comparison_img
