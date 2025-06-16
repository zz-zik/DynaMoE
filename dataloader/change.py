# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: change.py
@Time    : 2025/4/18 下午5:10
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 建筑物变化检测数据
@Usage   :
"""
import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiMaskChange(Dataset):
    def __init__(self, transform, train=False, test=False, **kwargs):
        self.data_path = kwargs.get('data_root', '')
        self.train = train
        self.test = test
        self.data_format = kwargs.get('data_format', 'default')  # default or custom
        self.transform = transform
        self.classes = kwargs.get('classes', [])  # list of change categories
        self.tensor = transforms.ToTensor()

        # 构建目录路径
        if self.data_format == "default":
            self.data_dir = os.path.join(self.data_path, 'test' if self.test else ('train' if self.train else 'val'))
            self.a_dir = os.path.join(self.data_dir, 'A')
            self.b_dir = os.path.join(self.data_dir, 'B')
            self.labels_dir = os.path.join(self.data_dir, 'masks')
        elif self.data_format == "custom":
            self.data_dir = self.data_path
            self.a_dir = os.path.join(self.data_path, 'A')
            self.b_dir = os.path.join(self.data_path, 'B')
            self.labels_dir = os.path.join(self.data_path, 'masks')
        else:
            raise ValueError(f"不支持的数据集格式：{self.data_format}")

        self.img_map = {}
        self.img_list = []

        # 加载图像路径
        if self.data_format == "default":
            a_img_paths = [f for f in os.listdir(self.a_dir) if f.endswith(('.png', '.jpg', '.tif'))]
            for filename in a_img_paths:
                a_img_path = os.path.join(self.a_dir, filename)
                b_img_path = os.path.join(self.b_dir, filename)
                label_class_paths = {
                    cls: os.path.join(self.labels_dir, cls, filename)
                    for cls in self.classes
                }
                if all(os.path.isfile(p) for p in [a_img_path, b_img_path] + list(label_class_paths.values())):
                    self.img_map[a_img_path] = (b_img_path, label_class_paths)
                    self.img_list.append(a_img_path)
        elif self.data_format == "custom":
            list_file = os.path.join(self.data_path, 'list', 'test.txt' if self.test else ('train.txt' if self.train else 'val.txt'))
            if not os.path.exists(list_file):
                raise FileNotFoundError(f"未找到列表文件：{list_file}")

            with open(list_file, 'r') as f:
                for line in f:
                    filename = line.strip()
                    if not filename:
                        continue
                    a_img_path = os.path.join(self.a_dir, filename)
                    b_img_path = os.path.join(self.b_dir, filename)
                    label_class_paths = {
                        cls: os.path.join(self.labels_dir, cls, filename)
                        for cls in self.classes
                    }
                    if all(os.path.isfile(p) for p in [a_img_path, b_img_path] + list(label_class_paths.values())):
                        self.img_map[a_img_path] = (b_img_path, label_class_paths)
                        self.img_list.append(a_img_path)

        self.img_list = sort_filenames_numerically(self.img_list)
        self.nSamples = len(self.img_list)

    def __len__(self):
        return self.nSamples

    @staticmethod
    def load_image(path):
        """使用 OpenCV 加载图像并转为 RGB 格式"""
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像：{path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_binary_mask(path):
        """加载二值掩码（0/1）"""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码：{path}")
        mask = (mask > 0).astype(np.uint8)
        return mask

    def __getitem__(self, index):
        a_img_path = self.img_list[index]
        b_img_path, label_class_paths = self.img_map[a_img_path]

        # Step 1: 图像读取（调用 load_image）
        a_img = self.load_image(a_img_path)
        b_img = self.load_image(b_img_path)

        # Step 2: 多类别掩码读取并合并为 [N, H, W] 的 NumPy 数组
        label_list = []
        for cls, mask_path in label_class_paths.items():
            mask = self.load_binary_mask(mask_path)  # [H, W], np.uint8
            label_list.append(mask)

        labels = np.stack(label_list, axis=-1)  # [H, W, N]

        # Step 3: 数据增强（传入 NumPy array）
        if self.transform:
            a_img, b_img, labels = self.transform(a_img, b_img, labels)

        return a_img, b_img, labels


def sort_filenames_numerically(filenames):
    def numeric_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))
        return (tuple(numbers), filename) if numbers else ((), filename)

    return sorted(filenames, key=numeric_key)


if __name__ == '__main__':
    from utils import load_config
    from dataloader.transforms import Transforms

    cfg = load_config("../configs/config.yaml")
    cfg.data.data_root = '../dataset/change_'

    transforms_train = Transforms(train=True, **cfg.data.transforms.to_dict())
    transforms_val = Transforms(train=False, **cfg.data.transforms.to_dict())

    train_dataset = MultiMaskChange(transform=transforms_train, train=True, **cfg.data.to_dict())
    val_dataset = MultiMaskChange(transform=transforms_val, train=False, **cfg.data.to_dict())
    # test_dataset = MultiMaskChange(transform=transforms_val, train=False, test=True, **cfg.data.to_dict())

    print('训练集样本数：', len(train_dataset))
    print('验证集样本数：', len(val_dataset))
    # print('测试集样本数：', len(test_dataset))

    img_a, img_b, label = train_dataset[0]
    print('训练集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '标注形状：', label.shape)

    img_a, img_b, label = val_dataset[0]
    print('验证集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '标注形状：', label.shape)

    # img_a, img_b, label = test_dataset[0]
    # print('测试集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '标注形状：',
    #       label)
