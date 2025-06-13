# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: loading_data.py
@Time    : 2025/4/18 下午5:10
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 加载数据
@Usage   :
"""
from dataloader import Transforms
from dataloader import MultiMaskChange


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(cfg):
    transforms_train = Transforms(train=True, **cfg.data.transforms.to_dict())
    transforms_val = Transforms(train=False, **cfg.data.transforms.to_dict())

    train_dataset = MultiMaskChange(transform=transforms_train, train=True, **cfg.data.to_dict())
    val_dataset = MultiMaskChange(transform=transforms_val, train=False, **cfg.data.to_dict())

    return train_dataset, val_dataset


# 测试
if __name__ == '__main__':
    from utils import load_config

    cfg = load_config("../configs/config.yaml")
    cfg.data.data_root = '../dataset/change_'
    train_dataset, val_dataset = loading_data(cfg)

    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))

    img_a, img_b, label = train_dataset[0]
    print('训练集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '标注形状：',
          {cls: label[cls].shape for cls in cfg.data.classes})

    img_a, img_b, label = val_dataset[0]
    print('验证集第1个样本a图像形状：', img_a.shape, 'b图像形状：', img_b.shape, '标注形状：',
          {cls: label[cls].shape for cls in cfg.data.classes})