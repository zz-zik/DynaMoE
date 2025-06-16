# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: misc.py
@Time    : 2025/6/10 上午10:33
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 配置文件
@Usage   : 
"""
import pprint
from typing import List
import torch.distributed as dist
import torch
import yaml
from torch import Tensor


class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = Config(**value)
            setattr(self, key, value)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def to_dict(self):
        """递归转换 Config 对象为 dict"""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    if config_dict is None:
        raise ValueError("配置文件为空或加载失败")
    cfg = Config(**config_dict)
    return cfg


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def _max_by_axis_pad(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    block = 128

    for i in range(2):
        maxes[i + 1] = ((maxes[i + 1] - 1) // block + 1) * block
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:

        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor


def collate_fn(batch):
    """
    自定义 collate 函数，支持多通道掩码标注格式
    输入 batch 形式为：
        [(imgA, imgB, label_tensor), ...]
        label_tensor = [C, H, W]
    输出：
        (nested_tensor_A, nested_tensor_B, labels_batched)
    """

    # 解包 batch
    images_a, images_b, labels = zip(*batch)

    # 转换为 tensor list
    images_a = list(images_a)  # List[Tensor(3, H, W)]
    images_b = list(images_b)  # List[Tensor(3, H, W)]

    # 构造 NestedTensor
    images_a = nested_tensor_from_tensor_list(images_a)
    images_b = nested_tensor_from_tensor_list(images_b)

    # 合并 label 为 batch 格式 [B, C, H, W]
    labels_batched = torch.stack(labels, dim=0)  # [B, C, H, W]

    return images_a, images_b, labels_batched
