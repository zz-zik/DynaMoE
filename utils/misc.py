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
import os
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


def parse_device(device_str):
    """解析设备配置字符串
    Args:
        device_str: 设备字符串，支持以下格式：
                   - 'cpu': 仅使用CPU
                   - 'cuda:0,1': 使用CUDA设备0和1
                   - '0,1,2': 使用CUDA设备0,1,2
    Returns:
        dict: 包含设备信息的字典
            - device_type: 'cpu' 或 'cuda'
            - device_ids: GPU设备ID列表（如果使用CUDA）
            - main_device: 主设备
            - distributed: 是否启用分布式训练
    """
    device_info = {
        'device_type': 'cpu',
        'device_ids': [],
        'main_device': 'cpu',
        'distributed': False
    }

    if device_str.lower() == 'cpu':
        return device_info

    if device_str.startswith('cuda:'):
        # 格式: 'cuda:0,1,2'
        device_ids_str = device_str[5:]
    else:
        # 格式: '0,1,2'
        device_ids_str = device_str

    try:
        device_ids = [int(x.strip()) for x in device_ids_str.split(',')]
        device_info.update({
            'device_type': 'cuda',
            'device_ids': device_ids,
            'main_device': f'cuda:{device_ids[0]}',
            'distributed': len(device_ids) > 1
        })
    except ValueError:
        print(f"Warning: Invalid device string '{device_str}', falling back to CPU")

    return device_info


def setup_distributed(device_info):
    """
    设置分布式训练环境

    Args:
        device_info: 设备信息字典

    Returns:
        bool: 是否成功初始化分布式训练
    """
    if not device_info['distributed']:
        return False

    if not torch.cuda.is_available():
        print("CUDA is not available, cannot use distributed training")
        return False

    # 检查环境变量
    if 'RANK' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(len(device_info['device_ids']))

        import torch.multiprocessing as mp
        mp.spawn(
            distributed_main,
            args=(device_info,),
            nprocs=len(device_info['device_ids']),
            join=True
        )
        return True

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    actual_device_id = device_info['device_ids'][local_rank]
    torch.cuda.set_device(actual_device_id)

    dist.init_process_group(backend='nccl')
    return True


def distributed_main(rank, device_info):
    """
    分布式训练主函数

    Args:
        rank: 进程排名
        device_info: 设备信息字典
    """
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    actual_device_id = device_info['device_ids'][rank]
    torch.cuda.set_device(actual_device_id)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=len(device_info['device_ids']),
        rank=rank
    )

    from engine.trainer_dp import Trainer
    from train import get_args_config

    cfg = get_args_config()
    cfg.device = f"cuda:{actual_device_id}"

    trainer = Trainer(cfg)
    trainer.run()
