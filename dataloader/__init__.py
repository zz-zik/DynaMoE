from dataloader.change import *
from dataloader.transforms import *
from dataloader.loading_data import *
import torch
from utils import collate_fn
from torch.utils.data import DataLoader, DistributedSampler
import logging


def build_dataset(cfg, distributed=False, world_size=1, rank=0):
    """
    构建数据集和数据加载器，支持分布式训练

    Args:
        cfg: 配置对象
        distributed: 是否使用分布式训练
        world_size: 总进程数
        rank: 当前进程排名

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    train_set, val_set = loading_data(cfg)

    if distributed:
        sampler_train = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        sampler_val = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

        data_loader_train = DataLoader(
            train_set,
            batch_size=cfg.training.batch_size,
            sampler=sampler_train,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )

        data_loader_val = DataLoader(
            val_set,
            batch_size=1,
            sampler=sampler_val,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.training.batch_size, drop_last=True)

        data_loader_train = DataLoader(
            train_set,
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        data_loader_val = DataLoader(
            val_set,
            batch_size=1,
            sampler=sampler_val,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

    if not distributed or rank == 0:
        logging.info("------------------------ preprocess dataset ------------------------")
        logging.info("Data_path: %s", cfg.data.data_root)
        logging.info("Data Transforms:\n %s", cfg.data.transforms)
        logging.info(f"# Train {train_set.nSamples}, Val {val_set.nSamples}")
        if distributed:
            logging.info(f"Distributed training: {world_size} processes, rank {rank}")

    return data_loader_train, data_loader_val

# 测试
if __name__ == '__main__':
    from utils import load_config
    from dataloader.transforms import Transforms

    cfg = load_config("../configs/config.yaml")
    cfg.data.data_root = '../dataset/change_'

    data_train, data_val = build_dataset(cfg)

    for i, (a_img, b_img, target) in enumerate(data_train):
        print('训练集第', i, '个样本a图像形状：', a_img.shape, 'b图像形状：', b_img.shape, '标注形状：', target.shape)
        if i == 2:
            break