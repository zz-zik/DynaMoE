# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: misc.py
@Time    : 2025/4/18 下午5:16
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    :
@Usage   :
"""
import argparse
import logging
import os
import sys
from datetime import datetime

import torch

from utils import load_config


def get_args_config():
    """
    参数包括了: input_dir weights_dir output_dir threshold
    """
    parser = argparse.ArgumentParser('SegChange')
    parser.add_argument('-c', '--config', type=str, required=True, help='The path of config file')
    args = parser.parse_args()
    if args.config is not None:
        cfg = load_config(args.config)
    else:
        raise ValueError('Please specify the config file')
    return cfg


def get_environment_info():
    info = (
        f"Environment info 🚀: "
        f"Python-{sys.version.split()[0]} torch-{torch.__version__} "
        f"CUDA:{torch.cuda.current_device()} "
        f"({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)}MiB)"
    )
    return info


def setup_logging(cfg, log_dirs):
    # 日期命名
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(log_dirs, exist_ok=True)
    log_file = os.path.join(log_dirs, f"{current_date}.log")

    level_str = cfg.logger.level
    level = getattr(logging, level_str.upper(), logging.INFO)  # 将字符串转换为日志级别常量

    # 创建日志格式
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 配置日志记录器
    logger = logging.getLogger()
    if logger.hasHandlers():  # 如果已经存在日志处理器，则先清除
        logger.handlers.clear()  # 清空默认的日志处理器
    logger.setLevel(level)

    # 添加控制台日志处理器
    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    # 如果指定了日志文件，添加文件日志处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # 确保日志文件的目录存在
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.info(f"日志已初始化，级别为 {logging.getLevelName(level)}。")
    return logger


def get_output_dir(output_dir, name):
    """
    创建唯一输出目录，若目录已存在则自动添加后缀

    Args:
        cfg: 配置对象，需包含 output_dir 和 name 属性

    Returns:
        :param name:
        :param output_dir:
    """
    base_output_dir = os.path.join(output_dir, name)

    suffix = 0
    while os.path.exists(base_output_dir):
        base_output_dir = f"{os.path.join(output_dir, name)}_{suffix}"
        suffix += 1

    os.makedirs(base_output_dir, exist_ok=True)  # 安全创建目录
    return base_output_dir


