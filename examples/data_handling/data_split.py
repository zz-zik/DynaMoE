# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: data_split.py
@Time    : 2025/6/13 下午5:56
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 数据拆分脚本
@Usage   :
data/
  ├── A/                      # 第一时相训练图像
  │── B/                      # 第二时相训练图像
  │── masks/                  # 训练标签（变化掩码）
  │   ├── building/     # 建筑变化掩码（0/1）
  │   ├── vegetation/   # 植被变化掩码（0/1）
  │   ├── road/         # 道路变化掩码（0/1）
  │   └── ...           # 后续新增类别可继续添加
  └── list                    # 列表文件
      ├── train.txt           # 训练集列表
      ├── val.txt             # 训练集列表
      └── test.txt            # 验证集列表
"""
import os
import random
import argparse


def split_data(data_base, ratios):
    """
    拆分数据集为训练集、验证集和测试集
    :param data_base: 根目录，包含 list 和 A 文件夹
    :param ratios: 拆分比例 [train_ratio, val_ratio, test_ratio]
    """
    # 定义路径
    a_dir = os.path.join(data_base, 'A')  # list 文件夹
    output_dir = os.path.join(data_base, 'list')   # 输出文件夹 A

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有文件名
    file_list = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]

    # 打乱文件列表
    random.shuffle(file_list)

    # 计算分割点
    total_files = len(file_list)
    train_end = int(total_files * ratios[0])
    val_end = train_end + int(total_files * ratios[1])

    # 分割数据集
    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]

    # 写入对应的 txt 文件
    def write_to_file(file_list, filename):
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            for item in file_list:
                f.write(f"{item}\n")

    write_to_file(train_files, 'train.txt')
    write_to_file(val_files, 'val.txt')
    write_to_file(test_files, 'test.txt')

    print(f"Split completed: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test files.")


if __name__ == "__main__":
    data_base = '../../dataset/change'  # 根目录路径
    ratios = [0.8, 0.1, 0.1]  # 拆分比例
    split_data(data_base, ratios)