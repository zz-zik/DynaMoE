# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: add_mask.py
@Time    : 2025/6/13 上午10:25
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 根据文件夹下的文件名，来判断masks文件夹下面的子文件夹里面的文件是否缺少，如果缺少，则补充全黑的二值化图
@Usage   : 
"""

import os
from PIL import Image
import numpy as np


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def generate_black_mask(size, save_path):
    """生成全黑二值图像（0 值）"""
    mask = np.zeros(size[::-1], dtype=np.uint8)  # 反转尺寸 (width, height) -> (height, width)
    img = Image.fromarray(mask)
    img.save(save_path)


def add_missing_masks(image_root, mask_root, file_types=None):
    """
    检查并补充缺失的 mask 文件
    :param image_root: 主图像文件夹路径，如 images/
    :param mask_root: 掩码文件夹路径，如 masks/
    :param file_types: 要处理的掩码类型列表，如 ['building', 'vegetation']
    """
    if file_types is None:
        file_types = ['building', 'vegetation', 'additions', 'landscapes', 'woodlands']

    # 获取主图像文件名（假设为主文件夹 A 中的文件）
    a_dir = os.path.join(image_root, 'A')
    if not os.path.exists(a_dir):
        print(f"警告：未找到 {a_dir}，程序退出")
        return

    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    base_names = [os.path.splitext(f)[0] for f in files]

    # 遍历每个掩码类型
    for file_type in file_types:
        mask_type_dir = os.path.join(mask_root, file_type)
        ensure_dir(mask_type_dir)

        for base_name in base_names:
            expected_file = os.path.join(mask_type_dir, f"{base_name}.png")

            if not os.path.exists(expected_file):
                # 尝试获取原图尺寸
                try:
                    with Image.open(os.path.join(a_dir, f"{base_name}.png")) as img:
                        size = img.size  # (width, height)
                except Exception as e:
                    print(f"无法打开原图以获取尺寸: {e}")
                    continue

                # 生成全黑二值图
                generate_black_mask(size, expected_file)
                print(f"已补充缺失的掩码文件: {expected_file}")


def main():
    image_root = '/sxs/zhoufei/DynaMoE/dataset/change2'
    mask_root = '/sxs/zhoufei/DynaMoE/dataset/change2/masks'

    add_missing_masks(image_root, mask_root)


if __name__ == "__main__":
    main()
