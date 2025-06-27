# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: updata_name.py
@Time    : 2025/6/12 下午6:06
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 更新文件名称
@Usage   : 
"""
import os
from tkinter import Tk, filedialog
from PIL import Image, UnidentifiedImageError
import numpy as np


def rename_files(root_dir, start_number, new_root_dir):
    subsets = ['additions', 'landscapes', 'woodlands']
    file_types = ['A', 'B', 'labels']

    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"警告：未找到{subset}文件夹，跳过")
            continue

        # 获取 A 文件夹中的文件列表作为基准
        a_dir = os.path.join(subset_dir, file_types[0])
        if not os.path.exists(a_dir):
            print(f"警告：未找到{subset}/A文件夹，跳过")
            continue

        files = os.listdir(a_dir)
        files.sort()  # 确保顺序一致

        for idx, file_name in enumerate(files):
            base_name = os.path.splitext(file_name)[0]
            new_name = f"{start_number + idx}"

            for file_type in file_types:
                file_type_dir = os.path.join(subset_dir, file_type)
                if not os.path.exists(file_type_dir):
                    print(f"警告：未找到{subset}/{file_type}文件夹，跳过")
                    continue

                old_path = os.path.join(file_type_dir, f"{base_name}.tif")

                # 如果文件不存在，尝试其他常见扩展名
                if not os.path.exists(old_path):
                    for ext in ['.png', '.jpeg', '.tif', '.jpg']:
                        temp_path = os.path.join(file_type_dir, f"{base_name}{ext}")
                        if os.path.exists(temp_path):
                            old_path = temp_path
                            break

                if not os.path.exists(old_path):
                    print(f"警告：未找到文件 {old_path}，跳过")
                    continue

                # 构造新文件路径
                new_file_name = f"{new_name}.png"
                new_file_type_dir = os.path.join(new_root_dir, subset, file_type)
                os.makedirs(new_file_type_dir, exist_ok=True)  # 创建目标文件夹
                new_path = os.path.join(new_file_type_dir, new_file_name)

                try:
                    with Image.open(old_path) as img:
                        # 如果成功打开，继续处理
                        if file_type == 'labels':
                            img_array = np.array(img)
                            img_array[img_array != 0] = 255
                            new_img = Image.fromarray(img_array.astype(np.uint8))
                            new_img.save(new_path)
                        else:
                            img.save(new_path)
                    print(f"保存: {old_path} -> {new_path}")
                except (UnidentifiedImageError, IOError) as e:
                    print(f"跳过无效文件: {old_path}，错误: {e}")
                    continue

        start_number += len(files)


def main():
    # 选择根目录
    root_dir = '/sxs/zhoufei/DynaMoE/dataset/add'
    new_root_dir = '/sxs/zhoufei/DynaMoE/dataset/Change_new/'  # 新的保存路径

    start_number = 795
    # 执行重命名和保存
    rename_files(root_dir, start_number, new_root_dir)
    print("重命名和保存完成！")


if __name__ == "__main__":
    main()
