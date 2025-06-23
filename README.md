<h2 align="center">
  A Mixture-of-Experts Framework with Dynamic Class Expansion
</h2>

## 简介

本研究提出了一种基于**空间感知门控机制的动态专家混合模型** （Dynamic Mixture of Experts with Spatial-aware Gating），旨在解决传统变化检测模型在面对**类别持续扩展需求时结构僵化、训练成本高、泛化能力受限** 的问题。现有方法通常采用固定输出层和多分类 Softmax 激活函数，导致新增变化类型时必须重新设计网络结构并重新训练整个模型，难以适应实际应用中不断演化的监测目标（如新增道路、水体、植被等变化类别）。为此，我们引入 MoE 架构，将每类变化建模为一个独立的“专家”子网络，并通过一个轻量级的空间感知门控模块，为输入图像中的每个像素动态分配最合适的专家组合。该架构具有高度模块化与可扩展性：当新增变化类别时，只需添加对应的新专家模块，并微调门控网络，无需修改已有专家和编码器参数，从而实现**零重构扩展** 与**增量式高效训练** 。此外，专家之间的解耦训练机制有效缓解了多任务学习中的特征干扰问题，提升了模型对复杂变化模式的建模能力。实验表明，该方法不仅在已有类别上保持稳定性能，还能快速适配新类别，展现出良好的实用性与部署潜力。

## 安装步骤

### 1. 创建虚拟环境

```bash
conda create -n moe python=3.10 -y
conda activate moe
```

### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

## 数据集介绍
多掩码二值图（Multi-Mask Binary Labeling）,其中每个变化类别都有一个独立的二值掩码图

✅ 优点：
- 每个专家只关注自己类别的掩码（与 MoE 架构匹配）
- 新增类别只需添加新的掩码文件，无需修改已有标注
- 支持任意数量的变化类别，适合持续扩展
- 可以同时标注多个类别发生变化的区域（比互斥更合理）

我们提供两种数据集结构格式：
### 1. 默认结构：
数据集的结构如下：
```text
dataset/
  ├── train/
  │   ├── A/                  # 第一时相训练图像
  │   ├── B/                  # 第二时相训练图像
  │   └── masks/              # 训练标签（变化掩码）
  │         ├── building/     # 建筑变化掩码（0/1）
  │         ├── vegetation/   # 植被变化掩码（0/1）
  │         ├── road/         # 道路变化掩码（0/1）
  │         └── ...           # 后续新增类别可继续添加
  ├── val/
  │   ├── A/                  # 第一时相验证图像
  │   ├── B/                  # 第二时相验证图像
  │   └── masks/              # 验证标签（变化掩码）
  │         ├── building/     # 建筑变化掩码（0/1）
  │         ├── vegetation/   # 植被变化掩码（0/1）
  │         ├── road/         # 道路变化掩码（0/1）
  │         └── ...           # 后续新增类别可继续添加
  └── test/
      ├── A/                  # 第一时相测试图像
      ├── B/                  # 第二时相测试图像
      └── masks/              # 测试标签（变化掩码）
            ├── building/     # 建筑变化掩码（0/1）
            ├── vegetation/   # 植被变化掩码（0/1）
            ├── road/         # 道路变化掩码（0/1）
            └── ...           # 后续新增类别可继续添加
```
修改参数文件[configs](./configs/config.yaml)中的`data_format`为`default`。

### 2. 自定义结构：
数据集的结构如下：
```text
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
```
修改参数文件[configs](./configs/config.yaml)中的`data_format`为`custom`。

## 训练

### 命令行训练
```bash
python train.py -c ./configs/config.yaml
```

分布式训练
```bash
CUDA_VISIBLE_DEVICES=1,2 python train.py -c configs/config.yaml
```

## 测试
```bash
python test.py -c ./configs/config.yaml
```

## 推理TIF
```bash
python infer.py -c ./configs/config.yaml
```

## 贡献

欢迎提交问题和代码改进。请确保遵循项目的代码风格和贡献指南。

## 许可证

本项目使用 [Apache License 2.0](LICENSE)
