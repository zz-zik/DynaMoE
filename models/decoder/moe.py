# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: moe.py
@Time    : 2025/6/10 下午3:43
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


class SpatialAwareGating(nn.Module):
    """空间感知门控网络 - 为每个像素动态分配专家权重"""

    def __init__(self,
                 input_channels: int,
                 num_experts: int,
                 hidden_dim: int = 256,
                 spatial_kernel_size: int = 3):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # 空间特征提取层 - 捕获局部上下文信息
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=spatial_kernel_size,
                      padding=spatial_kernel_size // 2, groups=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        # 全局上下文编码 - 通过全局平均池化获取全局信息
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, hidden_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 门控权重预测网络
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 4, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim // 4, num_experts, kernel_size=1)
        )

        # 温度参数用于控制门控分布的锐度
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            gates: 专家权重 [B, num_experts, H, W]
        """
        B, C, H, W = x.shape

        # 提取空间特征
        spatial_feat = self.spatial_conv(x)  # [B, hidden_dim//2, H, W]

        # 提取全局上下文并广播到空间维度
        global_feat = self.global_context(x)  # [B, hidden_dim//4, 1, 1]
        global_feat = global_feat.expand(-1, -1, H, W)  # [B, hidden_dim//4, H, W]

        # 融合局部和全局特征
        combined_feat = torch.cat([spatial_feat, global_feat], dim=1)

        # 预测门控权重
        gate_logits = self.gate_predictor(combined_feat)  # [B, num_experts, H, W]

        # 应用温度缩放的softmax激活
        gates = F.softmax(gate_logits / self.temperature, dim=1)

        return gates

    def add_expert(self):
        """动态添加新专家时扩展门控网络"""
        old_num_experts = self.num_experts
        self.num_experts += 1

        # 扩展最后一层卷积的输出通道
        old_conv = self.gate_predictor[-1]
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            self.num_experts,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # 复制原有权重并初始化新权重
        with torch.no_grad():
            new_conv.weight[:old_num_experts] = old_conv.weight
            new_conv.weight[old_num_experts:] = torch.randn_like(
                new_conv.weight[old_num_experts:]) * 0.01
            if new_conv.bias is not None:
                new_conv.bias[:old_num_experts] = old_conv.bias
                new_conv.bias[old_num_experts:] = 0.0

        self.gate_predictor[-1] = new_conv


class ChangeDetectionExpert(nn.Module):
    """单个变化检测专家网络"""

    def __init__(self,
                 input_channels: int,
                 output_channels: int = 1,
                 hidden_channels: int = 128):
        super().__init__()

        # 专家特定的特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # 残差连接增强特征学习
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
        )

        # 变化检测头
        self.change_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_channels // 2, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            change_prob: 变化概率图 [B, output_channels, H, W]
        """
        features = self.feature_extractor(x)
        change_prob = self.change_head(features)
        return change_prob


class ResidualBlock(nn.Module):
    """残差块用于增强特征学习"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DynamicMoE(nn.Module):
    """动态专家混合变化检测模型"""

    def __init__(self,
                 input_channels: int,
                 initial_expert_types: List[str] = ['building', 'vegetation', 'road'],
                 expert_hidden_dim: int = 128,
                 gate_hidden_dim: int = 256):
        super().__init__()

        self.input_channels = input_channels
        self.expert_types = initial_expert_types.copy()
        self.num_experts = len(initial_expert_types)

        # 初始化门控网络
        self.gating_network = SpatialAwareGating(
            input_channels=input_channels,
            num_experts=self.num_experts,
            hidden_dim=gate_hidden_dim
        )

        # 初始化专家网络
        self.experts = nn.ModuleDict()
        for expert_type in initial_expert_types:
            self.experts[expert_type] = ChangeDetectionExpert(
                input_channels=input_channels,
                hidden_channels=expert_hidden_dim
            )

        # 专家激活统计 - 用于负载均衡和专家利用率分析
        self.register_buffer('expert_usage', torch.zeros(self.num_experts))

    def forward(self, x: torch.Tensor, return_gates: bool = False) -> Dict:
        """
        Args:
            x: 输入特征图 [B, C, H, W]
            return_gates: 是否返回门控权重
        Returns:
            results: 包含最终预测和各专家预测的字典
        """
        B, C, H, W = x.shape

        # 获取门控权重
        gates = self.gating_network(x)  # [B, num_experts, H, W]

        # 更新专家使用统计
        if self.training:
            expert_usage = gates.mean(dim=[0, 2, 3])  # [num_experts]
            self.expert_usage = 0.9 * self.expert_usage + 0.1 * expert_usage

        # 获取所有专家的预测
        expert_outputs = []
        for i, expert_type in enumerate(self.expert_types):
            expert_pred = self.experts[expert_type](x)  # [B, 1, H, W]
            expert_outputs.append(expert_pred)

        # 堆叠专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, 1, H, W]

        # 获取最终预测[B, num_experts, H, W]
        prediction = expert_outputs.squeeze(2)

        results = {
            'prediction': prediction,
        }

        if return_gates:
            results['gates'] = gates

        return results

    def add_expert(self, expert_type: str, expert_hidden_dim: int = 128):
        """动态添加新专家 - 实现零重构扩展"""
        if expert_type in self.expert_types:
            print(f"Expert '{expert_type}' already exists!")
            return

        print(f"Adding new expert: {expert_type}")

        # 添加新专家网络
        self.experts[expert_type] = ChangeDetectionExpert(
            input_channels=self.input_channels,
            hidden_channels=expert_hidden_dim
        )

        # 更新专家列表
        self.expert_types.append(expert_type)
        self.num_experts += 1

        # 扩展门控网络
        self.gating_network.add_expert()

        # 扩展专家使用统计
        new_usage = torch.zeros(self.num_experts)
        new_usage[:-1] = self.expert_usage
        self.register_buffer('expert_usage', new_usage)

        # print(f"Successfully added expert '{expert_type}'. Total experts: {self.num_experts}")

    def remove_expert(self, expert_type: str):
        if expert_type not in self.expert_types:
            print(f"Expert '{expert_type}' does not exist!")
            return

        print(f"Removing expert: {expert_type}")

        # 移除专家
        del self.experts[expert_type]

        # 更新专家列表
        self.expert_types.remove(expert_type)
        self.num_experts -= 1

        # 调整门控网络
        old_conv = self.gating_network.gate_predictor[-1]
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            self.num_experts,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            indices = [i for i, et in enumerate(self.expert_types) if et != expert_type]
            new_conv.weight.data = old_conv.weight.data[indices]
            if new_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data[indices]

        self.gating_network.gate_predictor[-1] = new_conv

        # 更新专家使用统计
        new_usage = torch.zeros(self.num_experts)
        indices_tensor = torch.tensor([i for i, et in enumerate(self.expert_types)])
        new_usage = self.expert_usage[indices_tensor]
        self.register_buffer('expert_usage', new_usage)

        # print(f"Successfully removed expert '{expert_type}'. Total experts: {self.num_experts}")

    def freeze_existing_experts(self):
        """冻结现有专家参数 - 用于增量学习"""
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = False

    def unfreeze_all_experts(self):
        """解冻所有专家参数"""
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = True

    def get_expert_specialization(self, x: torch.Tensor) -> Dict:
        """分析专家特化程度"""
        with torch.no_grad():
            results = self.forward(x, return_gates=True)
            gates = results['gates']  # [B, num_experts, H, W]

            # 计算每个专家的激活程度
            expert_activation = gates.mean(dim=[0, 2, 3])  # [num_experts]

            # 计算门控权重的熵 - 衡量专家选择的确定性
            gate_entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=1).mean()

            specialization_info = {
                'expert_activation': {
                    expert_type: float(expert_activation[i])
                    for i, expert_type in enumerate(self.expert_types)
                },
                'gate_entropy': float(gate_entropy),
                'most_active_expert': self.expert_types[torch.argmax(expert_activation).item()],
                'expert_usage_history': {
                    expert_type: float(self.expert_usage[i])
                    for i, expert_type in enumerate(self.expert_types)
                }
            }

        return specialization_info


# 使用示例和测试代码
if __name__ == "__main__":
    """测试动态MoE模型"""
    print("=== 测试动态MoE变化检测模型 ===")

    # 创建模型
    model = DynamicMoE(
        input_channels=256,  # 假设输入特征图有256个通道
        initial_expert_types=['road', 'water', 'vegetation'],  #
        expert_hidden_dim=128,  # 专家隐藏层维度
        gate_hidden_dim=256  # 门控隐藏层维度
    )

    # 创建测试输入 (原图尺寸的特征图)
    batch_size, height, width = 2, 512, 512
    input_features = torch.randn(batch_size, 256, height, width)

    print(f"输入特征图尺寸: {input_features.shape}")
    print(f"初始专家数量: {model.num_experts}")
    print(f"专家类型: {model.expert_types}")

    # 前向传播
    results = model(input_features, return_gates=True)

    print(f"\n预测结果尺寸: {results['prediction'].shape}")
    print(f"门控权重尺寸: {results['gates'].shape}")

    # 分析专家特化
    specialization = model.get_expert_specialization(input_features)
    print(f"\n专家激活度: {specialization['expert_activation']}")
    print(f"门控熵: {specialization['gate_entropy']:.4f}")
    print(f"最活跃专家: {specialization['most_active_expert']}")

    # 测试动态添加专家
    print(f"\n=== 动态添加新专家 ===")
    model.freeze_existing_experts()
    model.add_expert('building')
    model.add_expert('forest')

    print(f"更新后专家数量: {model.num_experts}")
    print(f"更新后专家类型: {model.expert_types}")

    # 测试扩展后的前向传播
    results_expanded = model(input_features, return_gates=True)
    print(f"扩展后预测结果尺寸: {results_expanded['prediction'].shape}")
    print(f"扩展后门控权重尺寸: {results_expanded['gates'].shape}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"总参数量: {total_params / 1e6:,}")
    print(f"可训练参数量: {trainable_params / 1e6:,}")