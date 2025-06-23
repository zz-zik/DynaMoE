# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: loss.py
@Time    : 2025/6/12 下午5:47
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : DynaMoE 损失函数
@Usage   :
分割损失：


Moe损失：
交叉熵损失、负载均衡损失、辅助专家损失、门控熵损失
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoss(nn.Module):
    def __init__(
        self,
        use_moe: bool = True,
        weight_ce: float = 1.0,  # 分类损失权重
        weight_load: float = 0.01,  # 负载均衡损失权重
        weight_div: float = 0.001,  # 专家多样性损失权重
        weight_spa: float = 0.001,  # 稀疏性损失权重
        weight_gate_base: float = 0.0,  # 初始门控熵损失权重
        weight_gate_max: float = 0.05,  # 最大门控熵损失权重
        warmup_epochs: int = 10,  # 前多少个epoch逐渐增加权重
    ):
        super().__init__()
        self.use_moe = use_moe
        self.main_loss_weight = weight_ce
        self.load_balance_weight = weight_load
        self.diversity_weight = weight_div
        self.sparsity_weight = weight_spa
        self.gate_entropy_weight_base = weight_gate_base
        self.gate_entropy_weight_max = weight_gate_max
        self.warmup_epochs = warmup_epochs

        self.current_epoch = 0

        # 主损失
        # self.cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5, 0.8], device=device))  # 加入类别权重
        self.cls = nn.BCEWithLogitsLoss()

    def set_current_epoch(self, epoch: int):
        """
        设置当前训练轮次，用于动态调整 gate_entropy_weight
        """
        self.current_epoch = epoch

    def get_gate_entropy_weight(self) -> float:
        """
        根据当前训练轮次计算 gate_entropy_weight
        使用线性增长策略
        """
        if self.current_epoch < self.warmup_epochs:
            # 线性增长
            ratio = self.current_epoch / max(1, self.warmup_epochs)
            return self.gate_entropy_weight_base + ratio * (
                self.gate_entropy_weight_max - self.gate_entropy_weight_base
            )
        else:
            return self.gate_entropy_weight_max

    def forward(self, pred: dict, targets: torch.Tensor) -> dict:
        device = targets.device

        # 提取主任务预测结果
        if 'prediction' not in pred:
            raise KeyError("缺少关键字段 'prediction' 在 `pred` 字典中")
        predictions = pred['prediction']  # [B, num_experts, H, W]

        # 1. 主任务损失 - 变化检测损失
        expert_losses = self.cls(predictions, targets.float())  # 逐元素损失

        # 如果使用 MoE，则尝试提取 gates
        gates = None
        if self.use_moe:
            if 'gates' in pred:
                gates = pred['gates']  # [B, num_experts, H, W]
            else:
                raise ValueError("当 use_moe=True 时，必须提供 'gates' 字段在 pred 字典中")

            weighted_losses = expert_losses * gates  # [B, num_experts, H, W]
            main_loss = weighted_losses.sum(dim=1).mean()
        else:
            # 不使用 MoE 时：直接取所有专家的平均损失（等价于平均投票）
            main_loss = expert_losses.mean()

        # 初始化辅助损失为 0
        load_balance_loss = diversity_loss = sparsity_loss = gate_entropy_loss = torch.tensor(0.0, device=device)

        # 仅当 use_moe 启用时计算辅助损失
        if self.use_moe:
            if gates is None:
                raise ValueError("gates 不能为 None，当 use_moe=True 时必须提供门控权重")

            load_balance_loss = self.compute_load_balance_loss(gates)
            diversity_loss = self.compute_diversity_loss(predictions)
            sparsity_loss = self.compute_sparsity_loss(gates)
            gate_entropy_loss = self.compute_gate_entropy_loss(gates)

        # 获取当前 gate_entropy_weight
        current_gate_weight = self.get_gate_entropy_weight()

        # 总损失
        total_loss = (
            self.main_loss_weight * main_loss +
            self.load_balance_weight * load_balance_loss +
            self.diversity_weight * diversity_loss +
            self.sparsity_weight * sparsity_loss +
            current_gate_weight * gate_entropy_loss
        )

        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'load_balance_loss': load_balance_loss if self.use_moe else torch.tensor(0.0),
            'diversity_loss': diversity_loss if self.use_moe else torch.tensor(0.0),
            'sparsity_loss': sparsity_loss if self.use_moe else torch.tensor(0.0),
            'gate_entropy_loss': gate_entropy_loss if self.use_moe else torch.tensor(0.0),
        }


    def compute_load_balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        负载均衡损失 - 确保所有专家都被充分利用

        Args:
            gates: 门控权重 [B, num_experts, H, W]

        Returns:
            负载均衡损失
        """
        # 计算每个专家的平均激活程度
        expert_usage = gates.mean(dim=[0, 2, 3])  # [num_experts]

        # 计算使用率的方差 - 方差越小，负载越均衡
        uniform_usage = torch.ones_like(expert_usage) / expert_usage.size(0)
        load_balance_loss = F.mse_loss(expert_usage, uniform_usage)

        return load_balance_loss

    def compute_diversity_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        专家多样性损失 - 鼓励不同专家学习不同的特征表示
        Args:
            predictions: 专家预测 [B, num_experts, H, W]
        Returns:
            多样性损失
        """
        B, num_experts, H, W = predictions.shape

        # 计算专家预测之间的相关性
        # 将预测展平为向量
        pred_flat = predictions.view(B, num_experts, -1)  # [B, num_experts, H*W]

        # 计算专家间的余弦相似度
        pred_norm = F.normalize(pred_flat, dim=2)  # L2标准化

        # 计算相似度矩阵
        similarity_matrix = torch.bmm(pred_norm, pred_norm.transpose(1, 2))  # [B, num_experts, num_experts]

        # 提取上三角矩阵（排除对角线）
        mask = torch.triu(torch.ones(num_experts, num_experts), diagonal=1).bool()
        similarities = similarity_matrix[:, mask]  # [B, num_pairs]

        # 多样性损失 = 相似度的平方和（鼓励低相似度）
        diversity_loss = similarities.pow(2).mean()

        return diversity_loss

    def compute_sparsity_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        稀疏性损失 - 鼓励门控网络产生稀疏激活

        Args:
            gates: 门控权重 [B, num_experts, H, W]

        Returns:
            稀疏性损失
        """
        # 使用L1正则化鼓励稀疏性
        sparsity_loss = torch.mean(torch.sum(gates, dim=1))

        return sparsity_loss

    def compute_gate_entropy_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        计算门控权重的熵损失，鼓励更明确的专家选择

        Args:
            gates: 门控权重 [B, num_experts, H, W]

        Returns:
            gate_entropy_loss: 标量张量
        """
        gates = gates.clamp(min=1e-8)  # 防止 log(0)
        gate_entropy = - (gates * torch.log(gates)).sum(dim=1)  # 每个像素点的熵 [B, H, W]
        gate_entropy_loss = gate_entropy.mean()  # 取平均作为最终损失值
        return gate_entropy_loss
