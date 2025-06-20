from .encoder import *
from .decoder import *
from .dynamoe import DynaMoE
from .loss import MoELoss


def build_model(cfg, training=False):
    model = DynaMoE(cfg)
    if not training:
        return model

    # 创建损失函数
    losses = MoELoss(
        use_moe=cfg.model.use_moe,
        weight_ce=cfg.loss.weight_ce,
        weight_load=cfg.loss.weight_load,
        weight_div=cfg.loss.weight_div,
        weight_spa=cfg.loss.weight_spa,
        weight_gate_base=cfg.loss.weight_gate_base,
        weight_gate_max=cfg.loss.weight_gate_max,
        warmup_epochs=cfg.loss.warmup_epochs,
    )

    return model, losses