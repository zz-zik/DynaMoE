# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: dynamoe.py
@Time    : 2025/6/10 下午4:32
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : DynaMoE 模型
@Usage   : 
"""
from torch import nn, Tensor
import torch.nn.functional as F
from models import MambaDecoder


class DynaMoE(nn.Module):
    def __init__(self, cfg=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm_layer = norm_layer
        self.use_moe = cfg.model.use_moe  # 是否使用 MOE 模型
        if cfg.model.backbone == 'sigma_tiny':
            from models import vssm_tiny
            self.channels = [64, 128, 320, 512]
            self.backbone = vssm_tiny(norm_fuse=norm_layer)
        elif cfg.model.backbone == 'sigma_small':
            from models import vssm_small
            self.channels = [96, 192, 384, 768]
            self.backbone = vssm_small(norm_fuse=norm_layer)
        elif cfg.model.backbone == 'sigma_base':
            from models import vssm_base
            self.channels = [128, 256, 512, 1024]
            self.backbone = vssm_base(norm_fuse=norm_layer)
        else:
            raise ValueError('backbone not supported')
        self.decode_head = MambaDecoder(img_size=cfg.data.img_size, in_channels=self.channels,
                                        classes=cfg.data.classes, embed_dim=self.channels[0],
                                        deep_supervision=False, use_moe=cfg.model.use_moe,
                                        expert_hidden_dim=cfg.model.expert_hidden_dim,
                                        gate_hidden_dim=cfg.model.gate_hidden_dim,
                                        new_classes=cfg.model.new_classes)

    def forward(self, image1: Tensor, image2: Tensor):
        orisize = image1.shape

        x = self.backbone(image1, image2)
        out = self.decode_head.forward(x)
        # out可能是字典格式或者张量格式
        if isinstance(out, dict):
            for key in out:
                out[key] = F.interpolate(out[key], size=orisize[2:], mode='bilinear', align_corners=False)
        else:
            out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)

        return out


# 测试
if __name__ == '__main__':
    import torch
    from utils import load_config
    from deepspeed.profiling.flops_profiler import FlopsProfiler

    cfg = load_config("../configs/config.yaml")
    device = cfg.device  # 从配置中获取设备，例如 'cuda' 或 'cpu'

    image1 = torch.randn(2, 3, 512, 512).to(device)
    image2 = torch.randn(2, 3, 512, 512).to(device)

    model = DynaMoE(cfg).to(device)
    prof = FlopsProfiler(model)
    prof.start_profile()
    output = model(image1, image2)
    if isinstance(output, dict):
        print("output shapes:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print(f"output shape: {output.shape}")
    prof.stop_profile()
    print("GFlops: ", prof.get_total_flops() / (10 ** 9))
    prof.end_profile()
