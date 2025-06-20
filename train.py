# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: train.py
@Time    : 2025/6/13 下午5:45
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   : 
"""
from engine import Trainer
from utils import get_args_config


# TODO: DeepSpeed分布式训练
# TODO: 增量式训练
# 门控熵损失动态调整
def main():

    cfg = get_args_config()
    # 创建训练引擎并运行
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
