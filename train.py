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
from utils import *


# TODO: 增量式训练
def main():
    cfg = get_args_config()
    device_info = parse_device(cfg.device)

    if device_info['distributed']:
        # 确保在分布式模式下正确设置
        print(f"Setting up distributed training with devices: {device_info['device_ids']}")
        if setup_distributed(device_info):
            print("Distributed training setup completed.")
            return
        else:
            # 如果分布式设置失败，回退到单GPU模式
            print("Distributed setup failed, falling back to single GPU mode")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_info['device_ids'][0])
            trainer = Trainer(cfg)
            trainer.run()
    else:
        trainer = Trainer(cfg)
        trainer.run()


if __name__ == '__main__':
    main()
