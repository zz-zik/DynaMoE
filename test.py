# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: test.py
@Time    : 2025/6/16 下午4:30
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 
@Usage   : 
"""
from engine import Tester
from utils import get_args_config


def main():
    cfg = get_args_config()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
