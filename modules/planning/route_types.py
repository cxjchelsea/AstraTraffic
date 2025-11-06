# -*- coding: utf-8 -*-
"""
路径规划类型定义
"""
from enum import Enum
from typing import Literal

# 路径类型
RouteType = Literal["driving", "walking", "transit", "riding"]

# 路径策略
RouteStrategy = Literal[
    "fastest",      # 最快路径
    "shortest",     # 最短路径
    "avoid_traffic", # 避开拥堵
    "avoid_highway", # 不走高速
    "avoid_toll",   # 不走收费
]

