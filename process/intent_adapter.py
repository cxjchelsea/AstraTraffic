# -*- coding: utf-8 -*-
"""
intent_adapter.py
把“意图识别”封装成统一接口：
- 两种模式：规则意图（未上模型前临时使用）/ 模型推理（正式）
- 线程安全单例懒加载模型
- 注意：INTENT_TO_KB 来自 modules.retriever.rag_retriever（按你的现有结构）
返回：
    (label: str, score: float, topk: List[Tuple[str, float]])
"""

from __future__ import annotations
from typing import Tuple, List
import threading

from settings import USE_RULE_INTENT
# 你的映射定义在 rag_retriever 里（不是 retriever_adapter）
from modules.retriever.rag_retriever import INTENT_TO_KB

# -------------------------
# 1) 规则路由（临时跑通链路用）
#    规则标签名需与 INTENT_TO_KB 的 key 对齐
# -------------------------
# 可以按需增删关键字，但右侧标签必须出现在 INTENT_TO_KB 中
_RULES: List[Tuple[tuple, str]] = [
    (("限行", "处罚", "罚款", "记分", "专用道", "电子警察"), "交通法规"),
    (("信号", "配时", "相位", "绿信比", "潮汐车道", "可变车道", "诱导"), "信号配时"),
    (("公交", "地铁", "轨道", "换乘", "票价", "首班", "末班", "到站"), "公交规则"),
    (("停车", "泊位", "车场", "路侧"), "停车政策"),
    (("充电", "快充", "直流", "交流", "电桩"), "充电规范"),
    (("车路协同", "RSU", "OBU", "C-V2X", "V2X"), "车路协同"),
    # 如需把“路况查询/线路规划/停车空位/天气影响”等映射到静态库，也可加关键词规则
]

def _rule_intent(query: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    """极简关键词规则版意图识别（未上新模型前用于跑通链路）。"""
    q = (query or "").lower()
    for kws, label in _RULES:
        if any(k.lower() in q for k in kws):
            # 置信度给固定高值，TopK 只返回自身，便于 rag_core 走“高置信路径”
            return label, 0.95, [(label, 0.95)]
    # 未命中关键词 → 归入闲聊/未知；若你在 INTENT_TO_KB 中也维护了“闲聊其他”，会走对应路由
    return "闲聊其他", 0.20, [("闲聊其他", 0.20)]

# -------------------------
# 2) 模型路由（正式）
# -------------------------
_loader_lock = threading.Lock()
_singleton = None  # type: ignore

def _get_model():
    """延迟加载 PrimaryInfer（仅在 USE_RULE_INTENT=0 时触发）。"""
    global _singleton
    if _singleton is None:
        with _loader_lock:
            if _singleton is None:
                from modules.intent.intent_classify import PrimaryInfer
                _singleton = PrimaryInfer()
    return _singleton

# -------------------------
# 3) 对外统一接口
# -------------------------
def predict_intent(text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    返回：(label, score, topk)
    - USE_RULE_INTENT=1：使用规则意图（不依赖模型）
    - USE_RULE_INTENT=0：使用 PrimaryInfer 模型推理
    """
    if USE_RULE_INTENT:
        label, prob, topk = _rule_intent(text)
        # 保险：若规则返回的 label 不在 INTENT_TO_KB 中，降级为“闲聊其他”
        if label not in INTENT_TO_KB:
            label, prob, topk = "闲聊其他", 0.20, [("闲聊其他", 0.20)]
        return label, prob, topk

    clf = _get_model()
    out = clf.predict_one(text)
    label, prob = out["primary"]
    topk = out.get("topk_primary") or []
    return label, float(prob), [(l, float(p)) for l, p in topk]

# -------------------------
# 4) 可选：热重载（更新权重后无需重启进程）
# -------------------------
def reload_model() -> None:
    """替换 /data/models 下的权重后，调用以便下次预测重新加载。"""
    global _singleton
    with _loader_lock:
        _singleton = None
