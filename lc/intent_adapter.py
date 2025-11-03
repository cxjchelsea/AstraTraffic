# -*- coding: utf-8 -*-
"""
意图识别适配器（LangChain 版本）
直接调用 modules.intent，不使用 process adapter
"""
from typing import Tuple, List
import threading

from settings import USE_RULE_INTENT
from modules.retriever.rag_retriever import INTENT_TO_KB


# -------------------------
# 规则路由（临时跑通链路用）
# -------------------------
_RULES: List[Tuple[tuple, str]] = [
    (("限行", "处罚", "罚款", "记分", "专用道", "电子警察"), "交通法规"),
    (("信号", "配时", "相位", "绿信比", "潮汐车道", "可变车道", "诱导"), "信号配时"),
    (("公交", "地铁", "轨道", "换乘", "票价", "首班", "末班", "到站"), "公交规则"),
    (("停车", "泊位", "车场", "路侧"), "停车政策"),
    (("充电", "快充", "直流", "交流", "电桩"), "充电规范"),
    (("车路协同", "RSU", "OBU", "C-V2X", "V2X"), "车路协同"),
]


def _rule_intent(query: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    """极简关键词规则版意图识别（未上新模型前用于跑通链路）"""
    q = (query or "").lower()
    for kws, label in _RULES:
        if any(k.lower() in q for k in kws):
            return label, 0.95, [(label, 0.95)]
    
    # 未命中关键词 → 归入闲聊/未知
    if "闲聊其他" not in INTENT_TO_KB:
        return "交通法规", 0.20, [("交通法规", 0.20)]
    return "闲聊其他", 0.20, [("闲聊其他", 0.20)]


# -------------------------
# 模型路由（正式）
# -------------------------
_loader_lock = threading.Lock()
_singleton = None


def _get_model():
    """延迟加载 PrimaryInfer（仅在 USE_RULE_INTENT=0 时触发）"""
    global _singleton
    if _singleton is None:
        with _loader_lock:
            if _singleton is None:
                from modules.intent.intent_classify import PrimaryInfer
                _singleton = PrimaryInfer()
    return _singleton


# -------------------------
# 对外统一接口
# -------------------------
def predict_intent(text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    返回：(label, score, topk)
    - USE_RULE_INTENT=1：使用规则意图（不依赖模型）
    - USE_RULE_INTENT=0：使用 PrimaryInfer 模型推理（直接调用 modules）
    """
    if USE_RULE_INTENT:
        label, prob, topk = _rule_intent(text)
        # 保险：若规则返回的 label 不在 INTENT_TO_KB 中，降级为"闲聊其他"
        if label not in INTENT_TO_KB:
            label, prob, topk = "闲聊其他", 0.20, [("闲聊其他", 0.20)]
        return label, prob, topk

    clf = _get_model()
    out = clf.predict_one(text)
    label, prob = out["primary"]
    topk = out.get("topk_primary") or []
    
    # 保险：若不在映射中，降级
    if label not in INTENT_TO_KB:
        label, prob = "闲聊其他", 0.20
        topk = [("闲聊其他", 0.20)]
    
    return label, float(prob), [(l, float(p)) for l, p in topk]


# -------------------------
# 可选：热重载（更新权重后无需重启进程）
# -------------------------
def reload_model() -> None:
    """替换 /data/models 下的权重后，调用以便下次预测重新加载"""
    global _singleton
    with _loader_lock:
        _singleton = None

