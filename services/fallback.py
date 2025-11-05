# -*- coding: utf-8 -*-
"""
回退业务逻辑
职责：处理无文档时的fallback逻辑和友好答复生成
"""
from typing import Optional
from modules.rag_types.rag_types import IntentResult
from modules.config.tool_config import get_kb_intent_labels


def get_no_document_response(
    intent: Optional[IntentResult],
    has_realtime_info: bool = False,
) -> str:
    """
    获取无文档时的友好答复
    
    Args:
        intent: 意图识别结果（可选）
        has_realtime_info: 是否有实时信息（实时工具可能不需要文档）
    
    Returns:
        str: 友好答复文本
    """
    # 如果有实时信息，即使没有文档也可以生成答案（由调用方处理）
    if has_realtime_info:
        return ""
    
    intent_label = intent.label if intent else "未知"
    kb_labels = get_kb_intent_labels()
    
    # 如果是KB意图，提供KB相关的fallback
    if intent and intent.label in kb_labels:
        return "知识库里没有找到足够可靠的资料。请补充更具体的信息（道路/路段、时间段、线路名），我再查一次；或改问法规/配时/公交/停车/充电/车路协同等主题。"
    
    # 非KB意图的fallback
    return get_non_kb_response(intent_label)


def get_non_kb_response(intent_label: str) -> str:
    """
    获取非KB意图的友好答复
    
    Args:
        intent_label: 意图标签
    
    Returns:
        str: 友好答复文本
    """
    if intent_label in ("闲聊其他", "系统操作"):
        return "这是智慧交通助手。如果你想查询限行规则、信号配时、公交换乘、停车计费或充电规范，请描述更具体的道路/线路/时段/场景。"
    
    return "该问题暂未接入知识库。请补充更具体的信息（如道路/路段、时间段、线路名），或改问法规/配时/公交/停车/充电/车路协同等主题。"


def should_fallback(
    documents: list,
    quality_ok: bool,
    has_realtime_info: bool = False,
) -> bool:
    """
    判断是否应该使用fallback答复
    
    Args:
        documents: 检索到的文档列表
        quality_ok: 文档质量是否合格
        has_realtime_info: 是否有实时信息
    
    Returns:
        bool: 是否应该使用fallback
    """
    # 如果有实时信息，即使没有文档也可以生成答案，不需要fallback
    if has_realtime_info:
        return False
    
    # 如果没有文档或质量不合格，使用fallback
    return not documents or not quality_ok


def check_has_realtime_info(context: str) -> bool:
    """
    检查上下文中是否有实时信息
    
    Args:
        context: 上下文文本
    
    Returns:
        bool: 是否有实时信息
    """
    # 支持未来扩展其他实时工具（如实时公交、实时天气等）
    return "【实时路况信息】" in context or "【实时" in context

