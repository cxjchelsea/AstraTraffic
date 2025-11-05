# -*- coding: utf-8 -*-
"""
用户消息模板（底层实现）
职责：管理所有返回给用户的消息模板（输出给用户的内容）
包含：
- 实时工具的友好提示消息
- 错误消息和引导信息
注意：与prompt.py的区别在于，这里管理的是给用户看的消息，不是给LLM的prompt
"""
from typing import Optional


# ==================== 实时工具消息 ====================

def get_realtime_traffic_prefix() -> str:
    """获取实时路况信息前缀"""
    return "【实时路况信息】\n"


def get_no_road_name_message(prefix: Optional[str] = None) -> str:
    """
    获取无法提取道路名称的提示消息
    
    Args:
        prefix: 消息前缀，如果为None则使用默认前缀
    
    Returns:
        格式化的提示消息
    """
    if prefix is None:
        prefix = get_realtime_traffic_prefix()
    
    return (
        f"{prefix}无法从查询中提取道路名称。"
        f"请提供具体的道路名称，例如：\n"
        f"- '中关村大街的路况'\n"
        f"- '三环路拥堵情况'\n"
        f"- '长安街现在怎么样'"
    )


def get_traffic_api_failed_message(road_name: str, prefix: Optional[str] = None) -> str:
    """
    获取实时路况API调用失败的提示消息
    
    Args:
        road_name: 道路名称
        prefix: 消息前缀，如果为None则使用默认前缀
    
    Returns:
        格式化的提示消息
    """
    if prefix is None:
        prefix = get_realtime_traffic_prefix()
    
    return (
        f"{prefix}未能获取到'{road_name}'的实时路况信息。可能原因：\n"
        f"1) 道路名称不准确（请尝试使用完整路名，如'中关村大街'、'三环路'）\n"
        f"2) API服务暂时不可用\n"
        f"3) 该道路暂未接入实时路况系统"
    )


def get_traffic_success_message(traffic_info_text: str, prefix: Optional[str] = None) -> str:
    """
    获取成功获取路况信息的消息
    
    Args:
        traffic_info_text: 格式化的路况信息文本
        prefix: 消息前缀，如果为None则使用默认前缀
    
    Returns:
        格式化的消息
    """
    if prefix is None:
        prefix = get_realtime_traffic_prefix()
    
    return f"{prefix}{traffic_info_text}"

