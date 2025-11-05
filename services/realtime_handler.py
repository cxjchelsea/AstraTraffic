# -*- coding: utf-8 -*-
"""
实时数据处理业务逻辑
职责：处理外部实时数据（路况信息等）
"""
from typing import Optional
from modules.realtime import AmapTrafficAPI, TrafficInfo
from modules.config.settings import AMAP_API_KEY, AMAP_DEFAULT_CITY

# 全局客户端实例（单例模式）
_traffic_client: Optional[AmapTrafficAPI] = None


def get_traffic_client() -> Optional[AmapTrafficAPI]:
    """
    获取实时路况客户端实例（单例）
    
    Returns:
        AmapTrafficAPI实例，如果未配置API密钥则返回None
    """
    global _traffic_client
    
    if _traffic_client is not None:
        return _traffic_client
    
    if not AMAP_API_KEY:
        return None
    
    _traffic_client = AmapTrafficAPI(AMAP_API_KEY)
    return _traffic_client


def get_road_traffic(road_name: str, city: Optional[str] = None) -> Optional[TrafficInfo]:
    """
    查询指定道路的实时路况（适配层接口）
    
    Args:
        road_name: 道路名称
        city: 城市名称，如果为None则使用配置中的默认城市
        
    Returns:
        TrafficInfo对象，失败返回None
    """
    # 使用配置中的默认城市
    if city is None:
        city = AMAP_DEFAULT_CITY
    client = get_traffic_client()
    if client is None:
        return None
    return client.get_traffic_by_road(road_name, city)


def format_traffic_info(traffic: TrafficInfo) -> str:
    """
    格式化路况信息为文本（用于RAG上下文）
    
    Args:
        traffic: TrafficInfo对象
        
    Returns:
        格式化的文本字符串
    """
    parts = [f"道路: {traffic.road_name}"]
    if traffic.city:
        parts.append(f"城市: {traffic.city}")
    parts.append(f"拥堵等级: {traffic.congestion_level}")
    if traffic.speed:
        parts.append(f"平均速度: {traffic.speed}km/h")
    if traffic.description:
        parts.append(f"详情: {traffic.description}")
    return " | ".join(parts)


def extract_road_name_from_query(query: str) -> Optional[str]:
    """
    从用户查询中提取道路名称（使用LLM智能提取）
    
    Args:
        query: 用户查询文本
    
    Returns:
        提取的道路名称，如果未找到或LLM调用失败返回None
    """
    try:
        from modules.generator.prompt import build_road_name_extraction_prompt
        from adapters.llm import get_llm_client
        
        prompt = build_road_name_extraction_prompt(query)
        llm_client = get_llm_client()
        response = llm_client(prompt, temperature=0.1, max_tokens=30)
        
        # 清理响应：移除空白字符、引号等
        road_name = response.strip().strip('"').strip("'").strip()
        
        # 检查是否表示"无"或无效
        if not road_name or road_name.lower() in ["无", "none", "null", ""]:
            return None
        
        # 进一步清理：移除常见的后缀词
        for word in ["的路况", "怎么样", "如何", "怎样", "情况", "状况", "拥堵"]:
            if road_name.endswith(word):
                road_name = road_name[:-len(word)].strip()
        
        return road_name if road_name else None
        
    except Exception as e:
        print(f"[WARNING] LLM提取道路名称失败: {e}")
        return None

