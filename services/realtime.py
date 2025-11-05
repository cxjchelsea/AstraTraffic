# -*- coding: utf-8 -*-
"""
实时工具业务逻辑
职责：统一管理所有实时工具相关的业务逻辑
包含：
- 客户端管理（单例模式）
- 数据获取和格式化
- 工具执行器
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass

from modules.realtime import AmapTrafficAPI, TrafficInfo
from modules.realtime.map import AmapMapAPI, MapInfo, get_map_client
from modules.config.settings import AMAP_API_KEY, AMAP_DEFAULT_CITY
from services.tool_selector import ToolSelection
from modules.generator.messages import (
    get_realtime_traffic_prefix,
    get_no_road_name_message,
    get_traffic_api_failed_message,
    get_traffic_success_message,
)


# ==================== 客户端管理（单例） ====================

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


# ==================== 数据获取函数 ====================

def get_road_traffic(road_name: str, city: Optional[str] = None) -> Optional[TrafficInfo]:
    """
    查询指定道路的实时路况
    
    Args:
        road_name: 道路名称
        city: 城市名称，如果为None则使用配置中的默认城市
        
    Returns:
        TrafficInfo对象，失败返回None
    """
    if city is None:
        city = AMAP_DEFAULT_CITY
    client = get_traffic_client()
    if client is None:
        return None
    return client.get_traffic_by_road(road_name, city)


def get_location_map(location_name: str, city: Optional[str] = None) -> Optional[MapInfo]:
    """
    获取指定位置的地图信息
    
    Args:
        location_name: 地点名称
        city: 城市名称，如果为None则使用配置中的默认城市
        
    Returns:
        MapInfo对象，失败返回None
    """
    if city is None:
        city = AMAP_DEFAULT_CITY
    
    client = get_map_client()
    if client is None:
        return None
    
    return client.get_map_info(location_name, city)


# ==================== 格式化函数 ====================

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


def format_map_info_to_dict(map_info: MapInfo) -> Dict[str, Any]:
    """
    将MapInfo对象转换为字典格式（用于API返回）
    
    Args:
        map_info: MapInfo对象
        
    Returns:
        字典格式的地图数据
    """
    return {
        "location": {
            "lng": map_info.location.lng,
            "lat": map_info.location.lat
        },
        "zoom": map_info.zoom,
        "location_name": map_info.location_name,
        "markers": map_info.markers or [],
        "show_traffic": False  # 默认不显示路况图层
    }


# ==================== 信息提取函数 ====================

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


def extract_location_from_query(query: str) -> Optional[str]:
    """
    从用户查询中提取地点名称（使用LLM智能提取，失败时使用规则提取）
    
    Args:
        query: 用户查询文本
    
    Returns:
        提取的地点名称，如果未找到或LLM调用失败返回None
    """
    # 首先尝试规则提取（简单的fallback）
    location_name = _extract_location_by_rule(query)
    if location_name:
        print(f"[INFO] 使用规则提取地点名称: {location_name}")
        return location_name
    
    # 如果规则提取失败，使用LLM提取
    try:
        from modules.generator.prompt import build_location_extraction_prompt
        from adapters.llm import get_llm_client
        
        prompt = build_location_extraction_prompt(query)
        llm_client = get_llm_client()
        response = llm_client(prompt, temperature=0.1, max_tokens=30)
        
        # 清理响应：移除空白字符、引号等
        location_name = response.strip().strip('"').strip("'").strip()
        
        print(f"[INFO] LLM提取地点名称: {location_name}")
        
        # 检查是否表示"无"或无效
        if not location_name or location_name.lower() in ["无", "none", "null", ""]:
            return None
        
        # 进一步清理：移除常见的地图相关后缀词
        for word in ["的地图", "地图", "位置", "在哪里", "在哪", "查看", "看"]:
            if location_name.endswith(word):
                location_name = location_name[:-len(word)].strip()
        
        return location_name if location_name else None
        
    except Exception as e:
        print(f"[WARNING] LLM提取地点名称失败: {e}")
        # LLM失败时，再次尝试规则提取
        return _extract_location_by_rule(query)


def _extract_location_by_rule(query: str) -> Optional[str]:
    """
    使用规则从查询中提取地点名称（简单fallback）
    
    Args:
        query: 用户查询文本
    
    Returns:
        提取的地点名称，如果未找到返回None
    """
    import re
    
    # 移除常见的地图相关词汇
    patterns_to_remove = [
        r"查看(.+?)的地图",
        r"(.+?)的地图",
        r"(.+?)在哪里",
        r"(.+?)在哪",
        r"显示(.+?)地图",
        r"(.+?)位置",
        r"查看(.+?)",
        r"显示(.+?)",
    ]
    
    for pattern in patterns_to_remove:
        match = re.search(pattern, query)
        if match:
            location = match.group(1).strip()
            # 过滤掉太短的或明显不是地名的
            if len(location) >= 2 and location not in ["的", "地图", "位置"]:
                return location
    
    # 如果没有匹配到模式，尝试直接提取（去除常见词汇后的剩余部分）
    words_to_remove = ["查看", "显示", "地图", "位置", "在哪里", "在哪", "的"]
    remaining = query
    for word in words_to_remove:
        remaining = remaining.replace(word, "")
    
    remaining = remaining.strip()
    if len(remaining) >= 2:
        return remaining
    
    return None


# ==================== 工具执行器 ====================

@dataclass
class RealtimeToolResult:
    """实时工具执行结果"""
    success: bool
    context_text: str  # 添加到上下文的文本
    metadata: Dict[str, Any] = None  # 额外的元数据


class RealtimeToolExecutor:
    """实时工具执行器"""
    
    def __init__(self):
        """初始化实时工具执行器"""
        # 注册各个实时工具的处理函数
        self._tool_handlers = {
            "realtime_traffic": self._handle_traffic_tool,
            "realtime_map": self._handle_map_tool,
        }
    
    def execute_tool(
        self,
        tool_selection: ToolSelection,
        query: str,
        **kwargs
    ) -> Optional[RealtimeToolResult]:
        """
        执行实时工具
        
        Args:
            tool_selection: Tool选择结果
            query: 用户查询
            **kwargs: 额外参数（如城市名称等）
        
        Returns:
            RealtimeToolResult对象，如果工具不支持或执行失败返回None
        """
        handler = self._tool_handlers.get(tool_selection.tool)
        if not handler:
            return None
        
        return handler(query, tool_selection, **kwargs)
    
    def _handle_traffic_tool(
        self,
        query: str,
        tool_selection: ToolSelection,
        city: Optional[str] = None,
        **kwargs
    ) -> RealtimeToolResult:
        """
        处理实时路况工具
        
        Args:
            query: 用户查询
            tool_selection: Tool选择结果
            city: 城市名称（可选，默认使用配置中的城市）
        
        Returns:
            RealtimeToolResult对象
        """
        default_city = city or AMAP_DEFAULT_CITY
        
        # 从查询中提取道路名称
        road_name = extract_road_name_from_query(query)
        
        if not road_name:
            context_text = get_no_road_name_message()
            return RealtimeToolResult(
                success=False,
                context_text=context_text,
                metadata={"reason": "no_road_name"}
            )
        
        # 调用实时路况API
        traffic_info = get_road_traffic(road_name, default_city)
        
        if traffic_info:
            traffic_text = format_traffic_info(traffic_info)
            context_text = get_traffic_success_message(traffic_text)
            return RealtimeToolResult(
                success=True,
                context_text=context_text,
                metadata={
                    "road_name": road_name,
                    "city": default_city,
                    "congestion_level": traffic_info.congestion_level,
                    "speed": traffic_info.speed,
                }
            )
        else:
            context_text = get_traffic_api_failed_message(road_name)
            return RealtimeToolResult(
                success=False,
                context_text=context_text,
                metadata={
                    "road_name": road_name,
                    "city": default_city,
                    "reason": "api_failed"
                }
            )
    
    def _handle_map_tool(
        self,
        query: str,
        tool_selection: ToolSelection,
        city: Optional[str] = None,
        **kwargs
    ) -> RealtimeToolResult:
        """
        处理地图查看工具
        
        Args:
            query: 用户查询
            tool_selection: Tool选择结果
            city: 城市名称（可选，默认使用配置中的城市）
        
        Returns:
            RealtimeToolResult对象，包含地图数据和上下文文本
        """
        default_city = city or AMAP_DEFAULT_CITY
        
        # 从查询中提取地点名称
        location_name = extract_location_from_query(query)
        
        if not location_name:
            print(f"[WARNING] 无法从查询中提取地点名称: {query}")
            context_text = "【地图查看】\n无法从查询中提取地点名称。请提供具体的地点名称，例如：\n- '查看中关村的地图'\n- '天安门在哪里'\n- '北京市朝阳区的地图'"
            return RealtimeToolResult(
                success=False,
                context_text=context_text,
                metadata={"reason": "no_location_name"}
            )
        
        print(f"[INFO] 提取到地点名称: {location_name}, 城市: {default_city}")
        
        # 调用地图API
        map_info = get_location_map(location_name, default_city)
        
        if map_info:
            print(f"[INFO] 成功获取地图信息: {map_info.location_name} ({map_info.location.lng}, {map_info.location.lat})")
            
            context_text = (
                f"【地图查看】\n"
                f"位置：{map_info.location_name}\n"
                f"坐标：经度 {map_info.location.lng:.6f}, 纬度 {map_info.location.lat:.6f}\n"
                f"地址：{map_info.location.address or '暂无'}"
            )
            
            map_data_dict = format_map_info_to_dict(map_info)
            print(f"[INFO] 地图数据: {map_data_dict}")
            
            return RealtimeToolResult(
                success=True,
                context_text=context_text,
                metadata={
                    "location_name": location_name,
                    "city": default_city,
                    "map_data": map_data_dict,
                }
            )
        else:
            print(f"[WARNING] 未能找到位置信息: {location_name} 在 {default_city}")
            context_text = (
                f"【地图查看】\n"
                f"未能找到'{location_name}'的位置信息。可能原因：\n"
                f"1) 地点名称不准确（请尝试使用完整地名，如'中关村'、'天安门'）\n"
                f"2) API服务暂时不可用（请检查AMAP_API_KEY配置）\n"
                f"3) 该地点暂未在地图系统中"
            )
            return RealtimeToolResult(
                success=False,
                context_text=context_text,
                metadata={
                    "location_name": location_name,
                    "city": default_city,
                    "reason": "api_failed"
                }
            )


# ==================== 全局单例和便捷函数 ====================

_realtime_executor_instance: Optional[RealtimeToolExecutor] = None


def get_realtime_executor() -> RealtimeToolExecutor:
    """获取全局实时工具执行器实例"""
    global _realtime_executor_instance
    if _realtime_executor_instance is None:
        _realtime_executor_instance = RealtimeToolExecutor()
    return _realtime_executor_instance


def execute_realtime_tool(
    tool_selection: ToolSelection,
    query: str,
    **kwargs
) -> Optional[RealtimeToolResult]:
    """
    便捷函数：执行实时工具
    
    Args:
        tool_selection: Tool选择结果
        query: 用户查询
        **kwargs: 额外参数
    
    Returns:
        RealtimeToolResult对象
    """
    executor = get_realtime_executor()
    return executor.execute_tool(tool_selection, query, **kwargs)

