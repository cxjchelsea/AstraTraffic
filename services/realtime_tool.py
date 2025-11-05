# -*- coding: utf-8 -*-
"""
实时工具执行业务逻辑
职责：统一处理所有实时API工具的调用和执行
功能：
- 执行实时工具（如路况查询）
- 处理工具执行结果
- 格式化工具输出为上下文文本
注意：实际的API调用封装在 modules/realtime/traffic_api.py
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass

from services.tool_selector import ToolSelection
from services.realtime_handler import (
    get_road_traffic,
    format_traffic_info,
    extract_road_name_from_query,
)
from modules.config.settings import AMAP_DEFAULT_CITY
from modules.generator.messages import (
    get_realtime_traffic_prefix,
    get_no_road_name_message,
    get_traffic_api_failed_message,
    get_traffic_success_message,
)


@dataclass
class RealtimeToolResult:
    """实时工具执行结果"""
    success: bool
    context_text: str  # 添加到上下文的文本
    metadata: Dict[str, Any] = None  # 额外的元数据


class RealtimeToolExecutor:
    """实时工具执行器（执行层）"""
    
    def __init__(self):
        """初始化实时工具执行器"""
        # 注册各个实时工具的处理函数（使用tool_id字符串）
        self._tool_handlers = {
            "realtime_traffic": self._handle_traffic_tool,
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
            city: 城市名称（可选，默认北京市）
        
        Returns:
            RealtimeToolResult对象
        """
        # 使用配置中的默认城市
        default_city = city or AMAP_DEFAULT_CITY
        
        # 从查询中提取道路名称
        road_name = extract_road_name_from_query(query)
        
        if not road_name:
            # 无法提取道路名称
            context_text = get_no_road_name_message()
            return RealtimeToolResult(
                success=False,
                context_text=context_text,
                metadata={"reason": "no_road_name"}
            )
        
        # 调用实时路况API
        traffic_info = get_road_traffic(road_name, default_city)
        
        if traffic_info:
            # 成功获取路况信息
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
            # 查询失败
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


# 全局单例
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

