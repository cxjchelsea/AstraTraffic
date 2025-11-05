# -*- coding: utf-8 -*-
"""
业务逻辑模块
职责：实现业务逻辑（不依赖 LangChain 框架）
包含：
- tool_selector: 工具选择
- quality: 质量检查
- fallback: 回退逻辑
- input_handler: 输入处理
- realtime: 实时工具（客户端管理、数据获取、工具执行）
"""
from services.tool_selector import (
    ToolSelector,
    ToolSelection,
    select_tool,
    get_tool_selector,
)
from services.quality import check_hits_quality
from services.fallback import (
    get_no_document_response,
    get_non_kb_response,
    should_fallback,
    check_has_realtime_info,
)
from services.input_handler import get_history_manager
from services.realtime import (
    get_traffic_client,
    get_road_traffic,
    format_traffic_info,
    extract_road_name_from_query,
    get_location_map,
    extract_location_from_query,
    format_map_info_to_dict,
    RealtimeToolExecutor,
    RealtimeToolResult,
    get_realtime_executor,
    execute_realtime_tool,
)

__all__ = [
    "ToolSelector",
    "ToolSelection",
    "select_tool",
    "get_tool_selector",
    "check_hits_quality",
    "get_no_document_response",
    "get_non_kb_response",
    "should_fallback",
    "check_has_realtime_info",
    "get_history_manager",
    "get_traffic_client",
    "get_road_traffic",
    "format_traffic_info",
    "extract_road_name_from_query",
    "get_location_map",
    "extract_location_from_query",
    "format_map_info_to_dict",
    "RealtimeToolExecutor",
    "RealtimeToolResult",
    "get_realtime_executor",
    "execute_realtime_tool",
]

