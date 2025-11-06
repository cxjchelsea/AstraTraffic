# -*- coding: utf-8 -*-
"""
工具配置管理（底层实现）
职责：统一管理所有工具配置信息
包含：
- 工具ID定义（类型安全）
- 工具配置信息（描述、意图标签、知识库名称等）
- 工具配置查询接口
注意：这是配置信息，属于底层实现，不依赖适配层
"""
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass

# 工具ID类型定义（从配置中自动生成，确保类型安全）
ToolID = Literal[
    "kb_law",
    "kb_handbook",
    "kb_transit",
    "kb_parking",
    "kb_ev",
    "kb_iov",
    "kb_health",
    "kb_report",
    "realtime_traffic",
    "realtime_map",
    "route_planning",
    "none",
]


@dataclass
class ToolConfig:
    """工具配置信息"""
    tool_id: str  # 工具ID（如 "kb_law"）
    intent_label: str  # 意图标签（如 "交通法规"）
    description: str  # 工具描述（用于prompt）
    kb_name: Optional[str] = None  # 知识库名称（如果不是KB工具则为None）
    category: str = "kb"  # 工具类别：kb, realtime, special


# 工具配置定义（统一管理，不再硬编码在枚举中）
_TOOL_CONFIGS: Dict[str, ToolConfig] = {
    "kb_law": ToolConfig(
        tool_id="kb_law",
        intent_label="交通法规",
        description="交通法规知识库（限行、处罚、罚款、记分等）",
        kb_name="law",
        category="kb",
    ),
    "kb_handbook": ToolConfig(
        tool_id="kb_handbook",
        intent_label="信号配时",
        description="操作手册知识库（信号配时、诱导策略等）",
        kb_name="handbook",
        category="kb",
    ),
    "kb_transit": ToolConfig(
        tool_id="kb_transit",
        intent_label="公交规则",
        description="公共交通知识库（公交、地铁、换乘等）",
        kb_name="transit",
        category="kb",
    ),
    "kb_parking": ToolConfig(
        tool_id="kb_parking",
        intent_label="停车政策",
        description="停车政策知识库（停车、泊位等）",
        kb_name="parking",
        category="kb",
    ),
    "kb_ev": ToolConfig(
        tool_id="kb_ev",
        intent_label="充电规范",
        description="电动汽车知识库（充电、电桩等）",
        kb_name="ev",
        category="kb",
    ),
    "kb_iov": ToolConfig(
        tool_id="kb_iov",
        intent_label="车路协同",
        description="车路协同知识库（车路协同、V2X等）",
        kb_name="iov",
        category="kb",
    ),
    "kb_health": ToolConfig(
        tool_id="kb_health",
        intent_label="健康咨询",
        description="健康知识库（健康、医疗等）",
        kb_name="health",
        category="kb",
    ),
    "kb_report": ToolConfig(
        tool_id="kb_report",
        intent_label="报告查询",
        description="报告知识库（报告、分析等）",
        kb_name="report",
        category="kb",
    ),
    "realtime_traffic": ToolConfig(
        tool_id="realtime_traffic",
        intent_label="路况查询",
        description="实时路况API（查询当前道路拥堵情况）",
        kb_name=None,
        category="realtime",
    ),
    "realtime_map": ToolConfig(
        tool_id="realtime_map",
        intent_label="地图查看",
        description="实时地图查看API（查看指定位置的地图、路况、POI等）",
        kb_name=None,
        category="realtime",
    ),
    "route_planning": ToolConfig(
        tool_id="route_planning",
        intent_label="路径规划",
        description="路径规划API（规划从起点到终点的出行路线，支持驾车、步行、骑行等）",
        kb_name=None,
        category="realtime",
    ),
    "none": ToolConfig(
        tool_id="none",
        intent_label="日常对话",
        description="日常对话工具（用于问候语、简单对话等，不需要知识库和实时工具，直接由LLM友好回答）",
        kb_name=None,
        category="special",
    ),
}


def get_tool_config(tool_id: ToolID) -> Optional[ToolConfig]:
    """
    获取工具配置
    
    Args:
        tool_id: 工具ID
    
    Returns:
        ToolConfig对象，如果不存在返回None
    """
    return _TOOL_CONFIGS.get(tool_id)


def get_all_tool_ids() -> List[ToolID]:
    """
    获取所有工具ID列表
    
    Returns:
        List[str]: 工具ID列表
    """
    return list(_TOOL_CONFIGS.keys())


def get_tool_to_intent_map() -> Dict[str, str]:
    """
    获取Tool类型到意图标签的映射
    
    Returns:
        Dict[str, str]: Tool类型值到意图标签的映射
    """
    return {tool_id: config.intent_label for tool_id, config in _TOOL_CONFIGS.items()}


def get_tool_descriptions() -> List[Tuple[str, str]]:
    """
    获取所有工具的描述列表，用于构建tool selection prompt
    
    Returns:
        List[Tuple[str, str]]: (工具名称, 工具描述) 的列表
    """
    # 包含所有工具，包括none（用于日常对话）
    return [
        (config.tool_id, config.description)
        for tool_id, config in _TOOL_CONFIGS.items()
    ]


def get_kb_intent_labels() -> list[str]:
    """
    获取所有知识库意图标签列表（用于判断是否需要KB）
    
    Returns:
        list[str]: 知识库意图标签列表
    """
    # 从配置中获取所有KB工具的意图标签
    return [
        config.intent_label
        for config in _TOOL_CONFIGS.values()
        if config.category == "kb"
    ]


def tool_to_intent_label(tool_id: ToolID, default: str = "未知") -> str:
    """
    将工具ID转换为意图标签
    
    Args:
        tool_id: 工具ID
        default: 默认意图标签
    
    Returns:
        str: 意图标签
    """
    config = get_tool_config(tool_id)
    return config.intent_label if config else default


def get_tool_to_kb_name_map() -> Dict[str, Optional[str]]:
    """
    获取Tool类型到知识库名称的映射
    
    Returns:
        Dict[str, Optional[str]]: Tool类型值到知识库名称的映射（非KB工具为None）
    """
    return {tool_id: config.kb_name for tool_id, config in _TOOL_CONFIGS.items()}


def tool_to_kb_name(tool_id: ToolID) -> Optional[str]:
    """
    将工具ID转换为知识库名称
    
    Args:
        tool_id: 工具ID
    
    Returns:
        Optional[str]: 知识库名称，如果不是KB工具则返回None
    """
    config = get_tool_config(tool_id)
    return config.kb_name if config else None

