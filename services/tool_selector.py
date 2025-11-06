# -*- coding: utf-8 -*-
"""
工具选择业务逻辑
职责：基于LLM提示词智能决策调用哪个tool
功能：
- 使用LLM进行工具选择
- 处理工具选择结果
- 提供工具选择接口
注意：工具配置信息在 modules/config/tool_config.py

业务层归属：L3 决策层（轻量级决策）
- 当前功能：工具选择决策（基于LLM提示词）
- 当前模式：被动决策（用户查询触发）
- 未来增强：
  - 任务规划与分解
  - 多方案对比与评估
  - 风险评估与预测
  - 多Agent协调机制
"""
from typing import Optional
from dataclasses import dataclass

# 导入底层实现
from modules.generator.prompt import build_tool_selection_prompt
from modules.config.tool_config import (
    get_tool_descriptions,
    tool_to_kb_name,
    get_all_tool_ids,
    get_tool_config,
    ToolID,  # 类型定义
)
from modules.config.settings import LLM_API_BASE_URL


@dataclass
class ToolSelection:
    """Tool选择结果"""
    tool: ToolID  # 工具ID（类型安全），从intent_mapper配置中获取
    kb_name: Optional[str] = None  # 如果是知识库工具，这里是知识库名称
    confidence: float = 1.0        # 置信度
    reasoning: str = ""           # 选择理由


class ToolSelector:
    """基于LLM提示词的Tool选择器"""
    
    def __init__(self):
        """初始化Tool选择器（仅使用LLM）"""
        pass
    
    def select_tool(self, query: str) -> ToolSelection:
        """
        选择适合的tool（使用LLM提示词）
        
        Args:
            query: 用户查询
        
        Returns:
            ToolSelection对象
        """
        llm_selection = self._select_with_llm(query)
        if llm_selection:
            return llm_selection
        
        # 如果LLM调用失败，尝试基于关键词的简单规则匹配
        # 这是一个fallback机制，避免总是返回错误的默认工具
        fallback_tool = self._fallback_tool_selection(query)
        if fallback_tool:
            return fallback_tool
        
        # 如果规则匹配也失败，返回默认值（使用intent_mapper配置）
        default_tool_id = "kb_handbook"
        default_kb_name = tool_to_kb_name(default_tool_id)
        return ToolSelection(
            tool=default_tool_id,
            kb_name=default_kb_name,
            confidence=0.3,
            reasoning="LLM调用失败且规则匹配失败，使用默认handbook知识库"
        )
    
    def _fallback_tool_selection(self, query: str) -> Optional[ToolSelection]:
        """
        基于关键词的简单规则匹配（fallback机制）
        
        Args:
            query: 用户查询
        
        Returns:
            ToolSelection对象，如果匹配不到返回None
        """
        query_lower = query.lower()
        
        # 路径规划关键词
        if any(keyword in query_lower for keyword in ["从", "到", "怎么走", "路线", "路径", "规划", "怎么去"]):
            return ToolSelection(
                tool="route_planning",
                kb_name=None,
                confidence=0.6,
                reasoning="规则匹配：路径规划关键词"
            )
        
        # 路况查询关键词
        if any(keyword in query_lower for keyword in ["路况", "拥堵", "堵吗", "堵车", "这条路"]):
            return ToolSelection(
                tool="realtime_traffic",
                kb_name=None,
                confidence=0.6,
                reasoning="规则匹配：路况查询关键词"
            )
        
        # 地图查看关键词
        if any(keyword in query_lower for keyword in ["地图", "在哪里", "位置", "查看"]):
            return ToolSelection(
                tool="realtime_map",
                kb_name=None,
                confidence=0.6,
                reasoning="规则匹配：地图查看关键词"
            )
        
        # 交通法规关键词
        if any(keyword in query_lower for keyword in ["限行", "处罚", "罚款", "扣分", "记分", "法规", "规定"]):
            return ToolSelection(
                tool="kb_law",
                kb_name="law",
                confidence=0.6,
                reasoning="规则匹配：交通法规关键词"
            )
        
        # 充电相关关键词
        if any(keyword in query_lower for keyword in ["充电", "电桩", "电动车", "电车"]):
            return ToolSelection(
                tool="kb_ev",
                kb_name="ev",
                confidence=0.6,
                reasoning="规则匹配：充电相关关键词"
            )
        
        # 停车相关关键词
        if "停车" in query_lower:
            return ToolSelection(
                tool="kb_parking",
                kb_name="parking",
                confidence=0.6,
                reasoning="规则匹配：停车关键词"
            )
        
        # 公交相关关键词
        if any(keyword in query_lower for keyword in ["公交", "地铁", "换乘"]):
            return ToolSelection(
                tool="kb_transit",
                kb_name="transit",
                confidence=0.6,
                reasoning="规则匹配：公交相关关键词"
            )
        
        return None
    
    def _select_with_llm(self, query: str) -> Optional[ToolSelection]:
        """
        使用LLM提示词智能选择tool
        
        Args:
            query: 用户查询
        
        Returns:
            ToolSelection对象，如果LLM调用失败返回None
        """
        try:
            from adapters.llm import get_llm_client
            
            # 使用modules/generator/prompt.py中的prompt构建函数
            tool_descriptions = get_tool_descriptions()
            prompt = build_tool_selection_prompt(query, tool_descriptions)
            
            llm_client = get_llm_client()
            response = llm_client(prompt, temperature=0.1, max_tokens=50)
            tool_name = response.strip().lower()
            
            # 清理响应：移除可能的说明文字，只保留工具ID
            # 例如："route_planning" 或 "选择的工具：route_planning" → "route_planning"
            if "：" in tool_name or ":" in tool_name:
                parts = tool_name.split("：") if "：" in tool_name else tool_name.split(":")
                tool_name = parts[-1].strip()
            
            # 移除可能的引号
            tool_name = tool_name.strip('"').strip("'").strip()
            
            # 解析tool名称（支持直接匹配和模糊匹配）
            tool_id = self._parse_tool_name(tool_name)
            if tool_id:
                # 使用intent_mapper获取kb_name，不再硬编码
                kb_name = tool_to_kb_name(tool_id)
                
                return ToolSelection(
                    tool=tool_id,
                    kb_name=kb_name,
                    confidence=0.90,
                    reasoning=f"LLM选择：{tool_name}"
                )
            else:
                # 如果解析失败，打印调试信息
                print(f"[WARNING] 无法解析工具名称: '{tool_name}' (原始响应: '{response}')")
                print(f"[WARNING] 可用工具ID: {get_all_tool_ids()}")
        except Exception as e:
            # LLM调用失败，返回None
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"[WARNING] Tool选择器LLM调用失败 [{error_type}]: {error_msg}")
            # 如果是连接错误或服务器错误，提供更具体的提示
            if "ConnectionError" in error_type or "无法连接" in error_msg:
                print(f"  -> 建议: 检查Ollama服务是否在 {LLM_API_BASE_URL} 上运行")
            elif "HTTP" in error_msg or "500" in error_msg:
                print(f"  -> 建议: 检查模型名称是否正确，或Ollama服务是否正常运行")
        
        return None
    
    def _parse_tool_name(self, tool_name: str) -> Optional[ToolID]:
        """
        解析tool名称字符串为tool_id（支持直接匹配和模糊匹配）
        
        Args:
            tool_name: LLM返回的工具名称字符串
        
        Returns:
            tool_id字符串，如果匹配不到返回None
        """
        tool_name = tool_name.strip().lower()
        
        # 移除可能的标点符号和多余空格
        tool_name = tool_name.replace(".", "").replace("。", "").strip()
        
        # 直接匹配
        all_tool_ids = get_all_tool_ids()
        if tool_name in all_tool_ids:
            return tool_name
        
        # 模糊匹配：支持部分匹配
        # 例如："kb_law" 可以匹配 "law"、"kb_law"、"交通法规"等
        tool_id_map = {
            # 路径规划相关
            "route": "route_planning",
            "planning": "route_planning",
            "路径": "route_planning",
            "规划": "route_planning",
            "路线": "route_planning",
            
            # 路况查询相关
            "traffic": "realtime_traffic",
            "路况": "realtime_traffic",
            "拥堵": "realtime_traffic",
            
            # 地图查看相关
            "map": "realtime_map",
            "地图": "realtime_map",
            "位置": "realtime_map",
            
            # 知识库相关
            "law": "kb_law",
            "法规": "kb_law",
            "限行": "kb_law",
            "处罚": "kb_law",
            "罚款": "kb_law",
            
            "ev": "kb_ev",
            "充电": "kb_ev",
            "电桩": "kb_ev",
            "电动车": "kb_ev",
            
            "parking": "kb_parking",
            "停车": "kb_parking",
            
            "transit": "kb_transit",
            "公交": "kb_transit",
            "地铁": "kb_transit",
            
            "handbook": "kb_handbook",
            "配时": "kb_handbook",
            "信号": "kb_handbook",
            
            "iov": "kb_iov",
            "车路": "kb_iov",
            
            "health": "kb_health",
            "健康": "kb_health",
            
            "report": "kb_report",
            "报告": "kb_report",
            
            # 日常对话
            "none": "none",
            "对话": "none",
            "日常": "none",
        }
        
        # 尝试模糊匹配
        for key, tool_id in tool_id_map.items():
            if key in tool_name or tool_name in key:
                if tool_id in all_tool_ids:
                    return tool_id
        
        # 如果还是匹配不到，尝试从工具描述中匹配
        # 检查是否包含工具ID的前缀（如"kb_"、"realtime_"）
        for tool_id in all_tool_ids:
            if tool_id.replace("_", "") in tool_name.replace("_", ""):
                return tool_id
        
        return None


# 全局单例
_tool_selector_instance: Optional[ToolSelector] = None


def get_tool_selector() -> ToolSelector:
    """获取全局Tool选择器实例（仅使用LLM）"""
    global _tool_selector_instance
    if _tool_selector_instance is None:
        _tool_selector_instance = ToolSelector()
    return _tool_selector_instance


def select_tool(query: str, use_llm: bool = True) -> ToolSelection:
    """
    便捷函数：选择适合的tool（使用LLM提示词）
    
    Args:
        query: 用户查询
        use_llm: 是否使用LLM（已废弃，始终使用LLM）
    
    Returns:
        ToolSelection对象
    """
    selector = get_tool_selector()
    return selector.select_tool(query)
