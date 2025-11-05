# -*- coding: utf-8 -*-
"""
工具选择业务逻辑
职责：基于LLM提示词智能决策调用哪个tool
功能：
- 使用LLM进行工具选择
- 处理工具选择结果
- 提供工具选择接口
注意：工具配置信息在 modules/config/tool_config.py
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
        
        # 如果LLM调用失败，返回默认值（使用intent_mapper配置）
        default_tool_id = "kb_handbook"
        default_kb_name = tool_to_kb_name(default_tool_id)
        return ToolSelection(
            tool=default_tool_id,
            kb_name=default_kb_name,
            confidence=0.5,
            reasoning="LLM调用失败，使用默认handbook知识库"
        )
    
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
            
            # 解析tool名称（只做直接匹配，从intent_mapper配置中验证）
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
        解析tool名称字符串为tool_id（只做直接匹配，从配置中验证）
        
        Args:
            tool_name: LLM返回的工具名称字符串
        
        Returns:
            tool_id字符串，如果匹配不到返回None
        """
        tool_name = tool_name.strip().lower()
        
        # 从intent_mapper配置中验证工具是否存在，不再硬编码枚举
        if tool_name in get_all_tool_ids():
            return tool_name
        
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
