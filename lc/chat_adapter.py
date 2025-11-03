# -*- coding: utf-8 -*-
"""
对话管理适配器
将底层 modules/chat 模块适配为 LangChain 使用
"""
from typing import List, Tuple

# 导入底层实现
from modules.chat.history import ChatHistoryManager, get_history_manager as _get_history_manager
from modules.chat.query_rewriter import rewrite_query_with_history as _rewrite_query_with_history
from lc.llm_adapter import get_llm_client


def get_history_manager() -> ChatHistoryManager:
    """获取全局历史管理器实例（适配层）"""
    return _get_history_manager()


def rewrite_query_with_history(
    current_query: str,
    history: List[Tuple[str, str]],
    max_history_turns: int = 3
) -> str:
    """
    基于对话历史改写当前查询（适配层，自动获取LLM客户端）
    
    Args:
        current_query: 当前用户查询
        history: 对话历史列表
        max_history_turns: 最多使用多少轮历史（默认3轮）
    
    Returns:
        改写后的完整查询
    """
    llm_client = get_llm_client()
    return _rewrite_query_with_history(current_query, history, llm_client, max_history_turns)

