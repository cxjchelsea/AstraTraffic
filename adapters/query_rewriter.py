# -*- coding: utf-8 -*-
"""
查询改写适配器（LangChain接口适配）
职责：将底层查询改写逻辑适配为适配层接口
"""
from typing import List, Tuple
from modules.generator.query_rewriter import rewrite_query_with_history as _rewrite_query_with_history
from adapters.llm import get_llm_client


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

