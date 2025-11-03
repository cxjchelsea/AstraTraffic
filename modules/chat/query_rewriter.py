# -*- coding: utf-8 -*-
"""
查询改写模块（使用LLM）
基于对话历史将当前查询改写为完整查询
"""
from typing import List, Tuple, Optional, Callable


QUERY_REWRITE_PROMPT_TEMPLATE = """你是一个查询改写助手。根据对话历史，将用户的当前查询改写为完整、清晰的查询语句。

对话历史：
{history_text}

当前查询：{current_query}

要求：
1. 如果当前查询是省略或指代（如"那周末呢？"、"这个路段呢？"），请结合历史对话补充完整
2. 如果当前查询已经完整，可以适当优化表达，但不要改变原意
3. 只输出改写后的完整查询，不要输出其他内容
4. 如果历史对话与当前查询无关，直接返回当前查询（可适当优化）

改写后的完整查询："""


def format_history_for_rewrite(history: List[Tuple[str, str]]) -> str:
    """
    格式化对话历史用于查询改写
    
    Args:
        history: 对话历史列表，格式为 [(用户问题, 助手回答), ...]
    
    Returns:
        格式化后的历史文本
    """
    if not history:
        return "（无历史对话）"
    
    lines = []
    for i, (user_q, assistant_a) in enumerate(history, 1):
        # 截断过长的回答（只保留前100字）
        truncated_answer = assistant_a[:100] + "..." if len(assistant_a) > 100 else assistant_a
        lines.append(f"轮次{i}:")
        lines.append(f"  用户：{user_q}")
        lines.append(f"  助手：{truncated_answer}")
    
    return "\n".join(lines)


def rewrite_query_with_history(
    current_query: str,
    history: List[Tuple[str, str]],
    llm_client: Callable[[str, float, int], str],
    max_history_turns: int = 3
) -> str:
    """
    基于对话历史改写当前查询
    
    Args:
        current_query: 当前用户查询
        history: 对话历史列表
        llm_client: LLM客户端函数，签名: (prompt: str, temperature: float, max_tokens: int) -> str
        max_history_turns: 最多使用多少轮历史（默认3轮）
    
    Returns:
        改写后的完整查询
    """
    # 如果没有历史，直接返回原查询（可做简单优化）
    if not history:
        return current_query.strip()
    
    # 只使用最近的几轮历史
    recent_history = history[-max_history_turns:] if len(history) > max_history_turns else history
    
    # 格式化历史
    history_text = format_history_for_rewrite(recent_history)
    
    # 构建prompt
    prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(
        history_text=history_text,
        current_query=current_query
    )
    
    try:
        # 调用LLM进行改写
        rewritten = llm_client(prompt, temperature=0.3, max_tokens=200)
        
        # 清理输出
        rewritten = rewritten.strip()
        
        # 如果改写结果为空或异常，返回原查询
        if not rewritten or len(rewritten) < len(current_query) * 0.5:
            return current_query.strip()
        
        return rewritten
    except Exception as e:
        # LLM调用失败时，返回原查询
        print(f"[WARNING] 查询改写失败: {e}, 使用原查询")
        return current_query.strip()


def rewrite_query_with_session(
    current_query: str,
    session_id: str,
    history_manager,
    llm_client: Callable[[str, float, int], str],
    max_history_turns: int = 3
) -> str:
    """
    基于会话ID改写查询（便捷函数）
    
    Args:
        current_query: 当前用户查询
        session_id: 会话ID
        history_manager: 历史管理器实例
        llm_client: LLM客户端函数
        max_history_turns: 最多使用多少轮历史
    
    Returns:
        改写后的完整查询
    """
    history = history_manager.get_history(session_id, max_turns=max_history_turns)
    return rewrite_query_with_history(current_query, history, llm_client, max_history_turns)

